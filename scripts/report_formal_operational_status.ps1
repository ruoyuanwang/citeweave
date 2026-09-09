param()

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$base = Join-Path $workspaceRoot "experiments\graph_discovery_v2"
$runRoot = Join-Path $workspaceRoot "experiments\runs"

function Read-JsonIfPresent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-HashIfReadable {
    param([string]$Path)
    try {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
    } catch {
        return $null
    }
}

function Get-PanelMetadata {
    param(
        [string]$Name,
        [string]$Root,
        [int]$Planned
    )
    $complete = 0
    $failedParse = 0
    $attemptLog = 0
    $unexpectedStatus = 0
    $resultFiles = @()
    if (Test-Path -LiteralPath $Root) {
        $resultFiles = @(Get-ChildItem -LiteralPath $Root -Filter "results.json" -Recurse -File)
        foreach ($file in $resultFiles) {
            $payload = Read-JsonIfPresent -Path $file.FullName
            foreach ($record in @($payload.records)) {
                if ($record.status -eq "complete") {
                    $complete += 1
                } elseif ($record.status -eq "failed_parse") {
                    $failedParse += 1
                } else {
                    $unexpectedStatus += 1
                }
            }
            $attemptLog += @($payload.attempt_log).Count
        }
    }
    return [ordered]@{
        panel = $Name
        planned_cells = $Planned
        complete_cells = $complete
        remaining_cells = [Math]::Max(0, $Planned - $complete)
        failed_parse_cells = $failedParse
        attempt_log_records = $attemptLog
        unexpected_status_records = $unexpectedStatus
        result_files = $resultFiles.Count
        result_file_sha256 = @(
            $resultFiles | ForEach-Object {
                [ordered]@{
                    path = $_.FullName
                    sha256 = Get-HashIfReadable -Path $_.FullName
                }
            }
        )
    }
}

function Get-ProcessMatches {
    param([string]$Pattern)
    return @(
        Get-CimInstance Win32_Process |
            Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $Pattern } |
            ForEach-Object {
                [ordered]@{
                    process_id = $_.ProcessId
                    parent_process_id = $_.ParentProcessId
                    name = $_.Name
                    creation_date = $_.CreationDate
                }
            }
    )
}

function Get-ErrorLogMetadata {
    param([string]$Pattern)
    return @(
        Get-ChildItem -LiteralPath $runRoot -Filter $Pattern -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 3 |
            ForEach-Object {
                [ordered]@{
                    path = $_.FullName
                    bytes = $_.Length
                    last_write_time = $_.LastWriteTime.ToString("o")
                    sha256 = Get-HashIfReadable -Path $_.FullName
                }
            }
    )
}

$pipeline = Read-JsonIfPresent -Path (Join-Path $runRoot "formal_v3_confirmatory_pipeline_status.json")
$machine = Read-JsonIfPresent -Path (Join-Path $runRoot "machine_article_generation_status.json")
$claim = Read-JsonIfPresent -Path (Join-Path $runRoot "article_claim_review_preparation_status.json")

$articlePlanPath = Join-Path $workspaceRoot "experiments\article_quality_v2\machine_generation_v1\plan.json"
$articleTerminal = 0
$articlePlanCells = 0
if (Test-Path -LiteralPath $articlePlanPath) {
    $articlePlan = Read-JsonIfPresent -Path $articlePlanPath
    $articlePlanCells = @($articlePlan.cells).Count
    foreach ($cell in @($articlePlan.cells)) {
        $recordPath = Join-Path $cell.output_dir "execution_record.json"
        $record = Read-JsonIfPresent -Path $recordPath
        if ($null -ne $record -and $record.response_received -eq $true) {
            $articleTerminal += 1
        }
    }
}

$terminalAudits = @()
foreach ($name in @(
    "formal_v3_primary_terminal_audit.json",
    "formal_v3_replication_terminal_audit.json",
    "formal_v3_complexity_extension_terminal_audit.json"
)) {
    $path = Join-Path $base $name
    $audit = Read-JsonIfPresent -Path $path
    $terminalAudits += [ordered]@{
        name = $name
        present = $null -ne $audit
        status = if ($null -ne $audit) { $audit.status } else { $null }
        complete_cells = if ($null -ne $audit) { $audit.complete_cells } else { $null }
        terminal_parse_failure_cells = if ($null -ne $audit) { $audit.terminal_parse_failure_cells } else { $null }
        sha256 = if ($null -ne $audit) {
            Get-HashIfReadable -Path $path
        } else { $null }
    }
}

$result = [ordered]@{
    schema_version = 1
    report_scope = "operational_metadata_only_no_scientific_effects"
    observed_at_local = (Get-Date).ToString("o")
    pipeline = [ordered]@{
        status = $pipeline.status
        stage = $pipeline.stage
        updated_at_utc = $pipeline.updated_at_utc
    }
    processes = [ordered]@{
        formal_controller = Get-ProcessMatches -Pattern "run_identity_corrected_formal_pipeline.ps1"
        machine_article_watcher = Get-ProcessMatches -Pattern "run_machine_article_generation_after_formal.ps1"
        claim_review_watcher = Get-ProcessMatches -Pattern "prepare_article_claim_review_after_generation.ps1"
    }
    panels = @(
        Get-PanelMetadata -Name "primary" -Root (Join-Path $base "formal_v3_runs") -Planned 800
        Get-PanelMetadata -Name "replication" -Root (Join-Path $base "formal_v3_replication_runs") -Planned 120
        Get-PanelMetadata -Name "extension_incremental" -Root (Join-Path $base "formal_v3_complexity_extension_runs") -Planned 312
    )
    terminal_audits = $terminalAudits
    article_pipeline = [ordered]@{
        machine_status = $machine.status
        machine_stage = $machine.stage
        terminal_responses = $articleTerminal
        planned_responses = $articlePlanCells
        claim_preparation_status = $claim.status
        claim_preparation_stage = $claim.stage
    }
    error_logs = [ordered]@{
        formal = Get-ErrorLogMetadata -Pattern "formal_identity_corrected_*resume_*.stderr.log"
        machine_article = Get-ErrorLogMetadata -Pattern "machine_article_watcher_*.stderr.log"
        claim_review = Get-ErrorLogMetadata -Pattern "claim_review_watcher_*.stderr.log"
    }
}

$result | ConvertTo-Json -Depth 8
