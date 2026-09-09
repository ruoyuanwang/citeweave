param(
    [int]$PollSeconds = 60,
    [int]$MaximumTransportRounds = 10,
    [int]$MaximumParseAttempts = 3
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$integrityHold = Join-Path $workspaceRoot "experiments\graph_discovery_v2\formal_v3_author_identity_integrity_hold.json"
if (Test-Path -LiteralPath $integrityHold) {
    throw "Formal execution is on author-identity integrity hold. Corrected data, prospective amendment and promotion receipt are required before this hold can be archived. See $integrityHold"
}
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$apiKeyFile = (Resolve-Path ".\apikey.md").Path
$statusPath = Join-Path $workspaceRoot "experiments\runs\formal_v3_confirmatory_pipeline_status.json"

function Write-PipelineStatus {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )
    $payload = [ordered]@{
        schema_version = 1
        status = $Status
        stage = $Stage
        message = $Message
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        maximum_transport_rounds = $MaximumTransportRounds
        maximum_parse_attempts = $MaximumParseAttempts
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    Write-Output "[$([DateTime]::UtcNow.ToString('o'))] python $($Arguments -join ' ')"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Wait-ForReadiness {
    $paths = @(
        "experiments\graph_discovery_v2\formal_v3_readiness.json",
        "experiments\graph_discovery_v2\formal_v3_replication_readiness.json",
        "experiments\graph_discovery_v2\formal_v3_complexity_extension_readiness.json"
    )
    while ($true) {
        $states = @()
        foreach ($path in $paths) {
            if (Test-Path -LiteralPath $path) {
                $payload = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
                $states += [pscustomobject]@{ path = $path; status = $payload.status }
            } else {
                $states += [pscustomobject]@{ path = $path; status = "missing" }
            }
        }
        $summary = ($states | ForEach-Object { "$($_.path)=$($_.status)" }) -join "; "
        Write-PipelineStatus -Status "waiting" -Stage "readiness" -Message $summary
        Write-Output "[$([DateTime]::UtcNow.ToString('o'))] $summary"
        if (@($states | Where-Object { $_.status -ne "ready" }).Count -eq 0) {
            return
        }
        $builder = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*build_neural_dense_sidecars.py*neural_dense_semantic_v2*"
        }
        if (-not $builder) {
            Start-Sleep -Seconds 10
            $states = foreach ($path in $paths) {
                if (Test-Path -LiteralPath $path) {
                    Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
                } else {
                    [pscustomobject]@{ status = "missing"; blocking_reasons = @("missing") }
                }
            }
            if (@($states | Where-Object { $_.status -ne "ready" }).Count -gt 0) {
                $reasons = ($states | ForEach-Object { $_.blocking_reasons }) -join "; "
                throw "Neural prerequisite process ended without ready audits: $reasons"
            }
            return
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Invoke-PanelToTerminal {
    param(
        [string]$PanelName,
        [string[]]$RunnerArguments,
        [string]$PlanPath,
        [string]$AuditPath
    )
    for ($round = 1; $round -le $MaximumTransportRounds; $round++) {
        Write-PipelineStatus -Status "running" -Stage $PanelName -Message "execution round $round"
        Write-Output "[$([DateTime]::UtcNow.ToString('o'))] $PanelName round $round"
        & $python @RunnerArguments
        $runnerExit = $LASTEXITCODE
        if ($runnerExit -ne 0) {
            Write-Output "[$([DateTime]::UtcNow.ToString('o'))] $PanelName transport/process exit $runnerExit"
            if ($round -eq $MaximumTransportRounds) {
                throw "$PanelName exhausted $MaximumTransportRounds transport rounds"
            }
            Start-Sleep -Seconds ([Math]::Min(60, 5 * $round))
            continue
        }
        Invoke-CheckedPython @(
            "scripts\audit_formal_panel_terminal.py",
            "--plan", $PlanPath,
            "--maximum-parse-attempts", [string]$MaximumParseAttempts,
            "--output", $AuditPath
        )
        $audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
        if ($audit.status -eq "terminal") {
            Write-Output "[$([DateTime]::UtcNow.ToString('o'))] $PanelName terminal: complete=$($audit.complete_cells), terminal_parse_failures=$($audit.terminal_parse_failure_cells)"
            return
        }
        if ($audit.status -ne "needs_retry") {
            throw "$PanelName terminal audit status is $($audit.status)"
        }
        Write-Output "[$([DateTime]::UtcNow.ToString('o'))] $PanelName retry cells=$($audit.retry_cells.Count)"
    }
    throw "$PanelName did not reach a terminal state"
}

try {
    Write-PipelineStatus -Status "waiting" -Stage "readiness" -Message "waiting for all frozen prerequisites"
    Wait-ForReadiness

    $primaryPlan = "experiments\graph_discovery_v2\formal_v3_panel_plan.json"
    $replicationPlan = "experiments\graph_discovery_v2\formal_v3_replication_panel_plan.json"
    $extensionPlan = "experiments\graph_discovery_v2\formal_v3_complexity_extension_panel_plan.json"

    Invoke-CheckedPython @(
        "scripts\run_formal_v3_panel.py",
        "--protocol", "experiments\graph_discovery_v2\formal_v3_protocol.yml",
        "--neural-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense.yml",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
        "--results-root", "experiments\graph_discovery_v2\formal_v3_runs",
        "--readiness", "experiments\graph_discovery_v2\formal_v3_readiness.json",
        "--output-plan", $primaryPlan
    )
    Invoke-CheckedPython @(
        "scripts\run_formal_v3_replication_panel.py",
        "--protocol", "experiments\graph_discovery_v2\formal_v3_replication_protocol.yml",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_replication_benchmarks",
        "--results-root", "experiments\graph_discovery_v2\formal_v3_replication_runs",
        "--readiness", "experiments\graph_discovery_v2\formal_v3_replication_readiness.json",
        "--output-plan", $replicationPlan
    )
    Invoke-CheckedPython @(
        "scripts\run_formal_v3_complexity_extension_panel.py",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
        "--execution-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_runs",
        "--readiness", "experiments\graph_discovery_v2\formal_v3_complexity_extension_readiness.json",
        "--output-plan", $extensionPlan
    )
    $primary = Get-Content -LiteralPath $primaryPlan -Raw | ConvertFrom-Json
    $replication = Get-Content -LiteralPath $replicationPlan -Raw | ConvertFrom-Json
    $extension = Get-Content -LiteralPath $extensionPlan -Raw | ConvertFrom-Json
    if ($primary.status -ne "ready_to_execute" -or $primary.calls -ne 800) {
        throw "Primary plan must be ready with exactly 800 calls"
    }
    if ($replication.status -ne "ready_to_execute" -or $replication.calls -ne 120) {
        throw "Replication plan must be ready with exactly 120 calls"
    }
    if ($extension.status -ne "ready_to_execute" -or $extension.incremental_calls -ne 312) {
        throw "Extension plan must be ready with exactly 312 incremental calls"
    }

    Invoke-PanelToTerminal -PanelName "primary" -PlanPath $primaryPlan `
        -AuditPath "experiments\graph_discovery_v2\formal_v3_primary_terminal_audit.json" `
        -RunnerArguments @(
            "scripts\run_formal_v3_panel.py",
            "--protocol", "experiments\graph_discovery_v2\formal_v3_protocol.yml",
            "--neural-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense.yml",
            "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
            "--results-root", "experiments\graph_discovery_v2\formal_v3_runs",
            "--readiness", "experiments\graph_discovery_v2\formal_v3_readiness.json",
            "--api-key-file", $apiKeyFile,
            "--execute"
        )
    Invoke-PanelToTerminal -PanelName "replication" -PlanPath $replicationPlan `
        -AuditPath "experiments\graph_discovery_v2\formal_v3_replication_terminal_audit.json" `
        -RunnerArguments @(
            "scripts\run_formal_v3_replication_panel.py",
            "--protocol", "experiments\graph_discovery_v2\formal_v3_replication_protocol.yml",
            "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_replication_benchmarks",
            "--results-root", "experiments\graph_discovery_v2\formal_v3_replication_runs",
            "--readiness", "experiments\graph_discovery_v2\formal_v3_replication_readiness.json",
            "--api-key-file", $apiKeyFile,
            "--execute"
        )
    Invoke-PanelToTerminal -PanelName "complexity_extension_incremental" -PlanPath $extensionPlan `
        -AuditPath "experiments\graph_discovery_v2\formal_v3_complexity_extension_terminal_audit.json" `
        -RunnerArguments @(
            "scripts\run_formal_v3_complexity_extension_panel.py",
            "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
            "--execution-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_runs",
            "--readiness", "experiments\graph_discovery_v2\formal_v3_complexity_extension_readiness.json",
            "--api-key-file", $apiKeyFile,
            "--execute"
        )

    Write-PipelineStatus -Status "running" -Stage "merge_and_analysis" -Message "merging 672 cells and running frozen analyses"
    Invoke-CheckedPython @(
        "scripts\merge_formal_v3_complexity_extension_results.py",
        "--protocol", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol.yml",
        "--protocol-freeze", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol_freeze.json",
        "--execution-amendment", "experiments\graph_discovery_v2\formal_v3_complexity_extension_amendment_001_execution.yml",
        "--execution-amendment-freeze", "experiments\graph_discovery_v2\formal_v3_complexity_extension_amendment_001_execution_freeze.json",
        "--readiness", "experiments\graph_discovery_v2\formal_v3_complexity_extension_readiness.json",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
        "--primary-benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
        "--primary-results-root", "experiments\graph_discovery_v2\formal_v3_runs",
        "--replication-benchmark-root", "experiments\graph_discovery_v2\formal_v3_replication_benchmarks",
        "--replication-results-root", "experiments\graph_discovery_v2\formal_v3_replication_runs",
        "--incremental-results-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_runs",
        "--output-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_merged"
    )
    Invoke-CheckedPython @(
        "scripts\analyze_formal_v3_results.py",
        "--protocol", "experiments\graph_discovery_v2\formal_v3_protocol.yml",
        "--neural-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense.yml",
        "--statistics-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_003_statistics.yml",
        "--scale-pairs", "experiments\graph_discovery_v2\formal_v3_scale_pairs.json",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
        "--results-root", "experiments\graph_discovery_v2\formal_v3_runs",
        "--output", "experiments\graph_discovery_v2\formal_v3_analysis.json"
    )
    Invoke-CheckedPython @(
        "scripts\analyze_formal_v3_scale_replication.py",
        "--primary-benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
        "--primary-results-root", "experiments\graph_discovery_v2\formal_v3_runs",
        "--primary-scale-pairs", "experiments\graph_discovery_v2\formal_v3_scale_pairs.json",
        "--replication-benchmark-root", "experiments\graph_discovery_v2\formal_v3_replication_benchmarks",
        "--replication-results-root", "experiments\graph_discovery_v2\formal_v3_replication_runs",
        "--replication-scale-pairs", "experiments\graph_discovery_v2\formal_v3_replication_scale_pairs.json",
        "--output", "experiments\graph_discovery_v2\formal_v3_scale_replication_analysis.json"
    )
    Invoke-CheckedPython @(
        "scripts\analyze_formal_v3_complexity_extension.py",
        "--protocol", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol.yml",
        "--freeze", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol_freeze.json",
        "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
        "--results-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_merged",
        "--output", "experiments\graph_discovery_v2\formal_v3_complexity_extension_analysis.json"
    )
    Write-PipelineStatus -Status "complete" -Stage "complete" -Message "all panels terminal, merged, and analyzed"
} catch {
    Write-PipelineStatus -Status "failed" -Stage "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
