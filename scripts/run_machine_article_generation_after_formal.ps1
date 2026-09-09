param(
    [int]$PollSeconds = 60,
    [int]$MaximumTransportRounds = 10
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$apiKeyFile = (Resolve-Path ".\apikey.md").Path
$formalStatusPath = Join-Path $workspaceRoot "experiments\runs\formal_v3_confirmatory_pipeline_status.json"
$articleStatusPath = Join-Path $workspaceRoot "experiments\runs\machine_article_generation_status.json"
$planPath = "experiments\article_quality_v2\machine_generation_v1\plan.json"
$protocolPath = "experiments\article_quality_v2\machine_article_generation_protocol_v2.yml"
$freezePath = "experiments\article_quality_v2\machine_article_generation_protocol_v2_freeze.json"

function Write-ArticleStatus {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )
    [ordered]@{
        schema_version = 1
        status = $Status
        stage = $Stage
        message = $Message
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        maximum_transport_rounds = $MaximumTransportRounds
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $articleStatusPath -Encoding UTF8
}

function Get-ArticleTerminalCount {
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    $terminal = 0
    foreach ($cell in $plan.cells) {
        $recordPath = Join-Path $cell.output_dir "execution_record.json"
        if (Test-Path -LiteralPath $recordPath) {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            if ($record.request_sha256 -ne $cell.request_sha256) {
                throw "Article execution record request mismatch: $($cell.cell_id)"
            }
            if ($record.response_received -eq $true) {
                $terminal += 1
            }
        }
    }
    return $terminal
}

try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $formalStatusPath)) {
            Write-ArticleStatus -Status "waiting" -Stage "formal_graph_experiment" -Message "formal supervisor status missing"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $formal = Get-Content -LiteralPath $formalStatusPath -Raw | ConvertFrom-Json
        if ($formal.status -eq "complete") {
            break
        }
        Write-ArticleStatus -Status "waiting" -Stage "formal_graph_experiment" -Message "formal status=$($formal.status), stage=$($formal.stage)"
        Start-Sleep -Seconds $PollSeconds
    }

    for ($round = 1; $round -le $MaximumTransportRounds; $round++) {
        $before = Get-ArticleTerminalCount
        Write-ArticleStatus -Status "running" -Stage "machine_drafts" -Message "round=$round, terminal_before=$before/16"
        & $python "scripts\run_machine_article_generation.py" `
            "--plan" $planPath `
            "--protocol" $protocolPath `
            "--protocol-freeze" $freezePath `
            "--api-key-file" $apiKeyFile `
            "--execute"
        $runnerExit = $LASTEXITCODE
        $terminal = Get-ArticleTerminalCount
        if ($terminal -eq 16) {
            Write-ArticleStatus -Status "complete" -Stage "machine_drafts" -Message "16/16 provider responses terminal; CiteWeave drafts still require real human review"
            exit 0
        }
        if ($runnerExit -ne 0) {
            Write-Output "[$([DateTime]::UtcNow.ToString('o'))] article runner exit=$runnerExit, terminal=$terminal/16"
        }
        if ($round -lt $MaximumTransportRounds) {
            Start-Sleep -Seconds ([Math]::Min(60, 5 * $round))
        }
    }
    throw "Machine article generation exhausted transport rounds before 16 terminal responses"
} catch {
    Write-ArticleStatus -Status "failed" -Stage "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
