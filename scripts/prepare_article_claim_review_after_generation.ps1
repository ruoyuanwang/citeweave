param(
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$machineStatusPath = "experiments\runs\machine_article_generation_status.json"
$statusPath = "experiments\runs\article_claim_review_preparation_status.json"
$planPath = "experiments\article_quality_v2\machine_generation_v1\plan.json"
$reviewRoot = "experiments\article_quality_v2\article_claim_review_v1"
$rosterPath = Join-Path $reviewRoot "reviewer_roster.json"
$readinessPath = Join-Path $reviewRoot "readiness.json"
$protocolPath = "experiments\article_quality_v2\article_claim_review_protocol.yml"
$freezePath = "experiments\article_quality_v2\article_claim_review_protocol_freeze.json"

function Write-PreparationStatus {
    param([string]$Status, [string]$Stage, [string]$Message)
    [ordered]@{
        schema_version = 1
        status = $Status
        stage = $Stage
        message = $Message
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    while ($true) {
        if (-not (Test-Path -LiteralPath $machineStatusPath)) {
            Write-PreparationStatus -Status "waiting" -Stage "machine_generation" -Message "machine generation status missing"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $machine = Get-Content -LiteralPath $machineStatusPath -Raw | ConvertFrom-Json
        if ($machine.status -eq "failed") {
            throw "Machine article generation failed: $($machine.message)"
        }
        if ($machine.status -ne "complete") {
            Write-PreparationStatus -Status "waiting" -Stage "machine_generation" -Message "machine status=$($machine.status), stage=$($machine.stage)"
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        & $python "scripts\prepare_article_claim_review.py" `
            "--machine-plan" $planPath `
            "--roster" $rosterPath `
            "--output-root" $reviewRoot `
            "--protocol" $protocolPath `
            "--protocol-freeze" $freezePath `
            "--audit" `
            "--readiness-output" $readinessPath
        if ($LASTEXITCODE -ne 0) {
            throw "Article claim review readiness audit failed"
        }
        $readiness = Get-Content -LiteralPath $readinessPath -Raw | ConvertFrom-Json
        if ($readiness.status -eq "ready") {
            Write-PreparationStatus -Status "running" -Stage "packet_build" -Message "building 160 claim packets and 320 primary assignments"
            & $python "scripts\prepare_article_claim_review.py" `
                "--machine-plan" $planPath `
                "--roster" $rosterPath `
                "--output-root" $reviewRoot `
                "--protocol" $protocolPath `
                "--protocol-freeze" $freezePath
            if ($LASTEXITCODE -ne 0) {
                throw "Article claim packet build failed"
            }
            $manifest = Get-Content -LiteralPath (Join-Path $reviewRoot "manifest.json") -Raw | ConvertFrom-Json
            if ($manifest.packets -ne 160 -or $manifest.primary_reviews_required -ne 320) {
                throw "Article claim packet cardinality mismatch"
            }
            Write-PreparationStatus -Status "complete" -Stage "packet_build" -Message "160 claim packets ready for 320 primary reviews; adjudication remains disagreement-only"
            exit 0
        }
        if ($readiness.missing_drafts.Count -gt 0 -or $readiness.invalid_drafts.Count -gt 0) {
            Write-PreparationStatus -Status "blocked" -Stage "draft_validation" -Message ($readiness.blocking_reasons -join "; ")
            exit 1
        }
        Write-PreparationStatus -Status "waiting" -Stage "real_reviewer_roster" -Message ($readiness.blocking_reasons -join "; ")
        Start-Sleep -Seconds $PollSeconds
    }
} catch {
    Write-PreparationStatus -Status "failed" -Stage "failed" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
