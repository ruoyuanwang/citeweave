param(
    [datetime]$ResumeAfterUtc = [datetime]::UtcNow.Date.AddDays(1).AddSeconds(30)
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$registry = Join-Path $projectRoot "experiments\formal_datasets_openalex_title_abstract.yml"
$runDirectory = Join-Path $projectRoot "experiments\runs"
$stdout = $null
$stderr = $null
$downstreamStdout = Join-Path $runDirectory "remaining_formal_experiments_v1.stdout.log"
$downstreamStderr = Join-Path $runDirectory "remaining_formal_experiments_v1.stderr.log"
$status = Join-Path $runDirectory "openalex_title_abstract_harvest_v4_after_reset.status.json"
$attempts = @()

if (-not (Test-Path -LiteralPath $python)) {
    throw "Workspace Python executable is missing: $python"
}
if (-not (Test-Path -LiteralPath $registry)) {
    throw "Frozen registry is missing: $registry"
}
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

function Wait-UntilUtc {
    param([datetime]$TargetUtc)
    while ([datetime]::UtcNow -lt $TargetUtc) {
        Start-Sleep -Seconds 60
    }
}

$apiKey = [Environment]::GetEnvironmentVariable("OPENALEX_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "OPENALEX_API_KEY is not available in the Windows User environment."
}
$env:OPENALEX_API_KEY = $apiKey

$startedAt = [datetime]::UtcNow.ToString("o")
$nextAttemptUtc = $ResumeAfterUtc
$harvestExitCode = $null
$downstreamExitCode = $null
$attemptNumber = 0
Push-Location $projectRoot
try {
    while ($true) {
        Wait-UntilUtc -TargetUtc $nextAttemptUtc
        $attemptNumber += 1
        $attemptTag = "{0:D2}" -f $attemptNumber
        $stdout = Join-Path $runDirectory (
            "openalex_title_abstract_harvest_v4_after_reset.attempt_$attemptTag.stdout.log"
        )
        $stderr = Join-Path $runDirectory (
            "openalex_title_abstract_harvest_v4_after_reset.attempt_$attemptTag.stderr.log"
        )
        $attemptStartedAt = [datetime]::UtcNow.ToString("o")
        & $python "scripts\run_formal_pipeline.py" `
            "--registry" $registry `
            "--all-datasets" `
            "--stage" "harvest" `
            1> $stdout 2> $stderr
        $harvestExitCode = $LASTEXITCODE
        $attemptFinishedAt = [datetime]::UtcNow.ToString("o")
        $attempts += @{
            started_at_utc = $attemptStartedAt
            finished_at_utc = $attemptFinishedAt
            exit_code = $harvestExitCode
            stdout = $stdout
            stderr = $stderr
        }

        if ($harvestExitCode -eq 0) {
            break
        }

        $harvestError = Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue
        if ($harvestError -notmatch "API budget is exhausted") {
            break
        }

        $nextAttemptUtc = [datetime]::UtcNow.Date.AddDays(1).AddSeconds(30)
    }

    if ($harvestExitCode -eq 0) {
        $deepSeekKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
        if ([string]::IsNullOrWhiteSpace($deepSeekKey)) {
            throw "DEEPSEEK_API_KEY is not available in the Windows User environment."
        }
        $env:DEEPSEEK_API_KEY = $deepSeekKey
        & $python "scripts\run_remaining_formal_experiments.py" `
            1>> $downstreamStdout 2>> $downstreamStderr
        $downstreamExitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

@{
    started_at_utc = $startedAt
    finished_at_utc = [datetime]::UtcNow.ToString("o")
    harvest_exit_code = $harvestExitCode
    downstream_exit_code = $downstreamExitCode
    registry = $registry
    harvest_stdout = $stdout
    harvest_stderr = $stderr
    downstream_stdout = $downstreamStdout
    downstream_stderr = $downstreamStderr
    attempts = $attempts
} | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding UTF8

if ($harvestExitCode -ne 0) {
    exit $harvestExitCode
}
exit $downstreamExitCode
