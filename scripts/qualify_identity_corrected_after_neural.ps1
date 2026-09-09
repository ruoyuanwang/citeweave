param([ValidateRange(1, 60)][int]$PollSeconds = 60)
$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$statusPath = Join-Path $workspaceRoot "experiments\runs\formal_v3_confirmatory_pipeline_status.json"
$mutex = [System.Threading.Mutex]::new($false, "Local\CiteWeaveIdentityQualificationDProjectBibagent")
if (-not $mutex.WaitOne(0)) { throw "Another corrected qualifier is active." }
$paths = @(
    "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_identity_corrected\neural_dense_manifest.json",
    "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\complexity_extension_neural_dense_identity_corrected\neural_dense_manifest.json"
)
function Write-QualificationStatus([string]$Status, [string]$Stage, [string]$Message) {
    [ordered]@{
        schema_version = 1
        status = $Status
        stage = $Stage
        message = $Message
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        automatic_api_resume = $false
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
function Test-NeuralReady {
    foreach ($path in $paths) {
        if (-not (Test-Path -LiteralPath $path)) { return $false }
        if (-not (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json).passed) { return $false }
    }
    return $true
}
try {
    if (Test-Path -LiteralPath "experiments\graph_discovery_v2\formal_v3_identity_corrected_execution\promotion.json") {
        throw "Already promoted: use the explicit corrected executor, not qualification."
    }
    & $python scripts\prepare_identity_corrected_execution.py --freeze-only
    if ($LASTEXITCODE -ne 0) { throw "Frozen input validation failed." }
    while (-not (Test-NeuralReady)) {
        Write-QualificationStatus "waiting" "identity_corrected_neural" "Waiting for corrected primary and extension indexes; no API execution."
        $builder = Get-CimInstance Win32_Process | Where-Object {
            ($_.Name -eq "python.exe" -and $_.CommandLine -like "*build_neural_dense_sidecars.py*identity_corrected*") -or
            ($_.Name -in @("powershell.exe", "pwsh.exe") -and $_.CommandLine -like "*-File*run_identity_corrected_neural_prerequisites.ps1*")
        }
        if (-not $builder) {
            Start-Sleep -Seconds 10
            if (-not (Test-NeuralReady)) { throw "Corrected index builder ended before both manifests passed." }
        } else {
            Start-Sleep -Seconds $PollSeconds
        }
    }
    Write-QualificationStatus "running" "identity_corrected_qualification" "Checking data, token budgets, all three plans, and 360 reuse identities; no API execution."
    & $python scripts\prepare_identity_corrected_execution.py --prepare
    if ($LASTEXITCODE -ne 0) { throw "Corrected execution qualification failed. Inspect qualification logs." }
    Write-QualificationStatus "qualified" "awaiting_explicit_promotion" "All corrected prerequisites passed. Explicit review and promotion required before any formal provider call."
    Write-Output "Corrected qualification completed; intentionally no automatic API resume."
} catch {
    Write-QualificationStatus "failed" "identity_corrected_qualification" $_.Exception.Message
    Write-Error $_
    exit 1
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
