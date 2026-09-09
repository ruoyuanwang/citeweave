param([string]$DirectMLTarget = ".tmp\onnxruntime-directml")

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$directML = (Resolve-Path $DirectMLTarget).Path
$sourceRoot = (Resolve-Path ".\src").Path
$env:PYTHONPATH = "$directML;$sourceRoot"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$construction = Get-Content -Raw experiments\graph_discovery_v2\formal_v3_identity_corrected_benchmarks\construction_manifest.json | ConvertFrom-Json
if ($construction.status -ne "constructed_not_executed" -or $construction.records.Count -ne 4) {
    throw "All four corrected primary benchmarks must be constructed before indexing"
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    Write-Output "[$([DateTime]::UtcNow.ToString('o'))] python $($Arguments -join ' ')"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Corrected neural prerequisite failed: exit $LASTEXITCODE" }
}

$common = @(
    "--model", "BAAI/bge-m3", "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181",
    "--backend", "onnxruntime", "--onnx-provider", "DmlExecutionProvider", "--batch-size", "32",
    "--local-files-only", "--resume-existing", "--checkpoint-every-batches", "20",
    "--embedding-cache-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_identity_corrected_index_cache"
)
Invoke-CheckedPython (@(
    "scripts\build_neural_dense_sidecars.py",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_identity_corrected_benchmarks",
    "--workspace-root", "experiments\formal_v3_identity_corrected_workspaces\primary",
    "--output-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_identity_corrected"
) + $common)
Invoke-CheckedPython (@(
    "scripts\build_neural_dense_sidecars.py",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
    "--workspace-root", "experiments\formal_v3_identity_corrected_workspaces\primary",
    "--workspace-root", "experiments\formal_v3_identity_corrected_workspaces\replication",
    "--output-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\complexity_extension_neural_dense_identity_corrected"
) + $common)
Write-Output "[$([DateTime]::UtcNow.ToString('o'))] corrected neural materialization complete; formal API remains on integrity hold until promotion audits"
