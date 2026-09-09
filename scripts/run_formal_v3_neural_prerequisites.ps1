param(
    [string]$DirectMLTarget = ".tmp\onnxruntime-directml"
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspaceRoot
$python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$directML = (Resolve-Path $DirectMLTarget).Path
$sourceRoot = (Resolve-Path ".\src").Path
$env:PYTHONPATH = "$directML;$sourceRoot"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    Write-Output "[$([DateTime]::UtcNow.ToString('o'))] python $($Arguments -join ' ')"
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$commonEmbeddingArguments = @(
    "--model", "BAAI/bge-m3",
    "--model-revision", "5617a9f61b028005a4858fdac845db406aefb181",
    "--backend", "onnxruntime",
    "--onnx-provider", "DmlExecutionProvider",
    "--batch-size", "32",
    "--local-files-only",
    "--resume-existing",
    "--checkpoint-every-batches", "20",
    "--embedding-cache-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_semantic_v2_index_cache"
)

Invoke-CheckedPython (@(
    "scripts\build_neural_dense_sidecars.py",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
    "--workspace-root", "experiments\formal_v3_workspaces",
    "--output-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_semantic_v2"
) + $commonEmbeddingArguments)

Invoke-CheckedPython (@(
    "scripts\build_neural_dense_sidecars.py",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
    "--workspace-root", "experiments\formal_v3_workspaces",
    "--workspace-root", "experiments\formal_v3_replication_workspaces",
    "--output-root", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\complexity_extension_neural_dense_semantic_v2"
) + $commonEmbeddingArguments)

Invoke-CheckedPython @(
    "scripts\audit_formal_v3_readiness.py",
    "--protocol", "experiments\graph_discovery_v2\formal_v3_protocol.yml",
    "--freeze", "experiments\graph_discovery_v2\formal_v3_protocol_freeze.json",
    "--amendment", "experiments\graph_discovery_v2\formal_v3_amendment_001.yml",
    "--amendment-freeze", "experiments\graph_discovery_v2\formal_v3_amendment_001_freeze.json",
    "--neural-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense.yml",
    "--neural-amendment-freeze", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense_freeze.json",
    "--statistics-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_003_statistics.yml",
    "--statistics-amendment-freeze", "experiments\graph_discovery_v2\formal_v3_amendment_003_statistics_freeze.json",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_benchmarks",
    "--execution-root", "experiments\graph_discovery_v2\formal_v3_runs",
    "--tokenizer-manifest", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\deepseek_v4_tokenizer_manifest.json",
    "--neural-manifest", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\neural_dense_semantic_v2\neural_dense_manifest.json",
    "--output", "experiments\graph_discovery_v2\formal_v3_readiness.json"
)

Invoke-CheckedPython @(
    "scripts\audit_formal_v3_replication_readiness.py",
    "--protocol", "experiments\graph_discovery_v2\formal_v3_replication_protocol.yml",
    "--protocol-freeze", "experiments\graph_discovery_v2\formal_v3_replication_protocol_freeze.json",
    "--amendment", "experiments\graph_discovery_v2\formal_v3_replication_amendment_001_query.yml",
    "--amendment-freeze", "experiments\graph_discovery_v2\formal_v3_replication_amendment_001_query_freeze.json",
    "--query-judgment", "experiments\graph_discovery_v2\formal_v3_replication_query_judgment_amended.json",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_replication_benchmarks",
    "--scale-pairs", "experiments\graph_discovery_v2\formal_v3_replication_scale_pairs.json",
    "--scale-pairs-freeze", "experiments\graph_discovery_v2\formal_v3_replication_scale_pairs_freeze.json",
    "--execution-root", "experiments\graph_discovery_v2\formal_v3_replication_runs",
    "--tokenizer-manifest", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\deepseek_v4_tokenizer_manifest.json",
    "--output", "experiments\graph_discovery_v2\formal_v3_replication_readiness.json"
)

Invoke-CheckedPython @(
    "scripts\audit_formal_v3_complexity_extension_readiness.py",
    "--protocol", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol.yml",
    "--protocol-freeze", "experiments\graph_discovery_v2\formal_v3_complexity_extension_protocol_freeze.json",
    "--execution-amendment", "experiments\graph_discovery_v2\formal_v3_complexity_extension_amendment_001_execution.yml",
    "--execution-amendment-freeze", "experiments\graph_discovery_v2\formal_v3_complexity_extension_amendment_001_execution_freeze.json",
    "--neural-amendment", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense.yml",
    "--neural-amendment-freeze", "experiments\graph_discovery_v2\formal_v3_amendment_002_neural_dense_freeze.json",
    "--benchmark-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_benchmarks",
    "--execution-root", "experiments\graph_discovery_v2\formal_v3_complexity_extension_runs",
    "--tokenizer-manifest", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\deepseek_v4_tokenizer_manifest.json",
    "--neural-manifest", "experiments\graph_discovery_v2\formal_v3_execution_prerequisites\complexity_extension_neural_dense_semantic_v2\neural_dense_manifest.json",
    "--output", "experiments\graph_discovery_v2\formal_v3_complexity_extension_readiness.json"
)

Write-Output "[$([DateTime]::UtcNow.ToString('o'))] all neural prerequisites and readiness audits completed"
