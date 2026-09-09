param(
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedOutput = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $OutputPath))
$releaseRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot "experiments\releases"))
if (-not $resolvedOutput.StartsWith($releaseRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "Research candidate must be written under experiments/releases"
}
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Refusing to overwrite research candidate: $resolvedOutput"
}

$stagingRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "citeweave_candidate_" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $stagingRoot | Out-Null

try {
    $files = @()
    foreach ($directory in @("src", "scripts", "tests")) {
        $files += Get-ChildItem -LiteralPath (Join-Path $workspaceRoot $directory) `
            -Recurse -File |
            Where-Object {
                $_.FullName -notmatch "__pycache__|\.pyc$|\.pytest_cache|\.xml$"
            }
    }
    $files += Get-ChildItem -LiteralPath (Join-Path $workspaceRoot "docs") `
        -Recurse -File -Filter "*.md" |
        Where-Object { $_.Name -notlike "RESEARCH_CANDIDATE_HANDOFF_*.md" }
    foreach ($name in @("LICENSE", "README.md", "pyproject.toml")) {
        $files += Get-Item -LiteralPath (Join-Path $workspaceRoot $name)
    }
    $experimentFiles = @(
        "experiments\graph_discovery_v2\formal_v3_execution_amendment_009_utf8_tokenizer_io.yml",
        "experiments\graph_discovery_v2\formal_v3_execution_amendment_009_utf8_tokenizer_io_freeze.json",
        "experiments\graph_discovery_v2\formal_v3_execution_amendment_010_nested_operator_budget.yml",
        "experiments\graph_discovery_v2\formal_v3_execution_amendment_010_nested_operator_budget_freeze.json",
        "experiments\graph_discovery_v2\formal_v3_graph_program_answer_exposure_audit.json",
        "experiments\graph_discovery_v2\formal_v3_identity_corrected_execution\input_freeze.json",
        "experiments\graph_discovery_v2\formal_v3_mechanism_supplement_protocol.yml",
        "experiments\graph_discovery_v2\formal_v3_mechanism_supplement_protocol_freeze.json",
        "experiments\graph_discovery_v2\formal_v3_analysis.json",
        "experiments\graph_discovery_v2\formal_v3_scale_replication_analysis.json",
        "experiments\graph_discovery_v2\formal_v3_complexity_extension_analysis.json",
        "experiments\graph_discovery_v2\formal_v3_mechanism_supplement_analysis.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_development_v1_protocol.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_development_v1_protocol_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_development_v1_readiness.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_development_v1\construction_manifest.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_initialization.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_001_candidate_audit.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_001_candidate_audit_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_002_reserve_status.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_002_reserve_status_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_003_query_review.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_003_query_review_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_004_access_delivery.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_004_access_delivery_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_005_post_selection_execution.yml",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_amendment_005_post_selection_execution_freeze.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_candidate_audit.json",
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_reviewer_roster.json",
        "experiments\article_quality_v2\article_claim_review_amendment_002_shared_ui_compatibility.yml",
        "experiments\article_quality_v2\article_claim_review_amendment_002_shared_ui_compatibility_freeze.json",
        "experiments\article_quality_v2\article_claim_review_amendment_003_sequential_dependency_voi.yml",
        "experiments\article_quality_v2\article_claim_review_amendment_003_sequential_dependency_voi_freeze.json",
        "experiments\article_quality_v2\article_generation_diagnostic_amendment_001_hierarchical_section_parser.yml",
        "experiments\article_quality_v2\article_generation_diagnostic_amendment_001_hierarchical_section_parser_freeze.json",
        "experiments\article_quality_v2\machine_generation_v1\secondary_hierarchical_parser_diagnostic.json",
        "experiments\article_quality_v3\development_length_control_pilot_v1_protocol.yml",
        "experiments\article_quality_v3\development_length_control_pilot_v1_protocol_freeze.json",
        "experiments\article_quality_v3\development_length_control_pilot_v2_protocol.yml",
        "experiments\article_quality_v3\development_length_control_pilot_v2_protocol_freeze.json",
        "experiments\article_quality_v3\article_compiler_development_pilot_v1_protocol.yml",
        "experiments\article_quality_v3\article_compiler_development_pilot_v1_protocol_freeze.json",
        "experiments\article_quality_v3\article_compiler_repair_development_pilot_v1_protocol.yml",
        "experiments\article_quality_v3\article_compiler_repair_development_pilot_v1_protocol_freeze.json",
        "experiments\article_quality_v3\article_budget_optimizer_development_v1_protocol.yml",
        "experiments\article_quality_v3\article_budget_optimizer_development_v1_protocol_freeze.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v1_protocol.yml",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v1_protocol_freeze.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v1_plan.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v1_plan_freeze.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v1_analysis.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v2_protocol.yml",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v2_protocol_freeze.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v2_plan.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v2_plan_freeze.json",
        "experiments\article_quality_v3\article_prompt_compaction_audit_v1.json",
        "experiments\article_quality_v3\deterministic_machine_human_quality_diagnostic_v1.json",
        "experiments\article_quality_v3\deterministic_argument_structure_diagnostic_v2_protocol.yml",
        "experiments\article_quality_v3\deterministic_argument_structure_diagnostic_v2_protocol_freeze.json",
        "experiments\article_quality_v3\deterministic_argument_structure_diagnostic_v2.json",
        "experiments\article_quality_v3\live_sequential_voi_review_development_protocol.yml",
        "experiments\article_quality_v3\live_sequential_voi_review_development_protocol_freeze.json",
        "experiments\article_quality_v3\live_sequential_voi_review_readiness.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v3_protocol.yml",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v3_protocol_freeze.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v3_plan.json",
        "experiments\article_quality_v3\matched_article_compiler_claim_ready_development_v3_plan_freeze.json",
        "experiments\article_quality_v3\development_claim_voi_features_v1.json",
        "experiments\article_quality_v2\expert_evaluation_amendment_004_length_parity_and_density.yml",
        "experiments\article_quality_v2\expert_evaluation_amendment_004_length_parity_and_density_freeze.json",
        "experiments\article_quality_v2\expert_evaluation_amendment_005_role_separation.yml",
        "experiments\article_quality_v2\expert_evaluation_amendment_005_role_separation_freeze.json",
        "experiments\article_quality_v2\expert_evaluation_amendment_006_design_sensitivity_and_claim_scope.yml",
        "experiments\article_quality_v2\expert_evaluation_amendment_006_design_sensitivity_and_claim_scope_freeze.json",
        "experiments\article_quality_v2\expert_evaluation_amendment_007_blinded_evidence_collection.yml",
        "experiments\article_quality_v2\expert_evaluation_amendment_007_blinded_evidence_collection_freeze.json",
        "experiments\article_quality_v2\article_design_sensitivity_20260908.json",
        "experiments\human_review_v2\feedback_repair_trial_protocol.yml",
        "experiments\human_review_v2\feedback_repair_trial_protocol_freeze.json",
        "experiments\human_review_v2\review_policy_protocol.yml",
        "experiments\human_review_v2\review_policy_protocol_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_001_complementary_oversight.yml",
        "experiments\human_review_v2\review_policy_amendment_001_complementary_oversight_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_002_cluster_inference.yml",
        "experiments\human_review_v2\review_policy_amendment_002_cluster_inference_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_003_factorial_cluster_inference.yml",
        "experiments\human_review_v2\review_policy_amendment_003_factorial_cluster_inference_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_004_multidimensional_calibration.yml",
        "experiments\human_review_v2\review_policy_amendment_004_multidimensional_calibration_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_005_integrated_capability_and_factorial_execution.yml",
        "experiments\human_review_v2\review_policy_amendment_005_integrated_capability_and_factorial_execution_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_006_finite_benchmark_randomization.yml",
        "experiments\human_review_v2\review_policy_amendment_006_finite_benchmark_randomization_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_007_complementarity_baselines.yml",
        "experiments\human_review_v2\review_policy_amendment_007_complementarity_baselines_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_008_balanced_real_candidate_panel.yml",
        "experiments\human_review_v2\review_policy_amendment_008_balanced_real_candidate_panel_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_009_freeze_timestamp_correction.yml",
        "experiments\human_review_v2\review_policy_amendment_009_freeze_timestamp_correction_freeze.json",
        "experiments\human_review_v2\review_policy_amendment_010_ai_self_review_baseline.yml",
        "experiments\human_review_v2\review_policy_amendment_010_ai_self_review_baseline_freeze.json",
        "experiments\human_review_v2\oversight_ai_self_review_plan_v1.json",
        "experiments\human_review_v2\oversight_ai_self_review_plan_v1.freeze.json",
        "experiments\human_review_v2\oversight_design_sensitivity_20260907.json",
        "experiments\human_review_v2\source_relevance_warmup_protocol.yml",
        "experiments\human_review_v2\source_relevance_warmup_protocol_freeze.json",
        "experiments\human_review_v2\source_relevance_warmup_amendment_001_execution_integrity.yml",
        "experiments\human_review_v2\source_relevance_warmup_amendment_001_execution_integrity_freeze.json",
        "experiments\human_review_v2\source_relevance_warmup_amendment_002_closed_loop.yml",
        "experiments\human_review_v2\source_relevance_warmup_amendment_002_closed_loop_freeze.json",
        "experiments\human_review_v2\oversight_calibration_v1\manifest.json",
        "experiments\human_review_v2\oversight_heldout_selection_v1.json",
        "experiments\human_review_v2\oversight_heldout_selection_v2_balanced.json",
        "experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json",
        "experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\construction_audit_private.json"
    )
    foreach ($relativePath in $experimentFiles) {
        $files += Get-Item -LiteralPath (Join-Path $workspaceRoot $relativePath)
    }
    $robustBenchmarkRoot = Join-Path $workspaceRoot (
        "experiments\graph_discovery_v2\robust_graph_synthesis_development_v1"
    )
    $files += Get-ChildItem -LiteralPath $robustBenchmarkRoot -Recurse -File `
        -Filter "benchmark.json"
    $confirmatoryReviewPacketRoot = Join-Path $workspaceRoot (
        "experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_packets"
    )
    $files += Get-ChildItem -LiteralPath $confirmatoryReviewPacketRoot -Recurse -File
    $files = @($files | Sort-Object FullName -Unique)
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($workspaceRoot.Length).TrimStart("\")
        $target = Join-Path $stagingRoot $relativePath
        New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $stagingRoot,
        $resolvedOutput,
        [IO.Compression.CompressionLevel]::Optimal,
        $false
    )
} finally {
    $resolvedStaging = (Resolve-Path $stagingRoot).Path
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolvedStaging.StartsWith($temporaryRoot)) {
        throw "Unsafe staging path; refusing recursive cleanup"
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}

$archive = [IO.Compression.ZipFile]::OpenRead($resolvedOutput)
try {
    $prohibitedNames = @(
        $archive.Entries | Where-Object {
            $_.FullName -match "(?i)apikey|(^|[\\/])\.env($|[\\/])|results\.json|response\.json|execution_record\.json"
        }
    )
    $credentialPatternHits = 0
    # Require a token boundary so ordinary identifiers ending in "task-" do not
    # look like provider credentials merely because they contain the substring "sk-".
    $keyPattern = "(?<![A-Za-z0-9_-])" + "sk-" + "[A-Za-z0-9]{24,}"
    foreach ($entry in $archive.Entries) {
        if (
            $entry.Length -gt 0 -and
            $entry.FullName -match "\.(py|ps1|md|yml|yaml|json|toml|txt)$"
        ) {
            $reader = [IO.StreamReader]::new($entry.Open())
            try {
                $content = $reader.ReadToEnd()
                if ($content -cmatch $keyPattern) {
                    $credentialPatternHits += 1
                }
            } finally {
                $reader.Dispose()
            }
        }
    }
    if ($prohibitedNames.Count -ne 0 -or $credentialPatternHits -ne 0) {
        throw "Candidate archive failed credential/result exclusion scan"
    }
    $entryCount = $archive.Entries.Count
} finally {
    $archive.Dispose()
}

$item = Get-Item -LiteralPath $resolvedOutput
[ordered]@{
    schema_version = 1
    status = "packaged_and_scanned"
    path = $resolvedOutput
    entries = $entryCount
    bytes = $item.Length
    sha256 = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash.ToLower()
    prohibited_names = 0
    credential_pattern_hits = 0
} | ConvertTo-Json
