"""Post-hoc descriptive decomposition of the completed, frozen robustness matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path

from citeweave.formal_request import canonical_sha256
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/graph_discovery_v2/graph_phenomenon_robustness_v1"


def deletion_decomposition(baseline, variant):
    before = baseline["graph"]["largest_component_fraction"]
    after = baseline["measurements"]["hub_removal_resilience"][
        "largest_component_fraction_of_original_nodes"
    ]
    altered_before = variant["graph"]["largest_component_fraction"]
    altered_after = variant["measurements"]["hub_removal_resilience"][
        "largest_component_fraction_of_original_nodes"
    ]
    result = {
        "baseline_pre_deletion_fraction": before,
        "baseline_post_deletion_fraction": after,
        "variant_pre_deletion_fraction": altered_before,
        "variant_post_deletion_fraction": altered_after,
        "baseline_marginal_deletion_loss": before - after,
        "variant_marginal_deletion_loss": altered_before - altered_after,
        "pre_deletion_topology_change": altered_before - before,
        "additional_marginal_deletion_loss": (altered_before - altered_after) - (before - after),
        "total_post_deletion_change": altered_after - after,
    }
    assert math.isclose(
        result["total_post_deletion_change"],
        result["pre_deletion_topology_change"] - result["additional_marginal_deletion_loss"],
        abs_tol=1e-12,
    )
    return result


def main():
    summary_path = OUTPUT / "summary.json"
    summary = read_json(summary_path)
    if summary["status"] != "exploratory_diagnostics_complete" or summary["graph_cases"] != 144:
        raise ValueError("Completed 144-case matrix required")
    results = []
    for name, digest in summary["result_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("Result file drift")
        row = read_json(Path(name))
        if row["payload_sha256"] != canonical_sha256(
            {k: v for k, v in row.items() if k != "payload_sha256"}
        ):
            raise ValueError("Result payload drift")
        results.append(row)
    if len({(r["dataset_id"], r["variant"]["variant_id"]) for r in results}) != 144:
        raise ValueError("Duplicate or missing graph cases")
    baselines = {r["dataset_id"]: r for r in results if r["variant"]["family"] == "baseline"}
    graph_changes = [
        r for r in results if r["variant"]["family"] in {"minimum_weight", "edge_dropout"}
    ]
    windows = [r for r in results if r["variant"]["family"] == "temporal_window"]
    decompositions = [
        {
            "dataset_id": r["dataset_id"],
            "variant_id": r["variant"]["variant_id"],
            **deletion_decomposition(baselines[r["dataset_id"]], r),
        }
        for r in graph_changes
    ]
    thresholds = [r for r in decompositions if r["variant_id"].startswith("weight_")]
    result = {
        "schema_version": 1,
        "status": "posthoc_descriptive_implications_not_new_confirmatory_tests",
        "source_summary_sha256": sha256_file(summary_path),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "baseline_task_checks": sum(len(r["baseline_checks"]) for r in baselines.values()),
        "baseline_task_checks_passed": sum(
            c["passed"] for r in baselines.values() for c in r["baseline_checks"]
        ),
        "topology_changing_cases": len(graph_changes),
        "unchanged_path_distance": sum(
            r["comparison"]["multi_hop_connector"]["distance_equal"] for r in graph_changes
        ),
        "unchanged_fixed_edge_redundancy": sum(
            r["comparison"]["bridge_counterfactual"]["redundancy_equal"] for r in graph_changes
        ),
        "temporal_window_cases": len(windows),
        "changed_emerging_keyword": sum(
            not r["comparison"]["temporal_structural_shift"]["emerging_label_equal"]
            for r in windows
        ),
        "low_resolution_undefined_role_contrasts": sum(
            r["variant"]["variant_id"] == "resolution_0.5"
            and r["measurements"]["community_role_contrast"]["status"] != "measured"
            for r in results
        ),
        "threshold_maximum_total_post_deletion_decline": max(
            -r["total_post_deletion_change"] for r in thresholds
        ),
        "threshold_maximum_additional_marginal_deletion_loss": max(
            r["additional_marginal_deletion_loss"] for r in thresholds
        ),
        "deletion_decompositions": decompositions,
        "interpretation_guards": [
            "Community-seed, resolution-only, and temporal-window-only cases do not change topology; their unchanged path/deletion numbers are not counted as independent topology robustness evidence.",
            "A decline after both thresholding and hub deletion is not all caused by the hub deletion; report the pre-deletion topology shift and marginal deletion loss separately.",
            "All results concern fixed registered keyword graphs/anchors and selected perturbations, not arbitrary corpora, real-world interventions, or general graph reasoning ability.",
        ],
    }
    write_json(OUTPUT / "interpretation_audit.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "deletion_decompositions"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
