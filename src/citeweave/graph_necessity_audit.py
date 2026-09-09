from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file

_DECISIVE_KEYS = {
    "direct_edge_lookup": ("source_label", "target_label", "edge_weight"),
    "node_attribute_lookup": ("node_label", "importance", "weighted_degree"),
    "multi_hop_connector": (
        "intermediate_labels",
        "hops",
        "total_inverse_weight_distance",
    ),
    "bridge_counterfactual": (
        "alternate_hops_after_deletion",
        "component_increase",
    ),
    "community_role_contrast": (
        "dominant_community",
        "dominant_nodes",
        "outward_community",
        "outward_external_share",
    ),
    "hub_removal_resilience": (
        "replacement_hub",
        "components_after",
        "largest_component_fraction_of_original_nodes",
    ),
    "temporal_structural_shift": (
        "emerging_recent_documents",
        "emerging_growth_ratio",
        "established_weighted_degree",
        "same_community",
    ),
}

_OPERATOR_CLASSES = {
    "multi_hop_connector": {"path", "community_constraint"},
    "bridge_counterfactual": {"global_selection", "intervention"},
    "community_role_contrast": {"community_detection", "global_aggregation"},
    "hub_removal_resilience": {"global_selection", "intervention", "recompute"},
    "temporal_structural_shift": {"temporal_aggregation", "cross_layer_join"},
}

_OPERATOR_CLASS_BY_NAME = {
    "community_filter": {"community_constraint"},
    "shortest_path": {"path"},
    "path_projection": {"path"},
    "community_boundary_edges": {"community_constraint"},
    "edge_betweenness": {"global_selection"},
    "delete_edge_counterfactual": {"intervention", "recompute"},
    "louvain": {"community_detection"},
    "community_aggregate": {"global_aggregation"},
    "role_contrast": {"global_aggregation"},
    "weighted_degree_argmax": {"global_selection"},
    "delete_node": {"intervention"},
    "component_recompute": {"recompute"},
    "temporal_window_aggregate": {"temporal_aggregation"},
    "cross_layer_join": {"cross_layer_join"},
}


def _contains(value: Any, target: Any) -> bool:
    if isinstance(target, list):
        return all(_contains(value, item) for item in target)
    if isinstance(value, dict):
        return any(_contains(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, target) for item in value)
    if isinstance(value, bool) or isinstance(target, bool):
        return value is target
    return value == target


def _single_row_derivable(task: dict[str, Any]) -> bool:
    keys = _DECISIVE_KEYS[task["task_type"]]
    rows = task["contexts"]["flat_hybrid"].get("rows") or []
    return any(all(_contains(row, task["answer"][key]) for key in keys) for row in rows)


def _visible_evidence_ids(context: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    stack: list[Any] = [context]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            evidence_id = value.get("evidence_id")
            if isinstance(evidence_id, str):
                found.add(evidence_id)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return found


def audit_graph_necessity_task(task: dict[str, Any]) -> dict[str, Any]:
    task_type = task["task_type"]
    simple = task_type in {"direct_edge_lookup", "node_attribute_lookup"}
    trace = task.get("operator_trace") or []
    operator_names = [str(row.get("operator")) for row in trace]
    single_row = _single_row_derivable(task)
    required_classes = sorted(_OPERATOR_CLASSES.get(task_type) or [])
    observed_classes = sorted(
        {
            operator_class
            for operator_name in operator_names
            for operator_class in _OPERATOR_CLASS_BY_NAME.get(operator_name, set())
        }
    )
    missing_classes = sorted(set(required_classes) - set(observed_classes))
    condition_recall: dict[str, float] = {}
    gold = set(task.get("evidence_ids") or [])
    for condition in (
        "flat_hybrid",
        "graph_hierarchical_retrieval_v2",
        "graph_program",
    ):
        context = task["contexts"].get(condition) or {}
        visible = _visible_evidence_ids(context)
        condition_recall[condition] = len(gold & visible) / len(gold) if gold else 1.0
    passed = (
        len(trace) == 1 and not required_classes
        if simple
        else (
            not single_row
            and len(trace) >= 2
            and bool(required_classes)
            and not missing_classes
        )
    )
    return {
        "item_id": task["item_id"],
        "dataset_id": task["dataset_id"],
        "scale": task["scale"],
        "task_type": task_type,
        "registered_complexity": task["complexity"],
        "simple_control": simple,
        "intrinsic_single_record_lookup": simple,
        "decisive_answer_keys": list(_DECISIVE_KEYS[task_type]),
        "decisive_answer_single_flat_row": single_row,
        "operator_stages": len(trace),
        "operators": operator_names,
        "required_operator_classes": required_classes,
        "observed_operator_classes": observed_classes,
        "missing_operator_classes": missing_classes,
        "gold_evidence_recall": condition_recall,
        "necessity_certificate_passed": passed,
    }


def audit_graph_necessity_benchmarks(benchmark_root: Path) -> dict[str, Any]:
    construction_path = benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path)
    records: list[dict[str, Any]] = []
    benchmark_hashes = {}
    answers_by_anchor: dict[tuple[str, str], set[str]] = defaultdict(set)
    for dataset in construction["records"]:
        dataset_id = dataset["dataset_id"]
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        benchmark_hashes[dataset_id] = sha256_file(benchmark_path)
        for task in read_json(benchmark_path)["tasks"]:
            records.append(audit_graph_necessity_task(task))
            answers_by_anchor[(dataset_id, task["task_type"])].add(
                json.dumps(task["answer"], sort_keys=True, separators=(",", ":"))
            )
    simple = [row for row in records if row["simple_control"]]
    complex_rows = [row for row in records if not row["simple_control"]]
    counts = Counter(row["task_type"] for row in records)
    retrieval_diagnostics = {}
    for condition in (
        "flat_hybrid",
        "graph_hierarchical_retrieval_v2",
        "graph_program",
    ):
        retrieval_diagnostics[condition] = {
            "simple_mean_gold_evidence_recall": sum(
                row["gold_evidence_recall"][condition] for row in simple
            )
            / len(simple),
            "complex_mean_gold_evidence_recall": sum(
                row["gold_evidence_recall"][condition] for row in complex_rows
            )
            / len(complex_rows),
            "simple_full_gold_evidence_coverage": sum(
                row["gold_evidence_recall"][condition] == 1.0 for row in simple
            ),
            "complex_full_gold_evidence_coverage": sum(
                row["gold_evidence_recall"][condition] == 1.0
                for row in complex_rows
            ),
        }
    return {
        "schema_version": 1,
        "status": "passed" if all(row["necessity_certificate_passed"] for row in records) else "failed",
        "construction_manifest_sha256": sha256_file(construction_path),
        "benchmark_sha256": benchmark_hashes,
        "records": records,
        "summary": {
            "tasks": len(records),
            "task_type_counts": dict(sorted(counts.items())),
            "simple_controls": len(simple),
            "complex_tasks": len(complex_rows),
            "simple_intrinsic_single_record_lookups": sum(
                row["intrinsic_single_record_lookup"] for row in simple
            ),
            "simple_flat_hybrid_single_row_available": sum(
                row["decisive_answer_single_flat_row"] for row in simple
            ),
            "complex_flat_hybrid_single_row_available": sum(
                row["decisive_answer_single_flat_row"] for row in complex_rows
            ),
            "complex_with_two_or_more_operator_stages": sum(
                row["operator_stages"] >= 2 for row in complex_rows
            ),
            "complex_with_all_required_operator_classes": sum(
                not row["missing_operator_classes"] for row in complex_rows
            ),
            "complex_with_missing_operator_classes": sum(
                bool(row["missing_operator_classes"]) for row in complex_rows
            ),
            "scale_sensitive_dataset_task_anchors": sum(
                len(answers) > 1 for answers in answers_by_anchor.values()
            ),
            "dataset_task_anchors": len(answers_by_anchor),
            "retrieval_diagnostics_not_used_for_task_selection": retrieval_diagnostics,
            "all_certificates_passed": all(
                row["necessity_certificate_passed"] for row in records
            ),
        },
    }
