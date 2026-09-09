from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file, write_json


def _leaves(value: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            leaf
            for key, child in value.items()
            for leaf in _leaves(child, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list):
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _leaves(child, f"{path}[{index}]")
        ]
    return [(path, value)]


def _matches(answer: Any, candidate: Any) -> bool:
    if isinstance(answer, bool) or isinstance(candidate, bool):
        return type(answer) is type(candidate) and answer == candidate
    if isinstance(answer, (int, float)) and isinstance(candidate, (int, float)):
        return math.isclose(float(answer), float(candidate), rel_tol=1e-6, abs_tol=1e-6)
    return type(answer) is type(candidate) and answer == candidate


def _coverage(answer: dict[str, Any], exposed: Any) -> dict[str, Any]:
    answer_leaves = _leaves(answer)
    exposed_values = [value for _, value in _leaves(exposed)]
    matched = [
        path
        for path, answer_value in answer_leaves
        if any(_matches(answer_value, value) for value in exposed_values)
    ]
    return {
        "answer_leaf_count": len(answer_leaves),
        "matched_answer_leaf_count": len(matched),
        "matched_answer_paths": matched,
        "coverage": len(matched) / len(answer_leaves),
        "fully_exposed": len(matched) == len(answer_leaves),
    }


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _value_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return _matches(left, right)


def _operator(trace: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in trace if row.get("operator") == name)


def _derive_from_operator_trace(
    task_type: str,
    trace: list[dict[str, Any]],
    *,
    raw_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if task_type == "multi_hop_connector":
        shortest = _operator(trace, "shortest_path")
        projected = _operator(trace, "path_projection")
        path = shortest["path"]
        return {
            "source_label": path[0],
            "target_label": path[-1],
            "intermediate_labels": path[1:-1],
            "hops": projected["hops"],
            "total_inverse_weight_distance": shortest["total_distance"],
        }
    if task_type == "bridge_counterfactual":
        selected = _operator(trace, "edge_betweenness")["selected_edge"]
        deletion = _operator(trace, "delete_edge_counterfactual")
        answer = {
            "source_label": selected[0],
            "target_label": selected[1],
            "alternate_hops_after_deletion": deletion["alternate_hops"],
            "component_increase": (
                deletion["components_after"] - deletion["components_before"]
            ),
        }
        for row in raw_rows or []:
            labels = {row.get("source_label"), row.get("target_label")}
            if labels == set(selected) and "weight" in row:
                answer["edge_weight"] = row["weight"]
                break
        return answer
    if task_type == "community_role_contrast":
        metrics = {
            row["community"]: row
            for row in _operator(trace, "community_aggregate")["metrics"]
        }
        contrast = _operator(trace, "role_contrast")
        dominant = metrics[contrast["dominant"]]
        outward = metrics[contrast["outward"]]
        return {
            "dominant_community": dominant["community"],
            "dominant_representative": dominant["representative"],
            "dominant_nodes": dominant["nodes"],
            "outward_community": outward["community"],
            "outward_representative": outward["representative"],
            "outward_external_share": outward["external_share"],
        }
    if task_type == "hub_removal_resilience":
        hubs = [row for row in trace if row.get("operator") == "weighted_degree_argmax"]
        recompute = _operator(trace, "component_recompute")
        return {
            "removed_hub": hubs[0]["node"],
            "replacement_hub": hubs[-1]["node"],
            "components_after": recompute["components_after"],
            "largest_component_fraction_of_original_nodes": (
                recompute["largest_after"] / recompute["largest_before"]
            ),
        }
    if task_type == "temporal_structural_shift":
        emerging = _operator(trace, "temporal_window_aggregate")["selected"]
        established = _operator(trace, "weighted_degree_argmax")["selected"]
        joined = _operator(trace, "cross_layer_join")
        return {
            "emerging_label": emerging["node"],
            "emerging_recent_documents": emerging["recent_documents"],
            "emerging_growth_ratio": emerging["growth_ratio"],
            "established_label": established["node"],
            "established_weighted_degree": established["weighted_degree"],
            "same_community": joined["same_community"],
        }
    raise ValueError(f"Unsupported complex task type: {task_type}")


def _derivability(answer: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    matched = [
        key
        for key, value in answer.items()
        if key in derived and _value_equal(value, derived[key])
    ]
    return {
        "answer_field_count": len(answer),
        "matched_answer_field_count": len(matched),
        "matched_answer_fields": matched,
        "coverage": len(matched) / len(answer),
        "fully_derivable": len(matched) == len(answer),
    }


def audit_graph_program_answer_exposure(
    benchmark_paths: list[Path],
) -> dict[str, Any]:
    records = []
    benchmark_hashes = {}
    for benchmark_path in sorted(benchmark_paths):
        benchmark = read_json(benchmark_path)
        benchmark_hashes[str(benchmark_path)] = sha256_file(benchmark_path)
        for task in benchmark.get("tasks") or []:
            if int(task.get("complexity") or 0) <= 1:
                continue
            contexts = task["contexts"]
            exposures = {
                "operator_trace": contexts["graph_program"].get("operator_trace"),
                "operator_only_context": contexts["operator_only"],
                "flat_program_derived_rows": contexts["flat_program"].get(
                    "derived_rows"
                ),
                "flat_program_full_context": contexts["flat_program"],
            }
            trace = contexts["graph_program"]["operator_trace"]
            derivations = {
                "operator_trace": _derive_from_operator_trace(
                    task["task_type"], trace
                ),
                "operator_only_context": _derive_from_operator_trace(
                    task["task_type"], contexts["operator_only"]["operator_trace"]
                ),
                "flat_program_derived_rows": _derive_from_operator_trace(
                    task["task_type"], contexts["flat_program"]["derived_rows"]
                ),
                "flat_program_full_context": _derive_from_operator_trace(
                    task["task_type"],
                    contexts["flat_program"]["derived_rows"],
                    raw_rows=contexts["flat_program"].get("rows"),
                ),
                "graph_program_full_context": _derive_from_operator_trace(
                    task["task_type"],
                    trace,
                    raw_rows=contexts["graph_program"].get("edges"),
                ),
            }
            records.append(
                {
                    "dataset_id": benchmark["dataset_id"],
                    "item_id": task["item_id"],
                    "task_type": task["task_type"],
                    "scale": task["scale"],
                    "answer_leaf_paths": [path for path, _ in _leaves(task["answer"])],
                    "exposure": {
                        name: _coverage(task["answer"], value)
                        for name, value in exposures.items()
                    },
                    "derivability": {
                        name: _derivability(task["answer"], value)
                        for name, value in derivations.items()
                    },
                }
            )
    if not records:
        raise ValueError("No complex benchmark tasks found")
    task_type_counts = Counter(row["task_type"] for row in records)
    summaries = {}
    for exposure_name in records[0]["exposure"]:
        values = [row["exposure"][exposure_name] for row in records]
        summaries[exposure_name] = {
            "tasks": len(values),
            "fully_exposed_tasks": sum(row["fully_exposed"] for row in values),
            "mean_answer_leaf_coverage": sum(row["coverage"] for row in values)
            / len(values),
        }
    derivability_summaries = {}
    for derivation_name in records[0]["derivability"]:
        values = [row["derivability"][derivation_name] for row in records]
        derivability_summaries[derivation_name] = {
            "tasks": len(values),
            "fully_derivable_tasks": sum(row["fully_derivable"] for row in values),
            "mean_answer_field_coverage": sum(row["coverage"] for row in values)
            / len(values),
        }
    return {
        "schema_version": 1,
        "status": "audited_before_formal_model_outcomes",
        "formal_model_outcomes_inspected": False,
        "benchmarks": len(benchmark_paths),
        "benchmark_sha256": benchmark_hashes,
        "complex_tasks": len(records),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "exposure_summaries": summaries,
        "derivability_summaries": derivability_summaries,
        "records": records,
        "interpretation_guard": (
            "When the operator trace exposes all answer leaves, graph-program versus "
            "retrieval-only contrasts estimate the value of deterministic graph "
            "computation plus grounded synthesis, not unaided LLM graph reasoning. "
            "Flat-program and operator-only controls are required to isolate formatting "
            "and raw-provenance contributions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.benchmark_root.glob("*/benchmark.json"))
    result = audit_graph_program_answer_exposure(paths)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
