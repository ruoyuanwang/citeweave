from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .graph_discovery import (
    DiscoveryTask,
    _bridge_task,
    _communities,
    _community_task,
    _load_graph,
    _path_task,
    _rank_divergence_task,
    _resilience_task,
    _simple_control_from_complex,
    _temporal_structure_task,
)


class OperatorReplayError(RuntimeError):
    pass


def _replay_task(
    task: dict[str, Any],
    *,
    workspace: Path,
    graph: Any,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    common = {
        "dataset_id": task["dataset_id"],
        "network": task["network"],
        "scale": task["scale"],
        "communities": communities,
    }
    builders = {
        "multi_hop_connector": _path_task,
        "bridge_counterfactual": _bridge_task,
        "community_role_contrast": _community_task,
        "hub_removal_resilience": _resilience_task,
        "productivity_centrality_divergence": _rank_divergence_task,
    }
    if task["task_type"] == "temporal_structural_shift":
        return _temporal_structure_task(graph, workspace=workspace, **common)
    if task["task_type"] in {"direct_edge_lookup", "node_attribute_lookup"}:
        parent_builder = (
            _bridge_task
            if task["task_type"] == "direct_edge_lookup"
            else _resilience_task
        )
        parent = parent_builder(graph, **common)
        return (
            _simple_control_from_complex(graph, parent=parent)
            if parent is not None
            else None
        )
    builder = builders.get(task["task_type"])
    if builder is None:
        raise OperatorReplayError(f"Unsupported task type: {task['task_type']}")
    return builder(graph, **common)


def verify_benchmark_operator_replay(
    benchmark: dict[str, Any], *, workspace: Path
) -> dict[str, Any]:
    cache: dict[tuple[str, str], tuple[Any, dict[str, int]]] = {}
    records = []
    for task in benchmark["tasks"]:
        cache_key = (task["network"], task["scale"])
        if cache_key not in cache:
            graph, _ = _load_graph(workspace, task["network"], task["scale"])
            cache[cache_key] = (graph, _communities(graph))
        graph, communities = cache[cache_key]
        replayed = _replay_task(
            task,
            workspace=workspace,
            graph=graph,
            communities=communities,
        )
        mismatches = []
        if replayed is None:
            mismatches.append("task_missing_on_replay")
        else:
            expected = asdict(replayed)
            for field in (
                "item_id",
                "question",
                "answer",
                "answer_alternatives",
                "evidence_ids",
                "operator_trace",
                "interpretation_contract",
            ):
                if task.get(field) != expected.get(field):
                    mismatches.append(field)
        records.append(
            {
                "item_id": task["item_id"],
                "valid": not mismatches,
                "mismatches": mismatches,
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": benchmark["dataset_id"],
        "tasks": len(records),
        "valid_tasks": sum(record["valid"] for record in records),
        "invalid_tasks": sum(not record["valid"] for record in records),
        "records": records,
    }


def corrupt_operator_trace(task: dict[str, Any]) -> dict[str, Any]:
    corrupted = copy.deepcopy(task)
    trace = corrupted.get("operator_trace") or []
    if not trace:
        raise OperatorReplayError("Task has no operator trace to corrupt")

    def mutate(value: Any) -> tuple[Any, bool]:
        if isinstance(value, bool):
            return not value, True
        if isinstance(value, (int, float)):
            return value + 1, True
        if isinstance(value, str):
            return value + "__CORRUPTED", True
        if isinstance(value, list):
            for index, item in enumerate(value):
                replacement, changed = mutate(item)
                if changed:
                    result = list(value)
                    result[index] = replacement
                    return result, True
        if isinstance(value, dict):
            for key in sorted(value):
                if key == "operator":
                    continue
                replacement, changed = mutate(value[key])
                if changed:
                    result = dict(value)
                    result[key] = replacement
                    return result, True
        return value, False

    for index, step in enumerate(trace):
        replacement, changed = mutate(step)
        if changed:
            trace[index] = replacement
            return corrupted
    raise OperatorReplayError("Operator trace contains no mutable value")


def audit_fault_injection(
    benchmark: dict[str, Any], *, workspace: Path
) -> dict[str, Any]:
    clean = verify_benchmark_operator_replay(benchmark, workspace=workspace)
    detections = []
    for task in benchmark["tasks"]:
        single = {
            **benchmark,
            "tasks": [corrupt_operator_trace(task)],
        }
        result = verify_benchmark_operator_replay(single, workspace=workspace)
        record = result["records"][0]
        detections.append(
            {
                "item_id": task["item_id"],
                "detected": not record["valid"],
                "mismatches": record["mismatches"],
            }
        )
    return {
        "schema_version": 1,
        "clean_tasks": clean["tasks"],
        "clean_false_rejections": clean["invalid_tasks"],
        "injected_faults": len(detections),
        "detected_faults": sum(record["detected"] for record in detections),
        "detection_rate": (
            sum(record["detected"] for record in detections) / len(detections)
            if detections
            else 0.0
        ),
        "records": detections,
    }
