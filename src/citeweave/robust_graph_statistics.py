from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from .graph_discovery import score_discovery_response
from .io import read_json, sha256_file
from .robust_graph_synthesis import TASK_TYPES, answer_field_accuracy

CONDITIONS = (
    "flat_bm25",
    "graph_hierarchical_retrieval_v2",
    "flat_program",
    "graph_program",
    "operator_only",
)


class RobustGraphStatisticsError(ValueError):
    """Raised when development-result inputs violate the frozen panel contract."""


def _exact_sign_flip_p(effects: list[float]) -> float:
    if not effects:
        raise RobustGraphStatisticsError("No topic effects were supplied")
    observed = mean(effects)
    if observed <= 0:
        return 1.0
    null_means = [
        mean(sign * effect for sign, effect in zip(signs, effects))
        for signs in itertools.product((-1, 1), repeat=len(effects))
    ]
    return sum(value >= observed - 1e-12 for value in null_means) / len(null_means)


def _cluster_bootstrap_interval(
    effects: list[float], *, samples: int, seed: int
) -> list[float]:
    if samples < 100:
        raise RobustGraphStatisticsError("Cluster bootstrap requires at least 100 draws")
    generator = Random(seed)
    draws = sorted(
        mean(generator.choice(effects) for _ in effects) for _ in range(samples)
    )
    lower = draws[math.floor(0.025 * (samples - 1))]
    upper = draws[math.ceil(0.975 * (samples - 1))]
    return [lower, upper]


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return dict(sorted(adjusted.items()))


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _load_cells(
    construction_manifest_path: Path,
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    construction = read_json(construction_manifest_path)
    if construction.get("status") != "development_panel_constructed_before_model_outcomes":
        raise RobustGraphStatisticsError("Construction manifest is not the frozen development panel")
    if construction.get("topics") != 8 or construction.get("tasks") != 40:
        raise RobustGraphStatisticsError("Development panel must contain 8 topics and 40 tasks")

    all_cells: list[dict[str, Any]] = []
    tasks_by_id: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for panel_record in construction.get("records") or []:
        dataset_id = str(panel_record["dataset_id"])
        benchmark_path = Path(panel_record["benchmark"])
        if not benchmark_path.is_file():
            benchmark_path = construction_manifest_path.parent / dataset_id / "benchmark.json"
        if not benchmark_path.is_file() or sha256_file(benchmark_path) != panel_record.get("benchmark_sha256"):
            raise RobustGraphStatisticsError(f"Benchmark hash mismatch: {dataset_id}")
        benchmark = read_json(benchmark_path)
        tasks = benchmark.get("tasks") or []
        if {task.get("task_type") for task in tasks} != set(TASK_TYPES):
            raise RobustGraphStatisticsError(f"Task coverage mismatch: {dataset_id}")
        for task in tasks:
            tasks_by_id[str(task["item_id"])] = task

        result_path = run_root / dataset_id / "results.json"
        if not result_path.is_file():
            raise RobustGraphStatisticsError(f"Missing results: {dataset_id}")
        payload = read_json(result_path)
        run_manifest = payload.get("manifest") or {}
        if run_manifest.get("benchmark_sha256") != panel_record.get("benchmark_sha256"):
            raise RobustGraphStatisticsError(f"Run benchmark identity mismatch: {dataset_id}")
        if set(run_manifest.get("conditions") or []) != set(CONDITIONS):
            raise RobustGraphStatisticsError(f"Run condition mismatch: {dataset_id}")
        for record in payload.get("records") or []:
            if record.get("status", "complete") != "complete":
                continue
            key = (str(record.get("item_id")), str(record.get("condition")))
            if key in seen:
                raise RobustGraphStatisticsError(f"Duplicate completed cell: {key}")
            seen.add(key)
            task = tasks_by_id.get(key[0])
            if task is None or key[1] not in CONDITIONS:
                raise RobustGraphStatisticsError(f"Unexpected completed cell: {key}")
            response = record.get("response")
            if not isinstance(response, dict):
                raise RobustGraphStatisticsError(f"Missing parsed response: {key}")
            correct, total = answer_field_accuracy(task["answer"], response.get("answer"))
            score = score_discovery_response(task, response)
            usage = record.get("usage") or {}
            all_cells.append(
                {
                    "dataset_id": dataset_id,
                    "item_id": key[0],
                    "task_type": task["task_type"],
                    "condition": key[1],
                    "field_correct": correct,
                    "field_total": total,
                    "field_accuracy": correct / total,
                    "exact_accuracy": float(score["answer_exact"]),
                    "evidence_f1": float(score["evidence_f1"]),
                    "required_limitation_rate": float(score["has_required_limitation"]),
                    "abstention_rate": float(score["abstain"]),
                    "provider_tokens": float(usage.get("total_tokens") or 0),
                    "elapsed_seconds": float(record.get("elapsed_seconds") or 0),
                }
            )

    expected = {
        (item_id, condition)
        for item_id in tasks_by_id
        for condition in CONDITIONS
    }
    if seen != expected or len(all_cells) != 200:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise RobustGraphStatisticsError(
            f"Panel is incomplete: completed={len(all_cells)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    return all_cells, tasks_by_id


def analyze_robust_graph_synthesis(
    construction_manifest_path: Path,
    run_root: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20_260_908,
) -> dict[str, Any]:
    cells, _ = _load_cells(construction_manifest_path, run_root)
    metrics = (
        "field_accuracy",
        "exact_accuracy",
        "evidence_f1",
        "required_limitation_rate",
        "abstention_rate",
        "provider_tokens",
        "elapsed_seconds",
    )
    by_condition = {
        condition: {
            "cells": len(subset := [row for row in cells if row["condition"] == condition]),
            **{metric: _mean(subset, metric) for metric in metrics},
        }
        for condition in CONDITIONS
    }
    by_task_type = {
        task_type: {
            condition: _mean(
                [
                    row
                    for row in cells
                    if row["task_type"] == task_type and row["condition"] == condition
                ],
                "field_accuracy",
            )
            for condition in CONDITIONS
        }
        for task_type in TASK_TYPES
    }

    by_topic_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_topic_condition[(row["dataset_id"], row["condition"])].append(row)
    topics = sorted({row["dataset_id"] for row in cells})

    def contrast(left: str, right: str, metric: str, *, offset: int) -> dict[str, Any]:
        topic_effects = [
            _mean(by_topic_condition[(topic, left)], metric)
            - _mean(by_topic_condition[(topic, right)], metric)
            for topic in topics
        ]
        return {
            "left": left,
            "right": right,
            "metric": metric,
            "estimate": mean(topic_effects),
            "topic_effects": dict(zip(topics, topic_effects)),
            "exact_one_sided_sign_flip_p": _exact_sign_flip_p(topic_effects),
            "topic_cluster_bootstrap_95_interval": _cluster_bootstrap_interval(
                topic_effects, samples=bootstrap_samples, seed=seed + offset
            ),
        }

    contrasts = {
        "D1_computation_value": contrast("graph_program", "flat_bm25", "field_accuracy", offset=1),
        "D2_graph_retrieval_value": contrast(
            "graph_hierarchical_retrieval_v2", "flat_bm25", "field_accuracy", offset=2
        ),
        "D3_representation_only": contrast(
            "graph_program", "flat_program", "field_accuracy", offset=3
        ),
        "D4_raw_provenance_evidence": contrast(
            "graph_program", "operator_only", "evidence_f1", offset=4
        ),
        "D4_raw_provenance_field": contrast(
            "graph_program", "operator_only", "field_accuracy", offset=5
        ),
    }
    d3_interval = contrasts["D3_representation_only"]["topic_cluster_bootstrap_95_interval"]
    d4_interval = contrasts["D4_raw_provenance_field"]["topic_cluster_bootstrap_95_interval"]
    contrasts["D3_representation_only"]["equivalence_margin"] = 0.05
    contrasts["D3_representation_only"]["equivalent"] = d3_interval[0] >= -0.05 and d3_interval[1] <= 0.05
    contrasts["D4_raw_provenance_field"]["noninferiority_margin"] = -0.05
    contrasts["D4_raw_provenance_field"]["noninferior"] = d4_interval[0] >= -0.05
    holm = _holm(
        {
            name: contrasts[name]["exact_one_sided_sign_flip_p"]
            for name in (
                "D1_computation_value",
                "D2_graph_retrieval_value",
                "D4_raw_provenance_evidence",
            )
        }
    )
    for name, value in holm.items():
        contrasts[name]["holm_adjusted_p"] = value

    distinct_means = len({round(row["field_accuracy"], 12) for row in by_condition.values()})
    all_floor = all(row["field_accuracy"] == 0 for row in by_condition.values())
    all_ceiling = all(row["field_accuracy"] == 1 for row in by_condition.values())
    return {
        "schema_version": 1,
        "status": "complete_development_analysis",
        "confirmatory": False,
        "interpretation": "difficulty calibration and mechanism development only",
        "construction_manifest_sha256": sha256_file(construction_manifest_path),
        "completed_cells": len(cells),
        "topics": len(topics),
        "condition_summary": by_condition,
        "task_type_field_accuracy": by_task_type,
        "contrasts": contrasts,
        "multiplicity": {"method": "Holm", "adjusted_p": holm},
        "promotion_gate": {
            "all_200_cells_terminal": True,
            "at_least_three_distinct_condition_means": distinct_means >= 3,
            "neither_all_condition_floor_nor_all_condition_ceiling": not (all_floor or all_ceiling),
            "D3_equivalence": contrasts["D3_representation_only"]["equivalent"],
            "passed": distinct_means >= 3
            and not (all_floor or all_ceiling)
            and contrasts["D3_representation_only"]["equivalent"],
        },
    }
