from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from citeweave.graph_discovery import score_discovery_response
from citeweave.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_difference(
    records: list[dict[str, Any]],
    target: str,
    comparator: str,
    *,
    samples: int = 10_000,
    seed: int = 20260820,
) -> list[float]:
    topics = sorted({record["dataset_id"] for record in records})
    by_topic_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        by_topic_condition[(record["dataset_id"], record["condition"])].append(
            float(record["score"]["answer_exact"])
        )
    topic_effects = [
        _mean(by_topic_condition[(topic, target)])
        - _mean(by_topic_condition[(topic, comparator)])
        for topic in topics
    ]
    rng = np.random.default_rng(seed)
    draws = rng.choice(topic_effects, size=(samples, len(topic_effects)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _comparison(records: list[dict[str, Any]], target: str, comparator: str) -> dict[str, Any]:
    indexed = {
        (record["item_id"], record["condition"]): bool(record["score"]["answer_exact"])
        for record in records
    }
    item_ids = sorted(
        item_id
        for item_id, condition in indexed
        if condition == target and (item_id, comparator) in indexed
    )
    target_only = sum(indexed[(item_id, target)] and not indexed[(item_id, comparator)] for item_id in item_ids)
    comparator_only = sum(
        indexed[(item_id, comparator)] and not indexed[(item_id, target)] for item_id in item_ids
    )
    discordant = target_only + comparator_only
    p_value = (
        float(binomtest(min(target_only, comparator_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "target": target,
        "comparator": comparator,
        "pairs": len(item_ids),
        "target_only_correct": target_only,
        "comparator_only_correct": comparator_only,
        "accuracy_difference": _mean(
            [float(indexed[(item_id, target)]) for item_id in item_ids]
        )
        - _mean([float(indexed[(item_id, comparator)]) for item_id in item_ids]),
        "topic_bootstrap_95_ci": _bootstrap_difference(records, target, comparator),
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "cross_topic_panel_20260820",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "benchmarks",
    )
    parser.add_argument(
        "--additional-panel-root",
        action="append",
        type=Path,
        default=[],
        help="Merge another panel with the same item IDs for paired comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for panel_root in [args.panel_root, *args.additional_panel_root]:
        for topic_dir in sorted(path for path in panel_root.iterdir() if path.is_dir()):
            benchmark = read_json(args.benchmark_root / topic_dir.name / "benchmark.json")
            tasks = {task["item_id"]: task for task in benchmark["tasks"]}
            payload = read_json(topic_dir / "results.json")
            for record in payload["records"]:
                if record.get("status", "complete") != "complete":
                    failures.append({"dataset_id": topic_dir.name, **record})
                    continue
                rescored = dict(record)
                rescored["dataset_id"] = topic_dir.name
                rescored["score"] = score_discovery_response(
                    tasks[record["item_id"]], record["response"]
                )
                records.append(rescored)

    conditions = sorted({record["condition"] for record in records})
    overall = {}
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        overall[condition] = {
            "items": len(subset),
            "accuracy": _mean([float(record["score"]["answer_exact"]) for record in subset]),
            "mean_evidence_f1": _mean([record["score"]["evidence_f1"] for record in subset]),
            "mean_context_characters": _mean(
                [float(record["context_characters"]) for record in subset]
            ),
            "mean_elapsed_seconds": _mean([record["elapsed_seconds"] for record in subset]),
            "total_tokens": sum(
                int((record.get("usage") or {}).get("total_tokens") or 0) for record in subset
            ),
        }

    by_task_type = {}
    for task_type in sorted({record["task_type"] for record in records}):
        by_task_type[task_type] = {}
        for condition in conditions:
            subset = [
                record
                for record in records
                if record["task_type"] == task_type and record["condition"] == condition
            ]
            by_task_type[task_type][condition] = {
                "items": len(subset),
                "accuracy": _mean(
                    [float(record["score"]["answer_exact"]) for record in subset]
                ),
            }

    by_network = {}
    for network in sorted({record["network"] for record in records}):
        by_network[network] = {}
        for condition in conditions:
            subset = [
                record
                for record in records
                if record["network"] == network and record["condition"] == condition
            ]
            by_network[network][condition] = {
                "items": len(subset),
                "accuracy": _mean(
                    [float(record["score"]["answer_exact"]) for record in subset]
                ),
            }

    by_scale = {}
    for scale in ("small", "medium", "large"):
        if not any(record["scale"] == scale for record in records):
            continue
        by_scale[scale] = {}
        for condition in conditions:
            subset = [
                record
                for record in records
                if record["scale"] == scale and record["condition"] == condition
            ]
            by_scale[scale][condition] = {
                "items": len(subset),
                "accuracy": _mean(
                    [float(record["score"]["answer_exact"]) for record in subset]
                ),
                "mean_evidence_f1": _mean(
                    [record["score"]["evidence_f1"] for record in subset]
                ),
            }

    result = {
        "schema_version": 1,
        "topics": sorted({record["dataset_id"] for record in records}),
        "items_per_condition": len(records) // len(conditions),
        "completed_records": len(records),
        "failed_records": len(failures),
        "overall": overall,
        "by_task_type": by_task_type,
        "by_network": by_network,
        "by_scale": by_scale,
        "comparisons": [
            _comparison(records, "graph_program", comparator)
            for comparator in (
                "flat_tfidf",
                "flat_bm25",
                "flat_lsa",
                "flat_hybrid",
                "graph_hierarchical_retrieval",
                "graph_hierarchical_retrieval_v2",
                "graph_retrieval",
                "graph_query_retrieval",
                "flat_program",
                "operator_only",
                "flat_retrieval",
                "no_reference",
            )
            if comparator in conditions and "graph_program" in conditions
        ],
        "retrieval_comparisons": [
            _comparison(records, target, comparator)
            for target, comparator in (
                ("graph_hierarchical_retrieval_v2", "graph_hierarchical_retrieval"),
                ("graph_hierarchical_retrieval_v2", "graph_query_retrieval"),
                ("graph_hierarchical_retrieval_v2", "graph_retrieval"),
                ("graph_hierarchical_retrieval_v2", "flat_hybrid"),
                ("flat_hybrid", "flat_lsa"),
                ("flat_hybrid", "flat_bm25"),
                ("flat_hybrid", "flat_tfidf"),
            )
            if target in conditions and comparator in conditions
        ],
        "comparisons_by_scale": {
            scale: [
                _comparison(
                    [record for record in records if record["scale"] == scale],
                    "graph_program",
                    comparator,
                )
                for comparator in (
                    "flat_tfidf",
                    "flat_bm25",
                    "flat_lsa",
                    "flat_hybrid",
                    "graph_hierarchical_retrieval",
                    "graph_hierarchical_retrieval_v2",
                    "graph_query_retrieval",
                    "graph_retrieval",
                    "flat_program",
                    "operator_only",
                )
                if comparator in conditions and "graph_program" in conditions
            ]
            for scale in by_scale
        },
        "failures": failures,
        "interpretation": (
            "Exploratory frozen-prompt cross-topic panel. Five topic clusters are insufficient "
            "for a final publication claim; confidence intervals and p-values are diagnostic."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    # Keep the CLI portable on Windows terminals whose active code page is GBK;
    # the artifact itself remains UTF-8 through write_json.
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
