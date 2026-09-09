from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from citeweave.io import read_json, sha256_file, write_json


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Paired outcomes must have equal length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    statistic: Callable[[list[dict[str, Any]]], float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters = sorted({str(row["dataset_id"]) for row in rows})
    if not clusters:
        raise ValueError("Cluster bootstrap requires observations")
    by_cluster = {
        cluster: [row for row in rows if str(row["dataset_id"]) == cluster]
        for cluster in clusters
    }
    cluster_effects = {
        cluster: statistic(cluster_rows)
        for cluster, cluster_rows in by_cluster.items()
    }
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = generator.choice(clusters, size=len(clusters), replace=True)
        estimates.append(
            sum(cluster_effects[str(cluster)] for cluster in selected) / len(selected)
        )
    return {
        "estimate": sum(cluster_effects.values()) / len(cluster_effects),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(clusters),
        "cluster_weighting": "equal_dataset",
    }


def exact_cluster_signflip(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Sign-flip test requires cluster effects")
    observed = sum(values) / len(values)
    permutations = [
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    p_value = sum(value >= observed - 1e-12 for value in permutations) / len(
        permutations
    )
    return {
        "cluster_effects": values,
        "estimate": observed,
        "permutations": len(permutations),
        "p_value_one_sided": p_value,
        "minimum_attainable_p": 1.0 / len(permutations),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, p_values[key] * (total - rank))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _accuracy_difference(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["left"]) - float(row["right"]) for row in rows) / len(rows)


def _evidence_difference(rows: list[dict[str, Any]]) -> float:
    return sum(
        float(row["left_evidence_f1"]) - float(row["right_evidence_f1"])
        for row in rows
    ) / len(rows)


def matched_odds_ratio(mcnemar: dict[str, Any]) -> float | str:
    left_only = int(mcnemar["left_only"])
    right_only = int(mcnemar["right_only"])
    if right_only == 0:
        return "inf" if left_only else 1.0
    return left_only / right_only


def _record_outcome(record: dict[str, Any]) -> dict[str, Any]:
    complete = record.get("status", "complete") == "complete"
    return {
        "correct": bool(complete and (record.get("score") or {}).get("answer_exact")),
        "evidence_f1": (
            float((record.get("score") or {}).get("evidence_f1") or 0.0)
            if complete
            else 0.0
        ),
        "prompt_tokens": int((record.get("usage") or {}).get("prompt_tokens") or 0),
        "completion_tokens": int(
            (record.get("usage") or {}).get("completion_tokens") or 0
        ),
        "latency_seconds": float(record.get("elapsed_seconds") or 0.0),
        "status": record.get("status", "complete"),
    }


def _paired_rows(
    indexed: dict[tuple[str, str], dict[str, Any]],
    tasks: list[dict[str, Any]],
    *,
    left_condition: str,
    right_condition: str,
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        item_id = task["item_id"]
        try:
            left = _record_outcome(indexed[(item_id, left_condition)])
            right = _record_outcome(indexed[(item_id, right_condition)])
        except KeyError as exc:
            raise ValueError(f"Missing registered pair: {exc.args[0]}") from exc
        rows.append(
            {
                "dataset_id": task["dataset_id"],
                "item_id": item_id,
                "left": left["correct"],
                "right": right["correct"],
                "left_evidence_f1": left["evidence_f1"],
                "right_evidence_f1": right["evidence_f1"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--neural-amendment", type=Path, required=True)
    parser.add_argument("--statistics-amendment", type=Path, required=True)
    parser.add_argument("--scale-pairs", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weighting-amendment",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/"
            "formal_v3_analysis_amendment_007_equal_dataset_weighting.yml"
        ),
    )
    parser.add_argument(
        "--weighting-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/"
            "formal_v3_analysis_amendment_007_equal_dataset_weighting_freeze.json"
        ),
    )
    args = parser.parse_args()
    weighting_hash = sha256_file(args.weighting_amendment)
    if read_json(args.weighting_amendment_freeze).get("sha256") != weighting_hash:
        raise SystemExit("Equal-dataset weighting amendment differs from freeze")
    weighting = yaml.safe_load(args.weighting_amendment.read_text(encoding="utf-8"))
    if weighting.get("prior_statistics_sha256") != sha256_file(
        args.statistics_amendment
    ):
        raise SystemExit("Equal-dataset weighting amendment statistics mismatch")
    expected_implementation = weighting["implementations"][Path(__file__).name]
    if expected_implementation != sha256_file(Path(__file__)):
        raise SystemExit("Formal analysis implementation differs from amendment")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    neural = yaml.safe_load(args.neural_amendment.read_text(encoding="utf-8"))
    statistics = yaml.safe_load(args.statistics_amendment.read_text(encoding="utf-8"))
    neural_hash = sha256_file(args.neural_amendment)
    if neural_hash not in statistics["prior_amendments"]:
        raise SystemExit("Neural amendment is not in the frozen statistics chain")
    expected_conditions = {
        *protocol["conditions"]["primary"],
        *protocol["conditions"]["diagnostic"],
        neural["changes"]["neural_dense_baseline"]["condition_name"],
    }
    scale_pairs = read_json(args.scale_pairs)
    if sha256_file(args.scale_pairs) != statistics["scale_pair_artifact"]["sha256"]:
        raise SystemExit("Scale-pair artifact hash differs from frozen statistics amendment")
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    tasks = []
    records = []
    for construction_record in construction["records"]:
        dataset_id = construction_record["dataset_id"]
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        benchmark = read_json(benchmark_path)
        if sha256_file(benchmark_path) != construction_record["benchmark_sha256"]:
            raise SystemExit(f"Benchmark hash mismatch: {dataset_id}")
        tasks.extend(benchmark["tasks"])
        result_path = args.results_root / dataset_id / "results.json"
        if not result_path.is_file():
            raise SystemExit(f"Missing formal results: {result_path}")
        result = read_json(result_path)
        if result["manifest"]["benchmark_sha256"] != construction_record[
            "benchmark_sha256"
        ]:
            raise SystemExit(f"Result benchmark hash mismatch: {dataset_id}")
        if set(result["manifest"]["conditions"]) != expected_conditions:
            raise SystemExit(f"Result condition panel mismatch: {dataset_id}")
        records.extend(result["records"])
    indexed = {(row["item_id"], row["condition"]): row for row in records}
    if len(indexed) != len(records):
        raise SystemExit("Duplicate item-condition records in formal results")
    expected_keys = {
        (task["item_id"], condition)
        for task in tasks
        for condition in expected_conditions
    }
    if set(indexed) != expected_keys:
        missing = len(expected_keys - set(indexed))
        unexpected = len(set(indexed) - expected_keys)
        raise SystemExit(
            f"Incomplete formal panel: missing={missing}, unexpected={unexpected}"
        )
    samples = int(statistics["statistics"]["bootstrap_samples"])
    seed = int(statistics["statistics"]["bootstrap_seed"])

    h1_tasks = [task for task in tasks if 3 <= int(task["complexity"]) <= 5]
    h1_rows = _paired_rows(
        indexed,
        h1_tasks,
        left_condition="graph_program",
        right_condition="flat_hybrid",
    )
    h1_mcnemar = exact_mcnemar(
        [row["left"] for row in h1_rows], [row["right"] for row in h1_rows]
    )
    h1 = {
        "hypothesis": "H1",
        "items": len(h1_rows),
        "contrast": "graph_program_minus_flat_hybrid",
        **cluster_bootstrap(
            h1_rows, statistic=_accuracy_difference, samples=samples, seed=seed
        ),
        "evidence_f1_difference": cluster_bootstrap(
            h1_rows,
            statistic=_evidence_difference,
            samples=samples,
            seed=seed + 10,
        ),
        "mcnemar": h1_mcnemar,
        "matched_odds_ratio": matched_odds_ratio(h1_mcnemar),
    }

    h2_types = set(statistics["primary_hypotheses"]["H2"]["task_types"])
    h2_tasks = [task for task in tasks if task["task_type"] in h2_types]
    h2_rows = _paired_rows(
        indexed,
        h2_tasks,
        left_condition="graph_hierarchical_retrieval_v2",
        right_condition="flat_hybrid",
    )
    h2_mcnemar = exact_mcnemar(
        [row["left"] for row in h2_rows], [row["right"] for row in h2_rows]
    )
    h2 = {
        "hypothesis": "H2",
        "items": len(h2_rows),
        "contrast": "graph_hierarchical_retrieval_v2_minus_flat_hybrid",
        **cluster_bootstrap(
            h2_rows, statistic=_accuracy_difference, samples=samples, seed=seed + 1
        ),
        "evidence_f1_difference": cluster_bootstrap(
            h2_rows,
            statistic=_evidence_difference,
            samples=samples,
            seed=seed + 11,
        ),
        "mcnemar": h2_mcnemar,
        "matched_odds_ratio": matched_odds_ratio(h2_mcnemar),
    }

    task_by_id = {task["item_id"]: task for task in tasks}
    h3_group_rows = []
    h3_secondary_rows = []
    for pair in scale_pairs["pairs"]:
        outcomes = {}
        for scale in ("small", "medium", "large"):
            item_id = pair["item_ids"][scale]
            if item_id not in task_by_id:
                raise SystemExit(f"Scale-pair item missing from benchmark: {item_id}")
            for condition in ("graph_program", "flat_hybrid"):
                try:
                    outcomes[(scale, condition)] = _record_outcome(
                        indexed[(item_id, condition)]
                    )["correct"]
                except KeyError as exc:
                    raise SystemExit(f"Missing H3 result pair: {exc.args[0]}") from exc
        interaction = (
            float(outcomes[("large", "graph_program")])
            - float(outcomes[("large", "flat_hybrid")])
            - float(outcomes[("small", "graph_program")])
            + float(outcomes[("small", "flat_hybrid")])
        )
        h3_group_rows.append(
            {
                "dataset_id": pair["dataset_id"],
                "pair_id": pair["pair_id"],
                "interaction": interaction,
            }
        )
        medium_interaction = (
            float(outcomes[("medium", "graph_program")])
            - float(outcomes[("medium", "flat_hybrid")])
            - float(outcomes[("small", "graph_program")])
            + float(outcomes[("small", "flat_hybrid")])
        )
        h3_secondary_rows.append(
            {
                "dataset_id": pair["dataset_id"],
                "pair_id": pair["pair_id"],
                "interaction": medium_interaction,
            }
        )
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in h3_group_rows:
        by_dataset[row["dataset_id"]].append(row["interaction"])
    dataset_effects = [
        sum(values) / len(values) for _, values in sorted(by_dataset.items())
    ]
    h3_bootstrap = cluster_bootstrap(
        h3_group_rows,
        statistic=lambda rows: sum(row["interaction"] for row in rows) / len(rows),
        samples=samples,
        seed=seed + 2,
    )
    h3 = {
        "hypothesis": "H3",
        "groups": len(h3_group_rows),
        "items": len(h3_group_rows) * 2,
        "contrast": "small_to_large_graph_program_vs_flat_hybrid_interaction",
        **h3_bootstrap,
        "cluster_signflip": exact_cluster_signflip(dataset_effects),
        "secondary_small_to_medium": cluster_bootstrap(
            h3_secondary_rows,
            statistic=lambda rows: sum(row["interaction"] for row in rows) / len(rows),
            samples=samples,
            seed=seed + 12,
        ),
    }

    raw_p = {
        "H1": h1["mcnemar"]["p_value"],
        "H2": h2["mcnemar"]["p_value"],
        "H3": h3["cluster_signflip"]["p_value_one_sided"],
    }
    adjusted = holm_adjust(raw_p)
    for hypothesis in (h1, h2, h3):
        hypothesis["raw_p_value"] = raw_p[hypothesis["hypothesis"]]
        hypothesis["holm_adjusted_p_value"] = adjusted[hypothesis["hypothesis"]]

    summaries = {}
    for condition in sorted({row["condition"] for row in records}):
        subset = [row for row in records if row["condition"] == condition]
        outcomes = [_record_outcome(row) for row in subset]
        summaries[condition] = {
            "items": len(subset),
            "accuracy": sum(row["correct"] for row in outcomes) / len(outcomes),
            "mean_evidence_f1": sum(row["evidence_f1"] for row in outcomes)
            / len(outcomes),
            "prompt_tokens": sum(row["prompt_tokens"] for row in outcomes),
            "completion_tokens": sum(row["completion_tokens"] for row in outcomes),
            "mean_latency_seconds": sum(row["latency_seconds"] for row in outcomes)
            / len(outcomes),
            "failed_parse": sum(row["status"] != "complete" for row in outcomes),
        }
    output = {
        "schema_version": 1,
        "status": "analyzed",
        "protocol_sha256": sha256_file(args.protocol),
        "neural_dense_amendment_sha256": neural_hash,
        "statistics_amendment_sha256": sha256_file(args.statistics_amendment),
        "equal_dataset_weighting_amendment_sha256": weighting_hash,
        "scale_pairs_sha256": sha256_file(args.scale_pairs),
        "datasets": len(construction["records"]),
        "tasks": len(tasks),
        "records": len(records),
        "condition_summaries": summaries,
        "primary_hypotheses": {"H1": h1, "H2": h2, "H3": h3},
        "holm_family": {"raw": raw_p, "adjusted": adjusted},
        "interpretation_guard": (
            "H3 has four dataset clusters; its minimum exact one-sided p-value is 0.0625. "
            "Item-level pseudo-replication must not replace the registered cluster test."
        ),
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
