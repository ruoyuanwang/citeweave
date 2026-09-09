from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from citeweave.io import read_json, sha256_file, write_json


def exact_cluster_signflip(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Sign-flip test requires cluster effects")
    observed = sum(values) / len(values)
    permutations = [
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return {
        "estimate": observed,
        "cluster_effects": values,
        "clusters": len(values),
        "permutations": len(permutations),
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in permutations)
        / len(permutations),
        "minimum_attainable_p": 1.0 / len(permutations),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    clusters = sorted({row["dataset_id"] for row in rows})
    grouped = {
        cluster: [row for row in rows if row["dataset_id"] == cluster]
        for cluster in clusters
    }

    dataset_effects = {
        cluster: sum(row["interaction"] for row in cluster_rows) / len(cluster_rows)
        for cluster, cluster_rows in grouped.items()
    }

    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = generator.choice(clusters, size=len(clusters), replace=True)
        draws.append(
            sum(dataset_effects[str(cluster)] for cluster in selected) / len(selected)
        )
    return {
        "estimate": sum(dataset_effects.values()) / len(dataset_effects),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": len(clusters),
        "cluster_weighting": "equal_dataset",
    }


def _correct(record: dict[str, Any]) -> bool:
    return bool(
        record.get("status", "complete") == "complete"
        and (record.get("score") or {}).get("answer_exact")
    )


def _load_phase(
    *,
    phase: str,
    benchmark_root: Path,
    results_root: Path,
    scale_pair_path: Path,
) -> dict[str, Any]:
    construction = read_json(benchmark_root / "construction_manifest.json")
    scale_pairs = read_json(scale_pair_path)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    benchmark_hashes = {}
    for construction_record in construction["records"]:
        dataset_id = construction_record["dataset_id"]
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        benchmark_hash = sha256_file(benchmark_path)
        benchmark_hashes[dataset_id] = benchmark_hash
        if benchmark_hash != construction_record["benchmark_sha256"]:
            raise ValueError(f"{phase} benchmark hash mismatch: {dataset_id}")
        result_path = results_root / dataset_id / "results.json"
        if not result_path.is_file():
            raise ValueError(f"{phase} result missing: {result_path}")
        result = read_json(result_path)
        if result["manifest"]["benchmark_sha256"] != benchmark_hash:
            raise ValueError(f"{phase} result benchmark mismatch: {dataset_id}")
        for record in result["records"]:
            key = (record["item_id"], record["condition"])
            if key in indexed:
                raise ValueError(f"{phase} duplicate item-condition: {key}")
            indexed[key] = record
    if scale_pairs.get("benchmark_hashes") != benchmark_hashes:
        raise ValueError(f"{phase} scale-pair benchmark hashes mismatch")
    rows = []
    for pair in scale_pairs["pairs"]:
        outcomes = {}
        for scale in ("small", "large"):
            item_id = pair["item_ids"][scale]
            for condition in ("graph_program", "flat_hybrid"):
                key = (item_id, condition)
                if key not in indexed:
                    raise ValueError(f"{phase} missing registered H3 pair: {key}")
                outcomes[(scale, condition)] = _correct(indexed[key])
        interaction = (
            float(outcomes[("large", "graph_program")])
            - float(outcomes[("large", "flat_hybrid")])
            - float(outcomes[("small", "graph_program")])
            + float(outcomes[("small", "flat_hybrid")])
        )
        rows.append(
            {
                "phase": phase,
                "dataset_id": pair["dataset_id"],
                "pair_id": pair["pair_id"],
                "task_type": pair["task_type"],
                "interaction": interaction,
            }
        )
    return {
        "phase": phase,
        "scale_pairs_sha256": sha256_file(scale_pair_path),
        "groups": len(rows),
        "rows": rows,
    }


def _summarize(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_id"]].append(row["interaction"])
    dataset_effects = {
        dataset: sum(values) / len(values)
        for dataset, values in sorted(by_dataset.items())
    }
    return {
        "groups": len(rows),
        "datasets": len(dataset_effects),
        "dataset_effects": dataset_effects,
        "cluster_bootstrap": _cluster_bootstrap(rows, samples=samples, seed=seed),
        "cluster_signflip": exact_cluster_signflip(list(dataset_effects.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-benchmark-root", type=Path, required=True)
    parser.add_argument("--primary-results-root", type=Path, required=True)
    parser.add_argument("--primary-scale-pairs", type=Path, required=True)
    parser.add_argument("--replication-benchmark-root", type=Path, required=True)
    parser.add_argument("--replication-results-root", type=Path, required=True)
    parser.add_argument("--replication-scale-pairs", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--statistics-amendment",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/"
            "formal_v3_identity_corrected_statistics_v2/statistics.yml"
        ),
    )
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
        raise SystemExit("Scale-replication implementation differs from amendment")
    primary = _load_phase(
        phase="primary",
        benchmark_root=args.primary_benchmark_root,
        results_root=args.primary_results_root,
        scale_pair_path=args.primary_scale_pairs,
    )
    replication = _load_phase(
        phase="replication",
        benchmark_root=args.replication_benchmark_root,
        results_root=args.replication_results_root,
        scale_pair_path=args.replication_scale_pairs,
    )
    primary_summary = _summarize(
        primary["rows"], samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    replication_summary = _summarize(
        replication["rows"],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )
    combined_rows = [*primary["rows"], *replication["rows"]]
    combined_summary = _summarize(
        combined_rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed + 2
    )
    output = {
        "schema_version": 1,
        "status": "analyzed",
        "estimand": (
            "(graph_program_large-flat_hybrid_large)-"
            "(graph_program_small-flat_hybrid_small)"
        ),
        "equal_dataset_weighting_amendment_sha256": weighting_hash,
        "primary": {**primary_summary, "scale_pairs_sha256": primary["scale_pairs_sha256"]},
        "replication": {
            **replication_summary,
            "scale_pairs_sha256": replication["scale_pairs_sha256"],
        },
        "combined": combined_summary,
        "replication_minus_primary_effect": (
            replication_summary["cluster_bootstrap"]["estimate"]
            - primary_summary["cluster_bootstrap"]["estimate"]
        ),
        "interpretation_guard": (
            "The combined exact test treats dataset as the cluster. Scale-pair items are "
            "not independent replicates, and primary and replication estimates must also "
            "be reported separately."
        ),
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
