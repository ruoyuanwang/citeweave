"""Frozen, CPU-only exploratory robustness appendix for all eight article topics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from citeweave.formal_request import canonical_sha256
from citeweave.graph_discovery import _load_graph
from citeweave.graph_robustness import (
    baseline_checks,
    compare_variants,
    load_verified_trends,
    measure_variant,
    perturb_graph,
    registered_variants,
)
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
BENCHMARKS = BASE / "formal_v3_complexity_extension_benchmarks"
WORKSPACES = ROOT / "experiments/formal_v3_identity_corrected_workspaces"
OUTPUT = BASE / "graph_phenomenon_robustness_v1"
FREEZE = OUTPUT / "input_freeze.json"
PACKS = ROOT / "experiments/article_quality_v2/same_evidence_writer_packs_v4"


def verify_bindings(payload: dict) -> None:
    for name, expected in payload["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Robustness input drift: {name}")
    for package, version in payload["package_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Robustness runtime drift: {package}")


def freeze() -> dict:
    if FREEZE.exists():
        payload = read_json(FREEZE)
        verify_bindings(payload)
        return payload
    if OUTPUT.exists() and any(OUTPUT.glob("*/*.json")):
        raise ValueError("Cannot freeze a design after its robustness outcomes")
    construction = read_json(BENCHMARKS / "construction_manifest.json")
    if len(construction["records"]) != 8:
        raise ValueError("All eight topics required")
    paths = {
        BENCHMARKS / "construction_manifest.json",
        Path(__file__).resolve(),
        ROOT / "src/citeweave/graph_robustness.py",
        ROOT / "src/citeweave/graph_discovery.py",
        ROOT / "src/citeweave/formal_request.py",
        ROOT / "src/citeweave/io.py",
        ROOT / "tests/test_graph_robustness.py",
        PACKS / "manifest.json",
    }
    topics = []
    for row in construction["records"]:
        topic = row["dataset_id"]
        workspace = WORKSPACES / row["source_panel"] / topic
        original_root = (
            ROOT
            / "experiments"
            / (
                "formal_v3_workspaces"
                if row["source_panel"] == "primary"
                else "formal_v3_replication_workspaces"
            )
        )
        original_trends = original_root / topic / "analyses/keyword_trends.parquet"
        benchmark = BENCHMARKS / topic / "benchmark.json"
        pack = PACKS / topic / "evidence_pack.json"
        if sha256_file(benchmark) != row["benchmark_sha256"]:
            raise ValueError("Benchmark drift")
        paths.update(
            [
                benchmark,
                pack,
                workspace / "identity_correction_verification.json",
                workspace / "canonical/visualization/keyword_occurrences.parquet",
                workspace / "canonical/visualization/keyword_cooccurrence_edges.parquet",
                original_trends,
                workspace / "canonical/keywords.parquet",
                workspace / "canonical/works.parquet",
            ]
        )
        topics.append(
            {
                "dataset_id": topic,
                "workspace": str(workspace),
                "benchmark": str(benchmark),
                "article_pack": str(pack),
                "original_trends": str(original_trends),
            }
        )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "exploratory_sensitivity_design_frozen_before_its_outcomes",
        "variants": registered_variants(),
        "topics": topics,
        "planned_graph_cases": 8 * len(registered_variants()),
        "source_sha256": {str(path): sha256_file(path) for path in sorted(paths)},
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("networkx", "numpy", "pandas", "scipy", "scikit-learn", "duckdb")
        },
        "design_notes": [
            "One factor at a time; not a full factorial and no robustness p-values.",
            "All registered baseline vertices retained, including induced isolates.",
            "Frozen path endpoints, edge, and hub; edge highest-betweenness selection is not retested.",
            "Community and temporal roles reselected; memberships compared using ARI/Jaccard, not label IDs.",
            "Temporal candidate table remains the original top-keyword table; full-period topology retained.",
            "No changes to confirmatory benchmarks, scoring, writer inputs, or human review packets.",
            "This is not a model/human performance result or evidence of real-world causal resilience.",
        ],
        "references": [
            "https://arxiv.org/abs/0910.0165",
            "https://www.nature.com/articles/s41598-019-41695-z",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FREEZE.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload


def summarize(results: list[dict]) -> dict:
    families = {}
    for family in sorted({r["variant"]["family"] for r in results} - {"baseline"}):
        subset = [r for r in results if r["variant"]["family"] == family]
        comparisons = [r["comparison"] for r in subset]
        aris = [r["partition_adjusted_rand_index"] for r in comparisons]
        metrics = {}
        for comparison in comparisons:
            for task, values in comparison.items():
                if not isinstance(values, dict):
                    continue
                for metric, value in values.items():
                    if isinstance(value, bool):
                        counts = metrics.setdefault(f"{task}/{metric}", {"true": 0, "total": 0})
                        counts["true"] += int(value)
                        counts["total"] += 1
                    elif isinstance(value, (int, float)):
                        metrics.setdefault(f"{task}/{metric}", {"values": []})["values"].append(
                            value
                        )
        for metric in metrics.values():
            if "values" in metric:
                values = metric.pop("values")
                metric.update(
                    {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "n": len(values),
                    }
                )
        families[family] = {
            "graph_cases": len(subset),
            "ari_min": min(aris),
            "ari_max": max(aris),
            "ari_mean": sum(aris) / len(aris),
            "metrics": metrics,
        }
    return {
        "schema_version": 1,
        "status": "exploratory_diagnostics_complete",
        "graph_cases": len(results),
        "topics": len({r["dataset_id"] for r in results}),
        "families": families,
        "inferential_unit": "None; descriptive ranges across explicit perturbations, not independent trials.",
        "missing_or_noncomparable_cases": [
            {
                "dataset_id": row["dataset_id"],
                "variant_id": row["variant"]["variant_id"],
                "task": task,
                "status": values["status"],
            }
            for row in results
            for task, values in row["measurements"].items()
            if values["status"] != "measured"
        ],
    }


def execute(payload: dict) -> None:
    verify_bindings(payload)
    freeze_sha = sha256_file(FREEZE)
    all_results = []
    for topic in payload["topics"]:
        benchmark = read_json(Path(topic["benchmark"]))
        tasks = [t for t in benchmark["tasks"] if t["scale"] == "large" and t["complexity"] > 1]
        if len(tasks) != 5 or any(t["network"] != "keyword_cooccurrence" for t in tasks):
            raise ValueError("Expected five Large keyword phenomena per topic")
        pack = read_json(Path(topic["article_pack"]))
        # The article-pack source is frozen, and the corresponding benchmark is unchanged.
        if pack["source_benchmark_sha256"] != sha256_file(Path(topic["benchmark"])):
            raise ValueError("Article source benchmark differs")
        graph, _ = _load_graph(Path(topic["workspace"]), "keyword_cooccurrence", "large")
        trends = load_verified_trends(Path(topic["workspace"]), Path(topic["original_trends"]))
        baseline, base_partition = measure_variant(graph, tasks, trends, payload["variants"][0])
        checks = baseline_checks(baseline["measurements"], tasks)
        if not all(row["passed"] for row in checks):
            write_json(OUTPUT / topic["dataset_id"] / "baseline_failure.json", {"checks": checks})
            raise ValueError("Baseline arithmetic does not match registered article phenomena")
        for spec in payload["variants"]:
            started = time.perf_counter()
            output = OUTPUT / topic["dataset_id"] / f"{spec['variant_id']}.json"
            identity = canonical_sha256(
                {"freeze_sha256": freeze_sha, "dataset_id": topic["dataset_id"], "variant": spec}
            )
            if output.exists():
                row = read_json(output)
                if row.get("run_identity") != identity or row.get(
                    "payload_sha256"
                ) != canonical_sha256({k: v for k, v in row.items() if k != "payload_sha256"}):
                    raise ValueError(f"Existing robustness result drift: {output}")
            else:
                if spec["family"] == "baseline":
                    row, communities = baseline, base_partition
                else:
                    perturbed = perturb_graph(graph, spec)
                    row, communities = measure_variant(
                        perturbed,
                        tasks,
                        trends,
                        spec,
                        precomputed_partition=base_partition
                        if spec["family"] == "temporal_window"
                        else None,
                    )
                row = {
                    **row,
                    "dataset_id": topic["dataset_id"],
                    "run_identity": identity,
                    "baseline_checks": checks,
                    "comparison": compare_variants(baseline, row, base_partition, communities),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                row["payload_sha256"] = canonical_sha256(row)
                write_json(output, row)
            all_results.append(row)
            print(
                json.dumps(
                    {
                        "dataset_id": topic["dataset_id"],
                        "variant": spec["variant_id"],
                        "completed_cases": len(all_results),
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    verify_bindings(payload)
    if len(all_results) != payload["planned_graph_cases"]:
        raise ValueError("Incomplete sensitivity matrix")
    summary = summarize(all_results)
    summary["input_freeze_sha256"] = freeze_sha
    summary["result_sha256"] = {
        str(OUTPUT / r["dataset_id"] / f"{r['variant']['variant_id']}.json"): sha256_file(
            OUTPUT / r["dataset_id"] / f"{r['variant']['variant_id']}.json"
        )
        for r in all_results
    }
    write_json(OUTPUT / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze()
    else:
        if not FREEZE.is_file():
            raise SystemExit("Freeze the exploratory design before execution")
        lock = OUTPUT / "worker.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                "Worker lock exists; verify owner before manually clearing stale lock"
            ) from None
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            execute(read_json(FREEZE))
        finally:
            lock.unlink()
    print(json.dumps({"status": "ok", "python": sys.version.split()[0]}), flush=True)


if __name__ == "__main__":
    main()
