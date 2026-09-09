from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from citeweave.io import read_json, sha256_file, write_json


def build_replication_plan(
    *,
    protocol_path: Path,
    benchmark_root: Path,
    results_root: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    construction = read_json(benchmark_root / "construction_manifest.json")
    readiness = read_json(readiness_path)
    conditions = list(protocol["conditions"]["confirmatory"])
    datasets = []
    for record in construction["records"]:
        dataset_id = record["dataset_id"]
        benchmark = benchmark_root / dataset_id / "benchmark.json"
        if sha256_file(benchmark) != record["benchmark_sha256"]:
            raise ValueError(f"Replication benchmark hash mismatch: {dataset_id}")
        task_count = len(read_json(benchmark)["tasks"])
        datasets.append(
            {
                "dataset_id": dataset_id,
                "benchmark": str(benchmark.resolve()),
                "benchmark_sha256": record["benchmark_sha256"],
                "tasks": task_count,
                "conditions": len(conditions),
                "calls": task_count * len(conditions),
                "output": str((results_root / dataset_id).resolve()),
            }
        )
    return {
        "schema_version": 1,
        "status": "ready_to_execute" if readiness.get("status") == "ready" else "blocked",
        "protocol_sha256": sha256_file(protocol_path),
        "readiness_sha256": sha256_file(readiness_path),
        "readiness_status": readiness.get("status"),
        "blocking_reasons": readiness.get("blocking_reasons") or [],
        "conditions": conditions,
        "datasets": datasets,
        "tasks": sum(row["tasks"] for row in datasets),
        "calls": sum(row["calls"] for row in datasets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_replication_plan(
        protocol_path=args.protocol,
        benchmark_root=args.benchmark_root,
        results_root=args.results_root,
        readiness_path=args.readiness,
    )
    if args.output_plan:
        write_json(args.output_plan, plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if plan["status"] != "ready_to_execute":
        raise SystemExit(
            "Formal replication execution is blocked: "
            + "; ".join(plan["blocking_reasons"])
        )
    runner = Path(__file__).with_name("run_graph_discovery_experiment.py")
    for dataset in plan["datasets"]:
        command = [
            sys.executable,
            str(runner),
            "--benchmark",
            dataset["benchmark"],
            "--output",
            dataset["output"],
            "--readiness",
            str(args.readiness),
            "--conditions",
            *plan["conditions"],
            "--execute",
        ]
        if args.api_key_file:
            command.extend(["--api-key-file", str(args.api_key_file)])
        subprocess.run(command, check=True, shell=False)


if __name__ == "__main__":
    main()
