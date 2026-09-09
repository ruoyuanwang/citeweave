from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file, write_json

SIMPLE_TASKS = ("direct_edge_lookup", "node_attribute_lookup")
COMPLEX_TASKS = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)
ALL_CONDITIONS = (
    "flat_hybrid",
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
    "graph_program",
)
SUPPLEMENT_CONDITIONS = (
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
)


def build_extension_plan(
    *, benchmark_root: Path, execution_root: Path, readiness_path: Path
) -> dict[str, Any]:
    construction = read_json(benchmark_root / "construction_manifest.json")
    readiness = read_json(readiness_path)
    jobs = []
    for record in construction["records"]:
        dataset_id = record["dataset_id"]
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        if sha256_file(benchmark_path) != record["benchmark_sha256"]:
            raise ValueError(f"Complexity-extension benchmark hash mismatch: {dataset_id}")
        benchmark = read_json(benchmark_path)
        simple_count = sum(task["task_type"] in SIMPLE_TASKS for task in benchmark["tasks"])
        if simple_count != 6:
            raise ValueError(f"Complexity-extension simple-task count mismatch: {dataset_id}")
        jobs.append(
            {
                "job_id": f"{dataset_id}:simple_controls",
                "dataset_id": dataset_id,
                "source_panel": record["source_panel"],
                "benchmark": str(benchmark_path.resolve()),
                "benchmark_sha256": record["benchmark_sha256"],
                "task_types": list(SIMPLE_TASKS),
                "tasks": simple_count,
                "conditions": list(ALL_CONDITIONS),
                "calls": simple_count * len(ALL_CONDITIONS),
                "output": str(
                    (execution_root / dataset_id / "simple_controls").resolve()
                ),
            }
        )
        if record["source_panel"] == "replication":
            complex_count = sum(
                task["task_type"] in COMPLEX_TASKS for task in benchmark["tasks"]
            )
            if complex_count != 15:
                raise ValueError(
                    f"Complexity-extension complex-task count mismatch: {dataset_id}"
                )
            jobs.append(
                {
                    "job_id": f"{dataset_id}:complex_supplement",
                    "dataset_id": dataset_id,
                    "source_panel": "replication",
                    "benchmark": str(benchmark_path.resolve()),
                    "benchmark_sha256": record["benchmark_sha256"],
                    "task_types": list(COMPLEX_TASKS),
                    "tasks": complex_count,
                    "conditions": list(SUPPLEMENT_CONDITIONS),
                    "calls": complex_count * len(SUPPLEMENT_CONDITIONS),
                    "output": str(
                        (execution_root / dataset_id / "complex_supplement").resolve()
                    ),
                }
            )
    if sum(job["calls"] for job in jobs) != 312:
        raise ValueError("Complexity-extension incremental call total must equal 312")
    return {
        "schema_version": 1,
        "status": "ready_to_execute" if readiness.get("status") == "ready" else "blocked",
        "protocol_sha256": construction["protocol_sha256"],
        "execution_amendment_sha256": readiness.get("execution_amendment_sha256"),
        "readiness_sha256": sha256_file(readiness_path),
        "readiness_status": readiness.get("status"),
        "blocking_reasons": readiness.get("blocking_reasons") or [],
        "logical_cells": 672,
        "reused_cells": 360,
        "incremental_calls": 312,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_extension_plan(
        benchmark_root=args.benchmark_root,
        execution_root=args.execution_root,
        readiness_path=args.readiness,
    )
    if args.output_plan:
        write_json(args.output_plan, plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if plan["status"] != "ready_to_execute":
        raise SystemExit(
            "Complexity-extension execution is blocked: "
            + "; ".join(plan["blocking_reasons"])
        )
    runner = Path(__file__).with_name("run_graph_discovery_experiment.py")
    for job in plan["jobs"]:
        command = [
            sys.executable,
            str(runner),
            "--benchmark",
            job["benchmark"],
            "--output",
            job["output"],
            "--readiness",
            str(args.readiness),
            "--conditions",
            *job["conditions"],
        ]
        for task_type in job["task_types"]:
            command.extend(["--task-type", task_type])
        if args.api_key_file:
            command.extend(["--api-key-file", str(args.api_key_file)])
        command.append("--execute")
        subprocess.run(command, check=True, shell=False)


if __name__ == "__main__":
    main()
