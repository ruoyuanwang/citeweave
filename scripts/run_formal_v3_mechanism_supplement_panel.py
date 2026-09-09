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
SUPPLEMENT_CONDITIONS = ("flat_program", "operator_only")


def build_mechanism_supplement_plan(
    *,
    protocol_path: Path,
    protocol_freeze_path: Path,
    benchmark_root: Path,
    execution_root: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    if read_json(protocol_freeze_path).get("sha256") != protocol_hash:
        raise ValueError("Mechanism-supplement protocol differs from its freeze")
    construction = read_json(benchmark_root / "construction_manifest.json")
    if construction.get("datasets") != 8 or construction.get("tasks") != 168:
        raise ValueError("Mechanism supplement requires the frozen 8 x 21 task panel")
    readiness = read_json(readiness_path) if readiness_path.is_file() else {}
    jobs = []
    for record in construction.get("records") or []:
        dataset_id = record["dataset_id"]
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        if sha256_file(benchmark_path) != record["benchmark_sha256"]:
            raise ValueError(f"Mechanism benchmark hash mismatch: {dataset_id}")
        benchmark = read_json(benchmark_path)
        simple_count = sum(
            task["task_type"] in SIMPLE_TASKS for task in benchmark["tasks"]
        )
        complex_count = sum(
            task["task_type"] in COMPLEX_TASKS for task in benchmark["tasks"]
        )
        if (simple_count, complex_count) != (6, 15):
            raise ValueError(f"Mechanism task signature mismatch: {dataset_id}")
        for task in benchmark["tasks"]:
            if not set(SUPPLEMENT_CONDITIONS) <= set(task.get("contexts") or {}):
                raise ValueError(
                    f"Mechanism context missing for {dataset_id}:{task['item_id']}"
                )
        jobs.append(
            {
                "job_id": f"{dataset_id}:simple_same_computation",
                "dataset_id": dataset_id,
                "source_panel": record["source_panel"],
                "benchmark": str(benchmark_path.resolve()),
                "benchmark_sha256": record["benchmark_sha256"],
                "task_types": list(SIMPLE_TASKS),
                "tasks": simple_count,
                "conditions": list(SUPPLEMENT_CONDITIONS),
                "calls": simple_count * len(SUPPLEMENT_CONDITIONS),
                "output": str(
                    (execution_root / dataset_id / "simple_same_computation").resolve()
                ),
            }
        )
        if record["source_panel"] == "replication":
            jobs.append(
                {
                    "job_id": f"{dataset_id}:complex_same_computation",
                    "dataset_id": dataset_id,
                    "source_panel": "replication",
                    "benchmark": str(benchmark_path.resolve()),
                    "benchmark_sha256": record["benchmark_sha256"],
                    "task_types": list(COMPLEX_TASKS),
                    "tasks": complex_count,
                    "conditions": list(SUPPLEMENT_CONDITIONS),
                    "calls": complex_count * len(SUPPLEMENT_CONDITIONS),
                    "output": str(
                        (
                            execution_root
                            / dataset_id
                            / "complex_same_computation"
                        ).resolve()
                    ),
                }
            )
    if len(jobs) != 12 or sum(job["calls"] for job in jobs) != 216:
        raise ValueError("Mechanism supplement must contain 12 jobs and 216 calls")
    ready = bool(
        readiness.get("status") == "ready"
        and readiness.get("supplement_protocol_sha256") == protocol_hash
        and set(SUPPLEMENT_CONDITIONS)
        <= set(readiness.get("conditions") or [])
        and len(readiness.get("cell_identity_manifest") or []) == 336
    )
    return {
        "schema_version": 1,
        "status": "ready_to_execute" if ready else "blocked",
        "supplement_protocol_sha256": protocol_hash,
        "readiness_sha256": sha256_file(readiness_path)
        if readiness_path.is_file()
        else None,
        "readiness_status": readiness.get("status"),
        "blocking_reasons": []
        if ready
        else ["mechanism_supplement_readiness_not_ready"],
        "logical_cells": 504,
        "reused_cells": 288,
        "incremental_calls": 216,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_mechanism_supplement_plan(
        protocol_path=args.protocol,
        protocol_freeze_path=args.protocol_freeze,
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
        raise SystemExit("Mechanism-supplement execution is blocked by readiness")
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
