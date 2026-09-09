from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.graph_discovery import build_simple_control_benchmark
from citeweave.io import read_json, sha256_file, write_json

TASK_TYPES = (
    "direct_edge_lookup",
    "node_attribute_lookup",
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)


def _validate_matches(tasks: list[dict]) -> None:
    indexed = {task["item_id"]: task for task in tasks}
    simple = [task for task in tasks if task["complexity"] == 1]
    if len(simple) != 6:
        raise SystemExit("Each dataset must contain two simple controls at three scales")
    for task in simple:
        parent_id = task.get("matched_complex_item_id")
        if parent_id not in indexed:
            raise SystemExit(f"Simple control lacks its complex parent: {task['item_id']}")
        parent = indexed[parent_id]
        if task["task_type"] == "direct_edge_lookup":
            if task["evidence_ids"] != parent["evidence_ids"]:
                raise SystemExit(f"Edge control is not evidence-anchor matched: {task['item_id']}")
        elif task["evidence_ids"][0] != parent["evidence_ids"][0]:
            raise SystemExit(f"Node control is not hub-anchor matched: {task['item_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--primary-workspaces", type=Path, required=True)
    parser.add_argument("--replication-workspaces", type=Path, required=True)
    parser.add_argument("--primary-benchmarks", type=Path, required=True)
    parser.add_argument("--replication-benchmarks", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    freeze = read_json(args.freeze)
    if freeze.get("sha256") != protocol_hash:
        raise SystemExit("Complexity-extension protocol differs from freeze artifact")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_after_corpus_acquisition_before_extension_task_construction_and_model_outcomes"
    ):
        raise SystemExit("Complexity-extension protocol has an invalid freeze status")

    records = []
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        workspace_root = (
            args.primary_workspaces
            if dataset["source_panel"] == "primary"
            else args.replication_workspaces
        )
        workspace = workspace_root / dataset_id
        benchmark_root = (
            args.primary_benchmarks
            if dataset["source_panel"] == "primary"
            else args.replication_benchmarks
        )
        if not (workspace / "canonical" / "works.parquet").is_file():
            raise SystemExit(f"Missing frozen canonical workspace: {dataset_id}")
        frozen_benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        if not frozen_benchmark_path.is_file():
            raise SystemExit(f"Missing frozen source benchmark: {dataset_id}")
        frozen_benchmark = read_json(frozen_benchmark_path)
        complex_tasks = [
            task
            for task in frozen_benchmark["tasks"]
            if task["network"] == "keyword_cooccurrence"
            and task["task_type"] in TASK_TYPES
            and task["complexity"] > 1
        ]
        if len(complex_tasks) != 15:
            raise SystemExit(f"Frozen source benchmark lacks 15 complex tasks: {dataset_id}")
        output = args.output_root / dataset_id
        simple_component = build_simple_control_benchmark(
            workspace,
            output / "components" / "anchor_matched_simple_controls",
            parent_tasks=complex_tasks,
            record_budget=protocol["factorial_design"]["record_budget"],
        )
        tasks = [*complex_tasks, *simple_component["tasks"]]
        if len(tasks) != protocol["factorial_design"]["expected_tasks_per_dataset"]:
            raise SystemExit(f"Complexity-extension task count mismatch: {dataset_id}")
        _validate_matches(tasks)
        benchmark = {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "formal_v3_complexity_extension": {
                "confirmatory_extension": True,
                "status": "constructed_not_executed",
                "protocol_sha256": protocol_hash,
                "source_panel": dataset["source_panel"],
            },
            "design": {
                **frozen_benchmark["design"],
                "conditions": protocol["factorial_design"]["conditions"],
                "primary_hypotheses": ["C1", "C2"],
                "registered_anchor_matches": protocol["factorial_design"][
                    "registered_anchor_matches"
                ],
            },
            "scales": [
                row
                for row in frozen_benchmark["scales"]
                if row["network"] == "keyword_cooccurrence"
            ],
            "tasks": tasks,
        }
        benchmark_path = output / "benchmark.json"
        write_json(benchmark_path, benchmark)
        record = {
            "dataset_id": dataset_id,
            "source_panel": dataset["source_panel"],
            "status": "constructed_not_executed",
            "tasks": len(tasks),
            "simple_tasks": sum(task["complexity"] == 1 for task in tasks),
            "complex_tasks": sum(task["complexity"] > 1 for task in tasks),
            "benchmark_sha256": sha256_file(benchmark_path),
        }
        records.append(record)
        write_json(
            args.output_root / "construction_manifest.json",
            {
                "schema_version": 1,
                "status": "constructing",
                "protocol_sha256": protocol_hash,
                "records": records,
            },
        )
        print(dataset_id, len(tasks), record["simple_tasks"], record["complex_tasks"])

    manifest = {
        "schema_version": 1,
        "status": "constructed_not_executed",
        "protocol_sha256": protocol_hash,
        "datasets": len(records),
        "tasks": sum(row["tasks"] for row in records),
        "planned_calls": sum(row["tasks"] for row in records)
        * len(protocol["factorial_design"]["conditions"]),
        "records": records,
    }
    write_json(args.output_root / "construction_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
