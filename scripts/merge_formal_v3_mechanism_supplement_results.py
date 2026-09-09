from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from citeweave.formal_request import PROMPT_VERSION, task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json

CONDITIONS = ("graph_program", "flat_program", "operator_only")
IDENTITY_FIELDS = (
    "task_payload_sha256",
    "context_sha256",
    "request_messages_sha256",
)


def verify_cell_identity(
    record: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    mismatches = [field for field in IDENTITY_FIELDS if record.get(field) != expected.get(field)]
    return {"passed": not mismatches, "mismatches": mismatches}


def _result_index(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Required result file missing: {path}")
    payload = read_json(path)
    records = payload.get("records") or []
    indexed = {
        (str(record["item_id"]), str(record["condition"])): record
        for record in records
    }
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate final result cell: {path}")
    return indexed, payload.get("manifest") or {}


def merge_mechanism_supplement(
    *,
    benchmark_root: Path,
    primary_plan_path: Path,
    extension_merged_root: Path,
    supplement_plan_path: Path,
    extension_readiness_path: Path,
    supplement_readiness_path: Path,
) -> dict[str, Any]:
    construction = read_json(benchmark_root / "construction_manifest.json")
    primary_plan = read_json(primary_plan_path)
    supplement_plan = read_json(supplement_plan_path)
    extension_readiness = read_json(extension_readiness_path)
    supplement_readiness = read_json(supplement_readiness_path)
    if extension_readiness.get("status") != "ready":
        raise ValueError("Extension readiness is not ready")
    if supplement_readiness.get("status") != "ready":
        raise ValueError("Mechanism-supplement readiness is not ready")
    if supplement_plan.get("status") != "ready_to_execute":
        raise ValueError("Mechanism-supplement plan is not ready")
    extension_merge = read_json(extension_merged_root / "merge_manifest.json")
    if (
        extension_merge.get("status") != "merged_complete"
        or extension_merge.get("logical_cells") != 672
    ):
        raise ValueError("Complexity-extension merge is incomplete")

    extension_identities = {
        (str(row["dataset_id"]), str(row["item_id"]), str(row["condition"])): row
        for row in extension_readiness.get("cell_identity_manifest") or []
    }
    supplement_identities = {
        (str(row["dataset_id"]), str(row["item_id"]), str(row["condition"])): row
        for row in supplement_readiness.get("cell_identity_manifest") or []
    }
    if len(extension_identities) != 672 or len(supplement_identities) != 336:
        raise ValueError("Frozen cell identity manifests are incomplete")
    tokenizer_hash = supplement_readiness.get("tokenizer_manifest_sha256")
    primary_by_dataset = {
        str(row["dataset_id"]): row for row in primary_plan.get("datasets") or []
    }
    jobs = {
        str(row["job_id"]): row for row in supplement_plan.get("jobs") or []
    }
    tasks = []
    merged_records = []
    source_inventory: dict[str, str] = {}
    counts = {"extension_graph_program": 0, "primary_reuse": 0, "incremental": 0}

    for construction_record in construction.get("records") or []:
        dataset_id = str(construction_record["dataset_id"])
        source_panel = str(construction_record["source_panel"])
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        if sha256_file(benchmark_path) != construction_record.get("benchmark_sha256"):
            raise ValueError(f"Extension benchmark hash mismatch: {dataset_id}")
        benchmark = read_json(benchmark_path)
        extension_path = extension_merged_root / dataset_id / "results.json"
        extension_index, extension_manifest = _result_index(extension_path)
        if extension_manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
            raise ValueError(f"Extension result benchmark mismatch: {dataset_id}")
        source_inventory[str(extension_path.resolve())] = sha256_file(extension_path)

        primary_index: dict[tuple[str, str], dict[str, Any]] = {}
        primary_benchmark_tasks: dict[str, dict[str, Any]] = {}
        primary_path: Path | None = None
        if source_panel == "primary":
            plan_row = primary_by_dataset[dataset_id]
            primary_benchmark_path = Path(plan_row["benchmark"])
            primary_benchmark_tasks = {
                str(task["item_id"]): task
                for task in read_json(primary_benchmark_path)["tasks"]
            }
            primary_path = Path(plan_row["output"]) / "results.json"
            primary_index, primary_manifest = _result_index(primary_path)
            required = {
                "benchmark_sha256": sha256_file(primary_benchmark_path),
                "prompt_version": PROMPT_VERSION,
                "model": "deepseek-v4-pro",
                "temperature": 0,
                "thinking": "disabled",
                "tokenizer_manifest_sha256": tokenizer_hash,
            }
            mismatch = [key for key, value in required.items() if primary_manifest.get(key) != value]
            if mismatch:
                raise ValueError(f"Primary result manifest mismatch {dataset_id}: {mismatch}")
            source_inventory[str(primary_path.resolve())] = sha256_file(primary_path)

        simple_job = jobs[f"{dataset_id}:simple_same_computation"]
        simple_path = Path(simple_job["output"]) / "results.json"
        simple_index, simple_manifest = _result_index(simple_path)
        if (
            simple_manifest.get("benchmark_sha256") != sha256_file(benchmark_path)
            or set(simple_manifest.get("conditions") or [])
            != {"flat_program", "operator_only"}
            or simple_manifest.get("tokenizer_manifest_sha256") != tokenizer_hash
        ):
            raise ValueError(f"Simple supplement manifest mismatch: {dataset_id}")
        source_inventory[str(simple_path.resolve())] = sha256_file(simple_path)
        complex_index: dict[tuple[str, str], dict[str, Any]] = {}
        complex_path: Path | None = None
        if source_panel == "replication":
            complex_job = jobs[f"{dataset_id}:complex_same_computation"]
            complex_path = Path(complex_job["output"]) / "results.json"
            complex_index, complex_manifest = _result_index(complex_path)
            if (
                complex_manifest.get("benchmark_sha256") != sha256_file(benchmark_path)
                or set(complex_manifest.get("conditions") or [])
                != {"flat_program", "operator_only"}
                or complex_manifest.get("tokenizer_manifest_sha256") != tokenizer_hash
            ):
                raise ValueError(f"Complex supplement manifest mismatch: {dataset_id}")
            source_inventory[str(complex_path.resolve())] = sha256_file(complex_path)

        for task in benchmark["tasks"]:
            item_id = str(task["item_id"])
            tasks.append(
                {
                    "dataset_id": dataset_id,
                    "item_id": item_id,
                    "network": task["network"],
                    "scale": task["scale"],
                    "task_type": task["task_type"],
                    "complexity": task["complexity"],
                }
            )
            for condition in CONDITIONS:
                if condition == "graph_program":
                    source_path = extension_path
                    selected = extension_index.get((item_id, condition))
                    expected = extension_identities.get((dataset_id, item_id, condition))
                    source_kind = "extension_graph_program"
                elif task["complexity"] > 1 and source_panel == "primary":
                    source_task = primary_benchmark_tasks.get(item_id)
                    if source_task is None or task_payload_sha256(source_task) != task_payload_sha256(task):
                        raise ValueError(f"Primary/extension task mismatch: {item_id}")
                    source_path = primary_path
                    selected = primary_index.get((item_id, condition))
                    expected = supplement_identities.get((dataset_id, item_id, condition))
                    source_kind = "primary_reuse"
                elif task["complexity"] == 1:
                    source_path = simple_path
                    selected = simple_index.get((item_id, condition))
                    expected = supplement_identities.get((dataset_id, item_id, condition))
                    source_kind = "incremental"
                else:
                    source_path = complex_path
                    selected = complex_index.get((item_id, condition))
                    expected = supplement_identities.get((dataset_id, item_id, condition))
                    source_kind = "incremental"
                if selected is None or expected is None or source_path is None:
                    raise ValueError(f"Missing mechanism cell: {dataset_id}:{item_id}:{condition}")
                identity = verify_cell_identity(selected, expected)
                if not identity["passed"]:
                    raise ValueError(
                        f"Mechanism identity mismatch {dataset_id}:{item_id}:{condition}: "
                        f"{identity['mismatches']}"
                    )
                merged = dict(selected)
                merged["dataset_id"] = dataset_id
                merged["mechanism_provenance"] = {
                    "source_kind": source_kind,
                    "source_result_path": str(source_path.resolve()),
                    "source_result_sha256": source_inventory[str(source_path.resolve())],
                    "identity_audit": identity,
                }
                merged_records.append(merged)
                counts[source_kind] += 1
    if len(tasks) != 168 or len(merged_records) != 504:
        raise ValueError("Mechanism merge must contain 168 tasks and 504 cells")
    if counts != {
        "extension_graph_program": 168,
        "primary_reuse": 120,
        "incremental": 216,
    }:
        raise ValueError(f"Mechanism source accounting mismatch: {counts}")
    return {
        "schema_version": 1,
        "status": "merged_complete",
        "tasks": tasks,
        "records": merged_records,
        "logical_cells": len(merged_records),
        "source_accounting": counts,
        "source_inventory": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(source_inventory.items())
        ],
        "primary_plan_sha256": sha256_file(primary_plan_path),
        "extension_merge_manifest_sha256": sha256_file(
            extension_merged_root / "merge_manifest.json"
        ),
        "supplement_plan_sha256": sha256_file(supplement_plan_path),
        "extension_readiness_sha256": sha256_file(extension_readiness_path),
        "supplement_readiness_sha256": sha256_file(supplement_readiness_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--primary-plan", type=Path, required=True)
    parser.add_argument("--extension-merged-root", type=Path, required=True)
    parser.add_argument("--supplement-plan", type=Path, required=True)
    parser.add_argument("--extension-readiness", type=Path, required=True)
    parser.add_argument("--supplement-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_mechanism_supplement(
        benchmark_root=args.benchmark_root,
        primary_plan_path=args.primary_plan,
        extension_merged_root=args.extension_merged_root,
        supplement_plan_path=args.supplement_plan,
        extension_readiness_path=args.extension_readiness,
        supplement_readiness_path=args.supplement_readiness,
    )
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"tasks", "records"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
