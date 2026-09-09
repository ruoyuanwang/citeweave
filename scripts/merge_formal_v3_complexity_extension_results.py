from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from citeweave.formal_request import PROMPT_VERSION, task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json

CONDITIONS = (
    "flat_hybrid",
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
    "graph_program",
)


def verify_reuse_identity(
    record: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "task_payload_sha256",
        "context_sha256",
        "request_messages_sha256",
    )
    mismatches = [field for field in fields if record.get(field) != expected.get(field)]
    return {"passed": not mismatches, "mismatches": mismatches}


def _load_result(
    path: Path,
    *,
    expected_benchmark_sha256: str,
    readiness: dict[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"Required result file missing: {path}")
    payload = read_json(path)
    manifest = payload.get("manifest") or {}
    required_manifest = {
        "benchmark_sha256": expected_benchmark_sha256,
        "prompt_version": PROMPT_VERSION,
        "model": "deepseek-v4-pro",
        "temperature": 0,
        "thinking": "disabled",
        "tokenizer_manifest_sha256": readiness["tokenizer_manifest_sha256"],
    }
    mismatches = [
        key for key, value in required_manifest.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"Result manifest mismatch {path}: {mismatches}")
    indexed = {
        (record["item_id"], record["condition"]): record
        for record in payload.get("records") or []
    }
    if len(indexed) != len(payload.get("records") or []):
        raise ValueError(f"Duplicate result cell in {path}")
    return indexed, manifest, sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--execution-amendment-freeze", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--primary-benchmark-root", type=Path, required=True)
    parser.add_argument("--primary-results-root", type=Path, required=True)
    parser.add_argument("--replication-benchmark-root", type=Path, required=True)
    parser.add_argument("--replication-results-root", type=Path, required=True)
    parser.add_argument("--incremental-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    amendment_hash = sha256_file(args.execution_amendment)
    if read_json(args.protocol_freeze).get("sha256") != protocol_hash:
        raise SystemExit("Complexity-extension protocol freeze hash mismatch")
    if read_json(args.execution_amendment_freeze).get("sha256") != amendment_hash:
        raise SystemExit("Complexity-extension execution amendment freeze hash mismatch")
    readiness = read_json(args.readiness)
    if readiness.get("status") != "ready":
        raise SystemExit("Cannot merge complexity-extension results before readiness is ready")
    if readiness.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Complexity-extension readiness protocol hash mismatch")
    if readiness.get("execution_amendment_sha256") != amendment_hash:
        raise SystemExit("Complexity-extension readiness amendment hash mismatch")
    expected_identities = {
        (row["dataset_id"], row["item_id"], row["condition"]): row
        for row in readiness.get("cell_identity_manifest") or []
    }
    if len(expected_identities) != 672:
        raise SystemExit("Readiness lacks the frozen 672-cell identity manifest")

    construction = read_json(args.benchmark_root / "construction_manifest.json")
    merged_inventory = []
    total_reused = 0
    total_new = 0
    result_cache: dict[Path, tuple[dict, dict, str]] = {}

    def cached_result(path: Path, benchmark_sha256: str) -> tuple[dict, dict, str]:
        resolved = path.resolve()
        if resolved not in result_cache:
            result_cache[resolved] = _load_result(
                resolved,
                expected_benchmark_sha256=benchmark_sha256,
                readiness=readiness,
            )
        return result_cache[resolved]

    for construction_record in construction["records"]:
        dataset_id = construction_record["dataset_id"]
        source_panel = construction_record["source_panel"]
        extension_benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        extension_hash = sha256_file(extension_benchmark_path)
        if extension_hash != construction_record["benchmark_sha256"]:
            raise SystemExit(f"Extension benchmark hash mismatch: {dataset_id}")
        extension_benchmark = read_json(extension_benchmark_path)
        source_benchmark_root = (
            args.primary_benchmark_root
            if source_panel == "primary"
            else args.replication_benchmark_root
        )
        source_results_root = (
            args.primary_results_root
            if source_panel == "primary"
            else args.replication_results_root
        )
        source_benchmark_path = source_benchmark_root / dataset_id / "benchmark.json"
        source_benchmark_hash = sha256_file(source_benchmark_path)
        source_tasks = {
            task["item_id"]: task for task in read_json(source_benchmark_path)["tasks"]
        }
        source_result_path = source_results_root / dataset_id / "results.json"
        source_index, _, source_result_hash = cached_result(
            source_result_path, source_benchmark_hash
        )
        simple_path = (
            args.incremental_results_root
            / dataset_id
            / "simple_controls"
            / "results.json"
        )
        simple_index, _, simple_result_hash = cached_result(simple_path, extension_hash)
        supplement_index: dict[tuple[str, str], dict[str, Any]] = {}
        supplement_hash = None
        supplement_path = (
            args.incremental_results_root
            / dataset_id
            / "complex_supplement"
            / "results.json"
        )
        if source_panel == "replication":
            supplement_index, _, supplement_hash = cached_result(
                supplement_path, extension_hash
            )

        merged_records = []
        dataset_reused = 0
        dataset_new = 0
        for task in extension_benchmark["tasks"]:
            if task["complexity"] > 1:
                source_task = source_tasks.get(task["item_id"])
                if source_task is None or task_payload_sha256(
                    source_task
                ) != task_payload_sha256(task):
                    raise SystemExit(
                        f"Source/extension task payload mismatch: {task['item_id']}"
                    )
            for condition in CONDITIONS:
                key = (task["item_id"], condition)
                expected = expected_identities[(dataset_id, *key)]
                if task["complexity"] == 1:
                    selected = simple_index.get(key)
                    source_kind = "incremental_simple_control"
                    result_path = simple_path
                    result_hash = simple_result_hash
                    reused = False
                elif source_panel == "primary" or condition in {
                    "flat_hybrid",
                    "graph_program",
                }:
                    selected = source_index.get(key)
                    source_kind = f"reused_{source_panel}_complex"
                    result_path = source_result_path
                    result_hash = source_result_hash
                    reused = True
                else:
                    selected = supplement_index.get(key)
                    source_kind = "incremental_replication_complex_supplement"
                    result_path = supplement_path
                    result_hash = supplement_hash
                    reused = False
                if selected is None:
                    raise SystemExit(f"Missing extension logical cell: {dataset_id}:{key}")
                identity_audit = verify_reuse_identity(selected, expected)
                if not identity_audit["passed"]:
                    raise SystemExit(
                        f"Extension cell identity mismatch {dataset_id}:{key}: "
                        f"{identity_audit['mismatches']}"
                    )
                record = dict(selected)
                record["extension_provenance"] = {
                    "source_kind": source_kind,
                    "source_result_path": str(result_path.resolve()),
                    "source_result_sha256": result_hash,
                    "reuse": reused,
                    "reuse_equivalence_audit": identity_audit,
                }
                merged_records.append(record)
                dataset_reused += int(reused)
                dataset_new += int(not reused)
        if len(merged_records) != 84:
            raise SystemExit(f"Merged dataset must contain 84 cells: {dataset_id}")
        output_path = args.output_root / dataset_id / "results.json"
        write_json(
            output_path,
            {
                "manifest": {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "benchmark_sha256": extension_hash,
                    "conditions": list(CONDITIONS),
                    "protocol_sha256": protocol_hash,
                    "execution_amendment_sha256": amendment_hash,
                    "readiness_sha256": sha256_file(args.readiness),
                    "logical_cells": 84,
                    "reused_cells": dataset_reused,
                    "incremental_cells": dataset_new,
                },
                "records": merged_records,
            },
        )
        merged_inventory.append(
            {
                "dataset_id": dataset_id,
                "output": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
                "logical_cells": len(merged_records),
                "reused_cells": dataset_reused,
                "incremental_cells": dataset_new,
            }
        )
        total_reused += dataset_reused
        total_new += dataset_new
    if (total_reused, total_new) != (360, 312):
        raise SystemExit(
            f"Merged reuse accounting mismatch: reused={total_reused}, new={total_new}"
        )
    manifest = {
        "schema_version": 1,
        "status": "merged_complete",
        "protocol_sha256": protocol_hash,
        "execution_amendment_sha256": amendment_hash,
        "logical_cells": total_reused + total_new,
        "reused_cells": total_reused,
        "incremental_cells": total_new,
        "datasets": merged_inventory,
    }
    write_json(args.output_root / "merge_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
