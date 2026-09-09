from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from citeweave.graph_discovery import _load_graph
from citeweave.io import read_json, sha256_file, write_json

if __package__:
    from scripts.build_neural_dense_sidecars import _document_hash, _graph_rows
else:
    from build_neural_dense_sidecars import _document_hash, _graph_rows


def _schema_hash(row: dict[str, Any], document_schema: str) -> str:
    if document_schema == "semantic_entity_relation_json_v2":
        return _document_hash(row)
    if document_schema == "canonical_full_row_v1":
        document = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(document.encode()).hexdigest()
    raise ValueError(f"Unknown document schema: {document_schema}")


def _workspace_for(
    dataset_id: str, workspace_roots: list[Path]
) -> Path:
    workspace = next(
        (
            root / dataset_id
            for root in workspace_roots
            if (root / dataset_id / "canonical").is_dir()
        ),
        None,
    )
    if workspace is None:
        raise RuntimeError(f"No workspace root contains dataset {dataset_id}")
    return workspace


def audit_exact_scale_reuse(
    *,
    benchmark_root: Path,
    workspace_roots: list[Path],
    document_schema: str = "semantic_entity_relation_json_v2",
) -> dict[str, Any]:
    construction_path = benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path)
    records: list[dict[str, Any]] = []
    for dataset in construction["records"]:
        dataset_id = dataset["dataset_id"]
        workspace = _workspace_for(dataset_id, workspace_roots)
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        benchmark = read_json(benchmark_path)
        networks = sorted({task["network"] for task in benchmark["tasks"]})
        scales = sorted(
            {task["scale"] for task in benchmark["tasks"]} - {"large"}
        )
        for network in networks:
            large_graph, _ = _load_graph(workspace, network, "large")
            large_rows = _graph_rows(large_graph, network)
            large_hashes = {
                _schema_hash(row, document_schema) for row in large_rows
            }
            duplicate_large_semantic_documents = len(large_rows) - len(large_hashes)
            for scale in scales:
                target_graph, _ = _load_graph(workspace, network, scale)
                target_rows = _graph_rows(target_graph, network)
                reused = sum(
                    _schema_hash(row, document_schema) in large_hashes
                    for row in target_rows
                )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "network": network,
                        "target_scale": scale,
                        "target_rows": len(target_rows),
                        "large_duplicate_semantic_documents": (
                            duplicate_large_semantic_documents
                        ),
                        "reused_rows": reused,
                        "encoded_rows": len(target_rows) - reused,
                        "reuse_fraction": reused / len(target_rows),
                    }
                )
    target_rows = sum(row["target_rows"] for row in records)
    reused_rows = sum(row["reused_rows"] for row in records)
    return {
        "schema_version": 1,
        "method": (
            "exact_semantic_entity_relation_sha256_from_large_scale"
            if document_schema == "semantic_entity_relation_json_v2"
            else "exact_canonical_document_sha256_from_large_scale"
        ),
        "document_schema": document_schema,
        "source_construction_manifest_sha256": sha256_file(construction_path),
        "records": records,
        "summary": {
            "dataset_network_scale_cells": len(records),
            "target_rows": target_rows,
            "reused_rows": reused_rows,
            "encoded_rows": target_rows - reused_rows,
            "reuse_fraction": reused_rows / target_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--document-schema",
        choices=("canonical_full_row_v1", "semantic_entity_relation_json_v2"),
        default="semantic_entity_relation_json_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_exact_scale_reuse(
        benchmark_root=args.benchmark_root,
        workspace_roots=args.workspace_root,
        document_schema=args.document_schema,
    )
    write_json(args.output, report)
    print(report["summary"])


if __name__ == "__main__":
    main()
