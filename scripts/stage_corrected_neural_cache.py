"""Copy only unchanged Large graph caches after stopping the original worker.

Author caches are intentionally excluded; author indexes will be encoded afresh.
Small/Medium are re-materialized by the unchanged exact-scale-reuse implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from citeweave.bulk_acquisition import _pid_is_running
from citeweave.io import read_json, sha256_file, write_json

GRAPH_TABLES = {
    "keyword_cooccurrence": ("keyword_occurrences.parquet", "keyword_cooccurrence_edges.parquet"),
    "institution_collaboration": (
        "institution_productivity.parquet",
        "institution_collaboration_edges.parquet",
    ),
}


def copy_unchanged_cache(
    source: Path, destination: Path, old_workspace: Path, new_workspace: Path, network: str
) -> dict:
    if network not in GRAPH_TABLES:
        raise ValueError("Only unchanged keyword/institution caches may be copied")
    if destination.exists():
        raise ValueError("Refusing to overwrite a staged cache")
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(
        destination.resolve()
    ):
        raise ValueError("Cache source and destination must be disjoint")
    graph_proof = {}
    for name in GRAPH_TABLES[network]:
        old = old_workspace / "canonical" / "visualization" / name
        new = new_workspace / "canonical" / "visualization" / name
        if not pd.read_parquet(old).equals(pd.read_parquet(new)):
            raise ValueError(f"Graph table content/order changed: {name}")
        graph_proof[name] = {
            "old_sha256": sha256_file(old),
            "new_sha256": sha256_file(new),
            "ordered_rows_equal": True,
        }
    manifest_path = source / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest["status"] not in {"complete", "in_progress"}
        or manifest["document_serialization"] != "semantic_entity_relation_json_v2"
    ):
        raise ValueError("Unsupported source cache status/schema")
    complete = manifest["status"] == "complete"
    matrix_path = source / ("embeddings.npy" if complete else "embeddings.building.npy")
    matrix_hash = sha256_file(matrix_path)
    if complete and matrix_hash != manifest["embeddings_sha256"]:
        raise ValueError("Source embedding hash mismatch")
    matrix = np.load(matrix_path, mmap_mode="r")
    next_start = manifest["next_start"]
    if (
        matrix.shape != (manifest["row_count"], manifest["embedding_dimension"])
        or matrix.dtype != np.float32
    ):
        raise ValueError("Source matrix shape/dtype mismatch")
    if not 0 <= next_start <= len(matrix) or (complete and next_start != len(matrix)):
        raise ValueError("Invalid committed-prefix length")
    for start in range(0, next_start, 4096):
        if not np.isfinite(matrix[start : min(start + 4096, next_start)]).all():
            raise ValueError("Nonfinite values in committed cache prefix")
    del matrix
    manifest_hash = sha256_file(manifest_path)
    destination.mkdir(parents=True)
    shutil.copy2(matrix_path, destination / matrix_path.name)
    shutil.copy2(manifest_path, destination / "manifest.json")
    if (
        sha256_file(destination / matrix_path.name) != matrix_hash
        or sha256_file(destination / "manifest.json") != manifest_hash
    ):
        raise ValueError("Cache copy hash mismatch")
    if sha256_file(matrix_path) != matrix_hash or sha256_file(manifest_path) != manifest_hash:
        raise ValueError("Source cache changed during copy")
    return {
        "network": network,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_manifest_sha256": manifest_hash,
        "source_matrix_sha256": matrix_hash,
        "status": manifest["status"],
        "committed_rows": next_start,
        "total_rows": int(manifest["row_count"]),
        "graph_table_proof": graph_proof,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--old-workspace-root", type=Path, required=True)
    parser.add_argument("--new-workspace-root", type=Path, required=True)
    parser.add_argument("--stopped-worker-pid", type=int, required=True)
    args = parser.parse_args()
    if _pid_is_running(args.stopped_worker_pid):
        raise SystemExit("Stop the old neural worker before snapshotting its partial cache")
    if args.output_cache_root.exists():
        raise SystemExit("Output cache root already exists")
    rows = []
    for topic in sorted(args.source_cache_root.iterdir()):
        for network in GRAPH_TABLES:
            source = topic / f"{network}_large"
            if not (source / "manifest.json").is_file():
                continue
            row = copy_unchanged_cache(
                source,
                args.output_cache_root / topic.name / source.name,
                args.old_workspace_root / topic.name,
                args.new_workspace_root / topic.name,
                network,
            )
            rows.append({"dataset_id": topic.name, **row})
            print(
                json.dumps(
                    {
                        "topic": topic.name,
                        "network": network,
                        "committed_rows": row["committed_rows"],
                    }
                ),
                flush=True,
            )
    write_json(
        args.output_cache_root / "identity_correction_cache_snapshot.json",
        {
            "schema_version": 1,
            "status": "unchanged_large_caches_staged",
            "records": rows,
            "author_indexes": "not_copied_encode_afresh",
            "small_medium_indexes": "rematerialize_from_large_with_frozen_exact_reuse",
        },
    )


if __name__ == "__main__":
    main()
