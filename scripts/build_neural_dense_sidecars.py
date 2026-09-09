from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from citeweave.graph_discovery import (
    _communities,
    _edge_record,
    _load_graph,
    _node_record,
)
from citeweave.io import read_json, sha256_file, write_json
from citeweave.neural_dense_runtime import OnnxSentenceEncoder


def _tie_key(evidence_id: str) -> str:
    return hashlib.sha256(evidence_id.encode()).hexdigest()


def _embedding_document(row: dict[str, Any]) -> str:
    network = str(row["evidence_id"]).split(":", 1)[0]
    if "node_id" in row:
        semantic_record = {
            "network": network,
            "record_type": "node",
            "label": row["label"],
        }
    else:
        semantic_record = {
            "network": network,
            "record_type": "edge",
            "source_label": row["source_label"],
            "target_label": row["target_label"],
        }
    return json.dumps(
        semantic_record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _document_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(_embedding_document(row).encode()).hexdigest()


def _graph_rows(graph: Any, network: str) -> list[dict[str, Any]]:
    communities = _communities(graph)
    return [
        *[
            _node_record(graph, network, node, communities[node])
            for node in sorted(graph.nodes)
        ],
        *[
            _edge_record(graph, source, target)
            for source, target in sorted(graph.edges)
        ],
    ]


def _prepare_exact_embedding_reuse(
    *,
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_cache_path: Path,
    cache_identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, list[int], dict[str, Any]]:
    source_manifest_path = source_cache_path / "manifest.json"
    source_embeddings_path = source_cache_path / "embeddings.npy"
    source_manifest = read_json(source_manifest_path)
    if source_manifest.get("status") != "complete":
        raise RuntimeError(f"Reuse source cache is incomplete: {source_cache_path}")
    if source_manifest.get("cache_identity") != cache_identity:
        raise RuntimeError(f"Reuse source runtime identity mismatch: {source_cache_path}")
    if not source_embeddings_path.is_file() or sha256_file(
        source_embeddings_path
    ) != source_manifest.get("embeddings_sha256"):
        raise RuntimeError(f"Reuse source embedding hash mismatch: {source_cache_path}")
    if len(source_rows) != source_manifest.get("row_count"):
        raise RuntimeError(f"Reuse source row count mismatch: {source_cache_path}")
    source_row_ids_sha256 = hashlib.sha256(
        json.dumps(
            [row["evidence_id"] for row in source_rows], separators=(",", ":")
        ).encode()
    ).hexdigest()
    if source_row_ids_sha256 != source_manifest.get("row_ids_sha256"):
        raise RuntimeError(f"Reuse source row order mismatch: {source_cache_path}")
    source_by_hash: dict[str, int] = {}
    source_duplicate_semantic_documents = 0
    for index, row in enumerate(source_rows):
        document_hash = _document_hash(row)
        if document_hash in source_by_hash:
            source_duplicate_semantic_documents += 1
        else:
            source_by_hash[document_hash] = index
    matched: list[tuple[dict[str, Any], int]] = []
    unmatched: list[dict[str, Any]] = []
    for row in rows:
        source_index = source_by_hash.get(_document_hash(row))
        if source_index is None:
            unmatched.append(row)
        else:
            matched.append((row, source_index))
    reordered = [row for row, _ in matched] + unmatched
    source_embeddings = np.load(source_embeddings_path, mmap_mode="r")
    metadata = {
        "method": "exact_semantic_entity_relation_sha256_from_large_scale",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_embeddings_sha256": source_manifest["embeddings_sha256"],
        "source_row_ids_sha256": source_row_ids_sha256,
        "source_duplicate_semantic_documents": source_duplicate_semantic_documents,
        "reused_rows": len(matched),
        "encoded_rows": len(unmatched),
    }
    return reordered, source_embeddings, [index for _, index in matched], metadata


def _merge_rankings(
    current: list[tuple[float, str, dict[str, Any]]],
    *,
    scores: np.ndarray,
    rows: list[dict[str, Any]],
    limit: int,
) -> list[tuple[float, str, dict[str, Any]]]:
    candidates = [
        *current,
        *[
            (float(score), _tie_key(str(row["evidence_id"])), row)
            for score, row in zip(scores.tolist(), rows, strict=True)
        ],
    ]
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[:limit]


def _embedding_cache(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    row_ids_sha256: str,
    embedding_dimension: int,
    batch_size: int,
    cache_path: Path,
    cache_identity: dict[str, Any],
    checkpoint_every_batches: int,
    reuse_embeddings: np.ndarray | None = None,
    reuse_source_indices: list[int] | None = None,
    reuse_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "manifest.json"
    embeddings_path = cache_path / "embeddings.npy"
    building_path = cache_path / "embeddings.building.npy"
    expected = {
        "cache_identity": cache_identity,
        "row_count": len(rows),
        "row_ids_sha256": row_ids_sha256,
        "embedding_dimension": embedding_dimension,
        "dtype": "float32",
        "document_serialization": "semantic_entity_relation_json_v2",
    }
    if reuse_metadata is not None:
        expected["exact_embedding_reuse"] = reuse_metadata
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if manifest is not None and any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"Embedding cache identity mismatch: {cache_path}")
    if manifest is not None and manifest.get("status") == "complete":
        if not embeddings_path.is_file():
            raise RuntimeError(f"Completed embedding cache is missing: {embeddings_path}")
        if sha256_file(embeddings_path) != manifest.get("embeddings_sha256"):
            raise RuntimeError(f"Embedding cache SHA-256 mismatch: {embeddings_path}")
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embeddings.shape != (len(rows), embedding_dimension):
            raise RuntimeError(f"Embedding cache shape mismatch: {embeddings_path}")
        return embeddings, manifest
    next_start = int((manifest or {}).get("next_start", 0))
    if next_start:
        if not building_path.is_file():
            raise RuntimeError(f"Embedding cache checkpoint is missing: {building_path}")
        embeddings = np.lib.format.open_memmap(building_path, mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            building_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), embedding_dimension),
        )
        if reuse_metadata is not None:
            if reuse_embeddings is None or reuse_source_indices is None:
                raise ValueError("Exact reuse metadata requires embeddings and indices")
            reuse_count = len(reuse_source_indices)
            if reuse_count != int(reuse_metadata["reused_rows"]):
                raise ValueError("Exact reuse count mismatch")
            for start in range(0, reuse_count, 4096):
                indices = reuse_source_indices[start : start + 4096]
                embeddings[start : start + len(indices)] = reuse_embeddings[indices]
            embeddings.flush()
            next_start = reuse_count
            write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    **expected,
                    "next_start": next_start,
                },
            )
    if embeddings.shape != (len(rows), embedding_dimension):
        raise RuntimeError(f"Embedding cache checkpoint shape mismatch: {building_path}")
    for batch_index, start in enumerate(
        range(next_start, len(rows), batch_size), start=1
    ):
        batch_rows = rows[start : start + batch_size]
        documents = [_embedding_document(row) for row in batch_rows]
        batch_embeddings = np.asarray(
            model.encode(
                documents,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if batch_embeddings.shape != (len(batch_rows), embedding_dimension):
            raise RuntimeError("Embedding cache batch has an unexpected shape")
        embeddings[start : start + len(batch_rows)] = batch_embeddings
        next_start = start + len(batch_rows)
        if (
            batch_index % checkpoint_every_batches == 0
            or next_start == len(rows)
        ):
            embeddings.flush()
            write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    **expected,
                    "next_start": next_start,
                },
            )
    embeddings.flush()
    del embeddings
    os.replace(building_path, embeddings_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        **expected,
        "next_start": len(rows),
        "embeddings_sha256": sha256_file(embeddings_path),
    }
    write_json(manifest_path, manifest)
    return np.load(embeddings_path, mmap_mode="r"), manifest


def _build_group_contexts(
    *,
    model: Any,
    graph: Any,
    network: str,
    tasks: list[dict[str, Any]],
    model_name: str,
    model_revision: str,
    record_budget: int,
    batch_size: int,
    runtime_backend: str = "sentence-transformers-pytorch",
    runtime_precision: str = "float32",
    embedding_cache_path: Path | None = None,
    embedding_cache_identity: dict[str, Any] | None = None,
    reuse_embedding_cache_path: Path | None = None,
    reuse_graph: Any | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every_batches: int = 20,
) -> dict[str, dict[str, Any]]:
    rows = _graph_rows(graph, network)
    questions = [task["question"] for task in tasks]
    query_embeddings = model.encode(
        questions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    row_ids_sha256 = hashlib.sha256(
        json.dumps(
            [row["evidence_id"] for row in rows], separators=(",", ":")
        ).encode()
    ).hexdigest()
    cached_embeddings: np.ndarray | None = None
    reuse_embeddings: np.ndarray | None = None
    reuse_source_indices: list[int] | None = None
    reuse_metadata: dict[str, Any] | None = None
    if embedding_cache_path is not None:
        if embedding_cache_identity is None:
            raise ValueError("Embedding cache identity is required with a cache path")
        target_manifest_path = embedding_cache_path / "manifest.json"
        target_manifest = (
            read_json(target_manifest_path) if target_manifest_path.is_file() else None
        )
        if (
            reuse_embedding_cache_path is not None
            and reuse_graph is not None
            and (
                target_manifest is None
                or target_manifest.get("exact_embedding_reuse") is not None
            )
        ):
            (
                rows,
                reuse_embeddings,
                reuse_source_indices,
                reuse_metadata,
            ) = _prepare_exact_embedding_reuse(
                rows=rows,
                source_rows=_graph_rows(reuse_graph, network),
                source_cache_path=reuse_embedding_cache_path,
                cache_identity=embedding_cache_identity,
            )
            row_ids_sha256 = hashlib.sha256(
                json.dumps(
                    [row["evidence_id"] for row in rows], separators=(",", ":")
                ).encode()
            ).hexdigest()
        cached_embeddings, _ = _embedding_cache(
            model=model,
            rows=rows,
            row_ids_sha256=row_ids_sha256,
            embedding_dimension=int(np.asarray(query_embeddings).shape[1]),
            batch_size=batch_size,
            cache_path=embedding_cache_path,
            cache_identity=embedding_cache_identity,
            checkpoint_every_batches=checkpoint_every_batches,
            reuse_embeddings=reuse_embeddings,
            reuse_source_indices=reuse_source_indices,
            reuse_metadata=reuse_metadata,
        )
    rankings: list[list[tuple[float, str, dict[str, Any]]]] = [[] for _ in questions]
    next_start = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        expected = {
            "model": model_name,
            "model_revision": model_revision,
            "network": network,
            "task_ids": [task["item_id"] for task in tasks],
            "row_count": len(rows),
            "row_ids_sha256": row_ids_sha256,
            "record_budget": record_budget,
            "runtime_backend": runtime_backend,
            "runtime_precision": runtime_precision,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Neural checkpoint identity mismatch: {checkpoint_path}")
        next_start = int(checkpoint["next_start"])
        rankings = [
            [
                (float(item["score"]), str(item["tie_key"]), item["row"])
                for item in ranking
            ]
            for ranking in checkpoint["rankings"]
        ]
        if len(rankings) != len(questions) or not 0 <= next_start <= len(rows):
            raise RuntimeError(f"Malformed neural checkpoint: {checkpoint_path}")
    for batch_index, start in enumerate(
        range(next_start, len(rows), batch_size), start=1
    ):
        batch_rows = rows[start : start + batch_size]
        if cached_embeddings is None:
            documents = [_embedding_document(row) for row in batch_rows]
            document_embeddings = model.encode(
                documents,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            document_embeddings = cached_embeddings[start : start + len(batch_rows)]
        similarities = np.asarray(document_embeddings) @ np.asarray(query_embeddings).T
        for index in range(len(questions)):
            rankings[index] = _merge_rankings(
                rankings[index],
                scores=similarities[:, index],
                rows=batch_rows,
                limit=record_budget,
            )
        next_start = min(len(rows), start + len(batch_rows))
        if checkpoint_path is not None and (
            batch_index % checkpoint_every_batches == 0 or next_start == len(rows)
        ):
            write_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "status": (
                        "complete" if next_start == len(rows) else "in_progress"
                    ),
                    "model": model_name,
                    "model_revision": model_revision,
                    "network": network,
                    "task_ids": [task["item_id"] for task in tasks],
                    "row_count": len(rows),
                    "row_ids_sha256": row_ids_sha256,
                    "record_budget": record_budget,
                    "runtime_backend": runtime_backend,
                    "runtime_precision": runtime_precision,
                    "next_start": next_start,
                    "rankings": [
                        [
                            {"score": score, "tie_key": tie_key, "row": row}
                            for score, tie_key, row in ranking
                        ]
                        for ranking in rankings
                    ],
                },
            )
    return {
        task["item_id"]: {
            "representation": "flat_neural_dense_retrieval",
            "model": model_name,
            "model_revision": model_revision,
            "query_independent_index": True,
            "record_budget": record_budget,
            "similarity": "cosine",
            "rows": [row for _, _, row in ranking],
        }
        for task, ranking in zip(tasks, rankings, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        action="append",
        required=True,
        help="May be repeated; the first root containing a dataset is used.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embedding-cache-root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend",
        choices=("sentence-transformers-pytorch", "onnxruntime"),
        default="sentence-transformers-pytorch",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--dynamic-int8", action="store_true")
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--onnx-provider", default="CPUExecutionProvider")
    parser.add_argument("--onnx-intra-op-threads", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int, default=20)
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    try:
        import sentence_transformers
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required; do not substitute LSA for this baseline"
        ) from exc
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    dataset_ids = args.dataset or [row["dataset_id"] for row in construction["records"]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.dynamic_int8 and args.backend != "sentence-transformers-pytorch":
        raise ValueError("Dynamic INT8 is only implemented for the PyTorch backend")
    runtime_metadata: dict[str, Any] = {}
    if args.backend == "onnxruntime":
        model = OnnxSentenceEncoder(
            args.model,
            revision=args.model_revision,
            onnx_file=args.onnx_file,
            provider=args.onnx_provider,
            intra_op_threads=args.onnx_intra_op_threads,
            local_files_only=args.local_files_only,
        )
        runtime_metadata = model.runtime_metadata
    else:
        model = SentenceTransformer(
            args.model,
            revision=args.model_revision,
            device=args.device,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        )
    if args.dynamic_int8:
        model[0].auto_model = torch.ao.quantization.quantize_dynamic(
            model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    runtime_precision = (
        "torch_dynamic_qint8"
        if args.dynamic_int8
        else "onnx_float32"
        if args.backend == "onnxruntime"
        else "torch_float32"
    )
    embedding_cache_identity = {
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime_backend": args.backend,
        "runtime_precision": runtime_precision,
        "embedding_document_schema": "semantic_entity_relation_json_v2",
        **runtime_metadata,
    }
    artifacts = []
    index_artifacts: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        workspace = next(
            (
                root / dataset_id
                for root in args.workspace_root
                if (root / dataset_id / "canonical").is_dir()
            ),
            None,
        )
        if workspace is None:
            raise RuntimeError(f"No workspace root contains dataset {dataset_id}")
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        benchmark = read_json(benchmark_path)
        sidecar_path = args.output_root / dataset_id / "contexts.json"
        if args.resume_existing and sidecar_path.is_file():
            sidecar = read_json(sidecar_path)
            expected_ids = {task["item_id"] for task in benchmark["tasks"]}
            if (
                sidecar.get("source_benchmark_sha256")
                != sha256_file(benchmark_path)
                or sidecar.get("model") != args.model
                or sidecar.get("model_revision") != args.model_revision
                or sidecar.get("runtime_backend") != args.backend
                or sidecar.get("runtime_precision") != runtime_precision
                or set(sidecar.get("contexts") or {}) != expected_ids
            ):
                raise RuntimeError(f"Existing neural sidecar mismatch: {dataset_id}")
            artifacts.append(
                {
                    "artifact_type": "neural_context_sidecar",
                    "dataset_id": dataset_id,
                    "path": str(sidecar_path.relative_to(args.output_root)),
                    "sha256": sha256_file(sidecar_path),
                    "source_benchmark_sha256": sha256_file(benchmark_path),
                    "tasks": len(expected_ids),
                }
            )
            for index_artifact in sidecar.get("embedding_indexes") or []:
                index_artifacts.append(
                    {"dataset_id": dataset_id, **index_artifact}
                )
            print(dataset_id, len(expected_ids), "reused")
            continue
        record_budget = int(benchmark["design"]["record_budget"])
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for task in benchmark["tasks"]:
            grouped[(task["network"], task["scale"])].append(task)
        contexts: dict[str, dict[str, Any]] = {}
        dataset_index_artifacts: list[dict[str, Any]] = []
        for (network, scale), tasks in sorted(grouped.items()):
            graph, _ = _load_graph(workspace, network, scale)
            cache_path = (
                args.embedding_cache_root / dataset_id / f"{network}_{scale}"
                if args.embedding_cache_root is not None
                else None
            )
            reuse_cache_path = None
            reuse_graph = None
            if (
                cache_path is not None
                and scale != "large"
                and args.embedding_cache_root is not None
            ):
                candidate = (
                    args.embedding_cache_root / dataset_id / f"{network}_large"
                )
                candidate_manifest = candidate / "manifest.json"
                if candidate_manifest.is_file() and read_json(candidate_manifest).get(
                    "status"
                ) == "complete":
                    reuse_cache_path = candidate
                    reuse_graph, _ = _load_graph(workspace, network, "large")
            contexts.update(
                _build_group_contexts(
                    model=model,
                    graph=graph,
                    network=network,
                    tasks=tasks,
                    model_name=args.model,
                    model_revision=args.model_revision,
                    record_budget=record_budget,
                    batch_size=args.batch_size,
                    runtime_backend=args.backend,
                    runtime_precision=runtime_precision,
                    embedding_cache_path=cache_path,
                    embedding_cache_identity=embedding_cache_identity,
                    reuse_embedding_cache_path=reuse_cache_path,
                    reuse_graph=reuse_graph,
                    checkpoint_path=(
                        args.output_root
                        / dataset_id
                        / "checkpoints"
                        / f"{network}_{scale}.json"
                    ),
                    checkpoint_every_batches=args.checkpoint_every_batches,
                )
            )
            if cache_path is not None:
                cache_manifest_path = cache_path / "manifest.json"
                cache_manifest = read_json(cache_manifest_path)
                cache_artifact = {
                    "network": network,
                    "scale": scale,
                    "manifest_path": str(cache_manifest_path),
                    "manifest_sha256": sha256_file(cache_manifest_path),
                    "embeddings_sha256": cache_manifest["embeddings_sha256"],
                    "row_count": cache_manifest["row_count"],
                }
                dataset_index_artifacts.append(cache_artifact)
                index_artifacts.append(
                    {"dataset_id": dataset_id, **cache_artifact}
                )
        expected_ids = {task["item_id"] for task in benchmark["tasks"]}
        if set(contexts) != expected_ids:
            raise RuntimeError(f"Neural sidecar coverage mismatch for {dataset_id}")
        write_json(
            sidecar_path,
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "condition_name": "flat_neural_dense",
                "source_benchmark_sha256": sha256_file(benchmark_path),
                "model": args.model,
                "model_revision": args.model_revision,
                "runtime_precision": runtime_precision,
                "runtime_backend": args.backend,
                **runtime_metadata,
                "embedding_indexes": dataset_index_artifacts,
                "contexts": contexts,
            },
        )
        artifacts.append(
            {
                "artifact_type": "neural_context_sidecar",
                "dataset_id": dataset_id,
                "path": str(sidecar_path.relative_to(args.output_root)),
                "sha256": sha256_file(sidecar_path),
                "source_benchmark_sha256": sha256_file(benchmark_path),
                "tasks": len(contexts),
            }
        )
        print(dataset_id, len(contexts))
    manifest = {
        "schema_version": 1,
        "passed": len(artifacts) == len(dataset_ids),
        "neural": True,
        "query_independent_index": True,
        "condition_name": "flat_neural_dense",
        "model": args.model,
        "model_revision": args.model_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "runtime_precision": runtime_precision,
        "runtime_backend": args.backend,
        "device": args.device,
        "similarity": "cosine",
        **runtime_metadata,
        "embedding_indexes": index_artifacts,
        "artifacts": artifacts,
    }
    write_json(args.output_root / "neural_dense_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
