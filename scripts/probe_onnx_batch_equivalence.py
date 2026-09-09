from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from citeweave.graph_discovery import _load_graph
from citeweave.io import read_json, sha256_file, write_json
from citeweave.neural_dense_runtime import OnnxSentenceEncoder
from scripts.build_neural_dense_sidecars import _embedding_document, _graph_rows


def _top_k(scores: np.ndarray, limit: int) -> list[int]:
    return np.argsort(-scores, kind="stable")[:limit].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--scale", default="large")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="DmlExecutionProvider")
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, action="append", required=True)
    parser.add_argument("--sample-records", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    graph, _ = _load_graph(args.workspace, args.network, args.scale)
    rows = _graph_rows(graph, args.network)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["evidence_id"]).encode()).hexdigest(),
    )[: args.sample_records]
    documents = [_embedding_document(row) for row in ranked]
    benchmark = read_json(args.benchmark)
    queries = [task["question"] for task in benchmark["tasks"]]
    encoder = OnnxSentenceEncoder(
        args.model,
        revision=args.model_revision,
        provider=args.provider,
        local_files_only=args.local_files_only,
    )
    encoder.encode(documents[:8], batch_size=8, normalize_embeddings=True)
    started = time.perf_counter()
    reference = encoder.encode(
        documents,
        batch_size=args.reference_batch_size,
        normalize_embeddings=True,
    )
    reference_seconds = time.perf_counter() - started
    query_embeddings = encoder.encode(
        queries,
        batch_size=args.reference_batch_size,
        normalize_embeddings=True,
    )
    reference_scores = reference @ query_embeddings.T

    candidates: list[dict[str, Any]] = []
    for batch_size in args.candidate_batch_size:
        try:
            started = time.perf_counter()
            candidate = encoder.encode(
                documents,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            elapsed = time.perf_counter() - started
            maximum_difference = float(np.max(np.abs(reference - candidate)))
            cosines = np.sum(reference * candidate, axis=1)
            candidate_scores = candidate @ query_embeddings.T
            overlaps = []
            for query_index in range(len(queries)):
                reference_ranking = _top_k(
                    reference_scores[:, query_index], args.top_k
                )
                candidate_ranking = _top_k(
                    candidate_scores[:, query_index], args.top_k
                )
                overlaps.append(
                    len(set(reference_ranking) & set(candidate_ranking)) / args.top_k
                )
            candidates.append(
                {
                    "batch_size": batch_size,
                    "status": "complete",
                    "seconds": elapsed,
                    "records_per_second": len(documents) / elapsed,
                    "speedup_vs_reference": reference_seconds / elapsed,
                    "maximum_absolute_embedding_difference": maximum_difference,
                    "minimum_embedding_cosine": float(np.min(cosines)),
                    "minimum_top_k_overlap": min(overlaps),
                    "mean_top_k_overlap": float(np.mean(overlaps)),
                    "passed_equivalence": (
                        maximum_difference <= 1e-6
                        and float(np.min(cosines)) >= 0.999999
                        and min(overlaps) == 1.0
                    ),
                }
            )
        except (RuntimeError, ValueError, OSError) as error:  # pragma: no cover
            candidates.append(
                {
                    "batch_size": batch_size,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "passed_equivalence": False,
                }
            )
    result = {
        "schema_version": 1,
        "status": "batch_equivalence_probe_complete",
        "purpose": (
            "Pre-outcome runtime-only throughput and equivalence probe. No gold answers "
            "or generation outcomes are accessed."
        ),
        "platform": platform.platform(),
        "model": args.model,
        "model_revision": args.model_revision,
        "runtime": encoder.runtime_metadata,
        "workspace": str(args.workspace.resolve()),
        "network": args.network,
        "scale": args.scale,
        "benchmark_sha256": sha256_file(args.benchmark),
        "sample_selection": "lowest_sha256_evidence_id",
        "sample_records": len(documents),
        "queries": len(queries),
        "reference": {
            "batch_size": args.reference_batch_size,
            "seconds": reference_seconds,
            "records_per_second": len(documents) / reference_seconds,
        },
        "candidates": candidates,
    }
    result["passed"] = all(row["passed_equivalence"] for row in candidates)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
