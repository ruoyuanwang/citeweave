from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

from citeweave.io import read_json, sha256_file, write_json
from citeweave.neural_dense_runtime import OnnxSentenceEncoder


def _top_k(scores: np.ndarray, limit: int) -> list[int]:
    return np.argsort(-scores, kind="stable")[:limit].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-provider", default="CPUExecutionProvider")
    parser.add_argument("--candidate-provider", default="DmlExecutionProvider")
    parser.add_argument("--reference-threads", type=int, default=16)
    parser.add_argument("--reference-batch-size", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hardware-label", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    benchmark = read_json(args.benchmark)
    tasks = benchmark["tasks"]
    documents = [
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        for row in tasks[0]["contexts"]["flat_retrieval"]["rows"]
    ]
    queries = [task["question"] for task in tasks]
    reference = OnnxSentenceEncoder(
        args.model,
        revision=args.model_revision,
        provider=args.reference_provider,
        intra_op_threads=args.reference_threads,
        local_files_only=args.local_files_only,
    )
    started = time.perf_counter()
    reference_documents = reference.encode(
        documents,
        batch_size=args.reference_batch_size,
        normalize_embeddings=True,
    )
    reference_queries = reference.encode(
        queries,
        batch_size=args.reference_batch_size,
        normalize_embeddings=True,
    )
    reference_seconds = time.perf_counter() - started
    reference_metadata = reference.runtime_metadata
    del reference
    gc.collect()
    candidate = OnnxSentenceEncoder(
        args.model,
        revision=args.model_revision,
        provider=args.candidate_provider,
        local_files_only=args.local_files_only,
    )
    candidate.encode(
        documents[: min(8, len(documents))],
        batch_size=min(8, args.candidate_batch_size),
        normalize_embeddings=True,
    )
    started = time.perf_counter()
    candidate_documents = candidate.encode(
        documents,
        batch_size=args.candidate_batch_size,
        normalize_embeddings=True,
    )
    candidate_queries = candidate.encode(
        queries,
        batch_size=args.candidate_batch_size,
        normalize_embeddings=True,
    )
    candidate_seconds = time.perf_counter() - started
    candidate_repeat = candidate.encode(
        documents,
        batch_size=args.candidate_batch_size,
        normalize_embeddings=True,
    )
    document_cosines = np.sum(
        reference_documents * candidate_documents, axis=1
    )
    query_cosines = np.sum(reference_queries * candidate_queries, axis=1)
    reference_scores = reference_documents @ reference_queries.T
    candidate_scores = candidate_documents @ candidate_queries.T
    ranking_rows = []
    for index in range(len(queries)):
        reference_ranking = _top_k(reference_scores[:, index], args.top_k)
        candidate_ranking = _top_k(candidate_scores[:, index], args.top_k)
        ranking_rows.append(
            {
                "query_index": index,
                "top_k_overlap": len(
                    set(reference_ranking) & set(candidate_ranking)
                )
                / args.top_k,
                "reference_top_k": reference_ranking,
                "candidate_top_k": candidate_ranking,
            }
        )
    total_records = len(documents) + len(queries)
    result = {
        "schema_version": 1,
        "status": "runtime_equivalence_probe_complete",
        "purpose": (
            "Implementation-only equivalence and throughput validation. The probe "
            "does not compare experimental conditions, score a task against gold, "
            "or inspect any generation outcome."
        ),
        "source_benchmark": str(args.benchmark),
        "source_benchmark_sha256": sha256_file(args.benchmark),
        "model": args.model,
        "model_revision": args.model_revision,
        "hardware_label": args.hardware_label,
        "platform": platform.platform(),
        "documents": len(documents),
        "queries": len(queries),
        "reference": {
            **reference_metadata,
            "batch_size": args.reference_batch_size,
            "seconds": reference_seconds,
            "records_per_second": total_records / reference_seconds,
        },
        "candidate": {
            **candidate.runtime_metadata,
            "batch_size": args.candidate_batch_size,
            "seconds": candidate_seconds,
            "records_per_second": total_records / candidate_seconds,
        },
        "document_embedding_cosine": {
            "minimum": float(np.min(document_cosines)),
            "mean": float(np.mean(document_cosines)),
        },
        "query_embedding_cosine": {
            "minimum": float(np.min(query_cosines)),
            "mean": float(np.mean(query_cosines)),
        },
        "maximum_absolute_embedding_difference": float(
            np.max(np.abs(reference_documents - candidate_documents))
        ),
        "candidate_repeat_maximum_absolute_difference": float(
            np.max(np.abs(candidate_documents - candidate_repeat))
        ),
        "ranking_probe": ranking_rows,
        "minimum_top_k_overlap": min(
            row["top_k_overlap"] for row in ranking_rows
        ),
        "mean_top_k_overlap": float(
            np.mean([row["top_k_overlap"] for row in ranking_rows])
        ),
    }
    result["passed"] = (
        result["document_embedding_cosine"]["minimum"] >= 0.99999
        and result["query_embedding_cosine"]["minimum"] >= 0.99999
        and result["minimum_top_k_overlap"] == 1.0
        and result["candidate_repeat_maximum_absolute_difference"] == 0.0
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
