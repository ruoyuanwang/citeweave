from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from citeweave.io import write_json


def _rank(similarities: np.ndarray, limit: int) -> list[int]:
    return sorted(
        range(len(similarities)), key=lambda index: (-similarities[index], index)
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    documents = [
        json.dumps(
            {
                "evidence_id": f"synthetic:edge:{index:03d}",
                "source_label": f"Concept {index % 11}",
                "target_label": f"Mechanism {(index * 7) % 17}",
                "weight": float(1 + index % 13),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(64)
    ]
    queries = [
        "Which concept connects two mechanisms?",
        "Find a high-weight bridge edge.",
        "比较跨社区连接与局部连接。",
        "What structure may remain after a hub is removed?",
    ]
    model = SentenceTransformer(
        args.model,
        revision=args.model_revision,
        device="cpu",
        trust_remote_code=False,
        local_files_only=True,
    )
    started = time.perf_counter()
    fp_documents = model.encode(
        documents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    fp_queries = model.encode(
        queries,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    fp_seconds = time.perf_counter() - started
    model[0].auto_model = torch.ao.quantization.quantize_dynamic(
        model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    started = time.perf_counter()
    int8_documents = model.encode(
        documents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    int8_queries = model.encode(
        queries,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    int8_repeat = model.encode(
        documents,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    int8_seconds = time.perf_counter() - started
    cosines = np.sum(fp_documents * int8_documents, axis=1)
    ranking_rows = []
    for index, (fp_query, int8_query) in enumerate(
        zip(fp_queries, int8_queries, strict=True)
    ):
        fp_ranking = _rank(np.asarray(fp_documents) @ fp_query, 10)
        int8_ranking = _rank(np.asarray(int8_documents) @ int8_query, 10)
        ranking_rows.append(
            {
                "query_index": index,
                "top10_overlap": len(set(fp_ranking) & set(int8_ranking)) / 10,
                "fp32_top10": fp_ranking,
                "int8_top10": int8_ranking,
            }
        )
    result = {
        "schema_version": 1,
        "status": "runtime_precision_probe_complete",
        "purpose": (
            "Synthetic implementation validation only; contains no benchmark task, "
            "evidence record, retrieval outcome, or generation outcome."
        ),
        "model": args.model,
        "model_revision": args.model_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "synthetic_documents": len(documents),
        "synthetic_queries": len(queries),
        "document_embedding_cosine": {
            "minimum": float(np.min(cosines)),
            "mean": float(np.mean(cosines)),
        },
        "maximum_absolute_embedding_difference": float(
            np.max(np.abs(fp_documents - int8_documents))
        ),
        "int8_repeat_maximum_absolute_difference": float(
            np.max(np.abs(int8_documents - int8_repeat))
        ),
        "ranking_probe": ranking_rows,
        "minimum_top10_overlap": min(row["top10_overlap"] for row in ranking_rows),
        "fp32_seconds": fp_seconds,
        "int8_two_pass_seconds": int8_seconds,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
