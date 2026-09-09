from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_compiler_v2 import COMPILER_CONDITIONS
from citeweave.io import read_json, sha256_file, write_json


def build_plan(
    *,
    writer_input_root: Path,
    dataset_ids: list[str],
    output_root: Path,
    protocol_sha256: str,
) -> dict:
    articles = []
    for dataset_id in dataset_ids:
        writer_input = writer_input_root / dataset_id / "writer_input.json"
        if read_json(writer_input).get("dataset_id") != dataset_id:
            raise ValueError(f"Writer-input dataset mismatch: {dataset_id}")
        for condition in COMPILER_CONDITIONS:
            articles.append(
                {
                    "article_id": f"{dataset_id}__{condition}",
                    "dataset_id": dataset_id,
                    "condition": condition,
                    "writer_input": str(writer_input.resolve()),
                    "writer_input_sha256": sha256_file(writer_input),
                    "output_dir": str((output_root / dataset_id / condition).resolve()),
                }
            )
    return {
        "schema_version": 1,
        "status": "matched_article_compiler_claim_ready_development_plan",
        "confirmatory_replacement": False,
        "protocol_sha256": protocol_sha256,
        "conditions": list(COMPILER_CONDITIONS),
        "articles": articles,
        "provider_calls_per_article": 9,
        "maximum_provider_calls": 9 * len(articles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-input-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()
    protocol_sha256 = sha256_file(args.protocol)
    if read_json(args.protocol_freeze).get("sha256") != protocol_sha256:
        raise SystemExit("Matched article-compiler protocol differs from freeze")
    plan = build_plan(
        writer_input_root=args.writer_input_root,
        dataset_ids=args.dataset,
        output_root=args.output_root,
        protocol_sha256=protocol_sha256,
    )
    write_json(args.output_plan, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
