from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_oversight_readiness import audit_article_oversight_readiness
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-root", type=Path, required=True)
    parser.add_argument("--writer-input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for dataset_dir in sorted(path for path in args.article_root.iterdir() if path.is_dir()):
        article_path = dataset_dir / "draft.md"
        writer_input_path = args.writer_input_root / dataset_dir.name / "writer_input.json"
        if not article_path.is_file() or not writer_input_path.is_file():
            continue
        writer_input = read_json(writer_input_path)
        audit = audit_article_oversight_readiness(
            article_path.read_text(encoding="utf-8"),
            phenomenon_ids=sorted(
                row["phenomenon_id"] for row in writer_input["graph_phenomena"]
            ),
        )
        records.append(
            {
                "dataset_id": dataset_dir.name,
                "article_sha256": sha256_file(article_path),
                "writer_input_sha256": sha256_file(writer_input_path),
                **audit,
            }
        )
    report = {
        "schema_version": 1,
        "status": "article_oversight_readiness_audited",
        "articles": len(records),
        "ready": sum(
            row["status"] == "ready_for_claim_review_packetization"
            for row in records
        ),
        "records": records,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
