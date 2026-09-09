from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from citeweave.io import read_json, sha256_file, write_json
from citeweave.review_voi import build_claim_dependency_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.article_plan)
    rows = []
    article_records = []
    for article in plan["articles"]:
        output_dir = Path(article["output_dir"])
        execution = read_json(output_dir / "execution_record.json")
        readiness = read_json(output_dir / "oversight_readiness.json")
        if readiness["status"] != "ready_for_claim_review_packetization":
            raise RuntimeError(f"Article is not claim-ready: {article['article_id']}")
        draft_path = output_dir / "draft.md"
        writer_path = Path(article["writer_input"])
        features = build_claim_dependency_features(
            draft_path.read_text(encoding="utf-8"),
            article_id=article["article_id"],
            dataset_id=article["dataset_id"],
            writer_input=read_json(writer_path),
        )
        for row in features:
            row["article_id"] = article["article_id"]
            row["condition"] = article["condition"]
            rows.append(row)
        article_records.append(
            {
                "article_id": article["article_id"],
                "condition": article["condition"],
                "draft_sha256": sha256_file(draft_path),
                "writer_input_sha256": sha256_file(writer_path),
                "machine_gate_passed": execution["machine_gate_passed"],
                "claim_features": len(features),
            }
        )
    dependency_counts = Counter(
        (row["routing_scope_id"], dependency_id)
        for row in rows
        for dependency_id in row["dependency_ids"]
    )
    report = {
        "schema_version": 1,
        "status": "development_claim_voi_features_ready_without_human_outcomes",
        "scientific_role": "development_only_not_confirmatory",
        "article_plan_sha256": sha256_file(args.article_plan),
        "articles": len(article_records),
        "claim_features": len(rows),
        "unique_dependencies": len(dependency_counts),
        "shared_dependencies": sum(value > 1 for value in dependency_counts.values()),
        "maximum_dependency_fanout": max(dependency_counts.values()),
        "human_outcomes_present": 0,
        "routing_status": "blocked_until_real_resolved_reviews_and_preoutcome_time_estimates",
        "article_records": article_records,
        "features": rows,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
