from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_evidence_pack import build_article_evidence_pack
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--primary-workspaces", type=Path, required=True)
    parser.add_argument("--replication-workspaces", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--word-target", type=int, default=3000)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    records = []
    for row in construction["records"]:
        dataset_id = row["dataset_id"]
        workspace_root = (
            args.primary_workspaces
            if row["source_panel"] == "primary"
            else args.replication_workspaces
        )
        pack_path = args.output_root / dataset_id / "evidence_pack.json"
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        pack = None
        if args.resume_existing and pack_path.is_file():
            candidate = read_json(pack_path)
            figure_path = Path((candidate.get("figures") or [{}])[0].get("path", ""))
            if (
                candidate.get("passed") is True
                and candidate.get("source_selection_version")
                == "topic-task-bm25-v4-title-deduplicated"
                and candidate.get("source_benchmark_sha256")
                == sha256_file(benchmark_path)
                and figure_path.is_file()
                and sha256_file(figure_path)
                == candidate["figures"][0].get("sha256")
            ):
                pack = candidate
        if pack is None:
            pack = build_article_evidence_pack(
                benchmark_path=benchmark_path,
                workspace=workspace_root / dataset_id,
                output_dir=args.output_root / dataset_id,
                word_target=args.word_target,
            )
        records.append(
            {
                "dataset_id": dataset_id,
                "source_panel": row["source_panel"],
                "passed": pack["passed"],
                "phenomena": len(pack["graph_phenomena"]),
                "sources": len(pack["representative_sources"]),
                "abstract_excerpts": sum(
                    bool(source["abstract_excerpt"])
                    for source in pack["representative_sources"]
                ),
                "pack": str(pack_path.resolve()),
                "pack_sha256": sha256_file(pack_path),
                "figure_sha256": pack["figures"][0]["sha256"],
            }
        )
        print(dataset_id, records[-1]["sources"], records[-1]["abstract_excerpts"])
    manifest = {
        "schema_version": 1,
        "status": (
            "same_evidence_packs_ready"
            if len(records) == 8 and all(row["passed"] for row in records)
            else "incomplete"
        ),
        "source_construction_sha256": sha256_file(
            args.benchmark_root / "construction_manifest.json"
        ),
        "datasets": len(records),
        "article_conditions": 3,
        "planned_articles": len(records) * 3,
        "records": records,
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
