from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from citeweave.article_evidence_pack import ARTICLE_TASK_TYPES
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    records = []
    for row in manifest.get("records", []):
        pack_path = Path(row["pack"])
        pack = read_json(pack_path)
        source_ids = [source["reference_id"] for source in pack["representative_sources"]]
        normalized_titles = [
            re.sub(r"[^a-z0-9]+", " ", source["title"].casefold()).strip()
            for source in pack["representative_sources"]
        ]
        phenomena = pack["graph_phenomena"]
        figure_path = Path(pack["figures"][0]["path"])
        checks = {
            "pack_hash_matches_manifest": sha256_file(pack_path) == row["pack_sha256"],
            "figure_hash_matches_pack": (
                sha256_file(figure_path) == pack["figures"][0]["sha256"]
            ),
            "five_registered_phenomena": (
                [item["task_type"] for item in phenomena] == list(ARTICLE_TASK_TYPES)
            ),
            "fifteen_unique_sources": len(source_ids) == len(set(source_ids)) == 15,
            "fifteen_unique_normalized_titles": len(normalized_titles)
            == len(set(normalized_titles))
            == 15,
            "all_abstracts_nonempty": all(
                source["abstract_excerpt"].strip()
                for source in pack["representative_sources"]
            ),
            "three_sources_per_phenomenon": all(
                len(item["reference_ids"]) == len(set(item["reference_ids"])) == 3
                and set(item["reference_ids"]).issubset(source_ids)
                for item in phenomena
            ),
            "pack_quality_gate_passed": pack.get("passed") is True,
        }
        records.append(
            {
                "dataset_id": row["dataset_id"],
                "passed": all(checks.values()),
                "checks": checks,
                "phenomena": len(phenomena),
                "sources": len(source_ids),
                "figures": len(pack["figures"]),
            }
        )
    audit = {
        "schema_version": 1,
        "status": (
            "passed"
            if manifest.get("status") == "same_evidence_packs_ready"
            and len(records) == 8
            and all(row["passed"] for row in records)
            else "failed"
        ),
        "datasets": len(records),
        "phenomena": sum(row["phenomena"] for row in records),
        "sources_with_abstracts": sum(row["sources"] for row in records),
        "figures": sum(row["figures"] for row in records),
        "records": records,
    }
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
