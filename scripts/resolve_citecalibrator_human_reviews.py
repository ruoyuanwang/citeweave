from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from citeweave.citecalibrator_benchmark import resolve_human_reviews
from citeweave.io import sha256_file, write_json, write_jsonl


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--primary-reviews", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = _read_jsonl(args.cases)
    primary_reviews = _read_jsonl(args.primary_reviews)
    adjudications = _read_jsonl(args.adjudications)
    result = resolve_human_reviews(
        cases=cases,
        primary_reviews=primary_reviews,
        adjudications=adjudications,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "adjudication_packets.jsonl", result["adjudication_packets"])
    if result["status"] == "resolved":
        write_jsonl(args.output_dir / "gold.jsonl", result["gold"])
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "cases": result["cases"],
        "resolved": result["resolved"],
        "missing_primary_cases": len(result["missing_primary_case_ids"]),
        "adjudication_cases": len(result["adjudication_packets"]),
        "cases_sha256": sha256_file(args.cases),
        "primary_reviews_sha256": (
            sha256_file(args.primary_reviews) if args.primary_reviews.is_file() else None
        ),
        "adjudications_sha256": (
            sha256_file(args.adjudications)
            if args.adjudications and args.adjudications.is_file()
            else None
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
