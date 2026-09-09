from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.human_review_v2 import (
    build_adjudication_worklist,
    validate_review_return,
)
from citeweave.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.packet_root / "internal_manifest.json")
    reviewers = manifest["reviewers"]
    if len(reviewers) != 2:
        raise SystemExit("The registered double-review protocol requires exactly two reviewers")
    args.output.mkdir(parents=True, exist_ok=True)
    validated = []
    for reviewer in reviewers:
        return_path = args.packet_root / "returns" / f"{reviewer}.json"
        if not return_path.is_file():
            raise SystemExit(f"Missing frozen return for {reviewer}")
        result = validate_review_return(
            read_json(return_path),
            internal_manifest=manifest,
            packet_root=args.packet_root,
            require_complete=not args.allow_incomplete,
        )
        write_json(args.output / f"validated_{reviewer}.json", result)
        validated.append(result)
    worklist = build_adjudication_worklist(
        validated[0],
        validated[1],
        output_path=args.output / "adjudication_worklist.json",
    )
    summary = {
        "schema_version": 1,
        "reviewers": reviewers,
        "completed": {item["reviewer_code"]: item["completed"] for item in validated},
        "review_seconds": {
            item["reviewer_code"]: item["total_review_seconds"] for item in validated
        },
        "common_packets": worklist["common_packets"],
        "exact_agreement_rate": worklist["exact_agreement_rate"],
        "cohen_kappa": worklist["cohen_kappa"],
        "adjudication_required": worklist["adjudication_required"],
        "policy_compilation_unlocked": worklist["adjudication_required"] == 0,
        "note": (
            "If disagreements exist, freeze an adjudicated result file before compiling "
            "feedback rules; raw reviewer votes must not be silently majority-merged."
        ),
    }
    write_json(args.output / "review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
