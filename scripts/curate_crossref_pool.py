from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeweave.io import sha256_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a deterministic lexical relevance rule to a frozen Crossref pool."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--term", action="append", required=True)
    parser.add_argument("--year-from", type=int, required=True)
    parser.add_argument("--year-to", type=int, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def _publication_year(item: dict[str, Any]) -> int | None:
    for key in ("published", "issued", "published-online", "published-print"):
        value = item.get(key)
        parts = value.get("date-parts") if isinstance(value, dict) else None
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def _text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            *(item.get("title") or []),
            item.get("abstract") or "",
            *(item.get("subject") or []),
        ]
    ).casefold()


def main() -> None:
    args = parse_args()
    raw_paths = sorted(args.raw_dir.glob("crossref-page-*.json"))
    if not raw_paths:
        raise SystemExit(f"No Crossref raw pages found under {args.raw_dir}")

    terms = [term.casefold() for term in args.term]
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    source_records = 0
    excluded = {"duplicate": 0, "year": 0, "relevance": 0}
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in (payload.get("message") or {}).get("items") or []:
            source_records += 1
            identifier = (item.get("DOI") or item.get("URL") or "").casefold()
            if identifier and identifier in seen:
                excluded["duplicate"] += 1
                continue
            if identifier:
                seen.add(identifier)
            year = _publication_year(item)
            if year is None or not args.year_from <= year <= args.year_to:
                excluded["year"] += 1
                continue
            text = _text(item)
            if not all(term in text for term in terms):
                excluded["relevance"] += 1
                continue
            selected.append(item)

    eligible = len(selected)
    if args.max_records is not None:
        selected = selected[: args.max_records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, selected)
    manifest = {
        "curation_version": 1,
        "created_at": datetime.now(UTC),
        "source": "crossref",
        "selection_order": "Crossref relevance order preserved",
        "rule": {
            "fields": ["title", "abstract", "subject"],
            "case_sensitive": False,
            "must_contain_all_terms": terms,
            "year_from": args.year_from,
            "year_to": args.year_to,
            "max_records": args.max_records,
        },
        "raw_pages": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in raw_paths
        ],
        "counts": {
            "source_records": source_records,
            "unique_seen": len(seen),
            "eligible_before_cap": eligible,
            "selected": len(selected),
            "excluded": excluded,
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
