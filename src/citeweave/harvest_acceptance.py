from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from .bulk_acquisition import _read_raw_page
from .io import read_json, sha256_bytes, sha256_file, write_json
from .models import HarvestManifest, ProjectPaths, SourceName


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_bulk_harvest(root: Path) -> dict[str, Any]:
    """Verify a completed bulk crawl from checkpoint to staged JSONL."""
    paths = ProjectPaths(root)
    manifest_path = paths.audit / "harvest_manifest.json"
    if not manifest_path.exists():
        return {
            "project_root": str(paths.root),
            "passed": False,
            "passed_checks": 0,
            "total_checks": 1,
            "checks": [_check("harvest_manifest", False, "<missing>")],
        }
    harvest = HarvestManifest.model_validate(read_json(manifest_path))
    checks: list[dict[str, Any]] = []

    slices = sorted(harvest.slices, key=lambda item: item.date_from)
    coverage_errors: list[str] = []
    for previous, current in pairwise(slices):
        expected_next = date.fromisoformat(previous.date_to) + timedelta(days=1)
        if date.fromisoformat(current.date_from) != expected_next:
            coverage_errors.append(f"{previous.slice_id} -> {current.slice_id}")
    checks.append(
        _check(
            "non_overlapping_contiguous_slices",
            bool(slices) and not coverage_errors,
            {
                "slices": len(slices),
                "first": slices[0].date_from if slices else None,
                "last": slices[-1].date_to if slices else None,
                "errors": coverage_errors,
            },
        )
    )

    incomplete = [item.slice_id for item in slices if item.status != "complete"]
    count_errors = [
        {
            "slice": item.slice_id,
            "expected": item.expected_records,
            "received": item.received_records,
        }
        for item in slices
        if item.expected_records != item.received_records
    ]
    checks.append(
        _check(
            "slice_completeness",
            not incomplete and not count_errors,
            {"incomplete": incomplete, "count_errors": count_errors},
        )
    )

    page_errors: list[str] = []
    checked_pages = 0
    for shard in slices:
        expected_indices = list(range(1, len(shard.pages) + 1))
        actual_indices = [page.page_index for page in shard.pages]
        if actual_indices != expected_indices:
            page_errors.append(f"{shard.slice_id}: page index gap")
        if sum(page.records for page in shard.pages) != shard.received_records:
            page_errors.append(f"{shard.slice_id}: page count mismatch")
        for page in shard.pages:
            path = paths.root / page.raw_path
            if not path.exists():
                page_errors.append(f"{page.raw_path}: missing")
                continue
            payload = _read_raw_page(path)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            if sha256_bytes(encoded) != page.raw_sha256:
                page_errors.append(f"{page.raw_path}: hash mismatch")
            if path.stat().st_size != page.bytes_compressed:
                page_errors.append(f"{page.raw_path}: size mismatch")
            checked_pages += 1
    checks.append(
        _check(
            "raw_page_integrity",
            not page_errors and checked_pages == sum(len(item.pages) for item in slices),
            {"checked_pages": checked_pages, "errors": page_errors},
        )
    )

    staged_path = paths.root / harvest.staged_path if harvest.staged_path else None
    staged_lines = 0
    staged_unique = 0
    staged_error: str | None = None
    profile = {
        "records": 0,
        "with_doi": 0,
        "with_title": 0,
        "with_authors": 0,
        "with_abstract": 0,
        "with_references": 0,
        "reference_links": 0,
    }
    seen_path = paths.audit / "harvest_verify_seen.sqlite.tmp"
    seen_path.unlink(missing_ok=True)
    connection: sqlite3.Connection | None = None
    if staged_path and staged_path.exists():
        try:
            connection = sqlite3.connect(seen_path)
            connection.execute("CREATE TABLE seen (record_id TEXT PRIMARY KEY)")
            opener = gzip.open if staged_path.suffix == ".gz" else open
            with opener(staged_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    staged_lines += 1
                    doi = item.get("DOI") or item.get("doi")
                    if harvest.source == SourceName.crossref:
                        identifier = (doi or item.get("URL") or "").casefold()
                        title = item.get("title")
                        authors = item.get("author")
                        abstract = item.get("abstract")
                        references = item.get("reference") or []
                    elif harvest.source == SourceName.europe_pmc:
                        identifier = f"{item.get('source', '')}:{item.get('id', '')}"
                        title = item.get("title")
                        authors = (item.get("authorList") or {}).get("author")
                        abstract = item.get("abstractText")
                        references = item.get("_references") or []
                    else:
                        identifier = str(item.get("id") or "")
                        title = item.get("display_name") or item.get("title")
                        authors = item.get("authorships")
                        abstract = item.get("abstract_inverted_index")
                        references = item.get("referenced_works") or []
                    if not identifier:
                        identifier = f"anonymous:{sha256_bytes(line.encode('utf-8'))}"
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO seen(record_id) VALUES (?)",
                        (identifier,),
                    )
                    staged_unique += int(cursor.rowcount > 0)
                    profile["records"] += 1
                    profile["with_doi"] += int(bool(doi))
                    profile["with_title"] += int(bool(title))
                    profile["with_authors"] += int(bool(authors))
                    profile["with_abstract"] += int(bool(abstract))
                    profile["with_references"] += int(bool(references))
                    profile["reference_links"] += len(references)
                connection.commit()
            connection.close()
            connection = None
            write_json(paths.audit / "harvest_metadata_profile.json", profile)
        except (OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error) as exc:
            staged_error = str(exc)
        finally:
            if connection is not None:
                connection.close()
            seen_path.unlink(missing_ok=True)
    checks.append(
        _check(
            "staged_corpus_integrity",
            staged_path is not None
            and staged_path.exists()
            and staged_error is None
            and staged_lines == harvest.unique_records
            and staged_unique == harvest.unique_records
            and sha256_file(staged_path) == harvest.staged_sha256,
            {
                "path": str(staged_path) if staged_path else None,
                "lines": staged_lines,
                "unique_ids": staged_unique,
                "expected_lines": harvest.unique_records,
                "profile": profile,
                "error": staged_error,
            },
        )
    )

    aggregate_pass = (
        harvest.status == "complete"
        and harvest.planned_expected_records == sum(item.expected_records for item in slices)
        and harvest.received_records == sum(item.received_records for item in slices)
        and harvest.unique_records == harvest.planned_expected_records
    )
    checks.append(
        _check(
            "aggregate_completeness",
            aggregate_pass,
            {
                "status": harvest.status,
                "root_expected": harvest.root_expected_records,
                "planned_expected": harvest.planned_expected_records,
                "received": harvest.received_records,
                "unique": harvest.unique_records,
                "duplicates": harvest.duplicate_records,
            },
        )
    )

    passed = all(item["passed"] for item in checks)
    return {
        "project_root": str(paths.root),
        "passed": passed,
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
    }
