from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from .bulk_acquisition import _fingerprint, _read_raw_page
from .io import load_config, read_json, sha256_bytes, sha256_file, write_json
from .models import HarvestManifest, ProjectPaths, SourceName


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _payload_items(source: SourceName, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if source == SourceName.crossref:
        return list((payload.get("message") or {}).get("items") or [])
    if source == SourceName.europe_pmc:
        return list((payload.get("resultList") or {}).get("result") or [])
    return list(payload.get("results") or [])


def _payload_next_cursor(source: SourceName, payload: dict[str, Any]) -> str | None:
    if source == SourceName.crossref:
        return (payload.get("message") or {}).get("next-cursor")
    if source == SourceName.europe_pmc:
        return payload.get("nextCursorMark")
    return (payload.get("meta") or {}).get("next_cursor")


def _raw_identity(source: SourceName, item: dict[str, Any]) -> str:
    if source == SourceName.crossref:
        value = (item.get("DOI") or item.get("URL") or "").casefold()
    elif source == SourceName.europe_pmc:
        value = f"{item.get('source', '')}:{item.get('id', '')}".strip(":")
    else:
        value = str(item.get("id") or "")
    if value:
        return value
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"anonymous:{sha256_bytes(encoded)}"


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

    config_path = paths.root / "project.yml"
    config = load_config(config_path) if config_path.exists() else None
    checks.append(
        _check(
            "protocol_fingerprint",
            config is not None and _fingerprint(config) == harvest.query_fingerprint,
            {
                "project_config": str(config_path),
                "config_present": config is not None,
                "manifest_fingerprint": harvest.query_fingerprint,
                "computed_fingerprint": _fingerprint(config) if config is not None else None,
            },
        )
    )

    slices = sorted(harvest.slices, key=lambda item: item.date_from)
    coverage_errors: list[str] = []
    for previous, current in pairwise(slices):
        expected_next = date.fromisoformat(previous.date_to) + timedelta(days=1)
        if date.fromisoformat(current.date_from) != expected_next:
            coverage_errors.append(f"{previous.slice_id} -> {current.slice_id}")
    expected_first = f"{config.protocol.year_from:04d}-01-01" if config else None
    expected_last = f"{config.protocol.year_to:04d}-12-31" if config else None
    boundary_complete = bool(
        slices
        and config is not None
        and slices[0].date_from == expected_first
        and slices[-1].date_to == expected_last
    )
    checks.append(
        _check(
            "full_period_non_overlapping_contiguous_slices",
            boundary_complete and not coverage_errors,
            {
                "slices": len(slices),
                "first": slices[0].date_from if slices else None,
                "last": slices[-1].date_to if slices else None,
                "expected_first": expected_first,
                "expected_last": expected_last,
                "errors": coverage_errors,
            },
        )
    )

    incomplete = [item.slice_id for item in slices if item.status != "complete"]
    count_errors = [
        {
            "slice": item.slice_id,
            "planned_expected": item.expected_records,
            "cursor_snapshot_expected": item.cursor_snapshot_expected_records,
            "received": item.received_records,
        }
        for item in slices
        if (
            (
                item.cursor_snapshot_expected_records
                if item.cursor_snapshot_expected_records is not None
                else item.expected_records
            )
            != item.received_records
            and not (
                harvest.source == SourceName.openalex
                and item.cursor_exhausted
                and any(
                    f"OpenAlex cursor exhausted with count drift for {item.slice_id}"
                    in warning
                    for warning in harvest.warnings
                )
            )
        )
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
    raw_seen_path = paths.audit / "harvest_verify_raw_seen.sqlite.tmp"
    raw_seen_path.unlink(missing_ok=True)
    raw_connection = sqlite3.connect(raw_seen_path)
    raw_connection.execute("CREATE TABLE seen (record_id TEXT PRIMARY KEY)")
    raw_records = 0
    raw_unique = 0
    archived_pages_checked = 0
    try:
        for shard in slices:
            for restart in shard.restart_history:
                archived_indices = list(range(1, len(restart.pages) + 1))
                if [page.page_index for page in restart.pages] != archived_indices:
                    page_errors.append(
                        f"{shard.slice_id}: restart {restart.restart_index} page index gap"
                    )
                if sum(page.records for page in restart.pages) != restart.received_records:
                    page_errors.append(
                        f"{shard.slice_id}: restart {restart.restart_index} page count mismatch"
                    )
                archived_cursor = "*"
                for page in restart.pages:
                    if page.cursor_in != archived_cursor:
                        page_errors.append(
                            f"{shard.slice_id}: restart {restart.restart_index} "
                            f"cursor chain mismatch at page {page.page_index}"
                        )
                    path = paths.root / page.raw_path
                    if not path.exists():
                        page_errors.append(f"{page.raw_path}: archived page missing")
                        archived_cursor = page.cursor_out or ""
                        continue
                    try:
                        payload = _read_raw_page(path)
                        encoded = json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                        if sha256_bytes(encoded) != page.raw_sha256:
                            page_errors.append(f"{page.raw_path}: archived hash mismatch")
                        if path.stat().st_size != page.bytes_compressed:
                            page_errors.append(f"{page.raw_path}: archived size mismatch")
                        if len(_payload_items(harvest.source, payload)) != page.records:
                            page_errors.append(
                                f"{page.raw_path}: archived payload item count mismatch"
                            )
                        if _payload_next_cursor(harvest.source, payload) != page.cursor_out:
                            page_errors.append(
                                f"{page.raw_path}: archived response cursor mismatch"
                            )
                    except (
                        OSError,
                        EOFError,
                        gzip.BadGzipFile,
                        UnicodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        page_errors.append(
                            f"{page.raw_path}: archived unreadable ({type(exc).__name__})"
                        )
                    archived_cursor = page.cursor_out or ""
                    archived_pages_checked += 1
                if archived_cursor != restart.prior_cursor:
                    page_errors.append(
                        f"{shard.slice_id}: restart {restart.restart_index} "
                        "prior cursor does not continue its archived page chain"
                    )
            expected_indices = list(range(1, len(shard.pages) + 1))
            actual_indices = [page.page_index for page in shard.pages]
            if actual_indices != expected_indices:
                page_errors.append(f"{shard.slice_id}: page index gap")
            if sum(page.records for page in shard.pages) != shard.received_records:
                page_errors.append(f"{shard.slice_id}: page count mismatch")
            expected_cursor = "*"
            for page in shard.pages:
                if page.cursor_in != expected_cursor:
                    page_errors.append(
                        f"{shard.slice_id}: cursor chain mismatch at page {page.page_index}"
                    )
                path = paths.root / page.raw_path
                if not path.exists():
                    page_errors.append(f"{page.raw_path}: missing")
                    expected_cursor = page.cursor_out or ""
                    continue
                try:
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
                    items = _payload_items(harvest.source, payload)
                    if len(items) != page.records:
                        page_errors.append(f"{page.raw_path}: payload item count mismatch")
                    if _payload_next_cursor(harvest.source, payload) != page.cursor_out:
                        page_errors.append(f"{page.raw_path}: response cursor mismatch")
                    for item in items:
                        raw_records += 1
                        cursor = raw_connection.execute(
                            "INSERT OR IGNORE INTO seen(record_id) VALUES (?)",
                            (_raw_identity(harvest.source, item),),
                        )
                        raw_unique += int(cursor.rowcount > 0)
                except (
                    OSError,
                    EOFError,
                    gzip.BadGzipFile,
                    UnicodeError,
                    json.JSONDecodeError,
                    sqlite3.Error,
                ) as exc:
                    page_errors.append(f"{page.raw_path}: unreadable ({type(exc).__name__})")
                expected_cursor = page.cursor_out or ""
                checked_pages += 1
        raw_connection.commit()
    finally:
        raw_connection.close()
        raw_seen_path.unlink(missing_ok=True)
    raw_duplicates = raw_records - raw_unique
    if raw_records != harvest.received_records:
        page_errors.append("raw payload total does not match harvest.received_records")
    if raw_unique != harvest.unique_records or raw_duplicates != harvest.duplicate_records:
        page_errors.append("raw identity accounting does not match harvest deduplication totals")
    checks.append(
        _check(
            "raw_page_integrity",
            not page_errors and checked_pages == sum(len(item.pages) for item in slices),
            {
                "checked_pages": checked_pages,
                "archived_pages_checked": archived_pages_checked,
                "raw_records": raw_records,
                "raw_unique": raw_unique,
                "raw_duplicates": raw_duplicates,
                "errors": page_errors,
            },
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

    root_count_equal = harvest.root_expected_records == harvest.planned_expected_records
    count_drift_documented = any(
        "Count changed while planning" in warning for warning in harvest.warnings
    )
    cursor_drift_documented = any(
        "Cursor snapshot count changed from planning" in warning
        for warning in harvest.warnings
    )
    exhausted_count_drift_documented = any(
        "OpenAlex cursor exhausted with count drift" in warning
        for warning in harvest.warnings
    )
    aggregate_pass = (
        harvest.status == "complete"
        and (root_count_equal or count_drift_documented)
        and harvest.planned_expected_records == sum(item.expected_records for item in slices)
        and harvest.received_records == sum(item.received_records for item in slices)
        and (
            harvest.received_records == harvest.planned_expected_records
            or cursor_drift_documented
            or exhausted_count_drift_documented
        )
        and all(
            item.status == "complete"
            and item.cursor_exhausted
            and (
                item.received_records
                == (
                    item.cursor_snapshot_expected_records
                    if item.cursor_snapshot_expected_records is not None
                    else item.expected_records
                )
                or (
                    harvest.source == SourceName.openalex
                    and any(
                        f"OpenAlex cursor exhausted with count drift for {item.slice_id}"
                        in warning
                        for warning in harvest.warnings
                    )
                )
            )
            for item in slices
        )
        and harvest.unique_records + harvest.duplicate_records == harvest.received_records
    )
    checks.append(
        _check(
            "aggregate_completeness",
            aggregate_pass,
            {
                "status": harvest.status,
                "root_expected": harvest.root_expected_records,
                "planned_expected": harvest.planned_expected_records,
                "cursor_snapshot_expected": harvest.cursor_snapshot_expected_records,
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
