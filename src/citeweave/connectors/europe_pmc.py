from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..exceptions import AcquisitionError, CompletenessError
from ..io import atomic_write_bytes, sha256_bytes
from ..models import SourceName
from .base import AcquisitionResult, BaseConnector


class EuropePmcConnector(BaseConnector):
    source_name = SourceName.europe_pmc.value
    endpoint = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @staticmethod
    def _compile_query(protocol: Any) -> str:
        if protocol.query_mode == "phrase":
            terms = f'"{protocol.query_text}"'
        else:
            joiner = " AND " if protocol.query_mode == "all" else " OR "
            terms = joiner.join(f'"{term}"' for term in protocol.keywords)
            terms = f"({terms})"
        return f"{terms} AND FIRST_PDATE:[{protocol.year_from}-01-01 TO {protocol.year_to}-12-31]"

    def acquire(self, protocol: Any) -> AcquisitionResult:
        params: dict[str, Any] = {
            "query": self._compile_query(protocol),
            "format": "json",
            "resultType": "core",
            "pageSize": 1000,
            "cursorMark": "*",
        }
        manifest = self._new_manifest(self.source_name, params.copy())
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_paths = []
        cursor: str | None = "*"
        page_index = 0
        expected: int | None = None

        while cursor:
            params["cursorMark"] = cursor
            payload = self._request_json(self.endpoint, params=params)
            page_index += 1
            path, digest = self._save_raw_page(page_index, payload)
            raw_paths.append(path)
            manifest.raw_sha256.append(digest)
            if expected is None:
                expected = int(payload.get("hitCount", 0))
                manifest.expected_records = expected
            items = (payload.get("resultList") or {}).get("result") or []
            if not items:
                break
            for item in items:
                identifier = f"{item.get('source', '')}:{item.get('id', '')}"
                if identifier in seen:
                    manifest.duplicate_records += 1
                    continue
                seen.add(identifier)
                records.append(item)
            manifest.pages = page_index
            manifest.received_records += len(items)
            manifest.unique_records = len(records)
            if protocol.max_records and len(records) >= protocol.max_records:
                records = records[: protocol.max_records]
                manifest.unique_records = len(records)
                manifest.truncated = True
                manifest.warnings.append("Acquisition stopped at protocol.max_records.")
                break
            next_cursor = payload.get("nextCursorMark")
            cursor = next_cursor if next_cursor and next_cursor != cursor else None

        manifest.finished_at = datetime.now(UTC)
        manifest.drift = None if expected is None else len(records) - expected
        manifest.complete = bool(
            not manifest.truncated
            and expected is not None
            and len(records) >= expected
            and manifest.failed_pages == 0
        )
        if not manifest.complete and not manifest.truncated:
            raise CompletenessError(
                f"Europe PMC completeness failed: expected={expected}, unique={len(records)}"
            )
        if protocol.include_references and not manifest.truncated:
            records, reference_hashes, reference_paths, reference_warnings = (
                self._enrich_references(records)
            )
            manifest.raw_sha256.extend(reference_hashes)
            raw_paths.extend(reference_paths)
            manifest.warnings.extend(reference_warnings)
        return AcquisitionResult(records, manifest, raw_paths)

    def _fetch_reference_pages(
        self, item: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str | None]:
        source = item.get("source")
        identifier = item.get("id")
        key = f"{source}:{identifier}"
        if not source or not identifier or item.get("hasReferences") != "Y":
            return key, [], [], None
        endpoint = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{identifier}/references"
        )
        pages: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        expected: int | None = None
        try:
            page = 1
            while True:
                payload = self._request_json(
                    endpoint,
                    params={"format": "json", "page": page, "pageSize": 1000},
                )
                pages.append(payload)
                if expected is None:
                    expected = int(payload.get("hitCount", 0))
                batch = (payload.get("referenceList") or {}).get("reference") or []
                references.extend(batch)
                if not batch or len(references) >= expected:
                    break
                page += 1
            warning = None
            if expected is not None and len(references) < expected:
                warning = (
                    f"Reference enrichment incomplete for {key}: "
                    f"expected={expected}, received={len(references)}"
                )
            return key, references, pages, warning
        except (AcquisitionError, KeyError, TypeError, ValueError) as exc:
            return key, [], pages, f"Reference enrichment failed for {key}: {exc}"

    def _save_reference_page(
        self, key: str, page: int, payload: dict[str, Any]
    ) -> tuple[Path, str]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = sha256_bytes(encoded)
        safe_key = key.replace(":", "-").replace("/", "-")
        directory = self.raw_dir / "references"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_key}-page-{page:04d}-{digest[:12]}.json"
        if not path.exists():
            atomic_write_bytes(path, encoded)
        return path, digest

    def _enrich_references(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str], list[Path], list[str]]:
        by_key = {f"{item.get('source')}:{item.get('id')}": item for item in records}
        raw_hashes: list[str] = []
        raw_paths: list[Path] = []
        warnings: list[str] = []
        eligible = [item for item in records if item.get("hasReferences") == "Y"]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(self._fetch_reference_pages, item) for item in eligible]
            for future in as_completed(futures):
                key, references, pages, warning = future.result()
                if key in by_key:
                    by_key[key]["_references"] = references
                for page_index, payload in enumerate(pages, start=1):
                    path, digest = self._save_reference_page(key, page_index, payload)
                    raw_paths.append(path)
                    raw_hashes.append(digest)
                if warning:
                    warnings.append(warning)
        return records, raw_hashes, raw_paths, warnings
