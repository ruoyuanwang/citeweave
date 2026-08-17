from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..exceptions import CompletenessError
from ..models import SourceName
from .base import AcquisitionResult, BaseConnector


class CrossrefConnector(BaseConnector):
    source_name = SourceName.crossref.value
    endpoint = "https://api.crossref.org/works"

    def __init__(self, *args: Any, mailto: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mailto = mailto

    def _params(self, protocol: Any) -> dict[str, Any]:
        filters = [
            f"from-pub-date:{protocol.year_from}-01-01",
            f"until-pub-date:{protocol.year_to}-12-31",
        ]
        if protocol.document_types:
            filters.append(f"type:{protocol.document_types[0]}")
        params: dict[str, Any] = {
            "query.bibliographic": protocol.query_text,
            "filter": ",".join(filters),
            "rows": 1000,
            "cursor": "*",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def acquire(self, protocol: Any) -> AcquisitionResult:
        params = self._params(protocol)
        manifest = self._new_manifest(self.source_name, params.copy())
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_paths = []
        page_index = 0
        cursor: str | None = "*"
        expected: int | None = None

        if len(protocol.document_types) > 1:
            manifest.warnings.append(
                "Crossref API supports one type filter per run; only the first requested type was used."
            )

        while cursor:
            params["cursor"] = cursor
            payload = self._request_json(self.endpoint, params=params)
            page_index += 1
            path, digest = self._save_raw_page(page_index, payload)
            raw_paths.append(path)
            manifest.raw_sha256.append(digest)
            message = payload.get("message", {})
            if expected is None:
                expected = int(message.get("total-results", 0))
                manifest.expected_records = expected
            items = message.get("items") or []
            if not items:
                break

            for item in items:
                identifier = (item.get("DOI") or item.get("URL") or "").lower()
                if not identifier:
                    identifier = f"anonymous:{page_index}:{len(records)}"
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

            next_cursor = message.get("next-cursor")
            if len(items) < int(params["rows"]) or not next_cursor or next_cursor == cursor:
                cursor = None
            else:
                cursor = next_cursor

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
                f"Crossref completeness failed: expected={expected}, unique={len(records)}, "
                f"received={manifest.received_records}"
            )
        return AcquisitionResult(records=records, manifest=manifest, raw_paths=raw_paths)
