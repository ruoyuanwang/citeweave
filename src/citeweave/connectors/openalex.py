from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from ..exceptions import CompletenessError, ConfigurationError
from ..models import SourceName
from .base import AcquisitionResult, BaseConnector


class OpenAlexConnector(BaseConnector):
    source_name = SourceName.openalex.value
    endpoint = "https://api.openalex.org/works"

    def __init__(
        self,
        *args: Any,
        api_key: str | None = None,
        api_key_env: str = "OPENALEX_API_KEY",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.api_key = api_key or os.getenv(api_key_env)

    def acquire(self, protocol: Any) -> AcquisitionResult:
        filters = [
            f"from_publication_date:{protocol.year_from}-01-01",
            f"to_publication_date:{protocol.year_to}-12-31",
        ]
        if len(protocol.document_types) == 1:
            filters.append(f"type:{protocol.document_types[0]}")
        params: dict[str, Any] = {
            "search": protocol.query_text,
            "filter": ",".join(filters),
            "per_page": 100,
            "cursor": "*",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        manifest_query = {
            key: ("***" if key == "api_key" else value) for key, value in params.items()
        }
        manifest = self._new_manifest(self.source_name, manifest_query)
        if not self.api_key:
            manifest.warnings.append(
                "No OpenAlex API key configured; current OpenAlex quotas may reject the request."
            )
        if len(protocol.document_types) > 1:
            manifest.warnings.append(
                "Multiple OpenAlex work types are not compiled in v0.1; acquisition is unfiltered by type."
            )

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_paths = []
        cursor: str | None = "*"
        page_index = 0
        expected: int | None = None
        while cursor:
            params["cursor"] = cursor
            try:
                payload = self._request_json(self.endpoint, params=params)
            except Exception as exc:
                if "Insufficient budget" in str(exc) and not self.api_key:
                    raise ConfigurationError(
                        "OpenAlex requires an API key or available daily budget. "
                        "Set OPENALEX_API_KEY or choose Crossref/Europe PMC."
                    ) from exc
                raise
            page_index += 1
            path, digest = self._save_raw_page(page_index, payload)
            raw_paths.append(path)
            manifest.raw_sha256.append(digest)
            meta = payload.get("meta", {})
            if expected is None:
                expected = int(meta.get("count", 0))
                manifest.expected_records = expected
            items = payload.get("results") or []
            if not items:
                break
            for item in items:
                identifier = item.get("id")
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
            cursor = meta.get("next_cursor")

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
                f"OpenAlex completeness failed: expected={expected}, unique={len(records)}"
            )
        return AcquisitionResult(records, manifest, raw_paths)
