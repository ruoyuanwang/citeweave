from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from citeweave.bulk_acquisition import bulk_acquire, harvest_lock, iter_staged_records
from citeweave.connectors.base import BaseConnector
from citeweave.exceptions import AcquisitionError, InvalidCursorError
from citeweave.harvest_acceptance import verify_bulk_harvest
from citeweave.io import save_config
from citeweave.models import (
    AcquisitionPolicy,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)


class SyntheticCrossrefConnector(BaseConnector):
    source_name = SourceName.crossref.value
    endpoint = "https://synthetic.test/works"

    def __init__(
        self,
        raw_dir: Path,
        *,
        records_per_day: int,
        constant_cursor: bool = False,
        duplicate_ids: bool = False,
    ):
        super().__init__(raw_dir, max_retries=0)
        self.records_per_day = records_per_day
        self.constant_cursor = constant_cursor
        self.duplicate_ids = duplicate_ids
        self.requested_cursors: list[str | None] = []
        self._scroll_offsets: dict[str, int] = {}
        self.mailto = None

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover - bulk path only
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        match = re.search(
            r"from-pub-date:(\d{4}-\d{2}-\d{2}),until-pub-date:(\d{4}-\d{2}-\d{2})",
            params["filter"],
        )
        assert match
        start = date.fromisoformat(match.group(1))
        end = date.fromisoformat(match.group(2))
        total = ((end - start).days + 1) * self.records_per_day
        rows = int(params.get("rows", 0))
        if rows == 0:
            return {"message": {"total-results": total, "items": []}}
        cursor = params.get("cursor")
        self.requested_cursors.append(cursor)
        scroll_key = params["filter"]
        if self.constant_cursor:
            offset = 0 if cursor == "*" else self._scroll_offsets.get(scroll_key, 0)
        else:
            offset = 0 if cursor == "*" else int(str(cursor))
        size = max(0, min(rows, total - offset))
        prefix = f"{start:%Y%m%d}-{end:%Y%m%d}"
        items = [
            {
                "DOI": f"10.9999/{prefix}.{index // 2 if self.duplicate_ids else index}",
                "title": [f"Synthetic work {prefix} {index}"],
                "reference": [{"DOI": f"10.9999/ref.{index % 17}"}],
            }
            for index in range(offset, offset + size)
        ]
        self._scroll_offsets[scroll_key] = offset + size
        return {
            "message": {
                "total-results": total,
                "items": items,
                "next-cursor": "SCROLL" if self.constant_cursor else str(offset + size),
            }
        }


class DummyConnector(BaseConnector):
    source_name = SourceName.crossref.value
    endpoint = "https://synthetic.test/works"

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover
        raise NotImplementedError


class SyntheticOpenAlexDriftConnector(BaseConnector):
    source_name = SourceName.openalex.value
    endpoint = "https://synthetic.test/works"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir, max_retries=0)
        self.api_key = "test"

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        if int(params["per_page"]) == 1:
            return {"meta": {"count": 3, "next_cursor": None}, "results": []}
        return {
            "meta": {"count": 3, "next_cursor": None},
            "results": [
                {"id": f"https://openalex.org/W{index}", "display_name": f"Work {index}"}
                for index in range(4)
            ],
        }


class SyntheticOpenAlexEarlyCountConnector(BaseConnector):
    source_name = SourceName.openalex.value
    endpoint = "https://synthetic.test/works"

    def __init__(self, raw_dir: Path):
        super().__init__(raw_dir, max_retries=0)
        self.api_key = "test"
        self.requested_cursors: list[str] = []

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        if int(params["per_page"]) == 1 and params["cursor"] == "*":
            return {"meta": {"count": 1, "next_cursor": None}, "results": []}
        cursor = str(params["cursor"])
        self.requested_cursors.append(cursor)
        if cursor == "*":
            return {
                "meta": {"count": 1, "next_cursor": "second-page"},
                "results": [{"id": "https://openalex.org/W1", "display_name": "Work 1"}],
            }
        assert cursor == "second-page"
        return {
            "meta": {"count": 1, "next_cursor": None},
            "results": [{"id": "https://openalex.org/W2", "display_name": "Work 2"}],
        }


class SyntheticOpenAlexExpiredCursorConnector(BaseConnector):
    source_name = SourceName.openalex.value
    endpoint = "https://synthetic.test/works"

    def __init__(self, raw_dir: Path, *, reject_saved_once: bool):
        super().__init__(raw_dir, max_retries=0)
        self.api_key = "test"
        self.reject_saved_once = reject_saved_once
        self.requested_cursors: list[str] = []

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        if int(params["per_page"]) == 1:
            return {"meta": {"count": 2, "next_cursor": None}, "results": []}
        cursor = str(params["cursor"])
        self.requested_cursors.append(cursor)
        if cursor == "*":
            return {
                "meta": {"count": 2, "next_cursor": "saved-cursor"},
                "results": [
                    {
                        "id": "https://openalex.org/W-new-1",
                        "display_name": "New snapshot 1",
                    }
                ],
            }
        if self.reject_saved_once:
            self.reject_saved_once = False
            raise InvalidCursorError("explicit synthetic expired cursor")
        return {
            "meta": {"count": 2, "next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W-new-2",
                    "display_name": "New snapshot 2",
                }
            ],
        }


class SyntheticEuropePmcConnector(BaseConnector):
    source_name = SourceName.europe_pmc.value
    endpoint = "https://synthetic.test/search"

    def __init__(self, raw_dir: Path, *, records_per_day: int):
        super().__init__(raw_dir, max_retries=0)
        self.records_per_day = records_per_day
        self.requested_cursors: list[str | None] = []

    def acquire(self, protocol: SearchProtocol):  # pragma: no cover
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, headers
        match = re.search(
            r"FIRST_PDATE:\[(\d{4}-\d{2}-\d{2}) TO (\d{4}-\d{2}-\d{2})\]",
            params["query"],
        )
        assert match
        start = date.fromisoformat(match.group(1))
        end = date.fromisoformat(match.group(2))
        total = ((end - start).days + 1) * self.records_per_day
        if params.get("resultType") == "idlist":
            return {"hitCount": total, "resultList": {"result": []}}
        cursor = params.get("cursorMark")
        self.requested_cursors.append(cursor)
        offset = 0 if cursor == "*" else int(str(cursor))
        page_size = int(params["pageSize"])
        size = max(0, min(page_size, total - offset))
        items = [
            {"source": "MED", "id": f"{start:%Y%m%d}-{index}", "title": f"Work {index}"}
            for index in range(offset, offset + size)
        ]
        return {
            "hitCount": total,
            "nextCursorMark": str(offset + size),
            "resultList": {"result": items},
        }


def _config(*, target: int = 25_000, page_size: int = 1_000) -> ProjectConfig:
    return ProjectConfig(
        project_id="bulk-test",
        protocol=SearchProtocol(
            title="Synthetic large bibliometric crawl",
            keywords=["bibliometric"],
            year_from=2024,
            year_to=2024,
            source=SourceName.crossref,
            include_references=True,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            partition_strategy="adaptive_date",
            target_slice_records=target,
            page_size=page_size,
            max_retries=0,
        ),
    )


def test_bulk_harvest_streams_and_verifies_more_than_100k_records(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = _config()
    save_config(paths.root / "project.yml", config)
    connector = SyntheticCrossrefConnector(paths.raw, records_per_day=300)

    result = bulk_acquire(paths, config, connector)
    records = sum(1 for _ in iter_staged_records(result.staged_path))
    verification = verify_bulk_harvest(tmp_path)

    assert result.manifest.complete
    assert result.manifest.expected_records == 109_800
    assert records == 109_800
    assert result.manifest.pages >= 100
    assert verification["passed"], verification["checks"]
    assert verification["passed_checks"] == verification["total_checks"] == 6


def test_bulk_harvest_separates_raw_exhaustiveness_from_deduplication(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = _config(target=500_000)
    save_config(paths.root / "project.yml", config)
    connector = SyntheticCrossrefConnector(
        paths.raw,
        records_per_day=10,
        duplicate_ids=True,
    )

    result = bulk_acquire(paths, config, connector)
    verification = verify_bulk_harvest(tmp_path)

    assert result.manifest.complete
    assert result.manifest.received_records == result.manifest.expected_records == 3_660
    assert result.manifest.unique_records == 1_830
    assert result.manifest.duplicate_records == 1_830
    assert result.manifest.unique_records + result.manifest.duplicate_records == 3_660
    assert verification["passed"]


def test_openalex_uses_cursor_snapshot_count_when_index_drifts(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = ProjectConfig(
        project_id="openalex-drift",
        protocol=SearchProtocol(
            title="OpenAlex drift test",
            keywords=["bibliometric"],
            search_scope="title_abstract",
            year_from=2024,
            year_to=2024,
            source=SourceName.openalex,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            target_slice_records=500_000,
            page_size=100,
            max_retries=0,
        ),
    )
    save_config(paths.root / "project.yml", config)

    result = bulk_acquire(paths, config, SyntheticOpenAlexDriftConnector(paths.raw))
    verification = verify_bulk_harvest(tmp_path)

    assert result.manifest.complete
    assert result.manifest.expected_records == 3
    assert result.manifest.received_records == 4
    assert result.manifest.drift == 1
    assert verification["passed"], verification["checks"]
    assert any(
        "OpenAlex cursor exhausted with count drift" in warning
        for warning in result.manifest.warnings
    )


def test_openalex_exhausts_cursor_even_after_reaching_live_count(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = ProjectConfig(
        project_id="openalex-cursor-authority",
        protocol=SearchProtocol(
            title="OpenAlex cursor authority test",
            keywords=["bibliometric"],
            search_scope="title_abstract",
            year_from=2024,
            year_to=2024,
            source=SourceName.openalex,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            target_slice_records=500_000,
            page_size=100,
            max_retries=0,
        ),
    )
    save_config(paths.root / "project.yml", config)
    connector = SyntheticOpenAlexEarlyCountConnector(paths.raw)

    result = bulk_acquire(paths, config, connector)
    verification = verify_bulk_harvest(tmp_path)

    assert result.manifest.complete
    assert result.manifest.expected_records == 1
    assert result.manifest.received_records == 2
    assert connector.requested_cursors == ["*", "second-page"]
    assert verification["passed"], verification["checks"]


def test_openalex_explicit_expired_cursor_restarts_only_failed_slice(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = ProjectConfig(
        project_id="openalex-expired-cursor",
        protocol=SearchProtocol(
            title="OpenAlex expired cursor test",
            keywords=["bibliometric"],
            search_scope="title_abstract",
            year_from=2024,
            year_to=2024,
            source=SourceName.openalex,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            target_slice_records=500_000,
            page_size=100,
            max_retries=0,
            max_slice_restarts=2,
        ),
    )
    save_config(paths.root / "project.yml", config)
    first = SyntheticOpenAlexExpiredCursorConnector(paths.raw, reject_saved_once=False)
    partial = bulk_acquire(paths, config, first, page_budget=1)
    assert not partial.manifest.complete
    prior_raw_path = partial.raw_paths[0]
    prior_raw_hash = partial.manifest.raw_sha256[0]

    resumed = SyntheticOpenAlexExpiredCursorConnector(paths.raw, reject_saved_once=True)
    complete = bulk_acquire(paths, config, resumed, resume=True)
    verification = verify_bulk_harvest(tmp_path)
    harvest = __import__("json").loads(
        (paths.audit / "harvest_manifest.json").read_text(encoding="utf-8")
    )
    records = list(iter_staged_records(complete.staged_path))

    assert complete.manifest.complete
    assert resumed.requested_cursors == ["saved-cursor", "*", "saved-cursor"]
    assert [item["id"] for item in records] == [
        "https://openalex.org/W-new-1",
        "https://openalex.org/W-new-2",
    ]
    assert prior_raw_path.exists()
    restart = harvest["slices"][0]["restart_history"][0]
    assert restart["reason"] == "openalex_explicit_invalid_or_expired_cursor"
    assert restart["pages"][0]["raw_sha256"] == prior_raw_hash
    assert restart["pages"][0]["raw_path"] == prior_raw_path.relative_to(paths.root).as_posix()
    assert harvest["slices"][0]["restart_count"] == 1
    assert harvest["slices"][0]["failure_count"] == 1
    assert complete.manifest.failed_pages == 0
    assert verification["passed"], verification["checks"]
    raw_check = next(
        item for item in verification["checks"] if item["name"] == "raw_page_integrity"
    )
    assert raw_check["detail"]["archived_pages_checked"] == 1


def test_crossref_partial_harvest_restarts_only_incomplete_slice(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = _config(target=500_000)
    first = SyntheticCrossrefConnector(paths.raw, records_per_day=20)

    partial = bulk_acquire(paths, config, first, page_budget=2)
    assert not partial.manifest.complete
    assert partial.manifest.pages == 2

    resumed = SyntheticCrossrefConnector(paths.raw, records_per_day=20)
    complete = bulk_acquire(paths, config, resumed, resume=True)
    harvest = __import__("json").loads(
        (paths.audit / "harvest_manifest.json").read_text(encoding="utf-8")
    )

    assert complete.manifest.complete
    assert resumed.requested_cursors[0] == "*"
    assert harvest["slices"][0]["restart_count"] == 1
    assert sum(1 for _ in iter_staged_records(complete.staged_path)) == 7_320


def test_crossref_constant_scroll_cursor_is_not_treated_as_stalled(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    connector = SyntheticCrossrefConnector(
        paths.raw,
        records_per_day=10,
        constant_cursor=True,
    )

    result = bulk_acquire(paths, _config(target=500_000), connector)

    assert result.manifest.complete
    assert result.manifest.pages == 4
    assert connector.requested_cursors[1:] == ["SCROLL", "SCROLL", "SCROLL"]


def test_europe_pmc_resume_continues_from_saved_cursor(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = ProjectConfig(
        project_id="epmc-resume",
        protocol=SearchProtocol(
            title="Europe PMC resume test",
            keywords=["bibliometric"],
            year_from=2024,
            year_to=2024,
            source=SourceName.europe_pmc,
            include_references=False,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            target_slice_records=500_000,
            page_size=1_000,
            max_retries=0,
        ),
    )
    first = SyntheticEuropePmcConnector(paths.raw, records_per_day=10)
    partial = bulk_acquire(paths, config, first, page_budget=2)
    assert not partial.manifest.complete

    resumed = SyntheticEuropePmcConnector(paths.raw, records_per_day=10)
    complete = bulk_acquire(paths, config, resumed, resume=True)

    assert complete.manifest.complete
    assert resumed.requested_cursors[0] == "2000"
    assert sum(1 for _ in iter_staged_records(complete.staged_path)) == 3_660


def test_request_retries_429_and_server_error(monkeypatch, tmp_path: Path):
    statuses = iter([429, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(
            status,
            json={"ok": True} if status == 200 else {"error": status},
            headers={"Retry-After": "0"},
            request=request,
        )

    connector = DummyConnector(tmp_path, max_retries=3)
    connector.client.close()
    connector.client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("citeweave.connectors.base.time.sleep", lambda _: None)
    monkeypatch.setattr("citeweave.connectors.base.random.uniform", lambda _a, _b: 0)

    payload = connector._request_json(connector.endpoint, params={"rows": 0})

    assert payload == {"ok": True}
    assert connector.request_attempts == 3
    assert connector.retry_count == 2


def test_openalex_connector_classifies_only_explicit_invalid_cursor(tmp_path: Path):
    connector = SyntheticOpenAlexExpiredCursorConnector(
        tmp_path,
        reject_saved_once=False,
    )

    def invalid_cursor(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "Invalid cursor", "message": "The cursor has expired."},
            request=request,
        )

    connector.client.close()
    connector.client = httpx.Client(transport=httpx.MockTransport(invalid_cursor))
    with pytest.raises(InvalidCursorError):
        BaseConnector._request_json(connector, connector.endpoint, params={"cursor": "old"})

    def unrelated_bad_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "Invalid filter", "message": "Unknown field."},
            request=request,
        )

    connector.client.close()
    connector.client = httpx.Client(transport=httpx.MockTransport(unrelated_bad_request))
    with pytest.raises(AcquisitionError, match="request failed after retries"):
        BaseConnector._request_json(connector, connector.endpoint, params={"cursor": "old"})


def test_harvest_lock_rejects_concurrent_writer_and_recovers_stale_lock(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()

    with (
        harvest_lock(paths),
        pytest.raises(AcquisitionError, match="already running"),
        harvest_lock(paths),
    ):
        pass

    (paths.audit / "harvest.lock").write_text('{"pid": 999999999}', encoding="utf-8")
    with harvest_lock(paths):
        assert (paths.audit / "harvest.lock").exists()
    assert not (paths.audit / "harvest.lock").exists()
