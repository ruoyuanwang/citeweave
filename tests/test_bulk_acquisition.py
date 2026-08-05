from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from bibagent.bulk_acquisition import bulk_acquire, harvest_lock, iter_staged_records
from bibagent.connectors.base import BaseConnector
from bibagent.exceptions import AcquisitionError
from bibagent.harvest_acceptance import verify_bulk_harvest
from bibagent.models import (
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
    ):
        super().__init__(raw_dir, max_retries=0)
        self.records_per_day = records_per_day
        self.constant_cursor = constant_cursor
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
                "DOI": f"10.9999/{prefix}.{index}",
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
    connector = SyntheticCrossrefConnector(paths.raw, records_per_day=300)

    result = bulk_acquire(paths, _config(), connector)
    records = sum(1 for _ in iter_staged_records(result.staged_path))
    verification = verify_bulk_harvest(tmp_path)

    assert result.manifest.complete
    assert result.manifest.expected_records == 109_800
    assert records == 109_800
    assert result.manifest.pages >= 100
    assert verification["passed"]
    assert verification["passed_checks"] == verification["total_checks"] == 5


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
    monkeypatch.setattr("bibagent.connectors.base.time.sleep", lambda _: None)
    monkeypatch.setattr("bibagent.connectors.base.random.uniform", lambda _a, _b: 0)

    payload = connector._request_json(connector.endpoint, params={"rows": 0})

    assert payload == {"ok": True}
    assert connector.request_attempts == 3
    assert connector.retry_count == 2


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
