from __future__ import annotations

import gzip
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .connectors.base import AcquisitionResult, BaseConnector
from .exceptions import AcquisitionError, CompletenessError, ConfigurationError
from .io import atomic_write_bytes, read_json, sha256_bytes, sha256_file, write_json
from .models import (
    AcquisitionManifest,
    HarvestManifest,
    HarvestPage,
    HarvestSlice,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def harvest_lock(paths: ProjectPaths) -> Iterator[None]:
    """Prevent concurrent writers while allowing recovery from a dead process."""
    lock_path = paths.audit / "harvest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            existing_pid = 0
        if _pid_is_running(existing_pid):
            raise AcquisitionError(
                f"Another bulk harvest is already running with PID {existing_pid}."
            )
        lock_path.unlink(missing_ok=True)
    payload = json.dumps(
        {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()},
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise AcquisitionError("Another bulk harvest acquired the project lock.") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        if current.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)


class BulkSourceAdapter:
    """Translate one resumable harvesting algorithm to each supported API."""

    def __init__(self, connector: BaseConnector):
        self.connector = connector
        self.source = SourceName(connector.source_name)

    @property
    def endpoint(self) -> str:
        return str(self.connector.endpoint)

    @property
    def maximum_page_size(self) -> int:
        return 100 if self.source == SourceName.openalex else 1_000

    def default_page_size(self) -> int:
        return self.maximum_page_size

    def query_description(
        self,
        protocol: SearchProtocol,
        date_from: date,
        date_to: date,
    ) -> dict[str, Any]:
        params = self._params(protocol, date_from, date_to, page_size=1, cursor="*")
        return {
            key: ("***" if key == "api_key" else value)
            for key, value in params.items()
            if key not in {"cursor", "cursorMark", "rows", "per-page", "pageSize"}
        }

    def count(
        self,
        protocol: SearchProtocol,
        date_from: date,
        date_to: date,
    ) -> int:
        params = self._params(protocol, date_from, date_to, page_size=1, cursor="*")
        if self.source == SourceName.crossref:
            params["rows"] = 0
            params.pop("cursor", None)
        elif self.source == SourceName.europe_pmc:
            params["resultType"] = "idlist"
        payload = self.connector._request_json(self.endpoint, params=params)
        return self.expected(payload)

    def page(
        self,
        protocol: SearchProtocol,
        date_from: date,
        date_to: date,
        *,
        page_size: int,
        cursor: str,
    ) -> dict[str, Any]:
        params = self._params(protocol, date_from, date_to, page_size, cursor)
        return self.connector._request_json(self.endpoint, params=params)

    def _params(
        self,
        protocol: SearchProtocol,
        date_from: date,
        date_to: date,
        page_size: int,
        cursor: str,
    ) -> dict[str, Any]:
        start = date_from.isoformat()
        end = date_to.isoformat()
        if self.source == SourceName.europe_pmc:
            if protocol.query_mode == "phrase":
                terms = f'"{protocol.query_text}"'
            else:
                joiner = " AND " if protocol.query_mode == "all" else " OR "
                quoted_terms = [f'"{term}"' for term in protocol.keywords]
                terms = f"({joiner.join(quoted_terms)})"
            query = f"{terms} AND FIRST_PDATE:[{start} TO {end}]"
            if protocol.language:
                query += f" AND LANG:{protocol.language}"
            return {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": page_size,
                "cursorMark": cursor,
            }

        if self.source == SourceName.crossref:
            filters = [f"from-pub-date:{start}", f"until-pub-date:{end}"]
            if protocol.document_types:
                filters.append(f"type:{protocol.document_types[0]}")
            params: dict[str, Any] = {
                "query.bibliographic": protocol.query_text,
                "filter": ",".join(filters),
                "rows": page_size,
                "cursor": cursor,
            }
            mailto = getattr(self.connector, "mailto", None)
            if mailto:
                params["mailto"] = mailto
            return params

        if self.source == SourceName.openalex:
            filters = [
                f"from_publication_date:{start}",
                f"to_publication_date:{end}",
            ]
            if protocol.document_types:
                filters.append(f"type:{'|'.join(protocol.document_types)}")
            if protocol.language:
                filters.append(f"language:{protocol.language}")
            params = {
                "search": protocol.query_text,
                "filter": ",".join(filters),
                "per-page": page_size,
                "cursor": cursor,
            }
            api_key = getattr(self.connector, "api_key", None)
            if api_key:
                params["api_key"] = api_key
            return params

        raise ConfigurationError(f"Bulk acquisition is unsupported for {self.source.value}.")

    def expected(self, payload: dict[str, Any]) -> int:
        if self.source == SourceName.europe_pmc:
            return int(payload.get("hitCount", 0))
        if self.source == SourceName.crossref:
            return int((payload.get("message") or {}).get("total-results", 0))
        return int((payload.get("meta") or {}).get("count", 0))

    def items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.source == SourceName.europe_pmc:
            return list((payload.get("resultList") or {}).get("result") or [])
        if self.source == SourceName.crossref:
            return list((payload.get("message") or {}).get("items") or [])
        return list(payload.get("results") or [])

    def next_cursor(self, payload: dict[str, Any]) -> str | None:
        if self.source == SourceName.europe_pmc:
            return payload.get("nextCursorMark")
        if self.source == SourceName.crossref:
            return (payload.get("message") or {}).get("next-cursor")
        return (payload.get("meta") or {}).get("next_cursor")

    def page_is_last(
        self,
        items: list[dict[str, Any]],
        *,
        page_size: int,
        next_cursor: str | None,
        received: int,
        expected: int,
    ) -> bool:
        if not items or received >= expected:
            return True
        if self.source == SourceName.crossref:
            # Crossref returns a next cursor even after the final page.
            return len(items) < page_size
        return not next_cursor

    def identity(self, item: dict[str, Any]) -> str:
        if self.source == SourceName.europe_pmc:
            value = f"{item.get('source', '')}:{item.get('id', '')}".strip(":")
            if value:
                return value
        elif self.source == SourceName.crossref:
            value = (item.get("DOI") or item.get("URL") or "").casefold()
            if value:
                return value
        else:
            value = str(item.get("id") or "")
            if value:
                return value
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str).encode()
        return f"anonymous:{sha256_bytes(encoded)}"


def _fingerprint(config: ProjectConfig) -> str:
    payload = {
        "protocol": config.protocol.model_dump(mode="json"),
        "source": config.protocol.source.value,
        "partition_strategy": config.acquisition.partition_strategy,
        "target_slice_records": config.acquisition.target_slice_records,
        "page_size": config.acquisition.page_size,
    }
    return sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )


def _slice_id(date_from: date, date_to: date) -> str:
    return f"{date_from:%Y%m%d}-{date_to:%Y%m%d}"


def _plan_slices(
    adapter: BulkSourceAdapter,
    config: ProjectConfig,
) -> tuple[int, list[HarvestSlice], list[str]]:
    protocol = config.protocol
    policy = config.acquisition
    start = date(protocol.year_from, 1, 1)
    end = date(protocol.year_to, 12, 31)
    warnings: list[str] = []
    root_expected = adapter.count(protocol, start, end)

    if policy.partition_strategy == "none":
        return (
            root_expected,
            [
                HarvestSlice(
                    slice_id=_slice_id(start, end),
                    date_from=start.isoformat(),
                    date_to=end.isoformat(),
                    expected_records=root_expected,
                    status="complete" if root_expected == 0 else "pending",
                    cursor=None if root_expected == 0 else "*",
                )
            ],
            warnings,
        )

    if policy.partition_strategy == "year":
        slices = []
        for year in range(protocol.year_from, protocol.year_to + 1):
            year_start = date(year, 1, 1)
            year_end = date(year, 12, 31)
            expected = adapter.count(protocol, year_start, year_end)
            slices.append(
                HarvestSlice(
                    slice_id=_slice_id(year_start, year_end),
                    date_from=year_start.isoformat(),
                    date_to=year_end.isoformat(),
                    expected_records=expected,
                    status="complete" if expected == 0 else "pending",
                    cursor=None if expected == 0 else "*",
                )
            )
        return root_expected, slices, warnings

    leaves: list[HarvestSlice] = []

    def split(left: date, right: date, expected: int) -> None:
        if expected <= policy.target_slice_records or left == right:
            leaves.append(
                HarvestSlice(
                    slice_id=_slice_id(left, right),
                    date_from=left.isoformat(),
                    date_to=right.isoformat(),
                    expected_records=expected,
                    status="complete" if expected == 0 else "pending",
                    cursor=None if expected == 0 else "*",
                )
            )
            if expected > policy.target_slice_records:
                warnings.append(
                    f"Single-day slice {_slice_id(left, right)} contains {expected:,} records, "
                    "above target_slice_records."
                )
            return
        midpoint = left + timedelta(days=(right - left).days // 2)
        right_start = midpoint + timedelta(days=1)
        left_expected = adapter.count(protocol, left, midpoint)
        right_expected = adapter.count(protocol, right_start, right)
        if left_expected + right_expected != expected:
            warnings.append(
                f"Count changed while planning {left.isoformat()}..{right.isoformat()}: "
                f"parent={expected}, children={left_expected + right_expected}."
            )
        split(left, midpoint, left_expected)
        split(right_start, right, right_expected)

    split(start, end, root_expected)
    leaves.sort(key=lambda item: item.date_from)
    return root_expected, leaves, warnings


def _checkpoint(path: Path, manifest: HarvestManifest) -> None:
    manifest.updated_at = datetime.now(UTC)
    manifest.received_records = sum(item.received_records for item in manifest.slices)
    manifest.raw_bytes_compressed = sum(
        page.bytes_compressed for item in manifest.slices for page in item.pages
    )
    write_json(path, manifest.model_dump(mode="json"))


def _save_raw_page(
    paths: ProjectPaths,
    source: SourceName,
    slice_id: str,
    page_index: int,
    payload: dict[str, Any],
    *,
    compress: bool,
) -> tuple[Path, str, int]:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = sha256_bytes(encoded)
    directory = paths.raw / "harvest" / slice_id
    if compress:
        output = directory / f"{source.value}-page-{page_index:06d}-{digest[:12]}.json.gz"
        content = gzip.compress(encoded, compresslevel=6, mtime=0)
    else:
        output = directory / f"{source.value}-page-{page_index:06d}-{digest[:12]}.json"
        content = encoded
    if not output.exists():
        atomic_write_bytes(output, content)
    return output, digest, len(content)


def _read_raw_page(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_staged_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _assemble_staged_records(
    paths: ProjectPaths,
    manifest: HarvestManifest,
    adapter: BulkSourceAdapter,
) -> tuple[Path, int, int]:
    output = paths.staged / "source_records.jsonl.gz"
    temporary = output.with_suffix(output.suffix + ".tmp")
    seen_path = paths.staged / "harvest_seen.sqlite.tmp"
    temporary.unlink(missing_ok=True)
    seen_path.unlink(missing_ok=True)
    connection = sqlite3.connect(seen_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE seen (record_id TEXT PRIMARY KEY)")
    unique = 0
    duplicates = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as out:
            for shard in sorted(manifest.slices, key=lambda item: item.date_from):
                for page in sorted(shard.pages, key=lambda item: item.page_index):
                    payload = _read_raw_page(paths.root / page.raw_path)
                    for item in adapter.items(payload):
                        identifier = adapter.identity(item)
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO seen(record_id) VALUES (?)",
                            (identifier,),
                        )
                        if cursor.rowcount == 0:
                            duplicates += 1
                            continue
                        out.write(json.dumps(item, ensure_ascii=False, default=str))
                        out.write("\n")
                        unique += 1
                    connection.commit()
        os.replace(temporary, output)
    finally:
        connection.close()
        seen_path.unlink(missing_ok=True)
        Path(f"{seen_path}-wal").unlink(missing_ok=True)
        Path(f"{seen_path}-shm").unlink(missing_ok=True)
    return output, unique, duplicates


def _aggregate_acquisition_manifest(
    harvest: HarvestManifest,
    connector: BaseConnector,
) -> AcquisitionManifest:
    raw_hashes = [page.raw_sha256 for shard in harvest.slices for page in shard.pages]
    return AcquisitionManifest(
        source=harvest.source,
        query=harvest.query,
        started_at=harvest.created_at,
        finished_at=harvest.updated_at,
        expected_records=harvest.planned_expected_records,
        received_records=harvest.received_records,
        unique_records=harvest.unique_records,
        duplicate_records=harvest.duplicate_records,
        pages=len(raw_hashes),
        failed_pages=sum(item.failure_count for item in harvest.slices),
        complete=harvest.status == "complete",
        truncated=False,
        drift=harvest.unique_records - harvest.planned_expected_records,
        raw_sha256=raw_hashes,
        warnings=[
            *harvest.warnings,
            (
                f"Bulk acquisition used {len(harvest.slices)} date slices, "
                f"{connector.request_attempts} HTTP attempts and "
                f"{connector.retry_count} request retries."
            ),
        ],
    )


def bulk_acquire(
    paths: ProjectPaths,
    config: ProjectConfig,
    connector: BaseConnector,
    *,
    resume: bool = True,
    page_budget: int | None = None,
) -> AcquisitionResult:
    """Acquire a large result set with adaptive shards and per-page checkpoints."""
    if config.protocol.source == SourceName.import_file:
        raise ConfigurationError("Bulk API acquisition does not apply to import_file.")
    if config.protocol.max_records is not None:
        raise ConfigurationError(
            "Bulk acquisition is an all-records mode; protocol.max_records must be null."
        )
    if config.protocol.source == SourceName.crossref and len(config.protocol.document_types) > 1:
        raise ConfigurationError(
            "Crossref accepts one work-type filter per cursor query. Use one type or "
            "separate projects so every slice has an auditable, disjoint scope."
        )
    if config.protocol.source == SourceName.crossref and config.protocol.language:
        raise ConfigurationError(
            "Crossref bulk queries cannot enforce a language filter; remove protocol.language "
            "or choose a source with an API-level language filter."
        )
    if config.protocol.source == SourceName.europe_pmc and config.protocol.document_types:
        raise ConfigurationError(
            "Europe PMC bulk document-type compilation is not yet supported; remove the "
            "document_types filter instead of silently harvesting a broader corpus."
        )
    if config.protocol.source == SourceName.europe_pmc and config.protocol.include_references:
        raise ConfigurationError(
            "Europe PMC bulk search returns complete core publication metadata but not "
            "reference lists. Set include_references=false for the bulk metadata pass; "
            "reference enrichment must be run as a separate resumable queue."
        )

    adapter = BulkSourceAdapter(connector)
    policy = config.acquisition
    page_size = min(policy.page_size or adapter.default_page_size(), adapter.maximum_page_size)
    checkpoint_path = paths.audit / "harvest_manifest.json"
    fingerprint = _fingerprint(config)
    resumed_existing = checkpoint_path.exists()

    if resumed_existing:
        if not resume:
            raise ConfigurationError(
                f"Harvest checkpoint already exists at {checkpoint_path}; enable resume."
            )
        harvest = HarvestManifest.model_validate(read_json(checkpoint_path))
        if harvest.query_fingerprint != fingerprint:
            raise ConfigurationError(
                "Existing harvest checkpoint belongs to a different protocol or policy."
            )
        if harvest.status == "complete" and harvest.staged_path:
            staged_path = paths.root / harvest.staged_path
            if staged_path.exists() and sha256_file(staged_path) == harvest.staged_sha256:
                aggregate = _aggregate_acquisition_manifest(harvest, connector)
                return AcquisitionResult(
                    iter_staged_records(staged_path),
                    aggregate,
                    [paths.root / page.raw_path for s in harvest.slices for page in s.pages],
                    staged_path,
                )
    else:
        root_expected, slices, warnings = _plan_slices(adapter, config)
        start = date(config.protocol.year_from, 1, 1)
        end = date(config.protocol.year_to, 12, 31)
        harvest = HarvestManifest(
            source=config.protocol.source,
            query_fingerprint=fingerprint,
            query=adapter.query_description(config.protocol, start, end),
            page_size=page_size,
            partition_strategy=policy.partition_strategy,
            target_slice_records=policy.target_slice_records,
            root_expected_records=root_expected,
            planned_expected_records=sum(item.expected_records for item in slices),
            slices=slices,
            warnings=warnings,
        )
        _checkpoint(checkpoint_path, harvest)

    if resumed_existing and harvest.source == SourceName.crossref:
        for shard in harvest.slices:
            if shard.status in {"running", "failed"} and shard.pages:
                if shard.restart_count >= policy.max_slice_restarts:
                    raise AcquisitionError(
                        f"Crossref slice {shard.slice_id} exceeded max_slice_restarts."
                    )
                shard.restart_count += 1
                shard.status = "pending"
                shard.cursor = "*"
                shard.received_records = 0
                shard.pages = []
                shard.last_error = None
        _checkpoint(checkpoint_path, harvest)

    harvest.status = "running"
    _checkpoint(checkpoint_path, harvest)
    pages_this_run = 0
    for shard in harvest.slices:
        if shard.status == "complete":
            continue
        shard.status = "running"
        shard.started_at = shard.started_at or datetime.now(UTC)
        shard.last_error = None
        cursor = shard.cursor or "*"
        try:
            while cursor:
                if page_budget is not None and pages_this_run >= page_budget:
                    harvest.status = "partial"
                    shard.cursor = cursor
                    _checkpoint(checkpoint_path, harvest)
                    aggregate = _aggregate_acquisition_manifest(harvest, connector)
                    raw_paths = [
                        paths.root / page.raw_path for item in harvest.slices for page in item.pages
                    ]
                    return AcquisitionResult([], aggregate, raw_paths)
                payload = adapter.page(
                    config.protocol,
                    date.fromisoformat(shard.date_from),
                    date.fromisoformat(shard.date_to),
                    page_size=page_size,
                    cursor=cursor,
                )
                items = adapter.items(payload)
                next_cursor = adapter.next_cursor(payload)
                page_index = len(shard.pages) + 1
                raw_path, digest, compressed_bytes = _save_raw_page(
                    paths,
                    harvest.source,
                    shard.slice_id,
                    page_index,
                    payload,
                    compress=policy.compress_raw,
                )
                shard.received_records += len(items)
                shard.pages.append(
                    HarvestPage(
                        page_index=page_index,
                        cursor_in=cursor,
                        cursor_out=next_cursor,
                        records=len(items),
                        raw_path=raw_path.relative_to(paths.root).as_posix(),
                        raw_sha256=digest,
                        bytes_compressed=compressed_bytes,
                    )
                )
                pages_this_run += 1
                if adapter.page_is_last(
                    items,
                    page_size=page_size,
                    next_cursor=next_cursor,
                    received=shard.received_records,
                    expected=shard.expected_records,
                ):
                    shard.cursor = None
                    break
                cursor_stalled = next_cursor == cursor and harvest.source != SourceName.crossref
                if not next_cursor or cursor_stalled:
                    raise CompletenessError(
                        f"Cursor stopped before slice completion: {shard.slice_id} "
                        f"received={shard.received_records}, expected={shard.expected_records}."
                    )
                cursor = next_cursor
                shard.cursor = cursor
                _checkpoint(checkpoint_path, harvest)

            if shard.received_records != shard.expected_records:
                raise CompletenessError(
                    f"Slice {shard.slice_id} expected {shard.expected_records} records "
                    f"but received {shard.received_records}."
                )
            shard.status = "complete"
            shard.finished_at = datetime.now(UTC)
            shard.cursor = None
            _checkpoint(checkpoint_path, harvest)
        except Exception as exc:
            shard.status = "failed"
            shard.failure_count += 1
            shard.last_error = str(exc)[:1_000]
            harvest.status = "failed"
            _checkpoint(checkpoint_path, harvest)
            raise

    staged_path, unique, duplicates = _assemble_staged_records(paths, harvest, adapter)
    harvest.unique_records = unique
    harvest.duplicate_records = duplicates
    harvest.staged_path = staged_path.relative_to(paths.root).as_posix()
    harvest.staged_sha256 = sha256_file(staged_path)
    if unique != harvest.planned_expected_records:
        harvest.status = "failed"
        _checkpoint(checkpoint_path, harvest)
        raise CompletenessError(
            f"Bulk acquisition completeness failed: expected={harvest.planned_expected_records}, "
            f"unique={unique}, duplicates={duplicates}."
        )
    harvest.status = "complete"
    _checkpoint(checkpoint_path, harvest)
    aggregate = _aggregate_acquisition_manifest(harvest, connector)
    raw_paths = [paths.root / page.raw_path for item in harvest.slices for page in item.pages]
    return AcquisitionResult(
        iter_staged_records(staged_path),
        aggregate,
        raw_paths,
        staged_path,
    )
