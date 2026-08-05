from __future__ import annotations

import json
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import AcquisitionError
from ..io import atomic_write_bytes, sha256_bytes
from ..models import AcquisitionManifest, SearchProtocol


@dataclass
class AcquisitionResult:
    records: Iterable[dict[str, Any]]
    manifest: AcquisitionManifest
    raw_paths: list[Path]
    staged_path: Path | None = None


class BaseConnector(ABC):
    source_name: str

    def __init__(
        self,
        raw_dir: Path,
        *,
        timeout: float = 60,
        max_retries: int = 6,
        requests_per_second: float | None = None,
        user_agent: str = "CiteWeave/0.1 (metadata research workflow)",
    ):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.requests_per_second = requests_per_second
        self.request_attempts = 0
        self.retry_count = 0
        self.throttle_seconds = 0.0
        self._last_request_at = 0.0
        self._pace_lock = threading.Lock()
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def _pace(self) -> None:
        if self.requests_per_second is None:
            return
        minimum_interval = 1.0 / self.requests_per_second
        with self._pace_lock:
            now = time.monotonic()
            delay = max(0.0, self._last_request_at + minimum_interval - now)
            if delay:
                time.sleep(delay)
                self.throttle_seconds += delay
            self._last_request_at = time.monotonic()

    def _request_json(
        self, url: str, *, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._pace()
                self.request_attempts += 1
                response = self.client.get(url, params=params, headers=headers)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = AcquisitionError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                    if attempt >= self.max_retries:
                        break
                    retry_after = response.headers.get("retry-after")
                    base_delay = min(
                        float(retry_after) if retry_after else 2**attempt,
                        60,
                    )
                    delay = base_delay + random.uniform(0, min(1.0, base_delay * 0.25))
                    self.retry_count += 1
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                base_delay = min(2**attempt, 30)
                self.retry_count += 1
                time.sleep(base_delay + random.uniform(0, min(1.0, base_delay * 0.25)))
        raise AcquisitionError(f"{self.source_name} request failed after retries: {last_error}")

    def _save_raw_page(self, page_index: int, payload: dict[str, Any]) -> tuple[Path, str]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = sha256_bytes(encoded)
        path = self.raw_dir / f"{self.source_name}-page-{page_index:06d}-{digest[:12]}.json"
        if not path.exists():
            atomic_write_bytes(path, encoded)
        return path, digest

    @staticmethod
    def _new_manifest(source: str, query: dict[str, Any]) -> AcquisitionManifest:
        return AcquisitionManifest(
            source=source,
            query=query,
            started_at=datetime.now(UTC),
        )

    @abstractmethod
    def acquire(self, protocol: SearchProtocol) -> AcquisitionResult:
        raise NotImplementedError
