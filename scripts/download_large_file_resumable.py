from __future__ import annotations

import argparse
import hashlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_RESOLVE_LOCK = threading.Lock()


def _resolve_url(source_url: str) -> str:
    with _RESOLVE_LOCK, requests.head(
        source_url, allow_redirects=True, timeout=(20, 30)
    ) as response:
        response.raise_for_status()
        return response.url


def _download_part(
    *,
    source_url: str,
    initial_url: str,
    path: Path,
    start: int,
    end: int,
    retries: int,
    request_chunk_bytes: int,
) -> tuple[int, int]:
    expected = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_url = initial_url
    for attempt in range(retries):
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return start, expected
        if current > expected:
            raise RuntimeError(f"Part exceeds registered range: {path}")
        offset = start + current
        request_end = min(end, offset + request_chunk_bytes - 1)
        requested_bytes = request_end - offset + 1
        try:
            with requests.get(
                resolved_url,
                headers={"Range": f"bytes={offset}-{request_end}"},
                stream=True,
                timeout=(20, 30),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server did not honor Range request: {response.status_code}"
                    )
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size != current + requested_bytes:
                raise RuntimeError(f"Incomplete ranged response: {path}")
        except (OSError, requests.RequestException, RuntimeError):
            if attempt + 1 == retries:
                raise
            resolved_url = _resolve_url(source_url)
            time.sleep(min(10.0, 1.0 + attempt * 0.5))
    raise AssertionError("Retry loop terminated unexpectedly")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=100)
    parser.add_argument("--request-chunk-mb", type=int, default=16)
    args = parser.parse_args()
    if (
        args.expected_size <= 0
        or args.workers <= 0
        or args.request_chunk_mb <= 0
    ):
        raise ValueError("Expected size, workers, and request chunk must be positive")
    with requests.head(args.url, allow_redirects=True, timeout=(20, 30)) as response:
        response.raise_for_status()
        resolved_url = response.url
        observed_size = int(response.headers.get("Content-Length", 0))
        if observed_size != args.expected_size:
            raise RuntimeError(
                f"Remote size mismatch: {observed_size} != {args.expected_size}"
            )
    chunk_size = math.ceil(args.expected_size / args.workers)
    ranges = []
    for index in range(args.workers):
        start = index * chunk_size
        if start >= args.expected_size:
            break
        end = min(args.expected_size - 1, start + chunk_size - 1)
        ranges.append((index, start, end))
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {
            executor.submit(
                _download_part,
                source_url=args.url,
                initial_url=resolved_url,
                path=args.parts_dir / f"part-{index:03d}.bin",
                start=start,
                end=end,
                retries=args.retries,
                request_chunk_bytes=args.request_chunk_mb * 1024 * 1024,
            ): index
            for index, start, end in ranges
        }
        for future in as_completed(futures):
            index = futures[future]
            _, size = future.result()
            print(f"part {index + 1}/{len(ranges)} complete: {size} bytes", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assembled = args.output.with_name(f"{args.output.name}.assembling")
    digest = hashlib.sha256()
    size = 0
    with assembled.open("wb") as target:
        for index, start, end in ranges:
            part = args.parts_dir / f"part-{index:03d}.bin"
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise RuntimeError(f"Part size mismatch: {part}")
            with part.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    observed_sha256 = digest.hexdigest()
    if size != args.expected_size:
        raise RuntimeError(f"Assembled size mismatch: {size} != {args.expected_size}")
    if observed_sha256 != args.expected_sha256.lower():
        raise RuntimeError(
            f"Assembled SHA-256 mismatch: {observed_sha256} != {args.expected_sha256}"
        )
    os.replace(assembled, args.output)
    print(f"verified {args.output}: {size} bytes; sha256={observed_sha256}")


if __name__ == "__main__":
    main()
