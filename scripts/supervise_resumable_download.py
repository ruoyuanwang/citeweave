from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _part_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.glob("part-*.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-chunk-mb", type=int, default=4)
    parser.add_argument("--stall-seconds", type=float, default=60)
    parser.add_argument("--max-restarts", type=int, default=100)
    args = parser.parse_args()
    downloader = Path(__file__).resolve().with_name(
        "download_large_file_resumable.py"
    )
    command = [
        sys.executable,
        str(downloader),
        "--url",
        args.url,
        "--output",
        str(args.output),
        "--expected-size",
        str(args.expected_size),
        "--expected-sha256",
        args.expected_sha256,
        "--parts-dir",
        str(args.parts_dir),
        "--workers",
        str(args.workers),
        "--retries",
        "100",
        "--request-chunk-mb",
        str(args.request_chunk_mb),
    ]
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    for restart in range(args.max_restarts + 1):
        if args.output.is_file() and args.output.stat().st_size == args.expected_size:
            print(f"verified-size output already present: {args.output}")
            return
        before = _part_bytes(args.parts_dir)
        last_progress = time.monotonic()
        process = subprocess.Popen(command)
        while process.poll() is None:
            time.sleep(5)
            current = _part_bytes(args.parts_dir)
            if current > before:
                before = current
                last_progress = time.monotonic()
                print(
                    f"downloaded={current}/{args.expected_size} "
                    f"({100 * current / args.expected_size:.1f}%)",
                    flush=True,
                )
            if time.monotonic() - last_progress >= args.stall_seconds:
                print(f"stall detected; restarting worker pass {restart + 1}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
        if process.returncode == 0:
            if args.output.is_file() and args.output.stat().st_size == args.expected_size:
                return
            raise RuntimeError("Downloader exited zero without verified output")
    raise RuntimeError("Maximum download restart count exhausted")


if __name__ == "__main__":
    main()
