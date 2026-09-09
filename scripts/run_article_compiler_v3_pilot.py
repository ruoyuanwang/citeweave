from __future__ import annotations

import sys
from pathlib import Path

try:
    import run_article_compiler_v2_pilot as runner
except ModuleNotFoundError:
    import scripts.run_article_compiler_v2_pilot as runner

from citeweave.article_compiler_v3 import (
    build_compiler_repair_request,
    build_compiler_section_request,
)
from citeweave.io import read_json, sha256_file


def archive_failed_transport_attempts(plan_path: Path) -> int:
    archived = 0
    for article in read_json(plan_path)["articles"]:
        calls_dir = Path(article["output_dir"]) / "calls"
        for record_path in sorted(calls_dir.glob("*/execution_record.json")):
            record = read_json(record_path)
            if record.get("response_received") is not False:
                continue
            request_path = record_path.with_name("request.json")
            if sha256_file(request_path) != record.get("request_sha256"):
                raise RuntimeError(f"Failed-call request hash mismatch: {record_path}")
            attempt = 1
            while True:
                archive_path = record_path.with_name(
                    f"transport_failure_attempt_{attempt:03d}.json"
                )
                if not archive_path.exists():
                    break
                attempt += 1
            record_path.replace(archive_path)
            archived += 1
    return archived


def _consume_retry_flag() -> None:
    if "--retry-failed-call" not in sys.argv:
        return
    sys.argv.remove("--retry-failed-call")
    try:
        plan_path = Path(sys.argv[sys.argv.index("--plan") + 1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("--retry-failed-call requires --plan") from error
    archive_failed_transport_attempts(plan_path)


def main() -> None:
    _consume_retry_flag()
    runner.build_compiler_section_request = build_compiler_section_request
    runner.build_compiler_repair_request = build_compiler_repair_request
    runner.main()


if __name__ == "__main__":
    main()
