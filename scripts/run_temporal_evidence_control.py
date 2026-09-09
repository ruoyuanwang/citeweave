"""Explicit executor for the separate 96-cell temporal-information control.

No execution until all requests and this executor are frozen, the original panels
are promoted and analyzed, and scheduled machine drafts have finished using the
same provider. Completed responses are never resampled for quality.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from citeweave.formal_request import canonical_sha256, task_payload_sha256
from citeweave.graph_discovery import score_discovery_response
from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
OUTPUT = BASE / "temporal_shared_evidence_control_v1"
INPUT_FREEZE = OUTPUT / "execution_inputs_freeze.json"
EXECUTOR_FREEZE = OUTPUT / "executor_freeze.json"
RESULTS = OUTPUT / "results.json"


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify(bindings):
    for name, digest in bindings.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen control artifact drift: {name}")


def parse_response(raw):
    content = raw["choices"][0]["message"]["content"]
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(content)
    if not isinstance(result, dict):
        raise TypeError("Response must be a JSON object")
    return result


def score_control_response(task, parsed):
    evidence = parsed.get("evidence_ids") or []
    malformed = not isinstance(evidence, list) or any(
        not isinstance(item, str) for item in evidence
    )
    score = score_discovery_response(
        task, {**parsed, "evidence_ids": [] if malformed else evidence}
    )
    score["malformed_evidence_ids"] = malformed
    if parsed.get("item_id") != task["item_id"] or parsed.get("abstain"):
        score["answer_exact"] = False
    return score


def terminal(record):
    return record.get("status") == "complete" or (
        record.get("status") == "failed_parse" and int(record.get("attempt", 0)) >= 3
    )


def request_identity(cell, protocol):
    return canonical_sha256(
        {
            "model": protocol["model"],
            "temperature": protocol["temperature"],
            "max_tokens": protocol["max_tokens"],
            "thinking": {"type": "disabled"},
            "messages": cell["messages"],
            "stream": False,
        }
    )


def validate_saved(saved, cells, protocol):
    expected = {(r["item_id"], r["condition"]): request_identity(r, protocol) for r in cells}
    seen = set()
    for record in saved["records"]:
        key = (record["item_id"], record["condition"])
        if key in seen or record["request_sha256"] != expected.get(key):
            raise ValueError("Duplicate, unexpected, or drifted saved control result")
        seen.add(key)
    for attempt in saved.get("attempt_log", []):
        key = (attempt["item_id"], attempt["condition"])
        if attempt["request_sha256"] != expected.get(key):
            raise ValueError("Saved control attempt identity drift")


def freeze_executor():
    if EXECUTOR_FREEZE.exists():
        verify(read_json(EXECUTOR_FREEZE)["bindings"])
        return
    if not INPUT_FREEZE.is_file():
        raise SystemExit("All 96 exact requests must be materialized first")
    if RESULTS.exists():
        raise ValueError("Cannot freeze executor after control outcomes")
    verify(read_json(INPUT_FREEZE)["bindings"])
    paths = [
        INPUT_FREEZE,
        Path(__file__).resolve(),
        ROOT / "src/citeweave/graph_discovery.py",
        ROOT / "src/citeweave/formal_request.py",
        ROOT / "src/citeweave/io.py",
    ]
    with EXECUTOR_FREEZE.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "status": "executor_frozen_before_control_responses",
                "maximum_parse_attempts": 3,
                "httpx_version": httpx.__version__,
                "automatic_api_resume": False,
                "bindings": {str(p): sha256_file(p) for p in paths},
            },
            stream,
            indent=2,
        )


def execute(api_key_file):
    frozen = read_json(EXECUTOR_FREEZE)
    verify(frozen["bindings"])
    verify(read_json(INPUT_FREEZE)["bindings"])
    if frozen["httpx_version"] != httpx.__version__:
        raise ValueError("HTTP client runtime drift")
    for name in (
        "formal_v3_confirmatory_pipeline_status.json",
        "machine_article_generation_status.json",
    ):
        if read_json(ROOT / "experiments/runs" / name)["status"] != "complete":
            raise ValueError(
                "Original formal panels and scheduled machine drafts must finish before control execution"
            )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_identity_corrected_execution.py"),
            "--verify-promoted",
        ],
        cwd=ROOT,
        check=True,
    )
    protocol = read_json(OUTPUT / "protocol.json")
    cells = read_json(OUTPUT / "requests.json")["cells"]
    tasks = {
        t["item_id"]: t
        for rows in read_json(OUTPUT / "benchmarks.json")["datasets"].values()
        for t in rows
    }
    if len(cells) != 96 or len(tasks) != 24:
        raise ValueError("Control must retain 24 tasks and 96 cells")
    for cell in cells:
        if (
            canonical_sha256(cell["messages"]) != cell["messages_sha256"]
            or task_payload_sha256(tasks[cell["item_id"]]) != cell["task_payload_sha256"]
        ):
            raise ValueError("Frozen request/task mismatch")
    saved = (
        read_json(RESULTS)
        if RESULTS.exists()
        else {
            "schema_version": 1,
            "executor_freeze_sha256": sha256_file(EXECUTOR_FREEZE),
            "records": [],
            "attempt_log": [],
            "status": "running",
        }
    )
    if saved["executor_freeze_sha256"] != sha256_file(EXECUTOR_FREEZE):
        raise ValueError("Saved executor identity mismatch")
    validate_saved(saved, cells, protocol)
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        for line in api_key_file.read_text(encoding="utf-8").splitlines():
            label, separator, value = line.partition("=")
            if separator and label.strip().casefold() == "deepseek":
                key = value.strip()
                break
    if not key:
        raise ValueError("Configured DeepSeek key unavailable")
    with httpx.Client(timeout=240, follow_redirects=False) as client:
        for cell in cells:
            cell_key = (cell["item_id"], cell["condition"])
            previous = next(
                (r for r in saved["records"] if (r["item_id"], r["condition"]) == cell_key), None
            )
            if previous is not None and terminal(previous):
                continue
            first_attempt = int(previous["attempt"]) + 1 if previous else 1
            for attempt in range(first_attempt, 4):
                request = {
                    "model": protocol["model"],
                    "temperature": protocol["temperature"],
                    "max_tokens": protocol["max_tokens"],
                    "thinking": {"type": "disabled"},
                    "messages": cell["messages"],
                    "stream": False,
                }
                started = time.perf_counter()
                response = client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=request,
                )
                response.raise_for_status()
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"unparseable_provider_body": response.text}
                if not isinstance(raw, dict):
                    raw = {"unexpected_provider_payload": raw}
                # Do not persist provider headers or any accidental credential echo.
                raw = json.loads(json.dumps(raw, ensure_ascii=False).replace(key, "[REDACTED]"))
                record = {
                    "dataset_id": cell["dataset_id"],
                    "item_id": cell["item_id"],
                    "source_item_id": cell["source_item_id"],
                    "scale": cell["scale"],
                    "condition": cell["condition"],
                    "attempt": attempt,
                    "request_sha256": request_identity(cell, protocol),
                    "elapsed_seconds": time.perf_counter() - started,
                    "usage": raw.get("usage"),
                }
                try:
                    parsed = parse_response(raw)
                except (KeyError, IndexError, TypeError, ValueError):
                    record.update(
                        {
                            "status": "failed_parse",
                            "raw_response": raw,
                            "score": {"answer_exact": False},
                        }
                    )
                else:
                    score = score_control_response(tasks[cell["item_id"]], parsed)
                    record.update({"status": "complete", "response": parsed, "score": score})
                if previous is not None:
                    saved["attempt_log"].append(previous)
                    saved["records"].remove(previous)
                saved["records"].append(record)
                saved["terminal_cells"] = sum(terminal(r) for r in saved["records"])
                saved["status"] = "complete" if saved["terminal_cells"] == 96 else "running"
                validate_saved(saved, cells, protocol)
                atomic_json(RESULTS, saved)
                print(
                    json.dumps(
                        {
                            "dataset_id": cell["dataset_id"],
                            "condition": cell["condition"],
                            "attempt": attempt,
                            "terminal_cells": saved["terminal_cells"],
                        }
                    ),
                    flush=True,
                )
                previous = record
                if terminal(record):
                    break


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-executor", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    args = parser.parse_args()
    if args.freeze_executor:
        freeze_executor()
        return
    lock = OUTPUT / "executor.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("Control executor lock exists; verify ownership before recovery") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        execute(args.api_key_file)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
