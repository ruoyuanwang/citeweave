from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from citeweave.article_generation_diagnostic import (
    assess_generated_article_hierarchically,
)
from citeweave.io import read_json, sha256_file, write_json


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, candidate = line.partition(":")
        if separator and key.strip().casefold() == "deepseek" and candidate.strip():
            return candidate.strip()
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def _validate(
    plan_path: Path,
    plan_freeze_path: Path,
    protocol_path: Path,
    protocol_freeze_path: Path,
) -> dict:
    plan = read_json(plan_path)
    plan_freeze = read_json(plan_freeze_path)
    if plan_freeze.get("sha256") != sha256_file(plan_path):
        raise RuntimeError("Article-v3 pilot plan differs from freeze")
    if read_json(protocol_freeze_path).get("sha256") != sha256_file(protocol_path):
        raise RuntimeError("Article-v3 pilot protocol differs from freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Article-v3 pilot implementation mismatch: {label}")
    if plan.get("status") != "post_result_development_pilot_plan":
        raise RuntimeError("Article-v3 plan is not a development pilot")
    if len(plan.get("cells") or []) != 4:
        raise RuntimeError("Article-v3 development pilot requires exactly four cells")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path("apikey.md"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = _validate(
        args.plan, args.plan_freeze, args.protocol, args.protocol_freeze
    )
    if not args.execute:
        print(json.dumps({"status": "ready", "cells": len(plan["cells"])}, indent=2))
        return
    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    audit = {
        "schema_version": 1,
        "status": "executing",
        "cells": len(plan["cells"]),
        "executed": 0,
        "already_terminal": 0,
        "transport_failures": 0,
        "quality_gate_passes": 0,
    }
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        for cell in plan["cells"]:
            output_dir = Path(cell["output_dir"])
            record_path = output_dir / "execution_record.json"
            if record_path.is_file():
                existing = read_json(record_path)
                if existing.get("request_sha256") != cell["request_sha256"]:
                    raise RuntimeError(f"Existing request mismatch: {cell['cell_id']}")
                if existing.get("response_received") is True:
                    audit["already_terminal"] += 1
                    continue
            request_path = Path(cell["request"])
            writer_input_path = Path(cell["writer_input"])
            if sha256_file(request_path) != cell["request_sha256"]:
                raise RuntimeError(f"Request hash mismatch: {cell['cell_id']}")
            if sha256_file(writer_input_path) != cell["writer_input_sha256"]:
                raise RuntimeError(f"Writer-input hash mismatch: {cell['cell_id']}")
            request = read_json(request_path)
            started_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                )
                response.raise_for_status()
                raw = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                audit["transport_failures"] += 1
                write_json(
                    record_path,
                    {
                        "schema_version": 1,
                        "cell_id": cell["cell_id"],
                        "request_sha256": cell["request_sha256"],
                        "response_received": False,
                        "error_type": type(error).__name__,
                        "error": str(error).replace(key, "***")[:1000],
                    },
                )
                continue
            if key in json.dumps(raw, ensure_ascii=False):
                raise RuntimeError("Provider response unexpectedly contains the API key")
            raw_path = output_dir / "raw_response.json"
            write_json(raw_path, raw)
            article = str(raw["choices"][0]["message"]["content"]).strip() + "\n"
            draft_path = output_dir / "draft.md"
            draft_path.write_text(article, encoding="utf-8")
            assessment = assess_generated_article_hierarchically(
                article, writer_input=read_json(writer_input_path)
            )
            assessment_path = output_dir / "draft_assessment.json"
            write_json(assessment_path, assessment)
            passed = bool(assessment["passed_all_recomputed_gates"])
            audit["quality_gate_passes"] += int(passed)
            write_json(
                record_path,
                {
                    "schema_version": 1,
                    "cell_id": cell["cell_id"],
                    "dataset_id": cell["dataset_id"],
                    "condition": cell["condition"],
                    "status": "development_quality_gate_passed"
                    if passed
                    else "development_quality_gate_failed",
                    "confirmatory_replacement": False,
                    "request_sha256": cell["request_sha256"],
                    "writer_input_sha256": cell["writer_input_sha256"],
                    "response_received": True,
                    "generation_requests": 1,
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "raw_response_sha256": sha256_file(raw_path),
                    "draft_sha256": sha256_file(draft_path),
                    "draft_assessment_sha256": sha256_file(assessment_path),
                    "model": raw.get("model", request["model"]),
                    "finish_reason": raw["choices"][0].get("finish_reason"),
                    "usage": raw.get("usage"),
                    "provider_response_id_sha256": hashlib.sha256(
                        str(raw.get("id", "")).encode()
                    ).hexdigest(),
                },
            )
            audit["executed"] += 1
    audit["status"] = (
        "development_execution_complete"
        if audit["transport_failures"] == 0
        else "transport_failures_retry_same_requests_only"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
