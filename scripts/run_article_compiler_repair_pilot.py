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

from citeweave.article_compiler import SECTION_ORDER, assemble_article, normalize_section_body
from citeweave.article_generation_diagnostic import (
    WORD_PATTERN,
    assess_generated_article_hierarchically,
)
from citeweave.article_section_repair import REPAIR_SECTIONS, build_section_repair_request
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


def _validate(args: argparse.Namespace) -> dict:
    if read_json(args.protocol_freeze).get("sha256") != sha256_file(args.protocol):
        raise RuntimeError("Compiler-repair protocol differs from freeze")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Compiler-repair implementation mismatch: {label}")
    plan = read_json(args.plan)
    if read_json(args.plan_freeze).get("sha256") != sha256_file(args.plan):
        raise RuntimeError("Compiler-repair plan differs from freeze")
    if plan.get("status") != "posthoc_development_section_repair_plan":
        raise RuntimeError("Compiler-repair plan status is invalid")
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
    plan = _validate(args)
    if not args.execute:
        print(json.dumps({"status": "ready", "repair_calls": 4}, indent=2))
        return
    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    audit = {
        "schema_version": 1,
        "status": "executing",
        "repair_calls_executed": 0,
        "repair_calls_reused": 0,
        "transport_failures": 0,
        "repaired_articles_passed": 0,
    }
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        for row in plan["datasets"]:
            source_dir = Path(row["source_dir"])
            output_dir = Path(row["output_dir"])
            writer_input_path = Path(row["writer_input"])
            if sha256_file(source_dir / "draft.md") != row["source_draft_sha256"]:
                raise RuntimeError(f"Source draft mismatch: {row['dataset_id']}")
            if sha256_file(writer_input_path) != row["writer_input_sha256"]:
                raise RuntimeError(f"Writer input mismatch: {row['dataset_id']}")
            writer_input = read_json(writer_input_path)
            compiled = {
                section: (source_dir / "sections" / section.casefold() / "section.md")
                .read_text(encoding="utf-8")
                .strip()
                for section in SECTION_ORDER
            }
            repaired: dict[str, str] = {}
            failed = False
            for section in REPAIR_SECTIONS:
                repair_dir = output_dir / "repairs" / section.casefold()
                body_path = repair_dir / "section.md"
                record_path = repair_dir / "execution_record.json"
                if record_path.is_file():
                    existing = read_json(record_path)
                    if existing.get("response_received") is True:
                        if existing.get("section_sha256") != sha256_file(body_path):
                            raise RuntimeError(
                                f"Repair hash mismatch: {row['dataset_id']}:{section}"
                            )
                        repaired[section] = body_path.read_text(encoding="utf-8").strip()
                        audit["repair_calls_reused"] += 1
                        continue
                request = build_section_repair_request(
                    writer_input,
                    section=section,
                    original_body=compiled[section],
                    repaired_results=repaired.get("Results"),
                )
                request_path = repair_dir / "request.json"
                write_json(request_path, request)
                request_hash = sha256_file(request_path)
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
                    failed = True
                    write_json(
                        record_path,
                        {
                            "schema_version": 1,
                            "dataset_id": row["dataset_id"],
                            "section": section,
                            "request_sha256": request_hash,
                            "response_received": False,
                            "error_type": type(error).__name__,
                            "error": str(error).replace(key, "***")[:1000],
                        },
                    )
                    break
                if key in json.dumps(raw, ensure_ascii=False):
                    raise RuntimeError("Provider response unexpectedly contains API key")
                raw_path = repair_dir / "raw_response.json"
                write_json(raw_path, raw)
                body = normalize_section_body(
                    str(raw["choices"][0]["message"]["content"]), section
                )
                body_path.parent.mkdir(parents=True, exist_ok=True)
                body_path.write_text(body + "\n", encoding="utf-8")
                repaired[section] = body
                write_json(
                    record_path,
                    {
                        "schema_version": 1,
                        "dataset_id": row["dataset_id"],
                        "section": section,
                        "response_received": True,
                        "generation_requests": 1,
                        "request_sha256": request_hash,
                        "started_at": started_at,
                        "completed_at": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "raw_response_sha256": sha256_file(raw_path),
                        "section_sha256": sha256_file(body_path),
                        "section_word_count": len(WORD_PATTERN.findall(body)),
                        "finish_reason": raw["choices"][0].get("finish_reason"),
                        "usage": raw.get("usage"),
                        "provider_response_id_sha256": hashlib.sha256(
                            str(raw.get("id", "")).encode()
                        ).hexdigest(),
                    },
                )
                audit["repair_calls_executed"] += 1
            if failed:
                continue
            compiled.update(repaired)
            article = assemble_article(compiled)
            draft_path = output_dir / "draft.md"
            draft_path.parent.mkdir(parents=True, exist_ok=True)
            draft_path.write_text(article, encoding="utf-8")
            assessment = assess_generated_article_hierarchically(
                article, writer_input=writer_input
            )
            assessment_path = output_dir / "draft_assessment.json"
            write_json(assessment_path, assessment)
            passed = bool(assessment["passed_all_recomputed_gates"])
            audit["repaired_articles_passed"] += int(passed)
            write_json(
                output_dir / "execution_record.json",
                {
                    "schema_version": 1,
                    "dataset_id": row["dataset_id"],
                    "status": "repaired_compiler_quality_gate_passed"
                    if passed
                    else "repaired_compiler_quality_gate_failed",
                    "development_only": True,
                    "source_draft_sha256": row["source_draft_sha256"],
                    "draft_sha256": sha256_file(draft_path),
                    "draft_assessment_sha256": sha256_file(assessment_path),
                },
            )
    audit["status"] = (
        "posthoc_section_repair_development_complete"
        if audit["transport_failures"] == 0
        else "transport_failures_retry_identical_requests_only"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
