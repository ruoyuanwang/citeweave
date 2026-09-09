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

from citeweave.article_compiler import (
    GENERATION_ORDER,
    assemble_article,
    build_section_request,
    normalize_section_body,
)
from citeweave.article_generation_diagnostic import (
    WORD_PATTERN,
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


def _validate(args: argparse.Namespace) -> dict:
    if read_json(args.protocol_freeze).get("sha256") != sha256_file(args.protocol):
        raise RuntimeError("Article-compiler protocol differs from freeze")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Article-compiler implementation mismatch: {label}")
    plan = read_json(args.plan)
    if read_json(args.plan_freeze).get("sha256") != sha256_file(args.plan):
        raise RuntimeError("Article-compiler plan differs from freeze")
    if plan.get("status") != "post_result_article_compiler_development_plan":
        raise RuntimeError("Article-compiler plan is not a development plan")
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
        print(
            json.dumps(
                {
                    "status": "ready",
                    "datasets": len(plan["datasets"]),
                    "maximum_provider_calls": plan["maximum_provider_calls"],
                },
                indent=2,
            )
        )
        return
    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    summary = {
        "schema_version": 1,
        "status": "executing",
        "datasets": len(plan["datasets"]),
        "sections_executed": 0,
        "sections_reused": 0,
        "transport_failures": 0,
        "articles_passed": 0,
    }
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        for row in plan["datasets"]:
            output_dir = Path(row["output_dir"])
            writer_input_path = Path(row["writer_input"])
            if sha256_file(writer_input_path) != row["writer_input_sha256"]:
                raise RuntimeError(f"Writer-input mismatch: {row['dataset_id']}")
            writer_input = read_json(writer_input_path)
            compiled: dict[str, str] = {}
            dataset_failed = False
            for section in GENERATION_ORDER:
                section_dir = output_dir / "sections" / section.casefold()
                body_path = section_dir / "section.md"
                record_path = section_dir / "execution_record.json"
                if record_path.is_file():
                    existing = read_json(record_path)
                    if existing.get("response_received") is True:
                        if existing.get("section_sha256") != sha256_file(body_path):
                            raise RuntimeError(
                                f"Existing section hash mismatch: {row['dataset_id']}:{section}"
                            )
                        compiled[section] = body_path.read_text(encoding="utf-8").strip()
                        summary["sections_reused"] += 1
                        continue
                request = build_section_request(
                    writer_input, section=section, compiled_sections=compiled
                )
                request_path = section_dir / "request.json"
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
                    summary["transport_failures"] += 1
                    dataset_failed = True
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
                raw_path = section_dir / "raw_response.json"
                write_json(raw_path, raw)
                body = normalize_section_body(
                    str(raw["choices"][0]["message"]["content"]), section
                )
                body_path.parent.mkdir(parents=True, exist_ok=True)
                body_path.write_text(body + "\n", encoding="utf-8")
                compiled[section] = body
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
                summary["sections_executed"] += 1
            if dataset_failed:
                continue
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
            summary["articles_passed"] += int(passed)
            write_json(
                output_dir / "execution_record.json",
                {
                    "schema_version": 1,
                    "dataset_id": row["dataset_id"],
                    "status": "compiler_quality_gate_passed"
                    if passed
                    else "compiler_quality_gate_failed",
                    "confirmatory_replacement": False,
                    "writer_input_sha256": row["writer_input_sha256"],
                    "section_calls": len(GENERATION_ORDER),
                    "draft_sha256": sha256_file(draft_path),
                    "draft_assessment_sha256": sha256_file(assessment_path),
                },
            )
    summary["status"] = (
        "article_compiler_development_complete"
        if summary["transport_failures"] == 0
        else "transport_failures_retry_identical_section_requests_only"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
