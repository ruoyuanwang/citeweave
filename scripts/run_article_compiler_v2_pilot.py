from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from citeweave.article_budget_optimizer import optimize_article_word_budget
from citeweave.article_compiler import assemble_article, normalize_section_body
from citeweave.article_compiler_v2 import (
    build_compiler_repair_request,
    build_compiler_section_request,
)
from citeweave.article_generation_diagnostic import (
    WORD_PATTERN,
    assess_generated_article_hierarchically,
)
from citeweave.article_oversight_readiness import audit_article_oversight_readiness
from citeweave.io import read_json, sha256_file, write_json

CALL_SEQUENCE = (
    ("Methods", "base"),
    ("Results", "base"),
    ("Results", "repair"),
    ("Discussion", "base"),
    ("Discussion", "repair"),
    ("Limitations", "base"),
    ("Conclusion", "base"),
    ("Introduction", "base"),
    ("Abstract", "base"),
)


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, candidate = line.partition(":")
        if separator and key.strip().casefold() == "deepseek" and candidate.strip():
            return candidate.strip()
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_hash = sha256_file(args.protocol)
    if read_json(args.protocol_freeze).get("sha256") != protocol_hash:
        raise RuntimeError("Matched article-compiler protocol differs from freeze")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Matched article-compiler implementation mismatch: {label}")
    plan = read_json(args.plan)
    if read_json(args.plan_freeze).get("sha256") != sha256_file(args.plan):
        raise RuntimeError("Matched article-compiler plan differs from freeze")
    if plan.get("status") != "matched_article_compiler_claim_ready_development_plan":
        raise RuntimeError("Matched article-compiler plan is not the development plan")
    if plan.get("provider_calls_per_article") != len(CALL_SEQUENCE):
        raise RuntimeError("Plan does not precommit exactly nine calls per article")
    return plan


def _request_for_call(
    *,
    writer_input: dict[str, Any],
    condition: str,
    section: str,
    pass_type: str,
    compiled: dict[str, str],
    base_bodies: dict[str, str],
) -> dict[str, Any]:
    if pass_type == "base":
        return build_compiler_section_request(
            writer_input,
            condition=condition,
            section=section,
            compiled_sections=compiled,
        )
    return build_compiler_repair_request(
        writer_input,
        condition=condition,
        section=section,
        original_body=base_bodies[section],
        compiled_sections=compiled,
    )


def _execute_or_reuse_call(
    *,
    client: httpx.Client,
    endpoint: str,
    key: str,
    call_dir: Path,
    article_id: str,
    section: str,
    pass_type: str,
    request: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    request_path = call_dir / "request.json"
    body_path = call_dir / "section.md"
    record_path = call_dir / "execution_record.json"
    request_hash = hashlib.sha256(
        json.dumps(request, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    ).hexdigest()
    if record_path.is_file():
        record = read_json(record_path)
        if not record.get("response_received"):
            raise RuntimeError(f"Incomplete prior call must be retried explicitly: {call_dir}")
        if record.get("request_sha256") != request_hash:
            raise RuntimeError(f"Reconstructed request mismatch: {call_dir}")
        if record.get("section_sha256") != sha256_file(body_path):
            raise RuntimeError(f"Existing section hash mismatch: {call_dir}")
        return body_path.read_text(encoding="utf-8").strip(), record, True
    write_json(request_path, request)
    if sha256_file(request_path) != request_hash:
        raise RuntimeError(f"Request serialization hash mismatch: {call_dir}")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    try:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=request,
        )
        response.raise_for_status()
        raw = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        write_json(
            record_path,
            {
                "schema_version": 1,
                "article_id": article_id,
                "section": section,
                "pass_type": pass_type,
                "request_sha256": request_hash,
                "response_received": False,
                "error_type": type(error).__name__,
                "error": str(error).replace(key, "***")[:1000],
            },
        )
        raise
    if key in json.dumps(raw, ensure_ascii=False):
        raise RuntimeError("Provider response unexpectedly contains API key")
    raw_path = call_dir / "raw_response.json"
    write_json(raw_path, raw)
    body = normalize_section_body(str(raw["choices"][0]["message"]["content"]), section)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body + "\n", encoding="utf-8")
    record = {
        "schema_version": 1,
        "article_id": article_id,
        "section": section,
        "pass_type": pass_type,
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
        "usage": raw.get("usage") or {},
        "provider_response_id_sha256": hashlib.sha256(
            str(raw.get("id", "")).encode()
        ).hexdigest(),
    }
    write_json(record_path, record)
    return body, record, False


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
                    "articles": len(plan["articles"]),
                    "provider_calls_per_article": len(CALL_SEQUENCE),
                    "maximum_provider_calls": plan["maximum_provider_calls"],
                },
                indent=2,
            )
        )
        return
    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "executing",
        "articles": len(plan["articles"]),
        "calls_executed": 0,
        "calls_reused": 0,
        "transport_failures": 0,
        "machine_gate_passed": 0,
        "oversight_ready": 0,
        "fully_passed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "records": [],
    }
    try:
        with httpx.Client(timeout=600, follow_redirects=True) as client:
            for row in plan["articles"]:
                writer_input_path = Path(row["writer_input"])
                if sha256_file(writer_input_path) != row["writer_input_sha256"]:
                    raise RuntimeError(f"Writer-input mismatch: {row['article_id']}")
                writer_input = read_json(writer_input_path)
                output_dir = Path(row["output_dir"])
                compiled: dict[str, str] = {}
                base_bodies: dict[str, str] = {}
                call_records = []
                for call_index, (section, pass_type) in enumerate(CALL_SEQUENCE, start=1):
                    request = _request_for_call(
                        writer_input=writer_input,
                        condition=row["condition"],
                        section=section,
                        pass_type=pass_type,
                        compiled=compiled,
                        base_bodies=base_bodies,
                    )
                    call_dir = output_dir / "calls" / (
                        f"{call_index:02d}_{section.casefold()}_{pass_type}"
                    )
                    body, record, reused = _execute_or_reuse_call(
                        client=client,
                        endpoint=endpoint,
                        key=key,
                        call_dir=call_dir,
                        article_id=row["article_id"],
                        section=section,
                        pass_type=pass_type,
                        request=request,
                    )
                    summary["calls_reused" if reused else "calls_executed"] += 1
                    usage = record.get("usage") or {}
                    if not reused:
                        summary["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                        summary["completion_tokens"] += int(
                            usage.get("completion_tokens") or 0
                        )
                    call_records.append(record)
                    if pass_type == "base":
                        base_bodies[section] = body
                    if pass_type == "repair" or section not in {"Results", "Discussion"}:
                        compiled[section] = body
                raw_article = assemble_article(compiled)
                raw_path = output_dir / "assembled_before_budget.md"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw_article, encoding="utf-8")
                optimization_error = None
                try:
                    final_article, optimization = optimize_article_word_budget(raw_article)
                except ValueError as error:
                    final_article = raw_article
                    optimization_error = str(error)
                    optimization = {
                        "schema_version": 1,
                        "status": "budget_optimization_failed",
                        "error": optimization_error,
                        "original_word_count": len(WORD_PATTERN.findall(raw_article)),
                    }
                final_path = output_dir / "draft.md"
                final_path.write_text(final_article, encoding="utf-8")
                optimization_path = output_dir / "budget_optimization.json"
                write_json(optimization_path, optimization)
                assessment = assess_generated_article_hierarchically(
                    final_article, writer_input=writer_input
                )
                assessment_path = output_dir / "draft_assessment.json"
                write_json(assessment_path, assessment)
                phenomenon_ids = sorted(
                    item["phenomenon_id"] for item in writer_input["graph_phenomena"]
                )
                oversight = audit_article_oversight_readiness(
                    final_article, phenomenon_ids=phenomenon_ids
                )
                oversight_path = output_dir / "oversight_readiness.json"
                write_json(oversight_path, oversight)
                all_stop = all(record.get("finish_reason") == "stop" for record in call_records)
                machine_passed = bool(assessment["passed_all_recomputed_gates"])
                oversight_ready = oversight["status"] == "ready_for_claim_review_packetization"
                fully_passed = (
                    len(call_records) == len(CALL_SEQUENCE)
                    and all_stop
                    and optimization_error is None
                    and machine_passed
                    and oversight_ready
                )
                totals = {
                    "prompt_tokens": sum(
                        int((record.get("usage") or {}).get("prompt_tokens") or 0)
                        for record in call_records
                    ),
                    "completion_tokens": sum(
                        int((record.get("usage") or {}).get("completion_tokens") or 0)
                        for record in call_records
                    ),
                }
                article_record = {
                    "schema_version": 1,
                    "article_id": row["article_id"],
                    "dataset_id": row["dataset_id"],
                    "condition": row["condition"],
                    "status": "fully_qualified_for_confirmatory_planning"
                    if fully_passed
                    else "development_gate_failed",
                    "confirmatory_replacement": False,
                    "writer_input_sha256": row["writer_input_sha256"],
                    "provider_calls": len(call_records),
                    "all_finish_reason_stop": all_stop,
                    "usage": totals,
                    "assembled_before_budget_sha256": sha256_file(raw_path),
                    "draft_sha256": sha256_file(final_path),
                    "budget_optimization_sha256": sha256_file(optimization_path),
                    "draft_assessment_sha256": sha256_file(assessment_path),
                    "oversight_readiness_sha256": sha256_file(oversight_path),
                    "word_count": assessment["word_count"],
                    "machine_gate_passed": machine_passed,
                    "oversight_ready": oversight_ready,
                    "reviewable_claim_candidates": oversight["candidate_claims"],
                    "reviewable_claims_by_section": oversight[
                        "candidate_claims_by_section"
                    ],
                    "fully_passed": fully_passed,
                }
                write_json(output_dir / "execution_record.json", article_record)
                summary["machine_gate_passed"] += int(machine_passed)
                summary["oversight_ready"] += int(oversight_ready)
                summary["fully_passed"] += int(fully_passed)
                summary["records"].append(article_record)
    except (httpx.HTTPError, json.JSONDecodeError):
        summary["transport_failures"] += 1
        summary["status"] = "transport_failure_retry_identical_call_only"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise
    summary["status"] = "matched_article_compiler_development_complete"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
