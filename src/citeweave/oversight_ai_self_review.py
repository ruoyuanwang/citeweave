from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from .io import read_json, sha256_file, write_json
from .review_ui import ReviewStore
from .token_budget import CommandTokenizer

PACKET_ARMS = {
    "standard_claim_first": "standard_public_path",
    "adversarial_two_sided": "adversarial_public_path",
}
REQUIRED_RESPONSE_FIELDS = {
    "verdict",
    "decisive_evidence_ids",
    "invalid_operator_steps",
    "failure_mode",
    "minimal_rewrite",
    "rationale",
}
SYSTEM_PROMPT = """You are an independent scientific claim auditor. Evaluate the candidate response only against the supplied evidence and operator trace. Do not assume the candidate is correct, do not infer hidden gold information, and do not use outside knowledge.

Verdicts:
- supported: every substantive answer field is supported and no material qualification is missing.
- qualify: the core answer is defensible but needs the smallest supplied-evidence-bounded correction or limitation.
- reject: a core answer field conflicts with the supplied evidence or cannot be repaired by a local qualification.
- abstain: the supplied material is insufficient to decide.

Return one JSON object and no surrounding prose. It must contain exactly: verdict, decisive_evidence_ids, invalid_operator_steps, failure_mode, minimal_rewrite, rationale. Cite only evidence IDs visible in the packet. Use null for minimal_rewrite unless the verdict is qualify; qualify requires a nonempty minimal rewrite. Keep the rationale concise."""


class OversightAISelfReviewError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_ai_review_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    public = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"AUDIT_PACKET_JSON:\n{public}"},
    ]


def build_ai_review_request(
    packet: dict[str, Any],
    *,
    model: str = "deepseek-v4-pro",
    max_tokens: int = 1200,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_ai_review_messages(packet),
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def _load_public_packet(
    root: Path, record: dict[str, Any], packet_arm: str
) -> tuple[Path, dict[str, Any]]:
    path_field = PACKET_ARMS.get(packet_arm)
    if path_field is None:
        raise OversightAISelfReviewError(f"Unknown packet arm: {packet_arm}")
    hash_field = path_field.replace("path", "sha256")
    path = root / str(record[path_field])
    if not path.is_file() or sha256_file(path) != record[hash_field]:
        raise OversightAISelfReviewError(f"Public packet identity mismatch: {path}")
    packet = read_json(path)
    serialized = json.dumps(packet, ensure_ascii=False).casefold()
    if any(value in serialized for value in ("gold_verdict", "selected_condition")):
        raise OversightAISelfReviewError("Reviewer-facing packet leaks a hidden field")
    return path, packet


def build_ai_self_review_plan(
    *,
    packet_manifest_path: Path,
    tokenizer_manifest_path: Path,
    output_path: Path,
    tokenizer: CommandTokenizer | Any | None = None,
    model: str = "deepseek-v4-pro",
    max_prompt_tokens: int = 32768,
    max_completion_tokens: int = 1200,
) -> dict[str, Any]:
    if output_path.exists():
        raise OversightAISelfReviewError("Refusing to overwrite AI self-review plan")
    manifest = read_json(packet_manifest_path)
    records = manifest.get("records") or []
    if (
        manifest.get("status") != "prospective_factorial_packets_frozen_before_review"
        or manifest.get("study_role") != "fixed_outcome_balanced_real_candidate_benchmark"
        or manifest.get("human_outcomes_inspected") is not False
        or len(records) != 96
        or len({row.get("case_id") for row in records}) != 96
    ):
        raise OversightAISelfReviewError("AI baseline requires the balanced 96-case panel")
    tokenizer_manifest = read_json(tokenizer_manifest_path)
    if tokenizer_manifest.get("model") != model or tokenizer_manifest.get("passed") is not True:
        raise OversightAISelfReviewError("Tokenizer/model identity mismatch")
    counter = tokenizer or CommandTokenizer(
        tokenizer_manifest, manifest_dir=tokenizer_manifest_path.parent
    )
    if tokenizer is None and not counter.verify(
        tokenizer_manifest.get("verification_probes") or []
    )["passed"]:
        raise OversightAISelfReviewError("Frozen tokenizer probe failed")

    root = packet_manifest_path.resolve().parent
    cells = []
    for record in sorted(records, key=lambda row: str(row["case_id"])):
        case_id = str(record["case_id"])
        for packet_arm in PACKET_ARMS:
            packet_path, packet = _load_public_packet(root, record, packet_arm)
            request = build_ai_review_request(
                packet, model=model, max_tokens=max_completion_tokens
            )
            prompt_tokens = int(counter.count_messages(request["messages"]))
            if prompt_tokens > max_prompt_tokens:
                raise OversightAISelfReviewError(
                    f"Prompt exceeds {max_prompt_tokens} tokens: {case_id}/{packet_arm}"
                )
            cells.append(
                {
                    "cell_id": f"{case_id}__{packet_arm}",
                    "case_id": case_id,
                    "dataset_id": record["dataset_id"],
                    "packet_arm": packet_arm,
                    "packet_path": str(packet_path),
                    "packet_sha256": sha256_file(packet_path),
                    "messages_sha256": _canonical_hash(request["messages"]),
                    "request_sha256": _canonical_hash(request),
                    "prompt_tokens": prompt_tokens,
                }
            )
    if len(cells) != 192 or len({row["cell_id"] for row in cells}) != 192:
        raise OversightAISelfReviewError("AI self-review plan must contain 192 unique cells")
    result = {
        "schema_version": 1,
        "status": "ai_self_review_plan_frozen_before_provider_or_human_outcomes",
        "human_outcomes_inspected": False,
        "ai_review_outcomes_present": False,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "tokenizer_manifest_path": str(tokenizer_manifest_path.resolve()),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
        "model": model,
        "temperature": 0,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_tokens": max_completion_tokens,
        "cases": 96,
        "packet_arms": list(PACKET_ARMS),
        "planned_calls": 192,
        "prompt_token_summary": {
            "total": sum(row["prompt_tokens"] for row in cells),
            "minimum": min(row["prompt_tokens"] for row in cells),
            "maximum": max(row["prompt_tokens"] for row in cells),
        },
        "cells": cells,
    }
    write_json(output_path, result)
    return result


def freeze_ai_self_review_plan(plan_path: Path, freeze_path: Path) -> dict[str, Any]:
    if freeze_path.exists():
        raise OversightAISelfReviewError("Refusing to overwrite AI self-review freeze")
    plan = read_json(plan_path)
    if (
        plan.get("status")
        != "ai_self_review_plan_frozen_before_provider_or_human_outcomes"
        or plan.get("planned_calls") != 192
        or plan.get("human_outcomes_inspected") is not False
        or plan.get("ai_review_outcomes_present") is not False
    ):
        raise OversightAISelfReviewError("AI self-review plan is not freezable")
    result = {
        "schema_version": 1,
        "artifact": str(plan_path.resolve()),
        "sha256": sha256_file(plan_path),
        "status": plan["status"],
        "human_outcomes_inspected": False,
        "ai_review_outcomes_present": False,
        "planned_calls": 192,
    }
    write_json(freeze_path, result)
    return result


def parse_ai_review_content(content: str, packet: dict[str, Any]) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OversightAISelfReviewError("AI self-review response is not JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != REQUIRED_RESPONSE_FIELDS:
        raise OversightAISelfReviewError("AI self-review response fields differ from schema")
    try:
        ReviewStore._validate_answers("adversarial", parsed, packet)
    except (TypeError, ValueError) as exc:
        raise OversightAISelfReviewError(f"Invalid AI self-review response: {exc}") from exc
    return parsed


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, found, candidate = line.partition(":")
            if found and key.strip().casefold() == "deepseek" and candidate.strip():
                return candidate.strip()
    raise OversightAISelfReviewError("No DeepSeek API key is available")


def _balance(base_url: str, key: str) -> dict[str, Any]:
    response = httpx.get(
        base_url.rstrip("/") + "/user/balance",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "is_available": payload.get("is_available") is True,
        "balances": [
            {
                field: row.get(field)
                for field in (
                    "currency",
                    "total_balance",
                    "granted_balance",
                    "topped_up_balance",
                )
            }
            for row in payload.get("balance_infos") or []
        ],
    }


def _archive_transport_failure(path: Path) -> None:
    record = read_json(path)
    if record.get("status") != "transport_failure":
        raise OversightAISelfReviewError("Only transport failures may be retried")
    attempt = 1
    while path.with_name(f"transport_failure_attempt_{attempt:03d}.json").exists():
        attempt += 1
    path.replace(path.with_name(f"transport_failure_attempt_{attempt:03d}.json"))


def run_ai_self_review_plan(
    *,
    plan_path: Path,
    plan_freeze_path: Path,
    packet_manifest_path: Path,
    output_root: Path,
    api_key_file: Path,
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com",
    retry_failed_call: bool = False,
) -> dict[str, Any]:
    plan = read_json(plan_path)
    freeze = read_json(plan_freeze_path)
    if freeze.get("sha256") != sha256_file(plan_path):
        raise OversightAISelfReviewError("AI self-review plan differs from its freeze")
    if plan.get("packet_manifest_sha256") != sha256_file(packet_manifest_path):
        raise OversightAISelfReviewError("AI self-review packet manifest changed")
    if plan.get("planned_calls") != 192:
        raise OversightAISelfReviewError("AI self-review plan is incomplete")
    key = _api_key(api_key_file, api_key_env)
    balance = _balance(base_url, key)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "provider_balance_preflight.json", balance)
    if not balance["is_available"]:
        raise OversightAISelfReviewError("Provider balance reports unavailable")

    complete = parse_failures = transport_failures = 0
    for cell in plan["cells"]:
        cell_root = output_root / cell["packet_arm"] / cell["case_id"]
        record_path = cell_root / "execution_record.json"
        if record_path.exists():
            existing = read_json(record_path)
            if existing.get("request_sha256") != cell["request_sha256"]:
                raise OversightAISelfReviewError("Existing request identity mismatch")
            if existing.get("status") in {"complete", "terminal_parse_failure"}:
                complete += existing.get("status") == "complete"
                parse_failures += existing.get("status") == "terminal_parse_failure"
                continue
            if retry_failed_call:
                _archive_transport_failure(record_path)
            else:
                raise OversightAISelfReviewError(
                    "Transport failure exists; pass retry_failed_call for identical retry"
                )
        packet_path = Path(cell["packet_path"])
        if sha256_file(packet_path) != cell["packet_sha256"]:
            raise OversightAISelfReviewError("Planned packet changed")
        packet = read_json(packet_path)
        request = build_ai_review_request(
            packet,
            model=plan["model"],
            max_tokens=int(plan["max_completion_tokens"]),
        )
        if (
            _canonical_hash(request["messages"]) != cell["messages_sha256"]
            or _canonical_hash(request) != cell["request_sha256"]
        ):
            raise OversightAISelfReviewError("Reconstructed request identity mismatch")
        cell_root.mkdir(parents=True, exist_ok=True)
        write_json(cell_root / "request.json", request)
        try:
            response = httpx.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=request,
                timeout=180,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            write_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "transport_failure",
                    "response_received": False,
                    "request_sha256": cell["request_sha256"],
                    "error_type": type(exc).__name__,
                },
            )
            transport_failures += 1
            break
        write_json(cell_root / "provider_response.json", payload)
        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = parse_ai_review_content(content, packet)
        except (KeyError, IndexError, TypeError, OversightAISelfReviewError) as exc:
            write_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "terminal_parse_failure",
                    "response_received": True,
                    "request_sha256": cell["request_sha256"],
                    "provider_response_sha256": sha256_file(
                        cell_root / "provider_response.json"
                    ),
                    "error_type": type(exc).__name__,
                },
            )
            parse_failures += 1
            continue
        write_json(cell_root / "parsed_response.json", parsed)
        write_json(
            record_path,
            {
                "schema_version": 1,
                "status": "complete",
                "response_received": True,
                "request_sha256": cell["request_sha256"],
                "provider_response_sha256": sha256_file(
                    cell_root / "provider_response.json"
                ),
                "parsed_response_sha256": sha256_file(
                    cell_root / "parsed_response.json"
                ),
                "usage": payload.get("usage") or {},
            },
        )
        complete += 1
    summary = {
        "schema_version": 1,
        "status": (
            "terminal"
            if complete + parse_failures == 192 and transport_failures == 0
            else "incomplete"
        ),
        "plan_sha256": sha256_file(plan_path),
        "complete_cells": complete,
        "terminal_parse_failure_cells": parse_failures,
        "transport_failure_cells": transport_failures,
        "expected_cells": 192,
    }
    write_json(output_root / "execution_summary.json", summary)
    return summary
