from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import httpx

from citeweave.citecalibrator_benchmark import RISK_TYPES, validate_review
from citeweave.io import sha256_file, write_json, write_jsonl

SYSTEM_PROMPT = """You are a conservative claim-calibration reviewer for a bibliometric
research system. Use only the supplied evidence packet. Judge whether the atomic claim is
factually supported and whether its interpretation is calibrated to graph construction,
corpus scope, time window, and perturbation evidence. Correlation, co-occurrence, paths,
centrality, and communities do not establish causality or a domain mechanism.

Return one JSON object with exactly these fields:
action: accept|qualify|reject|abstain
risk_types: array selected from the supplied vocabulary
severity: minor|major|critical
unsupported_spans: exact claim substrings
decisive_evidence_ids: evidence IDs from the packet
minimal_revision: string for qualify, otherwise null

Use accept only when no material intervention is needed. Use qualify when a bounded rewrite
can preserve the claim. Use reject for fabricated or contradicted content. Use abstain when
the packet cannot resolve a material conflict. Output JSON only."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_fence(content: str) -> str:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _compact_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "atomic_claim": case["atomic_claim"],
        "local_context": case.get("local_context"),
        "graph_scope": case.get("graph_scope"),
        "phenomena": [
            {
                key: row.get(key)
                for key in (
                    "phenomenon_id",
                    "task_type",
                    "question",
                    "verified_answer",
                    "operator_trace",
                    "interpretation_contract",
                    "graph_evidence_ids",
                    "reference_ids",
                )
            }
            for row in case.get("phenomena") or []
        ],
        "sources": [
            {
                key: row.get(key)
                for key in (
                    "reference_id",
                    "title",
                    "year",
                    "doi",
                    "abstract_excerpt",
                )
            }
            for row in case.get("sources") or []
        ],
        "perturbation_profiles": case.get("perturbation_profiles"),
        "allowed_evidence_ids": case.get("allowed_evidence_ids"),
        "risk_type_vocabulary": list(RISK_TYPES),
    }


def _parse_prediction(
    content: str,
    *,
    case: dict[str, Any],
    condition: str,
    latency_seconds: float,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = json.loads(_strip_fence(content))
    required = {
        "action",
        "risk_types",
        "severity",
        "unsupported_spans",
        "decisive_evidence_ids",
        "minimal_revision",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Judge response fields differ from the required schema")
    validate_review(payload)
    invalid_ids = set(payload["decisive_evidence_ids"]) - set(
        case.get("allowed_evidence_ids") or []
    )
    if invalid_ids:
        raise ValueError(f"Judge cited unavailable evidence IDs: {sorted(invalid_ids)}")
    return {
        "case_id": case["case_id"],
        "condition": condition,
        **payload,
        "latency_seconds": latency_seconds,
        "usage": usage or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    all_cases = _read_jsonl(args.cases)
    selected = all_cases[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    condition = f"{args.model}_zero_shot_self_check"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    existing = _read_jsonl(output_path) if output_path.exists() and args.resume else []
    if output_path.exists() and not args.resume:
        raise SystemExit("Output exists; pass --resume to continue the same condition")
    by_id = {row["case_id"]: row for row in existing}

    with httpx.Client(timeout=180, follow_redirects=True) as client:
        for index, case in enumerate(selected, start=1):
            case_id = str(case["case_id"])
            if case_id in by_id:
                continue
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(_compact_case(case), ensure_ascii=False),
                    },
                ],
                "temperature": 0,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
            }
            record_dir = records_dir / case_id
            record_dir.mkdir(exist_ok=True)
            write_json(record_dir / "request.json", request)
            last_error: Exception | None = None
            for attempt in range(3):
                started = perf_counter()
                try:
                    response = client.post(
                        f"{args.base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request,
                    )
                    response.raise_for_status()
                    provider_payload = response.json()
                    latency = perf_counter() - started
                    write_json(record_dir / "provider_response.json", provider_payload)
                    content = provider_payload["choices"][0]["message"]["content"]
                    prediction = _parse_prediction(
                        content,
                        case=case,
                        condition=condition,
                        latency_seconds=latency,
                        usage=provider_payload.get("usage"),
                    )
                    by_id[case_id] = prediction
                    write_json(record_dir / "parsed_prediction.json", prediction)
                    write_jsonl(output_path, list(by_id.values()))
                    print(f"[{index}/{len(selected)}] {case_id}: {prediction['action']}")
                    break
                except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                    last_error = exc
                    write_json(
                        record_dir / f"failure_attempt_{attempt + 1}.json",
                        {"error_type": type(exc).__name__, "message": str(exc)[:1000]},
                    )
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
                        401,
                        402,
                        403,
                        404,
                    }:
                        break
                    if attempt < 2:
                        sleep(2**attempt)
            if case_id not in by_id:
                status_code = (
                    last_error.response.status_code
                    if isinstance(last_error, httpx.HTTPStatusError)
                    else None
                )
                write_json(
                    args.output_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "status": "blocked_api_judge",
                        "scientific_role": "prompt_only_self_check_baseline_not_gold",
                        "condition": condition,
                        "model": args.model,
                        "input_cases_sha256": sha256_file(args.cases),
                        "selected_cases": len(selected),
                        "completed_predictions": len(by_id),
                        "http_status": status_code,
                        "error_type": type(last_error).__name__,
                    },
                )
                raise RuntimeError(f"Judge failed for {case_id}: {last_error}")

    predictions = list(by_id.values())
    manifest = {
        "schema_version": 1,
        "status": "citecalibrator_api_judge_complete",
        "scientific_role": "prompt_only_self_check_baseline_not_gold",
        "condition": condition,
        "model": args.model,
        "temperature": 0,
        "input_cases": str(args.cases.resolve()),
        "input_cases_sha256": sha256_file(args.cases),
        "selected_offset": args.offset,
        "selected_limit": args.limit,
        "selected_cases": len(selected),
        "completed_predictions": len(predictions),
        "predictions_sha256": sha256_file(output_path),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
