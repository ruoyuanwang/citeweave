from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from citeweave.formal_request import (
    PROMPT_VERSION,
)
from citeweave.formal_request import (
    build_messages as _messages,
)
from citeweave.formal_request import (
    canonical_sha256 as _canonical_sha256,
)
from citeweave.formal_request import (
    task_payload_sha256 as _task_payload_sha256,
)
from citeweave.graph_discovery import score_discovery_response
from citeweave.io import read_json, sha256_file, write_json
from citeweave.token_budget import CommandTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _api_key(path: Path | None, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            for separator in ("=", ":"):
                key, found, candidate = line.partition(separator)
                if found and key.strip().casefold() == "deepseek":
                    value = candidate.strip()
                    if value:
                        return value
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def _parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload["choices"][0]["message"]["content"]
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


def _prepare_retry(
    records: list[dict[str, Any]],
    attempt_log: list[dict[str, Any]],
    *,
    item_id: str,
    condition: str,
) -> tuple[list[dict[str, Any]], int]:
    retained = []
    for record in records:
        if record["item_id"] == item_id and record["condition"] == condition:
            archived = dict(record)
            archived["archived_reason"] = "permitted_identical_retry"
            attempt_log.append(archived)
        else:
            retained.append(record)
    prior_attempts = sum(
        row["item_id"] == item_id and row["condition"] == condition
        for row in attempt_log
    )
    return retained, prior_attempts + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["no_reference", "flat_retrieval", "graph_retrieval", "graph_program"],
        choices=[
            "no_reference",
            "flat_retrieval",
            "flat_tfidf",
            "flat_bm25",
            "flat_lsa",
            "flat_neural_dense",
            "flat_hybrid",
            "graph_retrieval",
            "flat_program",
            "graph_query_retrieval",
            "graph_hierarchical_retrieval",
            "graph_hierarchical_retrieval_v2",
            "graph_program",
            "operator_only",
        ],
    )
    parser.add_argument("--scale", action="append")
    parser.add_argument("--task-type", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--readiness",
        type=Path,
        help="Required ready-status audit for confirmatory formal_v3 execution.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    benchmark = read_json(args.benchmark)
    formal_v3 = benchmark.get("formal_v3") or {}
    formal_replication = benchmark.get("formal_v3_replication") or {}
    formal_extension = benchmark.get("formal_v3_complexity_extension") or {}
    formal = formal_v3 or formal_replication or formal_extension
    readiness: dict[str, Any] | None = None
    tokenizer: CommandTokenizer | None = None
    context_token_budget: int | None = None
    neural_contexts: dict[str, dict[str, Any]] = {}
    extension_cell_identities: dict[tuple[str, str], dict[str, Any]] = {}
    if args.execute and formal:
        if not args.readiness or not args.readiness.is_file():
            raise SystemExit(
                "Formal v3 execution is blocked: provide a machine-verified --readiness audit"
            )
        readiness = read_json(args.readiness)
        if readiness.get("status") != "ready":
            raise SystemExit(
                "Formal v3 execution is blocked: readiness status is not ready"
            )
        hash_pairs = [("protocol_sha256", "protocol_sha256")]
        if formal_v3:
            hash_pairs.append(("amendment_sha256", "amendment_sha256"))
        elif formal_replication:
            hash_pairs.append(
                ("query_amendment_sha256", "query_amendment_sha256")
            )
        if any(
            readiness.get(readiness_key) != formal.get(benchmark_key)
            for readiness_key, benchmark_key in hash_pairs
        ):
            raise SystemExit(
                "Formal v3 execution is blocked: readiness hashes differ from benchmark"
            )
        if formal_extension:
            if not readiness.get("execution_amendment_sha256"):
                raise SystemExit(
                    "Complexity-extension execution is blocked: execution amendment missing"
                )
            extension_cell_identities = {
                (row["item_id"], row["condition"]): row
                for row in readiness.get("cell_identity_manifest") or []
                if row.get("dataset_id") == benchmark["dataset_id"]
            }
            registered_conditions = set(readiness.get("conditions") or [])
            if not set(args.conditions) <= registered_conditions:
                raise SystemExit(
                    "Complexity-extension execution is blocked: unregistered condition"
                )
        tokenizer_manifest = readiness.get("tokenizer_manifest") or {}
        tokenizer_manifest_path = Path(readiness["tokenizer_manifest_path"])
        tokenizer = CommandTokenizer(
            tokenizer_manifest,
            manifest_dir=tokenizer_manifest_path.parent,
        )
        verification = tokenizer.verify(tokenizer_manifest.get("verification_probes") or [])
        if not verification["passed"]:
            raise SystemExit(
                "Formal v3 execution is blocked: tokenizer command failed frozen probes"
            )
        message_verification = tokenizer.verify_message_probes(
            (readiness.get("tokenizer_api_usage_probes") or {}).get("records") or []
        )
        if not message_verification["passed"]:
            raise SystemExit(
                "Formal v3 execution is blocked: tokenizer failed API message probes"
            )
        context_token_budget = int(tokenizer_manifest["context_token_budget"])
        if "flat_neural_dense" in args.conditions:
            neural_manifest_path = Path(readiness["neural_dense_manifest_path"])
            neural_manifest = readiness.get("neural_dense_manifest") or {}
            artifact = next(
                (
                    row
                    for row in neural_manifest.get("artifacts") or []
                    if row.get("artifact_type") == "neural_context_sidecar"
                    and row.get("dataset_id") == benchmark["dataset_id"]
                ),
                None,
            )
            if artifact is None:
                raise SystemExit(
                    "Formal v3 execution is blocked: neural sidecar is missing for dataset"
                )
            sidecar_path = Path(artifact["path"])
            if not sidecar_path.is_absolute():
                sidecar_path = neural_manifest_path.parent / sidecar_path
            if sha256_file(sidecar_path) != artifact["sha256"]:
                raise SystemExit(
                    "Formal v3 execution is blocked: neural sidecar hash mismatch"
                )
            sidecar = read_json(sidecar_path)
            if sidecar.get("source_benchmark_sha256") != sha256_file(args.benchmark):
                raise SystemExit(
                    "Formal v3 execution is blocked: neural sidecar benchmark hash mismatch"
                )
            neural_contexts = sidecar.get("contexts") or {}
    tasks = [
        task
        for task in benchmark["tasks"]
        if (not args.scale or task["scale"] in args.scale)
        and (not args.task_type or task["task_type"] in args.task_type)
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    manifest = {
        "schema_version": 1,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "dataset_id": benchmark["dataset_id"],
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "temperature": 0,
        "thinking": "disabled",
        "conditions": args.conditions,
        "task_ids": [task["item_id"] for task in tasks],
        "calls": len(tasks) * len(args.conditions),
        "context_token_budget": context_token_budget,
        "tokenizer_manifest_sha256": (
            sha256_file(Path(readiness["tokenizer_manifest_path"]))
            if readiness
            else None
        ),
        "readiness_sha256": sha256_file(args.readiness) if args.readiness else None,
        "neural_dense_amendment_sha256": (
            readiness.get("neural_dense_amendment_sha256") if readiness else None
        ),
        "neural_runtime_amendment_sha256": (
            readiness.get("neural_runtime_amendment_sha256") if readiness else None
        ),
        "neural_execution_scope_amendment_sha256": (
            readiness.get("neural_execution_scope_amendment_sha256")
            if readiness
            else None
        ),
        "statistics_amendment_sha256": (
            readiness.get("statistics_amendment_sha256") if readiness else None
        ),
        "execution_amendment_sha256": (
            readiness.get("execution_amendment_sha256") if readiness else None
        ),
        "neural_dense_manifest_sha256": (
            sha256_file(Path(readiness["neural_dense_manifest_path"]))
            if readiness and "flat_neural_dense" in args.conditions
            else None
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run_manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    key = _api_key(args.api_key_file, args.api_key_env)
    results_path = args.output / "results.json"
    if results_path.is_file():
        saved = read_json(results_path)
        records = list(saved.get("records") or [])
        attempt_log = list(saved.get("attempt_log") or [])
    else:
        records = []
        attempt_log = []
    completed = {
        (record["item_id"], record["condition"])
        for record in records
        if record.get("status", "complete") == "complete"
    }
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=240, follow_redirects=True) as client:
        for task in tasks:
            for condition in args.conditions:
                if (task["item_id"], condition) in completed:
                    continue
                records, attempt = _prepare_retry(
                    records,
                    attempt_log,
                    item_id=task["item_id"],
                    condition=condition,
                )
                messages, context_audit = _messages(
                    task,
                    condition,
                    context_override=(
                        neural_contexts[task["item_id"]]
                        if condition == "flat_neural_dense"
                        else None
                    ),
                    tokenizer=tokenizer,
                    token_budget=context_token_budget,
                )
                request = {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 2200,
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }
                audit_identity = {
                    "task_payload_sha256": _task_payload_sha256(task),
                    "request_messages_sha256": _canonical_sha256(messages),
                }
                if formal_extension:
                    expected_identity = extension_cell_identities.get(
                        (task["item_id"], condition)
                    )
                    observed_identity = {
                        **audit_identity,
                        "context_sha256": context_audit["context_sha256"],
                    }
                    if expected_identity is None or any(
                        observed_identity[key] != expected_identity.get(key)
                        for key in observed_identity
                    ):
                        raise SystemExit(
                            "Complexity-extension execution is blocked: cell identity mismatch"
                        )
                started = time.perf_counter()
                response = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=request,
                )
                response.raise_for_status()
                raw = response.json()
                try:
                    parsed = _parse_response(raw)
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                    records.append(
                        {
                            "item_id": task["item_id"],
                            "network": task["network"],
                            "scale": task["scale"],
                            "task_type": task["task_type"],
                            "complexity": task["complexity"],
                            "condition": condition,
                            "attempt": attempt,
                            "status": "failed_parse",
                            **audit_identity,
                            **context_audit,
                            "elapsed_seconds": time.perf_counter() - started,
                            "usage": raw.get("usage"),
                            "finish_reason": raw.get("choices", [{}])[0].get("finish_reason"),
                            "raw_content": raw.get("choices", [{}])[0].get("message", {}).get("content"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    write_json(
                        results_path,
                        {
                            "manifest": manifest,
                            "records": records,
                            "attempt_log": attempt_log,
                        },
                    )
                    print(task["scale"], task["task_type"], condition, "failed_parse")
                    continue
                score = score_discovery_response(task, parsed)
                records.append(
                    {
                        "item_id": task["item_id"],
                        "network": task["network"],
                        "scale": task["scale"],
                        "task_type": task["task_type"],
                        "complexity": task["complexity"],
                        "condition": condition,
                        "attempt": attempt,
                        "status": "complete",
                        **audit_identity,
                        **context_audit,
                        "elapsed_seconds": time.perf_counter() - started,
                        "usage": raw.get("usage"),
                        "response": parsed,
                        "score": score,
                    }
                )
                write_json(
                    results_path,
                    {
                        "manifest": manifest,
                        "records": records,
                        "attempt_log": attempt_log,
                    },
                )
                print(task["scale"], task["task_type"], condition, score["answer_exact"])

    summary: dict[str, Any] = {}
    for condition in args.conditions:
        subset = [
            record
            for record in records
            if record["condition"] == condition and record.get("status", "complete") == "complete"
        ]
        if not subset:
            summary[condition] = {"items": 0}
            continue
        summary[condition] = {
            "items": len(subset),
            "answer_accuracy": sum(record["score"]["answer_exact"] for record in subset) / len(subset),
            "mean_evidence_f1": sum(record["score"]["evidence_f1"] for record in subset) / len(subset),
            "limitation_rate": sum(
                record["score"]["has_required_limitation"] for record in subset
            )
            / len(subset),
            "mean_context_characters": sum(record["context_characters"] for record in subset)
            / len(subset),
            "mean_elapsed_seconds": sum(record["elapsed_seconds"] for record in subset) / len(subset),
        }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
