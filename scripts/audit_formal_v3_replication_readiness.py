from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.token_budget import CommandTokenizer, verify_api_usage_probe_artifact


def _artifact_check(path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    if not path.is_file():
        return False, {}, [f"tokenizer_manifest_missing:{path}"]
    payload = read_json(path)
    reasons = []
    required = {
        "model": "deepseek-v4-pro",
        "passed": True,
        "verified_against_api_usage": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            reasons.append(f"tokenizer_{key}_mismatch")
    for artifact in payload.get("artifacts") or []:
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.is_file():
            reasons.append(f"tokenizer_artifact_missing:{artifact_path}")
        elif artifact.get("sha256") != sha256_file(artifact_path):
            reasons.append(f"tokenizer_artifact_hash_mismatch:{artifact_path}")
    return not reasons, payload, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--query-judgment", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scale-pairs", type=Path, required=True)
    parser.add_argument("--scale-pairs-freeze", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    amendment_hash = sha256_file(args.amendment)
    protocol_freeze = read_json(args.protocol_freeze)
    amendment_freeze = read_json(args.amendment_freeze)
    judgment = read_json(args.query_judgment)
    scale_pairs = read_json(args.scale_pairs)
    scale_freeze = read_json(args.scale_pairs_freeze)
    construction_path = args.benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path) if construction_path.is_file() else {}
    checks = {
        "protocol_hash_matches_freeze": protocol_freeze.get("protocol_sha256")
        == protocol_hash,
        "amendment_hash_matches_freeze": amendment_freeze.get("sha256")
        == amendment_hash,
        "query_judgment_approved": (
            judgment.get("protocol_sha256") == protocol_hash
            and judgment.get("query_amendment_sha256") == amendment_hash
            and judgment.get("judgment", {}).get("overall_approved") is True
        ),
        "construction_complete": (
            construction.get("status") == "constructed_not_executed"
            and construction.get("protocol_sha256") == protocol_hash
            and construction.get("query_amendment_sha256") == amendment_hash
        ),
        "scale_pairs_frozen": (
            scale_freeze.get("sha256") == sha256_file(args.scale_pairs)
            and scale_pairs.get("groups") == scale_freeze.get("groups")
            and scale_pairs.get("items") == scale_freeze.get("items")
            and scale_pairs.get("model_outcomes_inspected") is False
        ),
    }
    reasons = [key for key, passed in checks.items() if not passed]
    record_by_dataset = {
        row["dataset_id"]: row for row in construction.get("records") or []
    }
    expected_signatures = Counter(
        ("keyword_cooccurrence", scale, task_type)
        for scale in protocol["benchmark"]["scales"]
        for task_type in protocol["benchmark"]["task_types"]
    )
    benchmark_audits = []
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        row_reasons = []
        if not benchmark_path.is_file():
            row_reasons.append("benchmark_missing")
        else:
            benchmark = read_json(benchmark_path)
            embedded = benchmark.get("formal_v3_replication") or {}
            observed = Counter(
                (task["network"], task["scale"], task["task_type"])
                for task in benchmark["tasks"]
            )
            if observed != expected_signatures:
                row_reasons.append("task_signature_mismatch")
            if embedded.get("protocol_sha256") != protocol_hash:
                row_reasons.append("embedded_protocol_hash_mismatch")
            if embedded.get("query_amendment_sha256") != amendment_hash:
                row_reasons.append("embedded_amendment_hash_mismatch")
            if record_by_dataset.get(dataset_id, {}).get(
                "benchmark_sha256"
            ) != sha256_file(benchmark_path):
                row_reasons.append("benchmark_hash_mismatch")
            for task in benchmark["tasks"]:
                if not {"flat_hybrid", "graph_program"} <= set(task["contexts"]):
                    row_reasons.append("registered_context_missing")
                    break
        benchmark_audits.append(
            {"dataset_id": dataset_id, "passed": not row_reasons, "reasons": row_reasons}
        )
        reasons.extend(f"{dataset_id}:{reason}" for reason in row_reasons)
    checks["all_benchmarks_valid"] = all(row["passed"] for row in benchmark_audits)
    expected_hashes = {
        row["dataset_id"]: row["benchmark_sha256"]
        for row in construction.get("records") or []
    }
    checks["scale_pair_benchmark_hashes_match"] = (
        scale_pairs.get("benchmark_hashes") == expected_hashes
    )
    if not checks["scale_pair_benchmark_hashes_match"]:
        reasons.append("scale_pair_benchmark_hashes_mismatch")
    result_files = (
        sorted(args.execution_root.rglob("results.json"))
        if args.execution_root.exists()
        else []
    )
    checks["no_replication_results_present"] = not result_files
    if result_files:
        reasons.append("replication_results_already_present")

    tokenizer_ok, tokenizer, tokenizer_reasons = _artifact_check(
        args.tokenizer_manifest
    )
    verification: dict[str, Any] = {"passed": False, "probes": []}
    api_usage_probes: dict[str, Any] = {}
    message_verification: dict[str, Any] = {"passed": False, "probes": []}
    if tokenizer_ok:
        try:
            counter = CommandTokenizer(
                tokenizer, manifest_dir=args.tokenizer_manifest.parent
            )
            verification = counter.verify(tokenizer.get("verification_probes") or [])
            if not verification["passed"]:
                tokenizer_reasons.append("tokenizer_command_probe_mismatch")
            (
                api_usage_probes,
                message_verification,
                message_reasons,
            ) = verify_api_usage_probe_artifact(
                tokenizer,
                manifest_dir=args.tokenizer_manifest.parent,
                tokenizer=counter,
            )
            tokenizer_reasons.extend(message_reasons)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            tokenizer_reasons.append(
                f"tokenizer_command_failed:{type(exc).__name__}:{exc}"
            )
    checks["exact_model_tokenizer_ready"] = (
        tokenizer_ok
        and verification["passed"]
        and message_verification["passed"]
        and not tokenizer_reasons
    )
    checks["token_budget_declared"] = (
        isinstance(tokenizer.get("context_token_budget"), int)
        and tokenizer.get("context_token_budget", 0) > 0
    )
    reasons.extend(tokenizer_reasons)
    if not checks["token_budget_declared"]:
        reasons.append("tokenizer_context_token_budget_missing")
    reasons = list(dict.fromkeys(reasons))
    result = {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "blocking_reasons": reasons,
        "protocol_sha256": protocol_hash,
        "query_amendment_sha256": amendment_hash,
        "scale_pairs_sha256": sha256_file(args.scale_pairs),
        "benchmarks": benchmark_audits,
        "replication_result_files": [str(path.resolve()) for path in result_files],
        "tokenizer_manifest": tokenizer,
        "tokenizer_manifest_path": str(args.tokenizer_manifest.resolve()),
        "tokenizer_verification": verification,
        "tokenizer_api_usage_probes": api_usage_probes,
        "tokenizer_message_verification": message_verification,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
