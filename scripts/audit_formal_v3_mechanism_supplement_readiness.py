from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from citeweave.formal_request import (
    PROMPT_VERSION,
    build_messages,
    canonical_sha256,
    task_payload_sha256,
)
from citeweave.io import read_json, sha256_file, write_json
from citeweave.token_budget import CommandTokenizer, verify_api_usage_probe_artifact

CONDITIONS = ("flat_program", "operator_only")
EXPECTED_SOURCE_CELLS = {
    "primary": 800,
    "replication": 120,
    "extension": 312,
}


def _load_tokenizer(path: Path) -> tuple[
    dict[str, Any],
    CommandTokenizer | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
]:
    if not path.is_file():
        return {}, None, {}, {}, {}, [f"tokenizer_manifest_missing:{path}"]
    payload = read_json(path)
    reasons: list[str] = []
    if payload.get("model") != "deepseek-v4-pro":
        reasons.append("tokenizer_model_mismatch")
    if payload.get("passed") is not True:
        reasons.append("tokenizer_not_passed")
    if payload.get("verified_against_api_usage") is not True:
        reasons.append("tokenizer_not_verified_against_api_usage")
    tokenizer = None
    command_verification: dict[str, Any] = {"passed": False, "probes": []}
    api_probes: dict[str, Any] = {}
    message_verification: dict[str, Any] = {"passed": False, "probes": []}
    if not reasons:
        try:
            tokenizer = CommandTokenizer(payload, manifest_dir=path.parent)
            command_verification = tokenizer.verify(
                payload.get("verification_probes") or []
            )
            if not command_verification["passed"]:
                reasons.append("tokenizer_command_probe_mismatch")
            api_probes, message_verification, probe_reasons = (
                verify_api_usage_probe_artifact(
                    payload,
                    manifest_dir=path.parent,
                    tokenizer=tokenizer,
                )
            )
            reasons.extend(probe_reasons)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            reasons.append(f"tokenizer_command_failed:{type(exc).__name__}:{exc}")
    return (
        payload,
        tokenizer,
        command_verification,
        api_probes,
        message_verification,
        reasons,
    )


def audit_mechanism_supplement_readiness(
    *,
    supplement_protocol_path: Path,
    supplement_freeze_path: Path,
    extension_protocol_path: Path,
    benchmark_root: Path,
    tokenizer_manifest_path: Path,
    source_terminal_audits: dict[str, Path],
    execution_root: Path,
) -> dict[str, Any]:
    supplement_hash = sha256_file(supplement_protocol_path)
    supplement = yaml.safe_load(
        supplement_protocol_path.read_text(encoding="utf-8")
    )
    extension_hash = sha256_file(extension_protocol_path)
    freeze = read_json(supplement_freeze_path)
    checks = {
        "supplement_hash_matches_freeze": freeze.get("sha256") == supplement_hash,
        "extension_protocol_binding_matches": (
            supplement.get("source_bindings", {}).get(
                "complexity_extension_protocol_sha256"
            )
            == extension_hash
        ),
    }
    reasons = [key for key, passed in checks.items() if not passed]

    construction_path = benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path) if construction_path.is_file() else {}
    checks["construction_complete"] = (
        construction.get("status") == "constructed_not_executed"
        and construction.get("protocol_sha256") == extension_hash
        and construction.get("datasets") == 8
        and construction.get("tasks") == 168
    )
    if not checks["construction_complete"]:
        reasons.append("construction_not_complete")
    records = {
        str(row.get("dataset_id")): row for row in construction.get("records") or []
    }
    expected_signature = Counter(
        (scale, task_type)
        for scale in ("small", "medium", "large")
        for task_type in (
            "direct_edge_lookup",
            "node_attribute_lookup",
            "multi_hop_connector",
            "bridge_counterfactual",
            "community_role_contrast",
            "hub_removal_resilience",
            "temporal_structural_shift",
        )
    )
    benchmarks: dict[str, dict[str, Any]] = {}
    benchmark_audits = []
    for record in sorted(records.values(), key=lambda row: str(row["dataset_id"])):
        dataset_id = str(record["dataset_id"])
        path = benchmark_root / dataset_id / "benchmark.json"
        row_reasons: list[str] = []
        if not path.is_file():
            row_reasons.append("benchmark_missing")
        else:
            benchmark = read_json(path)
            embedded = benchmark.get("formal_v3_complexity_extension") or {}
            if sha256_file(path) != record.get("benchmark_sha256"):
                row_reasons.append("benchmark_hash_mismatch")
            if embedded.get("protocol_sha256") != extension_hash:
                row_reasons.append("embedded_extension_protocol_hash_mismatch")
            if Counter(
                (task["scale"], task["task_type"])
                for task in benchmark.get("tasks") or []
            ) != expected_signature:
                row_reasons.append("task_signature_mismatch")
            for task in benchmark.get("tasks") or []:
                if not set(CONDITIONS) <= set(task.get("contexts") or {}):
                    row_reasons.append("same_computation_context_missing")
                    break
            benchmarks[dataset_id] = benchmark
        benchmark_audits.append(
            {"dataset_id": dataset_id, "passed": not row_reasons, "reasons": row_reasons}
        )
        reasons.extend(f"{dataset_id}:{reason}" for reason in row_reasons)
    checks["all_eight_benchmarks_valid"] = (
        len(benchmark_audits) == 8 and all(row["passed"] for row in benchmark_audits)
    )
    if not checks["all_eight_benchmarks_valid"]:
        reasons.append("benchmark_panel_invalid")

    source_audits: dict[str, Any] = {}
    for name, path in source_terminal_audits.items():
        payload = read_json(path) if path.is_file() else {}
        source_audits[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path) if path.is_file() else None,
            "status": payload.get("status"),
            "expected_cells": payload.get("expected_cells"),
            "integrity_reasons": payload.get("integrity_reasons") or [],
        }
        passed = (
            payload.get("status") == "terminal"
            and payload.get("expected_cells") == EXPECTED_SOURCE_CELLS[name]
            and not (payload.get("integrity_reasons") or [])
        )
        checks[f"{name}_source_terminal"] = passed
        if not passed:
            reasons.append(f"{name}_source_not_terminal")

    result_files = (
        sorted(execution_root.rglob("results.json")) if execution_root.exists() else []
    )
    checks["no_supplement_results_present"] = not result_files
    if result_files:
        reasons.append("supplement_results_already_present")

    (
        tokenizer_payload,
        tokenizer,
        tokenizer_verification,
        tokenizer_api_probes,
        tokenizer_message_verification,
        tokenizer_reasons,
    ) = _load_tokenizer(tokenizer_manifest_path)
    checks["exact_model_tokenizer_ready"] = (
        tokenizer is not None
        and tokenizer_verification.get("passed") is True
        and tokenizer_message_verification.get("passed") is True
        and not tokenizer_reasons
    )
    reasons.extend(tokenizer_reasons)

    cells = []
    if (
        tokenizer is not None
        and checks["exact_model_tokenizer_ready"]
        and checks["all_eight_benchmarks_valid"]
    ):
        budget = int(tokenizer_payload["context_token_budget"])
        for dataset_id, benchmark in sorted(benchmarks.items()):
            for task in benchmark["tasks"]:
                for condition in CONDITIONS:
                    messages, context_audit = build_messages(
                        task,
                        condition,
                        tokenizer=tokenizer,
                        token_budget=budget,
                    )
                    cells.append(
                        {
                            "dataset_id": dataset_id,
                            "item_id": task["item_id"],
                            "condition": condition,
                            "task_payload_sha256": task_payload_sha256(task),
                            "context_sha256": context_audit["context_sha256"],
                            "request_messages_sha256": canonical_sha256(messages),
                        }
                    )
    checks["all_336_cell_identities_materialized"] = len(cells) == 336
    if not checks["all_336_cell_identities_materialized"]:
        reasons.append("cell_identity_manifest_not_materialized")

    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "blocking_reasons": reasons,
        # The generic runner compares this field with the embedded extension protocol.
        "protocol_sha256": extension_hash,
        "supplement_protocol_sha256": supplement_hash,
        "execution_amendment_sha256": supplement.get("source_bindings", {}).get(
            "complexity_execution_amendment_sha256"
        ),
        "source_terminal_audits": source_audits,
        "benchmarks": benchmark_audits,
        "supplement_result_files": [str(path.resolve()) for path in result_files],
        "tokenizer_manifest": tokenizer_payload,
        "tokenizer_manifest_path": str(tokenizer_manifest_path.resolve()),
        "tokenizer_manifest_sha256": (
            sha256_file(tokenizer_manifest_path)
            if tokenizer_manifest_path.is_file()
            else None
        ),
        "tokenizer_verification": tokenizer_verification,
        "tokenizer_api_usage_probes": tokenizer_api_probes,
        "tokenizer_message_verification": tokenizer_message_verification,
        "cell_identity_manifest": cells,
        "cell_identity_manifest_sha256": canonical_sha256(cells) if cells else None,
        "prompt_version": PROMPT_VERSION,
        "conditions": list(CONDITIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-protocol", type=Path, required=True)
    parser.add_argument("--supplement-freeze", type=Path, required=True)
    parser.add_argument("--extension-protocol", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--primary-terminal-audit", type=Path, required=True)
    parser.add_argument("--replication-terminal-audit", type=Path, required=True)
    parser.add_argument("--extension-terminal-audit", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_mechanism_supplement_readiness(
        supplement_protocol_path=args.supplement_protocol,
        supplement_freeze_path=args.supplement_freeze,
        extension_protocol_path=args.extension_protocol,
        benchmark_root=args.benchmark_root,
        tokenizer_manifest_path=args.tokenizer_manifest,
        source_terminal_audits={
            "primary": args.primary_terminal_audit,
            "replication": args.replication_terminal_audit,
            "extension": args.extension_terminal_audit,
        },
        execution_root=args.execution_root,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
