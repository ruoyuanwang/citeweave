from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.neural_index_audit import (
    validate_neural_embedding_indexes,
    validate_neural_runtime_amendments,
)
from citeweave.token_budget import CommandTokenizer, verify_api_usage_probe_artifact


def _task_signature(task: dict[str, Any]) -> tuple[str, str, str]:
    return task["network"], task["scale"], task["task_type"]


def _expected_signatures(protocol: dict[str, Any]) -> Counter[tuple[str, str, str]]:
    benchmark = protocol["benchmark"]
    signatures: Counter[tuple[str, str, str]] = Counter()
    for scale in benchmark["primary_scales"]:
        for task_type in benchmark["primary_task_types"]:
            signatures[(benchmark["primary_network"], scale, task_type)] += 1
    for network in benchmark["external_validity_networks"]:
        for task_type in benchmark["external_validity_task_types"]:
            signatures[(network, benchmark["external_validity_scale"], task_type)] += 1
    return signatures


def _artifact_check(
    path: Path,
    *,
    required_fields: dict[str, Any],
    label: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    if not path.is_file():
        return False, {}, [f"{label}_manifest_missing:{path}"]
    payload = read_json(path)
    reasons = []
    for key, expected in required_fields.items():
        if payload.get(key) != expected:
            reasons.append(
                f"{label}_{key}_mismatch:expected={expected!r},observed={payload.get(key)!r}"
            )
    for artifact in payload.get("artifacts") or []:
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.is_file():
            reasons.append(f"{label}_artifact_missing:{artifact_path}")
        elif artifact.get("sha256") != sha256_file(artifact_path):
            reasons.append(f"{label}_artifact_hash_mismatch:{artifact_path}")
    return not reasons, payload, reasons


def audit(
    *,
    protocol_path: Path,
    freeze_path: Path,
    amendment_path: Path,
    amendment_freeze_path: Path,
    neural_amendment_path: Path,
    neural_amendment_freeze_path: Path,
    statistics_amendment_path: Path,
    statistics_amendment_freeze_path: Path,
    benchmark_root: Path,
    execution_root: Path,
    tokenizer_manifest_path: Path,
    neural_manifest_path: Path,
    neural_runtime_amendment_path: Path,
    neural_runtime_amendment_freeze_path: Path,
    neural_scope_amendment_path: Path,
    neural_scope_amendment_freeze_path: Path,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    amendment_freeze = read_json(amendment_freeze_path)
    neural_amendment = yaml.safe_load(
        neural_amendment_path.read_text(encoding="utf-8")
    )
    neural_amendment_freeze = read_json(neural_amendment_freeze_path)
    statistics_amendment = yaml.safe_load(
        statistics_amendment_path.read_text(encoding="utf-8")
    )
    statistics_amendment_freeze = read_json(statistics_amendment_freeze_path)
    protocol_hash = sha256_file(protocol_path)
    amendment_hash = sha256_file(amendment_path)
    neural_amendment_hash = sha256_file(neural_amendment_path)
    statistics_amendment_hash = sha256_file(statistics_amendment_path)
    neural_runtime, neural_runtime_reasons = validate_neural_runtime_amendments(
        representation_amendment_path=neural_runtime_amendment_path,
        representation_freeze_path=neural_runtime_amendment_freeze_path,
        execution_scope_amendment_path=neural_scope_amendment_path,
        execution_scope_freeze_path=neural_scope_amendment_freeze_path,
    )
    checks: dict[str, bool] = {
        "protocol_hash_matches_freeze": freeze.get("protocol_sha256") == protocol_hash,
        "amendment_targets_protocol": amendment.get("base_protocol_sha256") == protocol_hash,
        "amendment_hash_matches_freeze": (
            amendment_freeze.get("amendment_sha256") == amendment_hash
        ),
        "amendment_pre_model": (
            amendment.get("model_outcomes_inspected") is False
            and amendment_freeze.get("model_outcomes_inspected") is False
        ),
        "neural_amendment_chain_valid": (
            neural_amendment.get("base_protocol_sha256") == protocol_hash
            and neural_amendment.get("prior_amendment_sha256") == amendment_hash
        ),
        "neural_amendment_hash_matches_freeze": (
            neural_amendment_freeze.get("amendment_sha256")
            == neural_amendment_hash
        ),
        "neural_amendment_pre_model": (
            neural_amendment.get("model_outcomes_inspected") is False
            and neural_amendment_freeze.get("model_outcomes_inspected") is False
        ),
        "statistics_amendment_chain_valid": (
            statistics_amendment.get("base_protocol_sha256") == protocol_hash
            and statistics_amendment.get("prior_amendments")
            == [amendment_hash, neural_amendment_hash]
        ),
        "statistics_amendment_hash_matches_freeze": (
            statistics_amendment_freeze.get("amendment_sha256")
            == statistics_amendment_hash
        ),
        "statistics_amendment_pre_model": (
            statistics_amendment.get("model_outcomes_inspected") is False
            and statistics_amendment_freeze.get("model_outcomes_inspected") is False
        ),
    }
    reasons = [key for key, passed in checks.items() if not passed]
    checks["neural_runtime_amendment_chain_valid"] = not neural_runtime_reasons
    reasons.extend(neural_runtime_reasons)
    construction_path = benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path) if construction_path.is_file() else {}
    checks["construction_complete"] = (
        construction.get("status") == "constructed_not_executed"
        and construction.get("protocol_sha256") == protocol_hash
        and construction.get("amendment_sha256") == amendment_hash
    )
    if not checks["construction_complete"]:
        reasons.append("construction_manifest_incomplete_or_hash_mismatch")

    expected = _expected_signatures(protocol)
    benchmark_audits = []
    record_by_dataset = {
        row["dataset_id"]: row for row in construction.get("records") or []
    }
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        benchmark_path = benchmark_root / dataset_id / "benchmark.json"
        row_reasons = []
        if not benchmark_path.is_file():
            row_reasons.append("benchmark_missing")
            benchmark_audits.append(
                {"dataset_id": dataset_id, "passed": False, "reasons": row_reasons}
            )
            reasons.append(f"{dataset_id}:benchmark_missing")
            continue
        benchmark = read_json(benchmark_path)
        observed = Counter(_task_signature(task) for task in benchmark["tasks"])
        manifest_record = record_by_dataset.get(dataset_id) or {}
        benchmark_hash = sha256_file(benchmark_path)
        if observed != expected:
            row_reasons.append("task_signature_distribution_mismatch")
        if len(benchmark["tasks"]) != protocol["benchmark"][
            "expected_tasks_per_eligible_dataset"
        ]:
            row_reasons.append("task_count_mismatch")
        if manifest_record.get("benchmark_sha256") != benchmark_hash:
            row_reasons.append("benchmark_hash_mismatch")
        if benchmark.get("formal_v3", {}).get("protocol_sha256") != protocol_hash:
            row_reasons.append("embedded_protocol_hash_mismatch")
        if benchmark.get("formal_v3", {}).get("amendment_sha256") != amendment_hash:
            row_reasons.append("embedded_amendment_hash_mismatch")
        structural_reasons = list(row_reasons)
        benchmark_audits.append(
            {
                "dataset_id": dataset_id,
                "passed": not row_reasons,
                "structural_passed": not structural_reasons,
                "reasons": row_reasons,
                "tasks": len(benchmark["tasks"]),
                "benchmark_sha256": benchmark_hash,
                "task_signatures": len(observed),
            }
        )
        reasons.extend(f"{dataset_id}:{reason}" for reason in row_reasons)
    checks["all_benchmarks_valid"] = all(row["passed"] for row in benchmark_audits)
    checks["benchmark_structure_valid"] = all(
        row.get("structural_passed", False) for row in benchmark_audits
    )
    scale_pair_spec = statistics_amendment["scale_pair_artifact"]
    scale_pair_path = statistics_amendment_path.parent / scale_pair_spec["path"]
    scale_pairs = read_json(scale_pair_path) if scale_pair_path.is_file() else {}
    checks["scale_pair_artifact_valid"] = (
        scale_pair_path.is_file()
        and sha256_file(scale_pair_path) == scale_pair_spec["sha256"]
        and scale_pairs.get("groups") == scale_pair_spec["groups"]
        and scale_pairs.get("items") == scale_pair_spec["items"]
        and scale_pairs.get("model_outcomes_inspected") is False
        and scale_pairs.get("benchmark_hashes")
        == {
            row["dataset_id"]: row["benchmark_sha256"]
            for row in construction.get("records") or []
        }
    )
    if not checks["scale_pair_artifact_valid"]:
        reasons.append("scale_pair_artifact_missing_or_hash_mismatch")

    result_files = sorted(execution_root.rglob("results.json")) if execution_root.exists() else []
    checks["no_confirmatory_results_present"] = not result_files
    if result_files:
        reasons.append("confirmatory_results_already_present")

    tokenizer_ok, tokenizer, tokenizer_reasons = _artifact_check(
        tokenizer_manifest_path,
        required_fields={
            "model": protocol["model_protocol"]["model"],
            "passed": True,
            "verified_against_api_usage": True,
        },
        label="tokenizer",
    )
    tokenizer_verification: dict[str, Any] = {"passed": False, "probes": []}
    tokenizer_api_usage_probes: dict[str, Any] = {}
    tokenizer_message_verification: dict[str, Any] = {"passed": False, "probes": []}
    if tokenizer_ok:
        try:
            counter = CommandTokenizer(tokenizer, manifest_dir=tokenizer_manifest_path.parent)
            tokenizer_verification = counter.verify(tokenizer.get("verification_probes") or [])
            if not tokenizer_verification["passed"]:
                tokenizer_reasons.append("tokenizer_command_probe_mismatch")
            (
                tokenizer_api_usage_probes,
                tokenizer_message_verification,
                message_reasons,
            ) = verify_api_usage_probe_artifact(
                tokenizer,
                manifest_dir=tokenizer_manifest_path.parent,
                tokenizer=counter,
            )
            tokenizer_reasons.extend(message_reasons)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            tokenizer_reasons.append(f"tokenizer_command_failed:{type(exc).__name__}:{exc}")
    tokenizer_ok = (
        tokenizer_ok
        and tokenizer_verification["passed"]
        and tokenizer_message_verification["passed"]
        and not tokenizer_reasons
    )
    checks["exact_model_tokenizer_ready"] = tokenizer_ok
    reasons.extend(tokenizer_reasons)
    neural_spec = neural_amendment["changes"]["neural_dense_baseline"]
    neural_ok, neural, neural_reasons = _artifact_check(
        neural_manifest_path,
        required_fields={
            "passed": True,
            "neural": True,
            "query_independent_index": True,
            "condition_name": "flat_neural_dense",
            "model": neural_spec["model"],
            "model_revision": neural_spec["model_revision"],
            "similarity": neural_spec["similarity"],
        },
        label="neural_dense",
    )
    if neural_ok:
        neural_reasons.extend(
            validate_neural_embedding_indexes(
                neural_manifest=neural,
                neural_manifest_path=neural_manifest_path,
            )
        )
        expected_by_dataset = {
            row["dataset_id"]: row for row in benchmark_audits
        }
        observed_datasets = set()
        for artifact in neural.get("artifacts") or []:
            if artifact.get("artifact_type") != "neural_context_sidecar":
                continue
            sidecar_path = Path(artifact["path"])
            if not sidecar_path.is_absolute():
                sidecar_path = neural_manifest_path.parent / sidecar_path
            sidecar = read_json(sidecar_path)
            dataset_id = sidecar.get("dataset_id")
            observed_datasets.add(dataset_id)
            benchmark_path = benchmark_root / str(dataset_id) / "benchmark.json"
            if dataset_id not in expected_by_dataset:
                neural_reasons.append(f"neural_dense_unknown_dataset:{dataset_id}")
                continue
            benchmark = read_json(benchmark_path)
            expected_ids = {task["item_id"] for task in benchmark["tasks"]}
            contexts = sidecar.get("contexts") or {}
            if set(contexts) != expected_ids:
                neural_reasons.append(f"neural_dense_task_coverage_mismatch:{dataset_id}")
            if sidecar.get("source_benchmark_sha256") != sha256_file(benchmark_path):
                neural_reasons.append(f"neural_dense_source_hash_mismatch:{dataset_id}")
            if any(
                context.get("representation") != "flat_neural_dense_retrieval"
                for context in contexts.values()
            ):
                neural_reasons.append(f"neural_dense_representation_mismatch:{dataset_id}")
        expected_datasets = {row["id"] for row in protocol["datasets"]}
        if observed_datasets != expected_datasets:
            neural_reasons.append("neural_dense_dataset_coverage_mismatch")
    neural_ok = neural_ok and not neural_reasons
    checks["neural_dense_baseline_ready"] = neural_ok
    reasons.extend(neural_reasons)
    checks["token_budget_declared"] = (
        isinstance(tokenizer.get("context_token_budget"), int)
        and tokenizer.get("context_token_budget", 0) > 0
    )
    if not checks["token_budget_declared"]:
        reasons.append("tokenizer_context_token_budget_missing")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "blocking_reasons": unique_reasons,
        "protocol_sha256": protocol_hash,
        "amendment_sha256": amendment_hash,
        "neural_dense_amendment_sha256": neural_amendment_hash,
        "neural_dense_amendment_path": str(neural_amendment_path.resolve()),
        "neural_runtime": neural_runtime,
        "neural_runtime_amendment_sha256": neural_runtime.get(
            "representation_amendment_sha256"
        ),
        "neural_execution_scope_amendment_sha256": neural_runtime.get(
            "execution_scope_amendment_sha256"
        ),
        "statistics_amendment_sha256": statistics_amendment_hash,
        "statistics_amendment_path": str(statistics_amendment_path.resolve()),
        "scale_pair_artifact_path": str(scale_pair_path.resolve()),
        "benchmarks": benchmark_audits,
        "confirmatory_result_files": [str(path.resolve()) for path in result_files],
        "tokenizer_manifest": tokenizer,
        "tokenizer_manifest_path": str(tokenizer_manifest_path.resolve()),
        "tokenizer_verification": tokenizer_verification,
        "tokenizer_api_usage_probes": tokenizer_api_usage_probes,
        "tokenizer_message_verification": tokenizer_message_verification,
        "neural_dense_manifest": neural,
        "neural_dense_manifest_path": str(neural_manifest_path.resolve()),
        "note": (
            "LSA is a latent semantic diagnostic and never satisfies the neural-dense gate. "
            "Status ready is required before confirmatory API execution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--neural-amendment", type=Path, required=True)
    parser.add_argument("--neural-amendment-freeze", type=Path, required=True)
    parser.add_argument("--statistics-amendment", type=Path, required=True)
    parser.add_argument("--statistics-amendment-freeze", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--neural-manifest", type=Path, required=True)
    parser.add_argument(
        "--neural-runtime-amendment",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/formal_v3_neural_runtime_amendment_003_semantic_representation.yml"
        ),
    )
    parser.add_argument(
        "--neural-runtime-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/formal_v3_neural_runtime_amendment_003_semantic_representation_freeze.json"
        ),
    )
    parser.add_argument(
        "--neural-scope-amendment",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/formal_v3_neural_runtime_amendment_004_execution_scope_clarification.yml"
        ),
    )
    parser.add_argument(
        "--neural-scope-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/formal_v3_neural_runtime_amendment_004_execution_scope_clarification_freeze.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        protocol_path=args.protocol,
        freeze_path=args.freeze,
        amendment_path=args.amendment,
        amendment_freeze_path=args.amendment_freeze,
        neural_amendment_path=args.neural_amendment,
        neural_amendment_freeze_path=args.neural_amendment_freeze,
        statistics_amendment_path=args.statistics_amendment,
        statistics_amendment_freeze_path=args.statistics_amendment_freeze,
        benchmark_root=args.benchmark_root,
        execution_root=args.execution_root,
        tokenizer_manifest_path=args.tokenizer_manifest,
        neural_manifest_path=args.neural_manifest,
        neural_runtime_amendment_path=args.neural_runtime_amendment,
        neural_runtime_amendment_freeze_path=args.neural_runtime_amendment_freeze,
        neural_scope_amendment_path=args.neural_scope_amendment,
        neural_scope_amendment_freeze_path=args.neural_scope_amendment_freeze,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
