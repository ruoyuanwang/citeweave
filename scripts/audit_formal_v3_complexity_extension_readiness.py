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
from citeweave.neural_index_audit import (
    validate_neural_embedding_indexes,
    validate_neural_runtime_amendments,
)
from citeweave.token_budget import CommandTokenizer, verify_api_usage_probe_artifact

CONDITIONS = (
    "flat_hybrid",
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
    "graph_program",
)


def _tokenizer_check(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"tokenizer_manifest_missing:{path}"]
    payload = read_json(path)
    reasons = []
    for key, expected in {
        "model": "deepseek-v4-pro",
        "passed": True,
        "verified_against_api_usage": True,
    }.items():
        if payload.get(key) != expected:
            reasons.append(f"tokenizer_{key}_mismatch")
    if not isinstance(payload.get("context_token_budget"), int) or int(
        payload.get("context_token_budget") or 0
    ) <= 0:
        reasons.append("tokenizer_context_token_budget_missing")
    for artifact in payload.get("artifacts") or []:
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.is_file():
            reasons.append(f"tokenizer_artifact_missing:{artifact_path}")
        elif sha256_file(artifact_path) != artifact.get("sha256"):
            reasons.append(f"tokenizer_artifact_hash_mismatch:{artifact_path}")
    return payload, reasons


def _neural_sidecars(
    *,
    manifest_path: Path,
    benchmark_hashes: dict[str, str],
    expected_model: str,
    expected_revision: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    if not manifest_path.is_file():
        return {}, {}, [f"neural_manifest_missing:{manifest_path}"]
    manifest = read_json(manifest_path)
    reasons = []
    if manifest.get("passed") is not True or manifest.get("neural") is not True:
        reasons.append("neural_manifest_not_passed")
    if manifest.get("model") != expected_model:
        reasons.append("neural_model_mismatch")
    if manifest.get("model_revision") != expected_revision:
        reasons.append("neural_model_revision_mismatch")
    reasons.extend(
        validate_neural_embedding_indexes(
            neural_manifest=manifest,
            neural_manifest_path=manifest_path,
        )
    )
    by_dataset = {}
    artifacts = [
        row
        for row in manifest.get("artifacts") or []
        if row.get("artifact_type") == "neural_context_sidecar"
    ]
    if {row.get("dataset_id") for row in artifacts} != set(benchmark_hashes):
        reasons.append("neural_dataset_coverage_mismatch")
    for artifact in artifacts:
        dataset_id = str(artifact["dataset_id"])
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file():
            reasons.append(f"neural_sidecar_missing:{dataset_id}")
            continue
        if sha256_file(path) != artifact.get("sha256"):
            reasons.append(f"neural_sidecar_hash_mismatch:{dataset_id}")
            continue
        sidecar = read_json(path)
        if sidecar.get("source_benchmark_sha256") != benchmark_hashes.get(dataset_id):
            reasons.append(f"neural_source_benchmark_mismatch:{dataset_id}")
            continue
        by_dataset[dataset_id] = sidecar
    return by_dataset, manifest, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--execution-amendment-freeze", type=Path, required=True)
    parser.add_argument("--neural-amendment", type=Path, required=True)
    parser.add_argument("--neural-amendment-freeze", type=Path, required=True)
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

    protocol_hash = sha256_file(args.protocol)
    execution_hash = sha256_file(args.execution_amendment)
    neural_hash = sha256_file(args.neural_amendment)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    execution = yaml.safe_load(args.execution_amendment.read_text(encoding="utf-8"))
    neural = yaml.safe_load(args.neural_amendment.read_text(encoding="utf-8"))
    neural_runtime, neural_runtime_reasons = validate_neural_runtime_amendments(
        representation_amendment_path=args.neural_runtime_amendment,
        representation_freeze_path=args.neural_runtime_amendment_freeze,
        execution_scope_amendment_path=args.neural_scope_amendment,
        execution_scope_freeze_path=args.neural_scope_amendment_freeze,
    )
    construction_path = args.benchmark_root / "construction_manifest.json"
    construction = read_json(construction_path) if construction_path.is_file() else {}
    checks = {
        "protocol_hash_matches_freeze": read_json(args.protocol_freeze).get("sha256")
        == protocol_hash,
        "execution_amendment_hash_matches_freeze": read_json(
            args.execution_amendment_freeze
        ).get("sha256")
        == execution_hash,
        "execution_amends_protocol": execution.get("amends_protocol_sha256")
        == protocol_hash,
        "neural_amendment_hash_matches_freeze": read_json(
            args.neural_amendment_freeze
        ).get("amendment_sha256")
        == neural_hash,
        "construction_complete": (
            construction.get("status") == "constructed_not_executed"
            and construction.get("protocol_sha256") == protocol_hash
            and construction.get("datasets") == 8
            and construction.get("tasks") == 168
            and construction.get("planned_calls") == 672
        ),
    }
    reasons = [key for key, passed in checks.items() if not passed]
    checks["neural_runtime_amendment_chain_valid"] = not neural_runtime_reasons
    reasons.extend(neural_runtime_reasons)
    records_by_dataset = {
        row["dataset_id"]: row for row in construction.get("records") or []
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
    benchmarks = {}
    benchmark_hashes = {}
    benchmark_audits = []
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        path = args.benchmark_root / dataset_id / "benchmark.json"
        row_reasons = []
        if not path.is_file():
            row_reasons.append("benchmark_missing")
        else:
            benchmark = read_json(path)
            benchmark_hash = sha256_file(path)
            embedded = benchmark.get("formal_v3_complexity_extension") or {}
            observed = Counter(
                (task["scale"], task["task_type"]) for task in benchmark["tasks"]
            )
            if observed != expected_signature:
                row_reasons.append("task_signature_mismatch")
            if embedded.get("protocol_sha256") != protocol_hash:
                row_reasons.append("embedded_protocol_hash_mismatch")
            if records_by_dataset.get(dataset_id, {}).get(
                "benchmark_sha256"
            ) != benchmark_hash:
                row_reasons.append("benchmark_hash_mismatch")
            task_index = {task["item_id"]: task for task in benchmark["tasks"]}
            for task in benchmark["tasks"]:
                if task["complexity"] == 1:
                    parent = task_index.get(task.get("matched_complex_item_id"))
                    if parent is None or task["evidence_ids"][0] != parent["evidence_ids"][0]:
                        row_reasons.append("simple_anchor_mismatch")
                        break
                if not {
                    "flat_hybrid",
                    "graph_hierarchical_retrieval_v2",
                    "graph_program",
                } <= set(task["contexts"]):
                    row_reasons.append("registered_context_missing")
                    break
            benchmarks[dataset_id] = benchmark
            benchmark_hashes[dataset_id] = benchmark_hash
        benchmark_audits.append(
            {"dataset_id": dataset_id, "passed": not row_reasons, "reasons": row_reasons}
        )
        reasons.extend(f"{dataset_id}:{reason}" for reason in row_reasons)
    checks["all_benchmarks_valid"] = all(row["passed"] for row in benchmark_audits)

    result_files = (
        sorted(args.execution_root.rglob("results.json"))
        if args.execution_root.exists()
        else []
    )
    checks["no_extension_results_present"] = not result_files
    if result_files:
        reasons.append("extension_results_already_present")

    tokenizer_payload, tokenizer_reasons = _tokenizer_check(args.tokenizer_manifest)
    tokenizer: CommandTokenizer | None = None
    tokenizer_verification: dict[str, Any] = {"passed": False, "probes": []}
    tokenizer_api_usage_probes: dict[str, Any] = {}
    tokenizer_message_verification: dict[str, Any] = {"passed": False, "probes": []}
    if not tokenizer_reasons:
        try:
            tokenizer = CommandTokenizer(
                tokenizer_payload, manifest_dir=args.tokenizer_manifest.parent
            )
            tokenizer_verification = tokenizer.verify(
                tokenizer_payload.get("verification_probes") or []
            )
            if not tokenizer_verification["passed"]:
                tokenizer_reasons.append("tokenizer_command_probe_mismatch")
            (
                tokenizer_api_usage_probes,
                tokenizer_message_verification,
                message_reasons,
            ) = verify_api_usage_probe_artifact(
                tokenizer_payload,
                manifest_dir=args.tokenizer_manifest.parent,
                tokenizer=tokenizer,
            )
            tokenizer_reasons.extend(message_reasons)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            tokenizer_reasons.append(
                f"tokenizer_command_failed:{type(exc).__name__}:{exc}"
            )
    checks["exact_model_tokenizer_ready"] = (
        not tokenizer_reasons
        and tokenizer_verification["passed"]
        and tokenizer_message_verification["passed"]
    )
    reasons.extend(tokenizer_reasons)

    neural_spec = neural["changes"]["neural_dense_baseline"]
    sidecars, neural_manifest, neural_reasons = _neural_sidecars(
        manifest_path=args.neural_manifest,
        benchmark_hashes=benchmark_hashes,
        expected_model=neural_spec["model"],
        expected_revision=neural_spec["model_revision"],
    )
    for dataset_id, sidecar in sidecars.items():
        expected_ids = {task["item_id"] for task in benchmarks[dataset_id]["tasks"]}
        if set(sidecar.get("contexts") or {}) != expected_ids:
            neural_reasons.append(f"neural_item_coverage_mismatch:{dataset_id}")
    checks["neural_dense_sidecars_ready"] = not neural_reasons
    reasons.extend(neural_reasons)

    cells = []
    if (
        tokenizer is not None
        and tokenizer_verification["passed"]
        and not neural_reasons
        and checks["all_benchmarks_valid"]
    ):
        budget = int(tokenizer_payload["context_token_budget"])
        for dataset_id, benchmark in sorted(benchmarks.items()):
            neural_contexts = sidecars[dataset_id]["contexts"]
            for task in benchmark["tasks"]:
                for condition in CONDITIONS:
                    messages, context_audit = build_messages(
                        task,
                        condition,
                        context_override=(
                            neural_contexts[task["item_id"]]
                            if condition == "flat_neural_dense"
                            else None
                        ),
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
    checks["all_672_cell_identities_materialized"] = len(cells) == 672
    if not checks["all_672_cell_identities_materialized"]:
        reasons.append("cell_identity_manifest_not_materialized")
    reasons = list(dict.fromkeys(reasons))
    result = {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "blocking_reasons": reasons,
        "protocol_sha256": protocol_hash,
        "execution_amendment_sha256": execution_hash,
        "neural_dense_amendment_sha256": neural_hash,
        "neural_runtime": neural_runtime,
        "neural_runtime_amendment_sha256": neural_runtime.get(
            "representation_amendment_sha256"
        ),
        "neural_execution_scope_amendment_sha256": neural_runtime.get(
            "execution_scope_amendment_sha256"
        ),
        "benchmarks": benchmark_audits,
        "extension_result_files": [str(path.resolve()) for path in result_files],
        "tokenizer_manifest": tokenizer_payload,
        "tokenizer_manifest_path": str(args.tokenizer_manifest.resolve()),
        "tokenizer_manifest_sha256": (
            sha256_file(args.tokenizer_manifest)
            if args.tokenizer_manifest.is_file()
            else None
        ),
        "tokenizer_verification": tokenizer_verification,
        "tokenizer_api_usage_probes": tokenizer_api_usage_probes,
        "tokenizer_message_verification": tokenizer_message_verification,
        "neural_dense_manifest": neural_manifest,
        "neural_dense_manifest_path": str(args.neural_manifest.resolve()),
        "cell_identity_manifest": cells,
        "cell_identity_manifest_sha256": canonical_sha256(cells) if cells else None,
        "prompt_version": PROMPT_VERSION,
        "conditions": list(CONDITIONS),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
