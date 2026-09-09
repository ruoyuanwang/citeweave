from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.robust_graph_synthesis import TASK_TYPES
from citeweave.token_budget import CommandTokenizer, canonical_context_json

ROOT = Path(__file__).resolve().parents[1]


def audit(
    protocol_path: Path,
    freeze_path: Path,
    benchmark_root: Path,
    tokenizer_manifest_path: Path | None = None,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    reasons: list[str] = []
    protocol_hash = sha256_file(protocol_path)
    if freeze.get("sha256") != protocol_hash:
        reasons.append("protocol_freeze_hash_mismatch")
    if protocol.get("status") != "frozen_after_deterministic_task_construction_before_any_model_outcomes":
        reasons.append("protocol_status_invalid")
    for label, binding in (protocol.get("implementation_bindings") or {}).items():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            reasons.append(f"implementation_hash_mismatch:{label}")

    manifest_path = benchmark_root / "construction_manifest.json"
    manifest = read_json(manifest_path)
    expected_manifest_hash = protocol["source_bindings"]["construction_manifest"]["sha256"]
    if sha256_file(manifest_path) != expected_manifest_hash:
        reasons.append("construction_manifest_hash_mismatch")
    if manifest.get("topics") != 8 or manifest.get("tasks") != 40:
        reasons.append("panel_size_mismatch")
    if manifest.get("planned_calls") != 200:
        reasons.append("planned_call_count_mismatch")

    expected_conditions = set(protocol["panel"]["conditions"])
    context_sizes: dict[str, list[int]] = {condition: [] for condition in expected_conditions}
    context_tokens: dict[str, list[int]] = {condition: [] for condition in expected_conditions}
    tokenizer = None
    if tokenizer_manifest_path is not None:
        tokenizer_manifest = read_json(tokenizer_manifest_path)
        expected_tokenizer_hash = protocol["source_bindings"]["tokenizer_manifest"]["sha256"]
        if sha256_file(tokenizer_manifest_path) != expected_tokenizer_hash:
            reasons.append("tokenizer_manifest_hash_mismatch")
        tokenizer = CommandTokenizer(
            tokenizer_manifest,
            manifest_dir=tokenizer_manifest_path.parent,
        )
        if not tokenizer.verify(tokenizer_manifest.get("verification_probes") or [])["passed"]:
            reasons.append("tokenizer_probe_failure")
    task_records = []
    for record in manifest.get("records") or []:
        path = Path(record["benchmark"])
        if not path.is_absolute():
            path = benchmark_root / record["dataset_id"] / "benchmark.json"
        if not path.is_file() or sha256_file(path) != record.get("benchmark_sha256"):
            reasons.append(f"benchmark_hash_mismatch:{record.get('dataset_id')}")
            continue
        benchmark = read_json(path)
        tasks = benchmark.get("tasks") or []
        observed_types = [task.get("task_type") for task in tasks]
        if sorted(observed_types) != sorted(TASK_TYPES):
            reasons.append(f"task_type_coverage_mismatch:{record['dataset_id']}")
        for task in tasks:
            contexts = task.get("contexts") or {}
            if set(contexts) != expected_conditions:
                reasons.append(f"condition_coverage_mismatch:{task.get('item_id')}")
                continue
            graph_program = contexts["graph_program"]
            flat_program = contexts["flat_program"]
            if graph_program.get("operator_trace") != flat_program.get("derived_rows"):
                reasons.append(f"program_trace_parity_failure:{task['item_id']}")
            if graph_program.get("evidence_rows") != flat_program.get("rows"):
                reasons.append(f"program_raw_evidence_parity_failure:{task['item_id']}")
            if "operator_trace" in contexts["graph_hierarchical_retrieval_v2"]:
                reasons.append(f"hierarchical_answer_trace_leak:{task['item_id']}")
            if "operator_trace" in contexts["flat_bm25"]:
                reasons.append(f"flat_answer_trace_leak:{task['item_id']}")
            raw_ids = {
                row["evidence_id"] for row in graph_program.get("evidence_rows") or []
            }
            if not set(task.get("evidence_ids") or []) <= raw_ids:
                reasons.append(f"gold_evidence_not_visible_to_program:{task['item_id']}")
            if len(task.get("evidence_ids") or []) > 8:
                reasons.append(f"gold_evidence_exceeds_prompt_cap:{task['item_id']}")
            if len(contexts["flat_bm25"].get("rows") or []) > protocol["panel"]["context_record_budget"]:
                reasons.append(f"flat_record_budget_exceeded:{task['item_id']}")
            for condition in (
                "graph_hierarchical_retrieval_v2",
                "graph_program",
                "flat_program",
            ):
                context_rows = (
                    contexts[condition].get("rows")
                    if condition == "flat_program"
                    else contexts[condition].get("evidence_rows")
                ) or []
                if len(context_rows) > protocol["panel"]["context_record_budget"]:
                    reasons.append(f"program_record_budget_exceeded:{task['item_id']}:{condition}")
            for condition, context in contexts.items():
                context_sizes[condition].append(
                    len(json.dumps(context, ensure_ascii=False, sort_keys=True))
                )
                if tokenizer is not None:
                    tokens = tokenizer.count(canonical_context_json(context))
                    context_tokens[condition].append(tokens)
                    if tokens > int(protocol["panel"]["context_token_budget"]):
                        reasons.append(
                            f"context_token_budget_exceeded:{task['item_id']}:{condition}:{tokens}"
                        )
            task_records.append(
                {
                    "item_id": task["item_id"],
                    "task_type": task["task_type"],
                    "answer_fields": len(task["answer"]),
                    "evidence_ids": len(task["evidence_ids"]),
                }
            )

    if len(task_records) != 40:
        reasons.append("audited_task_count_mismatch")
    difficulty = manifest.get("difficulty_diagnostic") or {}
    if difficulty.get("interior_rate_fields", 0) <= 0 or difficulty.get("distinct_rates", 0) < 3:
        reasons.append("continuous_difficulty_not_demonstrated")
    size_summary = {
        condition: {
            "minimum_characters": min(values) if values else 0,
            "maximum_characters": max(values) if values else 0,
            "mean_characters": sum(values) / len(values) if values else 0.0,
        }
        for condition, values in sorted(context_sizes.items())
    }
    token_summary = {
        condition: {
            "minimum_tokens": min(values) if values else None,
            "maximum_tokens": max(values) if values else None,
            "mean_tokens": sum(values) / len(values) if values else None,
        }
        for condition, values in sorted(context_tokens.items())
    }
    return {
        "schema_version": 1,
        "status": "ready_for_development_execution" if not reasons else "blocked",
        "confirmatory": False,
        "protocol_sha256": protocol_hash,
        "freeze_sha256": sha256_file(freeze_path),
        "construction_manifest_sha256": sha256_file(manifest_path),
        "topics": manifest.get("topics"),
        "tasks": len(task_records),
        "planned_calls": manifest.get("planned_calls"),
        "difficulty_diagnostic": difficulty,
        "context_size_diagnostic": size_summary,
        "context_token_diagnostic": token_summary,
        "parity_checks": {
            "flat_program_and_graph_program_share_raw_evidence": not any(
                reason.startswith("program_raw_evidence_parity_failure") for reason in reasons
            ),
            "flat_program_and_graph_program_share_operator_trace": not any(
                reason.startswith("program_trace_parity_failure") for reason in reasons
            ),
            "retrieval_conditions_hide_operator_trace": not any(
                "answer_trace_leak" in reason for reason in reasons
            ),
        },
        "blocking_reasons": reasons,
        "task_records": task_records,
        "execution_note": "Provider execution requires a separate live balance preflight.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.protocol,
        args.freeze,
        args.benchmark_root,
        args.tokenizer_manifest,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ready_for_development_execution":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
