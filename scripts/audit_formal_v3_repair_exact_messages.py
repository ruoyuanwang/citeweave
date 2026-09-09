from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from citeweave.formal_request import build_messages, canonical_sha256, task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
EXECUTION = BASE / "formal_v3_identity_corrected_execution"
EXPECTED_RESULT_HASHES = {
    "federated_learning_healthcare_2016_2025": (
        "d5508417680dbb35f5c25cc33d3f3fa62d9fd3428f8fc266d87e6882de8e43b7"
    ),
    "green_hydrogen_electrolysis_2010_2025": (
        "e758136ed4360f38da3d7afde1b987ed45b5fd260869b11ee3a38d3609e13a0f"
    ),
}


class FrozenInProcessTokenizer:
    """Load the frozen tokenizer once for exhaustive, exact request auditing."""

    def __init__(self, manifest: dict[str, Any], manifest_path: Path):
        runtime = manifest_path.parent / "deepseek_v4_tokenizer_runtime"
        encoding_dir = runtime / "encoding"
        sys.path.insert(0, str(encoding_dir))
        from encoding_dsv4 import encode_messages  # type: ignore[import-not-found]

        self._tokenizer = Tokenizer.from_file(str(runtime / "tokenizer.json"))
        self._encode_messages = encode_messages

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        rendered = self._encode_messages(
            messages,
            thinking_mode="chat",
            drop_thinking=True,
            add_default_bos_token=True,
        )
        return self.count(rendered)


def _verify_tokenizer_artifacts(
    manifest: dict[str, Any], manifest_path: Path, tokenizer: FrozenInProcessTokenizer
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for artifact in manifest.get("artifacts") or []:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            reasons.append(f"tokenizer_artifact_drift:{path}")
    for probe in manifest.get("verification_probes") or []:
        if tokenizer.count(str(probe["text"])) != int(probe["expected_tokens"]):
            reasons.append(f"tokenizer_probe_mismatch:{probe['probe_id']}")
    api_spec = manifest.get("api_usage_probe_artifact") or {}
    api_path = manifest_path.parent / str(api_spec.get("path") or "")
    if not api_path.is_file() or sha256_file(api_path) != api_spec.get("sha256"):
        reasons.append("tokenizer_api_usage_probe_artifact_drift")
    else:
        api_probes = read_json(api_path)
        for probe in api_probes.get("records") or []:
            if tokenizer.count_messages(probe["messages"]) != int(probe["prompt_tokens"]):
                reasons.append(f"tokenizer_message_probe_mismatch:{probe['probe_id']}")
    return not reasons, reasons


def _neural_contexts(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest_path = Path(readiness["neural_dense_manifest_path"])
    output: dict[str, dict[str, Any]] = {}
    for artifact in (readiness.get("neural_dense_manifest") or {}).get("artifacts") or []:
        if artifact.get("artifact_type") != "neural_context_sidecar":
            continue
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Neural sidecar drift: {path}")
        sidecar = read_json(path)
        output[str(sidecar["dataset_id"])] = sidecar["contexts"]
    return output


def _panel_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    specifications = (
        (
            "primary",
            EXECUTION / "primary_plan.json",
            EXECUTION / "primary_readiness.json",
            BASE / "formal_v3_identity_corrected_benchmarks",
        ),
        (
            "replication",
            EXECUTION / "replication_plan.json",
            EXECUTION / "replication_readiness.json",
            BASE / "formal_v3_replication_benchmarks",
        ),
        (
            "extension",
            EXECUTION / "extension_plan.json",
            EXECUTION / "extension_readiness.json",
            BASE / "formal_v3_complexity_extension_benchmarks",
        ),
    )
    for panel, plan_path, readiness_path, benchmark_root in specifications:
        plan = read_json(plan_path)
        readiness = read_json(readiness_path)
        conditions = (
            list(readiness["conditions"])
            if panel == "extension"
            else list(plan["conditions"])
        )
        neural = _neural_contexts(readiness) if "flat_neural_dense" in conditions else {}
        if panel == "extension":
            dataset_ids = sorted({str(row["dataset_id"]) for row in plan["jobs"]})
        else:
            dataset_ids = [str(row["dataset_id"]) for row in plan["datasets"]]
        for dataset_id in dataset_ids:
            benchmark_path = benchmark_root / dataset_id / "benchmark.json"
            benchmark = read_json(benchmark_path)
            for task in benchmark["tasks"]:
                for condition in conditions:
                    cells.append(
                        {
                            "panel": panel,
                            "dataset_id": dataset_id,
                            "benchmark_path": benchmark_path,
                            "task": task,
                            "condition": condition,
                            "context_override": (
                                neural[dataset_id][task["item_id"]]
                                if condition == "flat_neural_dense"
                                else None
                            ),
                        }
                    )
    return cells


def _completed_primary_records() -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    reasons: list[str] = []
    results_root = BASE / "formal_v3_runs"
    for dataset_id, expected_hash in EXPECTED_RESULT_HASHES.items():
        path = results_root / dataset_id / "results.json"
        if not path.is_file() or sha256_file(path) != expected_hash:
            reasons.append(f"preserved_result_hash_mismatch:{dataset_id}")
            continue
        payload = read_json(path)
        if payload.get("attempt_log"):
            reasons.append(f"unexpected_attempt_log:{dataset_id}")
        for record in payload.get("records") or []:
            if record.get("status", "complete") != "complete":
                reasons.append(f"noncomplete_preserved_record:{dataset_id}")
                continue
            key = (str(record["item_id"]), str(record["condition"]))
            if key in completed:
                reasons.append(f"duplicate_preserved_record:{key[0]}:{key[1]}")
            completed[key] = record
    if len(completed) != 380:
        reasons.append(f"preserved_complete_count_mismatch:{len(completed)}")
    return completed, reasons


def audit() -> dict[str, Any]:
    manifest_path = (
        BASE
        / "formal_v3_execution_prerequisites/deepseek_v4_tokenizer_manifest.json"
    )
    manifest = read_json(manifest_path)
    tokenizer = FrozenInProcessTokenizer(manifest, manifest_path)
    tokenizer_passed, reasons = _verify_tokenizer_artifacts(
        manifest, manifest_path, tokenizer
    )
    completed, completed_reasons = _completed_primary_records()
    reasons.extend(completed_reasons)
    budget = int(manifest["context_token_budget"])
    identities: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    changed_completed: list[dict[str, str]] = []
    cells = _panel_cells()
    for cell_index, cell in enumerate(cells, start=1):
        if cell_index == 1 or cell_index % 100 == 0 or cell_index == len(cells):
            print(
                json.dumps(
                    {
                        "stage": "exact_message_audit",
                        "assembled_or_attempted": cell_index - 1,
                        "total": len(cells),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        task = cell["task"]
        condition = cell["condition"]
        try:
            messages, context_audit = build_messages(
                task,
                condition,
                context_override=cell["context_override"],
                tokenizer=tokenizer,  # type: ignore[arg-type]
                token_budget=budget,
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "panel": cell["panel"],
                    "item_id": task["item_id"],
                    "condition": condition,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        request_hash = canonical_sha256(messages)
        record = {
            "panel": cell["panel"],
            "dataset_id": cell["dataset_id"],
            "item_id": task["item_id"],
            "condition": condition,
            "task_payload_sha256": task_payload_sha256(task),
            "request_messages_sha256": request_hash,
            "context_sha256": context_audit["context_sha256"],
            "full_context_tokens": context_audit["full_context_tokens"],
            "budgeted_context_tokens": context_audit["budgeted_context_tokens"],
            "prompt_tokens": tokenizer.count_messages(messages),
            "retained_records": context_audit["retained_records"],
            "original_records": context_audit["original_records"],
        }
        identities.append(record)
        old = completed.get((task["item_id"], condition))
        if (
            cell["panel"] == "primary"
            and old is not None
            and old.get("request_messages_sha256") != request_hash
        ):
            changed_completed.append(
                {
                    "item_id": task["item_id"],
                    "condition": condition,
                    "preserved_sha256": str(old.get("request_messages_sha256")),
                    "reconstructed_sha256": request_hash,
                }
            )
    counts = Counter(row["panel"] for row in identities)
    expected_counts = {"primary": 800, "replication": 120, "extension": 672}
    if dict(counts) != expected_counts:
        reasons.append(f"logical_cell_count_mismatch:{dict(counts)}")
    if failures:
        reasons.append(f"exact_message_assembly_failures:{len(failures)}")
    if changed_completed:
        reasons.append(f"completed_request_identity_changes:{len(changed_completed)}")
    over_budget = [
        row
        for row in identities
        if row["budgeted_context_tokens"] > budget
    ]
    if over_budget:
        reasons.append(f"context_budget_violations:{len(over_budget)}")
    duplicate_keys = [
        key
        for key, count in Counter(
            (row["panel"], row["item_id"], row["condition"]) for row in identities
        ).items()
        if count != 1
    ]
    if duplicate_keys:
        reasons.append(f"duplicate_logical_cell_identities:{len(duplicate_keys)}")
    unique_reasons = list(dict.fromkeys(reasons))
    rendered_identities = json.dumps(
        identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema_version": 1,
        "status": "ready_for_repair_qualification" if not unique_reasons else "blocked",
        "blocking_reasons": unique_reasons,
        "scientific_effects_inspected": False,
        "provider_calls": 0,
        "tokenizer_passed": tokenizer_passed,
        "tokenizer_manifest_path": str(manifest_path.resolve()),
        "tokenizer_manifest_sha256": sha256_file(manifest_path),
        "context_token_budget": budget,
        "logical_cells": dict(counts),
        "expected_logical_cells": expected_counts,
        "assembled_cells": len(identities),
        "assembly_failures": failures,
        "context_budget_violations": len(over_budget),
        "maximum_budgeted_context_tokens": max(
            (row["budgeted_context_tokens"] for row in identities), default=0
        ),
        "maximum_prompt_tokens": max(
            (row["prompt_tokens"] for row in identities), default=0
        ),
        "trimmed_cells": sum(
            row["retained_records"] != row["original_records"] for row in identities
        ),
        "preserved_complete_cells": len(completed),
        "completed_request_identity_changes": changed_completed,
        "cell_identity_manifest_sha256": hashlib.sha256(
            rendered_identities.encode("utf-8")
        ).hexdigest(),
        "cell_identity_manifest": identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EXECUTION / "repair_010_exact_message_audit.json",
    )
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cell_identity_manifest"}, ensure_ascii=False, indent=2))
    if result["status"] != "ready_for_repair_qualification":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
