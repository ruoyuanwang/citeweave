from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from .formal_request import task_payload_sha256
from .io import read_json, sha256_file, write_json

_TASK_SEVERITY = {
    "multi_hop_connector": "medium",
    "bridge_counterfactual": "critical",
    "community_role_contrast": "high",
    "hub_removal_resilience": "critical",
    "temporal_structural_shift": "high",
}
_HELDOUT_ISSUES = (
    "evidence_relevance",
    "graph_answer_consistency",
    "causal_overreach",
    "counterevidence_coverage",
)


class OversightHeldoutPacketError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _case_id(item_id: str) -> str:
    return "HO-" + hashlib.sha256(item_id.encode()).hexdigest()[:20]


def _select_tasks(tasks: list[dict[str, Any]], *, dataset_id: str, seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["task_type"] in _TASK_SEVERITY:
            by_type[str(task["task_type"])].append(task)
    if set(by_type) != set(_TASK_SEVERITY):
        raise OversightHeldoutPacketError(f"Dataset {dataset_id} lacks a complex task type")
    ordered_types = sorted(by_type)
    rotation = int(hashlib.sha256(dataset_id.encode()).hexdigest()[:8], 16) % len(
        ordered_types
    )
    three_case_types = {
        ordered_types[rotation], ordered_types[(rotation + 1) % len(ordered_types)]
    }
    selected = []
    for task_type in ordered_types:
        quota = 3 if task_type in three_case_types else 2
        candidates = sorted(by_type[task_type], key=lambda row: str(row["item_id"]))
        if len(candidates) < quota:
            raise OversightHeldoutPacketError(
                f"Dataset {dataset_id} lacks {quota} cases for {task_type}"
            )
        rng = Random(f"{seed}:selection:{dataset_id}:{task_type}")
        rng.shuffle(candidates)
        selected.extend(candidates[:quota])
    if len(selected) != 12:
        raise AssertionError("Held-out selection must contain 12 tasks per dataset")
    return sorted(selected, key=lambda row: str(row["item_id"]))


def freeze_heldout_case_selection(
    *,
    primary_construction_manifest_path: Path,
    replication_construction_manifest_path: Path,
    primary_benchmark_root: Path,
    replication_benchmark_root: Path,
    primary_results_root: Path,
    replication_results_root: Path,
    output_path: Path,
    seed: int = 20260907,
) -> dict[str, Any]:
    """Freeze task identities without reading any model result or human outcome."""
    if output_path.exists():
        raise OversightHeldoutPacketError("Refusing to overwrite held-out selection")
    panels = (
        (
            "primary",
            primary_construction_manifest_path,
            primary_benchmark_root,
            primary_results_root,
        ),
        (
            "replication",
            replication_construction_manifest_path,
            replication_benchmark_root,
            replication_results_root,
        ),
    )
    records = []
    benchmark_receipts = []
    for source_panel, construction_path, benchmark_root, results_root in panels:
        construction = read_json(construction_path)
        if construction.get("status") != "constructed_not_executed":
            raise OversightHeldoutPacketError("Construction manifest status changed")
        for dataset in sorted(
            construction.get("records") or [], key=lambda row: str(row["dataset_id"])
        ):
            dataset_id = str(dataset["dataset_id"])
            benchmark_path = benchmark_root / dataset_id / "benchmark.json"
            if (
                not benchmark_path.is_file()
                or sha256_file(benchmark_path) != dataset["benchmark_sha256"]
            ):
                raise OversightHeldoutPacketError(f"Benchmark hash mismatch: {dataset_id}")
            benchmark = read_json(benchmark_path)
            selected = _select_tasks(
                benchmark.get("tasks") or [], dataset_id=dataset_id, seed=seed
            )
            benchmark_receipts.append(
                {
                    "dataset_id": dataset_id,
                    "source_panel": source_panel,
                    "benchmark_path": str(benchmark_path.resolve()),
                    "benchmark_sha256": sha256_file(benchmark_path),
                    "selected_tasks": 12,
                }
            )
            issue_offset = int(
                hashlib.sha256(f"{seed}:issue:{dataset_id}".encode()).hexdigest()[:8],
                16,
            ) % len(_HELDOUT_ISSUES)
            for index, task in enumerate(selected):
                issue = _HELDOUT_ISSUES[(issue_offset + index) % len(_HELDOUT_ISSUES)]
                severity = _TASK_SEVERITY[str(task["task_type"])]
                risk = {
                    "medium": 0.20,
                    "high": 0.35,
                    "critical": 0.55,
                }[severity]
                records.append(
                    {
                        "case_id": _case_id(str(task["item_id"])),
                        "item_id": task["item_id"],
                        "dataset_id": dataset_id,
                        "source_panel": source_panel,
                        "benchmark_path": str(benchmark_path.resolve()),
                        "benchmark_sha256": sha256_file(benchmark_path),
                        "task_payload_sha256": task_payload_sha256(task),
                        "results_path": str(
                            (results_root / dataset_id / "results.json").resolve()
                        ),
                        "condition": "graph_program",
                        "network": task["network"],
                        "scale": task["scale"],
                        "task_type": task["task_type"],
                        "complexity": task["complexity"],
                        "domain": dataset_id,
                        "issue_type": issue,
                        "severity": severity,
                        "predecision_error_risk": risk,
                        "estimated_default_review_seconds": 45.0
                        + 15.0 * int(task["complexity"]),
                        "descendants": len(task.get("operator_trace") or []),
                    }
                )
    if len(records) != 96 or len({row["case_id"] for row in records}) != 96:
        raise OversightHeldoutPacketError("Held-out selection must contain 96 unique cases")
    issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        issue_counts[row["dataset_id"]][row["issue_type"]] += 1
    if any(
        counts != Counter({issue: 3 for issue in _HELDOUT_ISSUES})
        for counts in issue_counts.values()
    ):
        raise OversightHeldoutPacketError("Each dataset must contain three cases per issue")
    result = {
        "schema_version": 1,
        "status": "heldout_case_selection_frozen_during_machine_execution_before_human_review",
        "selection_reads_model_results": False,
        "machine_effects_inspected_by_selector": False,
        "human_outcomes_inspected": False,
        "source_condition": "graph_program",
        "seed": seed,
        "datasets": 8,
        "cases": 96,
        "cases_per_dataset": 12,
        "issues": list(_HELDOUT_ISSUES),
        "primary_construction_manifest_sha256": sha256_file(
            primary_construction_manifest_path
        ),
        "replication_construction_manifest_sha256": sha256_file(
            replication_construction_manifest_path
        ),
        "benchmark_receipts": benchmark_receipts,
        "records": sorted(records, key=lambda row: str(row["case_id"])),
    }
    write_json(output_path, result)
    return result


def _validate_terminal_audits(paths: list[Path]) -> list[dict[str, str]]:
    receipts = []
    for path in paths:
        audit = read_json(path)
        if (
            audit.get("status") != "terminal"
            or audit.get("terminal_parse_failure_cells") != 0
            or audit.get("retry_cells")
            or audit.get("integrity_reasons")
            or audit.get("complete_cells") != audit.get("expected_cells")
        ):
            raise OversightHeldoutPacketError(f"Formal panel is not cleanly terminal: {path}")
        receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if len(receipts) != 3:
        raise OversightHeldoutPacketError("Primary, replication, and extension audits are required")
    return receipts


def _evidence_sets(task: dict[str, Any], response: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    context = task["contexts"]["graph_program"]
    graph_records = [
        {"evidence_type": "graph_node", **row} for row in context.get("nodes") or []
    ] + [{"evidence_type": "graph_edge", **row} for row in context.get("edges") or []]
    cited = set(map(str, response.get("evidence_ids") or []))
    direct = [
        row
        for row in graph_records
        if str(row.get("evidence_id") or "") in cited
        or str(row.get("node_id") or "") in cited
    ]
    registered = set(map(str, task.get("evidence_ids") or []))
    check_records = [
        row
        for row in graph_records
        if str(row.get("evidence_id") or "") in registered
        or str(row.get("node_id") or "") in registered
    ]
    trace = [
        {"evidence_id": f"TRACE-{index}", "evidence_type": "operator_trace", "value": row}
        for index, row in enumerate(task.get("operator_trace") or [])
    ]
    contract = task["interpretation_contract"]
    interpretation = [
        {
            "evidence_id": "INTERPRETATION-LIMIT",
            "evidence_type": "required_limitation",
            "text": contract["required_limitation"],
        },
        *[
            {
                "evidence_id": f"FORBIDDEN-{index}",
                "evidence_type": "forbidden_inference",
                "text": value,
            }
            for index, value in enumerate(contract.get("forbidden") or [])
        ],
    ]
    first = [*direct, *trace]
    second = [*check_records, *interpretation]
    if not first:
        first = trace
    return first, second


def _gold_verdict(score: dict[str, Any]) -> str:
    if not score.get("answer_exact") or score.get("abstain"):
        return "reject"
    if (
        float(score.get("evidence_precision") or 0.0) >= 1.0 - 1e-12
        and float(score.get("evidence_recall") or 0.0) >= 1.0 - 1e-12
        and score.get("has_required_limitation") is True
    ):
        return "supported"
    return "qualify"


def materialize_heldout_factorial_packets(
    *,
    selection_path: Path,
    terminal_audit_paths: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightHeldoutPacketError("Refusing to overwrite held-out packets")
    selection = read_json(selection_path)
    if (
        selection.get("status")
        != "heldout_case_selection_frozen_during_machine_execution_before_human_review"
        or selection.get("selection_reads_model_results") is not False
        or selection.get("human_outcomes_inspected") is not False
        or selection.get("cases") != 96
    ):
        raise OversightHeldoutPacketError("Held-out selection is not frozen")
    audit_receipts = _validate_terminal_audits(terminal_audit_paths)
    benchmark_cache: dict[str, dict[str, dict[str, Any]]] = {}
    result_cache: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    records = []
    gold_counts: Counter[str] = Counter()
    for selected in selection["records"]:
        benchmark_path = Path(selected["benchmark_path"])
        if sha256_file(benchmark_path) != selected["benchmark_sha256"]:
            raise OversightHeldoutPacketError("Selected benchmark changed")
        key = str(benchmark_path.resolve())
        if key not in benchmark_cache:
            benchmark_cache[key] = {
                task["item_id"]: task for task in read_json(benchmark_path)["tasks"]
            }
        task = benchmark_cache[key].get(selected["item_id"])
        if task is None or task_payload_sha256(task) != selected["task_payload_sha256"]:
            raise OversightHeldoutPacketError("Selected task identity changed")
        results_path = Path(selected["results_path"])
        result_key = str(results_path.resolve())
        if result_key not in result_cache:
            payload = read_json(results_path)
            if payload.get("manifest", {}).get("benchmark_sha256") != selected[
                "benchmark_sha256"
            ]:
                raise OversightHeldoutPacketError("Result manifest benchmark mismatch")
            result_cache[result_key] = {
                (row["item_id"], row["condition"]): row
                for row in payload.get("records") or []
            }
        generation = result_cache[result_key].get(
            (selected["item_id"], selected["condition"])
        )
        if generation is None or generation.get("status", "complete") != "complete":
            raise OversightHeldoutPacketError("Selected model output is not complete")
        if generation.get("task_payload_sha256") != selected["task_payload_sha256"]:
            raise OversightHeldoutPacketError("Selected generation task hash mismatch")
        response = generation.get("response")
        score = generation.get("score")
        if not isinstance(response, dict) or not isinstance(score, dict):
            raise OversightHeldoutPacketError("Selected generation lacks response or score")
        evidence_a, evidence_b = _evidence_sets(task, response)
        case_id = selected["case_id"]
        common = {
            "schema_version": 1,
            "packet_type": "prospective_factorial_graph_claim_audit",
            "case_id": case_id,
            "dataset_id": selected["dataset_id"],
            "domain": selected["domain"],
            "issue_type": selected["issue_type"],
            "severity": selected["severity"],
            "question": task["question"],
            "candidate_response": response,
            "operator_trace": task["operator_trace"],
            "generation_condition_hidden": True,
            "gold_answer_hidden": True,
            "review_focus": selected["issue_type"],
            "response_schema": {
                "verdict": ["supported", "qualify", "reject", "abstain"],
                "decisive_evidence_ids": "list[string]",
                "invalid_operator_steps": "list[integer]",
                "failure_mode": "string",
                "minimal_rewrite": "string|null",
                "rationale": "string",
            },
        }
        standard = {
            **common,
            "evidence_set_a": [*evidence_a, *evidence_b],
            "evidence_set_b": [],
            "review_instructions": [
                "Read the candidate claim before checking the flat evidence list.",
                "Choose supported, qualify, reject, or abstain and identify decisive evidence.",
            ],
        }
        swap = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0
        adversarial = {
            **common,
            "evidence_set_a": evidence_b if swap else evidence_a,
            "evidence_set_b": evidence_a if swap else evidence_b,
            "review_instructions": [
                "Audit both evidence sets independently before deciding.",
                "Do not assume which evidence set supports or challenges the candidate.",
                "Choose supported, qualify, reject, or abstain and identify decisive evidence.",
            ],
        }
        standard_path = output_root / "standard" / f"{case_id}.json"
        adversarial_path = output_root / "adversarial" / f"{case_id}.json"
        internal_path = output_root / "internal" / f"{case_id}.json"
        write_json(standard_path, standard)
        write_json(adversarial_path, adversarial)
        gold_verdict = _gold_verdict(score)
        write_json(
            internal_path,
            {
                "schema_version": 1,
                "case_id": case_id,
                "item_id": selected["item_id"],
                "gold_verdict": gold_verdict,
                "deterministic_score": score,
                "benchmark_answer_sha256": _canonical_hash(task["answer"]),
                "role_map": {
                    "evidence_set_a": "registered_checks" if swap else "candidate_and_trace",
                    "evidence_set_b": "candidate_and_trace" if swap else "registered_checks",
                },
                "human_outcomes_inspected": False,
            },
        )
        gold_counts[gold_verdict] += 1
        records.append(
            {
                **{
                    key: selected[key]
                    for key in (
                        "case_id",
                        "item_id",
                        "dataset_id",
                        "domain",
                        "issue_type",
                        "severity",
                        "predecision_error_risk",
                        "estimated_default_review_seconds",
                        "descendants",
                    )
                },
                "candidate_output_sha256": _canonical_hash(response),
                "generation_record_sha256": _canonical_hash(generation),
                "standard_public_path": str(standard_path.relative_to(output_root)),
                "standard_public_sha256": sha256_file(standard_path),
                "adversarial_public_path": str(
                    adversarial_path.relative_to(output_root)
                ),
                "adversarial_public_sha256": sha256_file(adversarial_path),
                "internal_path": str(internal_path.relative_to(output_root)),
                "internal_sha256": sha256_file(internal_path),
            }
        )
    result = {
        "schema_version": 1,
        "status": "prospective_factorial_packets_frozen_before_review",
        "study_role": "prospective_heldout_candidate_outputs",
        "human_outcomes_inspected": False,
        "selection_sha256": sha256_file(selection_path),
        "terminal_audits": audit_receipts,
        "datasets": 8,
        "cases": 96,
        "same_evidence_across_packet_arms": True,
        "generation_condition_blinded": True,
        "gold_answer_blinded": True,
        "internal_gold_verdict_counts": dict(sorted(gold_counts.items())),
        "records": records,
    }
    write_json(output_root / "manifest.json", result)
    return result
