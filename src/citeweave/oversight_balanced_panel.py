from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .formal_request import task_payload_sha256
from .io import read_json, sha256_file, write_json
from .oversight_heldout_packets import (
    OversightHeldoutPacketError,
    _canonical_hash,
    _evidence_sets,
    _gold_verdict,
    _validate_terminal_audits,
)

_CONDITION_PRIORITY = (
    "graph_program",
    "flat_program",
    "operator_only",
    "graph_hierarchical_retrieval_v2",
    "graph_query_retrieval",
    "flat_hybrid",
    "flat_neural_dense",
    "flat_bm25",
)


def _choose_reject_ids(rows: list[dict[str, Any]], *, seed: int) -> set[str]:
    if len(rows) != 12:
        raise OversightHeldoutPacketError("Each dataset must retain exactly 12 tasks")
    mandatory = {
        str(row["case_id"])
        for row in rows
        if set(row["available_gold_verdicts"]) == {"reject"}
    }
    eligible = [
        str(row["case_id"])
        for row in rows
        if "reject" in set(row["available_gold_verdicts"])
        and str(row["case_id"]) not in mandatory
    ]
    needed = 6 - len(mandatory)
    if needed < 0 or len(eligible) < needed:
        raise OversightHeldoutPacketError("A balanced six-reject subset is infeasible")
    row_index = {str(row["case_id"]): row for row in rows}
    candidates = []
    for extra in itertools.combinations(sorted(eligible), needed):
        chosen = mandatory | set(extra)
        issues = Counter(row_index[case_id]["issue_type"] for case_id in chosen)
        issue_imbalance = sum((2 * issues[name] - 3) ** 2 for name in sorted(issues))
        task_types = {row_index[case_id]["task_type"] for case_id in chosen}
        tie = hashlib.sha256(
            f"{seed}:reject-subset:{'|'.join(sorted(chosen))}".encode()
        ).hexdigest()
        candidates.append((issue_imbalance, -len(task_types), tie, chosen))
    return min(candidates, key=lambda value: value[:3])[3]


def _choose_generation(
    generations: list[dict[str, Any]], *, target_verdict: str
) -> dict[str, Any]:
    eligible = [
        row for row in generations if _gold_verdict(row["score"]) == target_verdict
    ]
    if not eligible:
        raise OversightHeldoutPacketError(
            f"No real candidate output has target verdict {target_verdict}"
        )
    priority = {condition: index for index, condition in enumerate(_CONDITION_PRIORITY)}
    return min(
        eligible,
        key=lambda row: (
            priority.get(str(row["condition"]), len(priority)),
            str(row["condition"]),
        ),
    )


def build_balanced_heldout_selection(
    *, base_selection_path: Path, output_path: Path, seed: int = 20260908
) -> dict[str, Any]:
    if output_path.exists():
        raise OversightHeldoutPacketError("Refusing to overwrite balanced selection")
    base = read_json(base_selection_path)
    if (
        base.get("status")
        != "heldout_case_selection_frozen_during_machine_execution_before_human_review"
        or base.get("cases") != 96
        or base.get("human_outcomes_inspected") is not False
    ):
        raise OversightHeldoutPacketError("Base task selection is not valid")
    result_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    enriched = []
    for selected in base["records"]:
        results_path = Path(selected["results_path"])
        key = str(results_path.resolve())
        if key not in result_cache:
            payload = read_json(results_path)
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in payload.get("records") or []:
                if row.get("status", "complete") == "complete":
                    grouped[str(row["item_id"])].append(row)
            result_cache[key] = grouped
        generations = result_cache[key].get(str(selected["item_id"])) or []
        valid = [
            row
            for row in generations
            if row.get("task_payload_sha256") == selected["task_payload_sha256"]
            and isinstance(row.get("score"), dict)
            and isinstance(row.get("response"), dict)
        ]
        verdicts = sorted({_gold_verdict(row["score"]) for row in valid})
        if not valid or not set(verdicts) <= {"qualify", "reject"}:
            raise OversightHeldoutPacketError("Candidate output set is invalid")
        enriched.append(
            {
                **selected,
                "task_type": selected["task_type"],
                "available_gold_verdicts": verdicts,
                "_generations": valid,
            }
        )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_dataset[str(row["dataset_id"])].append(row)
    if len(by_dataset) != 8:
        raise OversightHeldoutPacketError("Balanced panel requires eight datasets")

    records = []
    condition_counts: Counter[str] = Counter()
    for dataset_id, rows in sorted(by_dataset.items()):
        reject_ids = _choose_reject_ids(rows, seed=seed)
        for row in rows:
            target = "reject" if row["case_id"] in reject_ids else "qualify"
            generation = _choose_generation(
                row.pop("_generations"), target_verdict=target
            )
            condition = str(generation["condition"])
            condition_counts[condition] += 1
            records.append(
                {
                    **{key: value for key, value in row.items() if key != "condition"},
                    "selected_condition": condition,
                    "target_gold_verdict": target,
                    "generation_record_sha256": _canonical_hash(generation),
                }
            )
    gold = Counter(row["target_gold_verdict"] for row in records)
    if gold != Counter({"qualify": 48, "reject": 48}):
        raise OversightHeldoutPacketError("Balanced selection must be exactly 48/48")
    result = {
        "schema_version": 2,
        "status": "outcome_balanced_real_candidate_selection_frozen_before_human_review",
        "study_role": "fixed_balanced_benchmark_for_randomized_human_oversight",
        "selection_reads_model_results": True,
        "selection_reads_human_outcomes": False,
        "human_outcomes_inspected": False,
        "base_task_selection_sha256": sha256_file(base_selection_path),
        "seed": seed,
        "datasets": 8,
        "cases": 96,
        "cases_per_dataset": 12,
        "gold_target_per_dataset": {"qualify": 6, "reject": 6},
        "condition_priority": list(_CONDITION_PRIORITY),
        "selected_condition_counts": dict(sorted(condition_counts.items())),
        "records": sorted(records, key=lambda row: str(row["case_id"])),
        "scope_guard": (
            "Result-aware class balancing occurs before human outcomes. Randomized "
            "packet/routing effects apply to this fixed benchmark, not natural output prevalence."
        ),
    }
    write_json(output_path, result)
    return result


def materialize_balanced_heldout_packets(
    *,
    selection_path: Path,
    terminal_audit_paths: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightHeldoutPacketError("Refusing to overwrite balanced packets")
    selection = read_json(selection_path)
    if (
        selection.get("status")
        != "outcome_balanced_real_candidate_selection_frozen_before_human_review"
        or selection.get("selection_reads_human_outcomes") is not False
        or selection.get("human_outcomes_inspected") is not False
        or selection.get("cases") != 96
    ):
        raise OversightHeldoutPacketError("Balanced selection is not frozen")
    audit_receipts = _validate_terminal_audits(terminal_audit_paths)
    benchmark_cache: dict[str, dict[str, dict[str, Any]]] = {}
    result_cache: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    records = []
    private_gold: Counter[str] = Counter()
    private_conditions: Counter[str] = Counter()
    for selected in selection["records"]:
        benchmark_path = Path(selected["benchmark_path"])
        if sha256_file(benchmark_path) != selected["benchmark_sha256"]:
            raise OversightHeldoutPacketError("Selected benchmark changed")
        benchmark_key = str(benchmark_path.resolve())
        if benchmark_key not in benchmark_cache:
            benchmark_cache[benchmark_key] = {
                task["item_id"]: task for task in read_json(benchmark_path)["tasks"]
            }
        task = benchmark_cache[benchmark_key].get(selected["item_id"])
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
        condition = selected["selected_condition"]
        generation = result_cache[result_key].get((selected["item_id"], condition))
        if (
            generation is None
            or generation.get("status", "complete") != "complete"
            or generation.get("task_payload_sha256") != selected["task_payload_sha256"]
            or _canonical_hash(generation) != selected["generation_record_sha256"]
        ):
            raise OversightHeldoutPacketError("Selected generation identity mismatch")
        response, score = generation.get("response"), generation.get("score")
        if not isinstance(response, dict) or not isinstance(score, dict):
            raise OversightHeldoutPacketError("Selected generation lacks response or score")
        gold_verdict = _gold_verdict(score)
        if gold_verdict != selected["target_gold_verdict"]:
            raise OversightHeldoutPacketError("Selected gold verdict changed")
        evidence_a, evidence_b = _evidence_sets(task, response)
        case_id = selected["case_id"]
        common = {
            "schema_version": 2,
            "packet_type": "balanced_real_candidate_graph_claim_audit",
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
        write_json(
            internal_path,
            {
                "schema_version": 2,
                "case_id": case_id,
                "item_id": selected["item_id"],
                "selected_condition": condition,
                "gold_verdict": gold_verdict,
                "deterministic_score": score,
                "benchmark_answer_sha256": _canonical_hash(task["answer"]),
                "role_map": {
                    "evidence_set_a": (
                        "registered_checks" if swap else "candidate_and_trace"
                    ),
                    "evidence_set_b": (
                        "candidate_and_trace" if swap else "registered_checks"
                    ),
                },
                "human_outcomes_inspected": False,
            },
        )
        private_gold[gold_verdict] += 1
        private_conditions[condition] += 1
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
    if private_gold != Counter({"qualify": 48, "reject": 48}):
        raise OversightHeldoutPacketError("Materialized gold is not exactly balanced")
    result = {
        "schema_version": 2,
        "status": "prospective_factorial_packets_frozen_before_review",
        "study_role": "fixed_outcome_balanced_real_candidate_benchmark",
        "human_outcomes_inspected": False,
        "selection_reads_model_results": True,
        "selection_sha256": sha256_file(selection_path),
        "terminal_audits": audit_receipts,
        "datasets": 8,
        "cases": 96,
        "same_evidence_across_packet_arms": True,
        "generation_condition_blinded": True,
        "gold_answer_blinded": True,
        "gold_distribution_hidden_from_reviewers": True,
        "records": records,
        "scope_guard": selection["scope_guard"],
    }
    write_json(output_root / "manifest.json", result)
    write_json(
        output_root / "construction_audit_private.json",
        {
            "schema_version": 1,
            "status": "private_pre_review_construction_audit",
            "manifest_sha256": sha256_file(output_root / "manifest.json"),
            "gold_verdict_counts": dict(sorted(private_gold.items())),
            "selected_condition_counts": dict(sorted(private_conditions.items())),
            "human_outcomes_inspected": False,
        },
    )
    return result
