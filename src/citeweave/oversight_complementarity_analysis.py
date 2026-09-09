from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file

VERDICTS = {"supported", "qualify", "reject", "abstain"}


class OversightComplementarityError(ValueError):
    pass


def _exact_sign_flip(effects: list[float]) -> dict[str, Any]:
    if not effects:
        raise OversightComplementarityError("At least one dataset effect is required")
    observed = statistics.fmean(effects)
    null = [
        statistics.fmean(sign * value for sign, value in zip(signs, effects, strict=True))
        for signs in itertools.product((-1.0, 1.0), repeat=len(effects))
    ]
    return {
        "estimate": observed,
        "datasets": len(effects),
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in null)
        / len(null),
        "enumerated_sign_patterns": len(null),
    }


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def analyze_complementarity_cases(
    cases: list[dict[str, Any]],
    *,
    minimum_datasets: int = 8,
    cases_per_dataset: int = 12,
) -> dict[str, Any]:
    """Compare final oversight with unaudited AI and one primary judgment.

    Primary accuracy is averaged within case before cases and datasets are averaged,
    preventing intentionally dual-reviewed critical cases from receiving extra weight.
    """

    if len({str(row.get("case_id")) for row in cases}) != len(cases):
        raise OversightComplementarityError("Case identities must be unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewer_rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    case_results = []
    for row in cases:
        dataset_id = str(row.get("dataset_id") or "")
        gold = str(row.get("gold_verdict") or "")
        ai = str(row.get("ai_verdict") or "")
        final = str(row.get("final_verdict") or "")
        primaries = list(row.get("primary_decisions") or [])
        if not dataset_id or {gold, ai, final} - VERDICTS or len(primaries) not in {1, 2}:
            raise OversightComplementarityError("Invalid case verdict or primary team")
        primary_correct = []
        primary_seconds = 0.0
        for primary in primaries:
            reviewer = str(primary.get("reviewer_id") or "")
            verdict = str(primary.get("verdict") or "")
            seconds = float(primary.get("review_seconds") or 0.0)
            if not reviewer or verdict not in VERDICTS or seconds <= 0:
                raise OversightComplementarityError("Invalid primary decision")
            correct = int(verdict == gold)
            primary_correct.append(correct)
            primary_seconds += seconds
            reviewer_rows[reviewer].append((correct, seconds))
        team_seconds = float(row.get("team_review_seconds") or 0.0)
        if team_seconds < primary_seconds or team_seconds <= 0:
            raise OversightComplementarityError("Team time cannot be below primary time")
        result = {
            "case_id": str(row["case_id"]),
            "dataset_id": dataset_id,
            "ai_correct": int(ai == gold),
            "mean_primary_correct": statistics.fmean(primary_correct),
            "team_correct": int(final == gold),
            "primary_decisions": len(primaries),
            "team_review_seconds": team_seconds,
            "team_rescue": int(final == gold and any(value == 0 for value in primary_correct)),
            "team_harm": int(final != gold and any(value == 1 for value in primary_correct)),
        }
        grouped[dataset_id].append(result)
        case_results.append(result)

    if len(grouped) != minimum_datasets or any(
        len(rows) != cases_per_dataset for rows in grouped.values()
    ):
        raise OversightComplementarityError(
            "Complementarity analysis requires the exact balanced held-out panel"
        )

    dataset_effects = []
    for dataset_id, rows in sorted(grouped.items()):
        ai_accuracy = statistics.fmean(row["ai_correct"] for row in rows)
        primary_accuracy = statistics.fmean(
            row["mean_primary_correct"] for row in rows
        )
        team_accuracy = statistics.fmean(row["team_correct"] for row in rows)
        dataset_effects.append(
            {
                "dataset_id": dataset_id,
                "unaudited_ai_accuracy": ai_accuracy,
                "case_averaged_single_primary_accuracy": primary_accuracy,
                "final_team_accuracy": team_accuracy,
                "team_minus_unaudited_ai": team_accuracy - ai_accuracy,
                "team_minus_single_primary": team_accuracy - primary_accuracy,
            }
        )

    effects = {
        "C1_team_minus_unaudited_ai": [
            row["team_minus_unaudited_ai"] for row in dataset_effects
        ],
        "C2_team_minus_single_primary": [
            row["team_minus_single_primary"] for row in dataset_effects
        ],
    }
    tests = {name: _exact_sign_flip(values) for name, values in effects.items()}
    adjusted = _holm(
        {name: row["p_value_one_sided"] for name, row in tests.items()}
    )
    for name, row in tests.items():
        row["holm_adjusted_p"] = adjusted[name]
        row["passed"] = row["estimate"] > 0 and adjusted[name] < 0.05

    reviewer_diagnostics = {
        reviewer: {
            "decisions": len(rows),
            "accuracy": statistics.fmean(correct for correct, _ in rows),
            "correct_per_review_minute": sum(correct for correct, _ in rows)
            / (sum(seconds for _, seconds in rows) / 60.0),
        }
        for reviewer, rows in sorted(reviewer_rows.items())
    }
    total_team_seconds = sum(row["team_review_seconds"] for row in case_results)
    return {
        "schema_version": 1,
        "status": "complementarity_baselines_analyzed",
        "cases": len(case_results),
        "datasets": len(grouped),
        "dataset_weighting": "equal_dataset",
        "single_primary_weighting": "mean_within_case_then_equal_case",
        "registered_tests": tests,
        "joint_complementarity_passed": all(row["passed"] for row in tests.values()),
        "dataset_effects": dataset_effects,
        "pooled_diagnostics": {
            "unaudited_ai_accuracy": statistics.fmean(
                row["ai_correct"] for row in case_results
            ),
            "case_averaged_single_primary_accuracy": statistics.fmean(
                row["mean_primary_correct"] for row in case_results
            ),
            "final_team_accuracy": statistics.fmean(
                row["team_correct"] for row in case_results
            ),
            "team_rescue_cases": sum(row["team_rescue"] for row in case_results),
            "team_harm_cases": sum(row["team_harm"] for row in case_results),
            "final_team_correct_per_review_minute": sum(
                row["team_correct"] for row in case_results
            )
            / (total_team_seconds / 60.0),
        },
        "reviewer_diagnostics_descriptive_only": reviewer_diagnostics,
        "case_results": case_results,
        "interpretation_guard": (
            "This tests the final audited decision against unaudited model acceptance "
            "and a case-averaged first-pass assisted-human judgment. It is not a "
            "human-only writing comparison, and reviewer-level maxima are descriptive."
        ),
    }


def load_complementarity_cases(
    *,
    packet_manifest_path: Path,
    assignment_root: Path,
    primary_validation_path: Path,
    final_outcomes_path: Path,
) -> list[dict[str, Any]]:
    packet_manifest = read_json(packet_manifest_path)
    assignment_path = assignment_root / "assignment_manifest.json"
    assignment = read_json(assignment_path)
    primary = read_json(primary_validation_path)
    outcomes = read_json(final_outcomes_path)
    if assignment.get("packet_manifest_sha256") != sha256_file(packet_manifest_path):
        raise OversightComplementarityError("Assignment/packet identity mismatch")
    if primary.get("assignment_manifest_sha256") != sha256_file(assignment_path):
        raise OversightComplementarityError("Primary/assignment identity mismatch")
    if outcomes.get("assignment_manifest_sha256") != sha256_file(assignment_path):
        raise OversightComplementarityError("Outcome/assignment identity mismatch")
    if outcomes.get("primary_validation_sha256") != sha256_file(primary_validation_path):
        raise OversightComplementarityError("Outcome/primary identity mismatch")

    packet_root = packet_manifest_path.resolve().parent
    packet_index = {
        str(row["case_id"]): row for row in packet_manifest.get("records") or []
    }
    route_index = {
        str(row["case_id"]): row for row in assignment.get("case_routes") or []
    }
    primary_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary.get("validated_results") or []:
        primary_index[str(row["packet_id"])].append(row)
    outcome_index = {
        str(row["case_id"]): row for row in outcomes.get("records") or []
    }
    diagnostic_index = {
        str(row["case_id"]): row for row in outcomes.get("diagnostics") or []
    }
    case_ids = set(packet_index)
    if not case_ids or not (
        case_ids == set(route_index) == set(primary_index) == set(outcome_index) == set(diagnostic_index)
    ):
        raise OversightComplementarityError("Complementarity case identities differ")

    cases = []
    for case_id in sorted(case_ids):
        packet_row = packet_index[case_id]
        internal_path = packet_root / str(packet_row["internal_path"])
        standard_path = packet_root / str(packet_row["standard_public_path"])
        if sha256_file(internal_path) != packet_row["internal_sha256"]:
            raise OversightComplementarityError("Private gold hash mismatch")
        if sha256_file(standard_path) != packet_row["standard_public_sha256"]:
            raise OversightComplementarityError("Public packet hash mismatch")
        gold = str(read_json(internal_path)["gold_verdict"])
        candidate = read_json(standard_path)["candidate_response"]
        ai_verdict = "abstain" if candidate.get("abstain") is True else "supported"
        route = route_index[case_id]
        primaries = primary_index[case_id]
        if {str(row["reviewer_code"]) for row in primaries} != set(
            map(str, route.get("primary_reviewers") or [])
        ):
            raise OversightComplementarityError("Primary reviewer identities differ")
        diagnostic = diagnostic_index[case_id]
        if diagnostic.get("gold_verdict") != gold:
            raise OversightComplementarityError("Final diagnostic gold mismatch")
        cases.append(
            {
                "case_id": case_id,
                "dataset_id": route["dataset_id"],
                "gold_verdict": gold,
                "ai_verdict": ai_verdict,
                "final_verdict": diagnostic["final_verdict"],
                "team_review_seconds": outcome_index[case_id]["review_seconds"],
                "primary_decisions": [
                    {
                        "reviewer_id": row["reviewer_code"],
                        "verdict": row["verdict"],
                        "review_seconds": row["review_seconds"],
                    }
                    for row in primaries
                ],
            }
        )
    return cases
