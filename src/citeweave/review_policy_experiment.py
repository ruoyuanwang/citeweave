from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

from .review_learning import PolicyReplayObservation, audit_policy_promotion

REVIEW_POLICY_CONDITIONS = (
    "always_review",
    "static_risk",
    "raw_memory_prompt",
    "review_compiler_active",
)


@dataclass(frozen=True)
class ReviewPolicyOutcome:
    case_id: str
    dataset_id: str
    condition: Literal[
        "always_review",
        "static_risk",
        "raw_memory_prompt",
        "review_compiler_active",
    ]
    sequence_index: int
    severity: Literal["low", "medium", "high", "critical"]
    predecision_error_risk: float
    original_correct: bool
    final_correct: bool
    route: Literal["human_review", "auto_accept", "auto_correct"]
    review_seconds: float
    issue_type: str
    active_feedback_ids: tuple[str, ...] = ()
    semantic_neighbor: bool = False
    unrelated: bool = False


def _exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    tail = (
        sum(
            math.comb(discordant, value)
            for value in range(min(left_only, right_only) + 1)
        )
        / (2**discordant)
        if discordant
        else 0.5
    )
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_value_two_sided": min(1.0, 2.0 * tail),
    }


def _exact_cluster_signflip(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cluster sign-flip requires dataset effects")
    observed = sum(values) / len(values)
    draws = [
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return {
        "estimate": observed,
        "clusters": len(values),
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in draws)
        / len(draws),
        "minimum_attainable_p": 1.0 / len(draws),
    }


def _cluster_bootstrap_difference(
    rows: list[tuple[ReviewPolicyOutcome, ReviewPolicyOutcome]],
    *,
    metric: Callable[[ReviewPolicyOutcome, ReviewPolicyOutcome], float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters = sorted({left.dataset_id for left, _ in rows})
    grouped = {
        cluster: [pair for pair in rows if pair[0].dataset_id == cluster]
        for cluster in clusters
    }

    cluster_effects = {
        cluster: sum(metric(left, right) for left, right in pairs) / len(pairs)
        for cluster, pairs in grouped.items()
    }

    generator = Random(seed)
    draws = []
    for _ in range(samples):
        selected = generator.choices(clusters, k=len(clusters))
        draws.append(
            sum(cluster_effects[cluster] for cluster in selected) / len(selected)
        )
    draws.sort()
    low = draws[int(0.025 * (samples - 1))]
    high = draws[int(0.975 * (samples - 1))]
    return {
        "estimate": sum(cluster_effects.values()) / len(cluster_effects),
        "ci_low": low,
        "ci_high": high,
        "clusters": len(clusters),
        "cluster_weighting": "equal_dataset",
        "dataset_effects": cluster_effects,
    }


def _aurc(rows: list[ReviewPolicyOutcome]) -> float:
    ordered = sorted(rows, key=lambda row: (row.predecision_error_risk, row.case_id))
    errors = 0
    selective_risks = []
    for index, row in enumerate(ordered, start=1):
        errors += not row.original_correct
        selective_risks.append(errors / index)
    return sum(selective_risks) / len(selective_risks)


def _correction_lag(
    rows: list[ReviewPolicyOutcome], *, stability_window: int
) -> dict[str, Any]:
    by_feedback: dict[str, list[ReviewPolicyOutcome]] = defaultdict(list)
    for row in rows:
        if not row.semantic_neighbor:
            continue
        for feedback_id in row.active_feedback_ids:
            by_feedback[feedback_id].append(row)
    observed = []
    censored = 0
    for feedback_rows in by_feedback.values():
        ordered = sorted(feedback_rows, key=lambda row: row.sequence_index)
        lag = next(
            (
                start + 1
                for start in range(len(ordered) - stability_window + 1)
                if all(
                    row.final_correct
                    for row in ordered[start : start + stability_window]
                )
            ),
            None,
        )
        if lag is None:
            censored += 1
        else:
            observed.append(lag)
    return {
        "feedback_signals": len(by_feedback),
        "stability_window": stability_window,
        "observed_lags": len(observed),
        "censored_lags": censored,
        "mean_observed_lag": sum(observed) / len(observed) if observed else None,
    }


def analyze_review_policy_panel(
    outcomes: list[ReviewPolicyOutcome],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260820,
    stability_window: int = 3,
) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("Review policy panel is empty")
    indexed = {(row.case_id, row.condition): row for row in outcomes}
    if len(indexed) != len(outcomes):
        raise ValueError("Duplicate case-condition outcomes")
    case_ids = sorted({row.case_id for row in outcomes})
    expected = {
        (case_id, condition)
        for case_id in case_ids
        for condition in REVIEW_POLICY_CONDITIONS
    }
    if set(indexed) != expected:
        raise ValueError("Every case must have exactly four registered condition outcomes")
    for case_id in case_ids:
        case_rows = [indexed[(case_id, condition)] for condition in REVIEW_POLICY_CONDITIONS]
        invariant = {
            (
                row.dataset_id,
                row.sequence_index,
                row.severity,
                row.original_correct,
                row.issue_type,
            )
            for row in case_rows
        }
        if len(invariant) != 1:
            raise ValueError(f"Paired-case invariant mismatch: {case_id}")
    for row in outcomes:
        if not 0.0 <= row.predecision_error_risk <= 1.0:
            raise ValueError("Predecision risk must lie in [0, 1]")
        if row.route == "human_review" and row.review_seconds <= 0:
            raise ValueError("Human-review routes require positive server-timed seconds")
        if row.route != "human_review" and row.review_seconds != 0:
            raise ValueError("Automatic routes cannot claim human review time")
        if row.condition == "always_review" and row.route != "human_review":
            raise ValueError("always_review must route every case to a human")
        if row.severity == "critical" and row.route != "human_review":
            raise ValueError("Critical cases may never be automated")

    summaries = {}
    for condition in REVIEW_POLICY_CONDITIONS:
        rows = [indexed[(case_id, condition)] for case_id in case_ids]
        autoaccepted = [row for row in rows if row.route == "auto_accept"]
        original_errors = [row for row in rows if not row.original_correct]
        semantic_feedback_rows = [
            row for row in rows if row.semantic_neighbor and row.active_feedback_ids
        ]
        summaries[condition] = {
            "cases": len(rows),
            "datasets": len({row.dataset_id for row in rows}),
            "original_accuracy": sum(row.original_correct for row in rows) / len(rows),
            "final_accuracy": sum(row.final_correct for row in rows) / len(rows),
            "review_request_rate": sum(row.route == "human_review" for row in rows)
            / len(rows),
            "review_seconds": sum(row.review_seconds for row in rows),
            "errors_corrected": sum(
                not row.original_correct and row.final_correct for row in rows
            ),
            "error_correction_rate": (
                sum(row.final_correct for row in original_errors) / len(original_errors)
                if original_errors
                else None
            ),
            "introduced_regressions": sum(
                row.original_correct and not row.final_correct for row in rows
            ),
            "autoaccepted_cases": len(autoaccepted),
            "unsafe_autoaccepts": sum(not row.final_correct for row in autoaccepted),
            "unsafe_autoaccept_rate": (
                sum(not row.final_correct for row in autoaccepted) / len(autoaccepted)
                if autoaccepted
                else None
            ),
            "predecision_aurc": _aurc(rows),
            "post_feedback_accuracy": (
                sum(row.final_correct for row in semantic_feedback_rows)
                / len(semantic_feedback_rows)
                if semantic_feedback_rows
                else None
            ),
            "correction_lag": _correction_lag(
                rows, stability_window=stability_window
            ),
        }

    always = [indexed[(case_id, "always_review")] for case_id in case_ids]

    def paired_contrast(
        left_condition: str, right_condition: str, *, seed_offset: int
    ) -> dict[str, Any]:
        left = [indexed[(case_id, left_condition)] for case_id in case_ids]
        right = [indexed[(case_id, right_condition)] for case_id in case_ids]
        pairs = list(zip(left, right, strict=True))
        return {
            "contrast": f"{left_condition}_minus_{right_condition}",
            "accuracy_difference": _cluster_bootstrap_difference(
                pairs,
                metric=lambda left_row, right_row: (
                    float(left_row.final_correct) - float(right_row.final_correct)
                ),
                samples=bootstrap_samples,
                seed=bootstrap_seed + seed_offset,
            ),
            "review_seconds_saved_per_case": _cluster_bootstrap_difference(
                pairs,
                metric=lambda left_row, right_row: (
                    right_row.review_seconds - left_row.review_seconds
                ),
                samples=bootstrap_samples,
                seed=bootstrap_seed + seed_offset + 100,
            ),
            "mcnemar": _exact_mcnemar(
                [row.final_correct for row in left],
                [row.final_correct for row in right],
            ),
            "review_seconds_saved_by_left": summaries[right_condition][
                "review_seconds"
            ]
            - summaries[left_condition]["review_seconds"],
        }

    comparisons = {}
    for offset, condition in enumerate(REVIEW_POLICY_CONDITIONS[1:], start=1):
        comparisons[condition] = paired_contrast(
            condition, "always_review", seed_offset=offset
        )

    primary = paired_contrast(
        "review_compiler_active", "raw_memory_prompt", seed_offset=20
    )
    primary["quality_noninferiority_margin"] = 0.02
    primary["minimum_confirmatory_datasets"] = 8
    primary["confirmatory_dataset_gate_passed"] = (
        primary["accuracy_difference"]["clusters"] >= 8
    )
    quality_effects = list(primary["accuracy_difference"]["dataset_effects"].values())
    labor_effects = list(
        primary["review_seconds_saved_per_case"]["dataset_effects"].values()
    )
    primary["quality_noninferiority_exact"] = _exact_cluster_signflip(
        [effect + 0.02 for effect in quality_effects]
    )
    primary["labor_superiority_exact"] = _exact_cluster_signflip(labor_effects)
    primary["quality_noninferiority_ci_passed"] = (
        primary["accuracy_difference"]["ci_low"] >= -0.02
    )
    primary["quality_noninferiority_passed"] = (
        primary["confirmatory_dataset_gate_passed"]
        and primary["quality_noninferiority_exact"]["p_value_one_sided"] < 0.05
    )
    primary["labor_saving_passed"] = (
        primary["confirmatory_dataset_gate_passed"]
        and primary["review_seconds_saved_per_case"]["estimate"] > 0
        and primary["labor_superiority_exact"]["p_value_one_sided"] < 0.05
    )
    primary["joint_success"] = (
        primary["quality_noninferiority_passed"]
        and primary["labor_saving_passed"]
    )
    registered_contrasts = {
        "primary_active_vs_raw_memory": primary,
        "active_vs_always_review": comparisons["review_compiler_active"],
        "active_vs_static_risk": paired_contrast(
            "review_compiler_active", "static_risk", seed_offset=21
        ),
        "raw_memory_vs_static_risk": paired_contrast(
            "raw_memory_prompt", "static_risk", seed_offset=22
        ),
    }

    active = [indexed[(case_id, "review_compiler_active")] for case_id in case_ids]
    replay = [
        PolicyReplayObservation(
            case_id=candidate.case_id,
            dataset_id=candidate.dataset_id,
            severity=candidate.severity,
            baseline_correct=baseline.final_correct,
            candidate_correct=candidate.final_correct,
            baseline_review_seconds=baseline.review_seconds,
            candidate_review_seconds=candidate.review_seconds,
            candidate_route=candidate.route,
            unrelated=candidate.unrelated,
        )
        for candidate, baseline in zip(active, always, strict=True)
    ]
    return {
        "schema_version": 1,
        "status": "analyzed",
        "cases": len(case_ids),
        "records": len(outcomes),
        "conditions": list(REVIEW_POLICY_CONDITIONS),
        "condition_summaries": summaries,
        "paired_vs_always_review": comparisons,
        "registered_contrasts": registered_contrasts,
        "active_policy_promotion_audit": audit_policy_promotion(replay),
        "interpretation_guard": (
            "Policy replay estimates routing value against independently adjudicated labels; "
            "it is not evidence that simulated or researcher-authored feedback equals real "
            "reviewers. Cases within a dataset are precision observations, not independent "
            "inferential replicates; primary joint success requires both eight-dataset exact "
            "quality noninferiority and exact labor superiority."
        ),
    }
