from __future__ import annotations

from dataclasses import replace

import pytest

from citeweave.review_policy_experiment import (
    REVIEW_POLICY_CONDITIONS,
    ReviewPolicyOutcome,
    analyze_review_policy_panel,
)


def _panel(cases: int = 80) -> list[ReviewPolicyOutcome]:
    rows = []
    for index in range(cases):
        original_correct = index % 5 != 0
        for condition in REVIEW_POLICY_CONDITIONS:
            review = (
                condition == "always_review"
                or (condition == "review_compiler_active" and not original_correct)
                or condition == "raw_memory_prompt"
            )
            rows.append(
                ReviewPolicyOutcome(
                    case_id=f"C{index:03d}",
                    dataset_id=f"D{index % 8}",
                    condition=condition,
                    sequence_index=index,
                    severity="medium",
                    predecision_error_risk=0.9 if not original_correct else 0.1,
                    original_correct=original_correct,
                    final_correct=True if review else original_correct,
                    route="human_review" if review else "auto_accept",
                    review_seconds=10.0 if review else 0.0,
                    issue_type="causal_overreach",
                    active_feedback_ids=("F1",) if index >= 5 else (),
                    semantic_neighbor=index >= 5,
                    unrelated=index % 7 == 0,
                )
            )
    return rows


def test_four_condition_panel_is_paired_and_reports_labor_quality() -> None:
    result = analyze_review_policy_panel(_panel(), bootstrap_samples=100, bootstrap_seed=7)
    active = result["condition_summaries"]["review_compiler_active"]
    assert result["cases"] == 80
    assert result["records"] == 320
    assert active["final_accuracy"] == 1
    assert active["review_request_rate"] == 0.2
    assert active["correction_lag"]["mean_observed_lag"] == 1
    assert result["paired_vs_always_review"]["review_compiler_active"][
        "review_seconds_saved_by_left"
    ] == 640
    assert result["registered_contrasts"]["primary_active_vs_raw_memory"][
        "joint_success"
    ]
    primary = result["registered_contrasts"]["primary_active_vs_raw_memory"]
    assert primary["accuracy_difference"]["cluster_weighting"] == "equal_dataset"
    assert primary["quality_noninferiority_exact"]["minimum_attainable_p"] == 1 / 256
    assert primary["labor_superiority_exact"]["p_value_one_sided"] == 1 / 256


def test_unequal_case_counts_do_not_change_equal_dataset_estimand() -> None:
    rows = _panel(cases=9)
    for index, row in enumerate(rows):
        if (
            row.dataset_id == "D0" and row.condition == "raw_memory_prompt"
        ) or (
            row.dataset_id == "D1" and row.condition == "review_compiler_active"
        ):
            rows[index] = replace(row, final_correct=False)
    result = analyze_review_policy_panel(rows, bootstrap_samples=100, bootstrap_seed=7)
    primary = result["registered_contrasts"]["primary_active_vs_raw_memory"]
    assert primary["accuracy_difference"]["estimate"] == 0.0
    assert primary["accuracy_difference"]["clusters"] == 8


def test_incomplete_or_critical_auto_routing_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        analyze_review_policy_panel(_panel()[:-1], bootstrap_samples=10)
    rows = _panel()
    for index in range(4):
        original = rows[index]
        changes = {**original.__dict__, "severity": "critical"}
        if original.condition != "always_review":
            changes.update({"route": "auto_accept", "review_seconds": 0.0})
        rows[index] = ReviewPolicyOutcome(**changes)
    with pytest.raises(ValueError, match="Critical"):
        analyze_review_policy_panel(rows, bootstrap_samples=10)
