from __future__ import annotations

from collections import Counter

import pytest

from citeweave.oversight_factorial_assignment import _allocate_arms
from citeweave.oversight_randomization_analysis import (
    OversightRandomizationError,
    analyze_oversight_randomization,
)


def _inputs() -> tuple[list[dict], list[dict], dict]:
    packets = []
    for dataset in range(8):
        for index in range(12):
            packets.append(
                {
                    "case_id": f"D{dataset}-C{index}",
                    "dataset_id": f"D{dataset}",
                    "domain": f"domain-{dataset}",
                    "issue_type": f"issue-{index % 4}",
                    "severity": ("low", "medium", "high", "critical")[index % 4],
                }
            )
    seed = 20260907
    arms = _allocate_arms(packets, seed=seed)
    routes = []
    outcomes = []
    for packet in packets:
        case_id = packet["case_id"]
        packet_arm, assignment_arm = arms[case_id]
        routes.append(
            {
                "case_id": case_id,
                "packet_arm": packet_arm,
                "assignment_arm": assignment_arm,
            }
        )
        outcomes.append(
            {
                "case_id": case_id,
                "dataset_id": packet["dataset_id"],
                "packet_arm": packet_arm,
                "assignment_arm": assignment_arm,
                "adjudicated_final_correct": packet_arm == "adversarial_two_sided",
                "review_seconds": (
                    60.0 if assignment_arm == "capability_cost_router" else 120.0
                ),
            }
        )
    assignment = {
        "status": "frozen_before_heldout_review",
        "heldout_outcomes_inspected": False,
        "seed": seed,
        "case_routes": routes,
    }
    return outcomes, packets, assignment


def test_randomization_analysis_reproduces_design_and_detects_large_effects() -> None:
    outcomes, packets, assignment = _inputs()
    result = analyze_oversight_randomization(
        outcomes, packets, assignment, permutations=999, permutation_seed=11
    )
    assert result["cases"] == 96
    assert result["datasets"] == 8
    assert result["registered_contrasts"]["B1"]["estimate"] == pytest.approx(1.0)
    assert result["registered_contrasts"]["B1"]["randomization_p_value"] <= 0.01
    assert "96-case benchmark population" in result["scope_guard"]


def test_randomization_draws_keep_three_cases_in_every_cell() -> None:
    _, packets, _ = _inputs()
    for seed in (1, 2, 3):
        arms = _allocate_arms(packets, seed=seed)
        by_dataset = Counter(
            (packet["dataset_id"], *arms[packet["case_id"]]) for packet in packets
        )
        assert set(by_dataset.values()) == {3}


def test_randomization_analysis_rejects_arm_or_seed_tampering() -> None:
    outcomes, packets, assignment = _inputs()
    outcomes[0]["packet_arm"] = (
        "standard_claim_first"
        if outcomes[0]["packet_arm"] == "adversarial_two_sided"
        else "adversarial_two_sided"
    )
    with pytest.raises(OversightRandomizationError, match="differ"):
        analyze_oversight_randomization(outcomes, packets, assignment, permutations=999)

    outcomes, packets, assignment = _inputs()
    assignment["seed"] += 1
    with pytest.raises(OversightRandomizationError, match="cannot be reproduced"):
        analyze_oversight_randomization(outcomes, packets, assignment, permutations=999)


def test_randomization_analysis_rejects_invalid_panel_or_low_permutations() -> None:
    outcomes, packets, assignment = _inputs()
    with pytest.raises(OversightRandomizationError, match="Exactly 96"):
        analyze_oversight_randomization(
            outcomes[:-1], packets, assignment, permutations=999
        )
    with pytest.raises(OversightRandomizationError, match="At least 999"):
        analyze_oversight_randomization(outcomes, packets, assignment, permutations=998)
