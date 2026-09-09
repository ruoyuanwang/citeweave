from __future__ import annotations

import pytest

from citeweave.oversight_design_sensitivity import (
    OversightSensitivityError,
    build_oversight_sensitivity_grid,
    simulate_packet_accuracy_power,
)


def test_sensitivity_is_deterministic_and_power_rises_with_effect() -> None:
    kwargs = {
        "datasets": 8,
        "cases_per_factor_level_per_dataset": 6,
        "baseline_accuracy": 0.6,
        "risk_differences": (0.1, 0.2, 0.3),
        "simulations": 1_000,
        "seed": 7,
    }
    first = simulate_packet_accuracy_power(**kwargs)
    second = simulate_packet_accuracy_power(**kwargs)
    assert first == second
    power = [row["power"] for row in first["results"]]
    assert power == sorted(power)
    assert first["conservative_holm_threshold"] == pytest.approx(0.05 / 3)


def test_grid_labels_current_design_without_reading_outcomes() -> None:
    result = build_oversight_sensitivity_grid(simulations=1_000, seed=3)
    assert result["status"] == "prospective_design_sensitivity_no_outcomes_read"
    assert result["current_design"] == {
        "datasets": 8,
        "cases_per_factor_level_per_dataset": 6,
    }
    assert len(result["designs"]) == 4


def test_sensitivity_rejects_invalid_inputs() -> None:
    with pytest.raises(OversightSensitivityError, match="2 to 12"):
        simulate_packet_accuracy_power(
            datasets=13,
            cases_per_factor_level_per_dataset=6,
            baseline_accuracy=0.6,
            risk_differences=(0.1,),
            simulations=1_000,
        )
    with pytest.raises(OversightSensitivityError, match="positive and feasible"):
        simulate_packet_accuracy_power(
            datasets=8,
            cases_per_factor_level_per_dataset=6,
            baseline_accuracy=0.9,
            risk_differences=(0.1,),
            simulations=1_000,
        )
