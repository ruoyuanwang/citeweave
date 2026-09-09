from __future__ import annotations

import pytest

from citeweave.article_design_sensitivity import (
    ArticleSensitivityError,
    build_article_sensitivity_report,
    simulate_standardized_topic_power,
)


def test_standardized_topic_power_is_deterministic_and_monotone() -> None:
    kwargs = {
        "topics": 8,
        "standardized_effects": (0.25, 0.75, 1.25),
        "simulations": 1_000,
        "seed": 5,
    }
    first = simulate_standardized_topic_power(**kwargs)
    second = simulate_standardized_topic_power(**kwargs)
    assert first == second
    power = [row["power"] for row in first["results"]]
    assert power == sorted(power)
    assert first["conservative_holm_threshold"] == pytest.approx(0.05 / 3)


def test_article_report_preserves_topic_as_replication_unit() -> None:
    report = build_article_sensitivity_report(simulations=1_000, seed=9)
    assert report["registered_topics"] == 8
    assert report["a1_binary_claim_sensitivity"]["datasets"] == 8
    assert [row["topics"] for row in report["a2_a3_standardized_topic_sensitivity"]] == [
        8,
        12,
    ]
    assert "do not increase" in report["scope_guard"]


def test_article_sensitivity_rejects_invalid_inputs() -> None:
    with pytest.raises(ArticleSensitivityError, match="2 to 12"):
        simulate_standardized_topic_power(
            topics=13, standardized_effects=(1.0,), simulations=1_000
        )
    with pytest.raises(ArticleSensitivityError, match="positive"):
        simulate_standardized_topic_power(
            topics=8, standardized_effects=(0.0,), simulations=1_000
        )
