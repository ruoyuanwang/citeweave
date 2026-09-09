from __future__ import annotations

from scripts.analyze_formal_v3_results import (
    cluster_bootstrap,
    exact_cluster_signflip,
    exact_mcnemar,
    holm_adjust,
)


def test_exact_mcnemar_and_holm_are_deterministic() -> None:
    result = exact_mcnemar(
        [True, True, True, True, False],
        [False, False, False, True, False],
    )
    assert result["left_only"] == 3
    assert result["right_only"] == 0
    assert result["p_value"] == 0.25
    adjusted = holm_adjust({"H1": 0.01, "H2": 0.04, "H3": 0.02})
    assert adjusted == {"H1": 0.03, "H3": 0.04, "H2": 0.04}


def test_cluster_bootstrap_resamples_datasets_not_items() -> None:
    rows = [
        {"dataset_id": "a", "value": 1.0},
        {"dataset_id": "a", "value": 1.0},
        {"dataset_id": "b", "value": -1.0},
    ]
    result = cluster_bootstrap(
        rows,
        statistic=lambda sample: sum(row["value"] for row in sample) / len(sample),
        samples=1000,
        seed=7,
    )
    assert result["estimate"] == 0.0
    assert result["clusters"] == 2
    assert result["cluster_weighting"] == "equal_dataset"
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_four_cluster_signflip_cannot_reach_point_zero_five() -> None:
    result = exact_cluster_signflip([1.0, 1.0, 1.0, 1.0])
    assert result["p_value_one_sided"] == 1 / 16
    assert result["minimum_attainable_p"] == 1 / 16
