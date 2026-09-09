from __future__ import annotations

from scripts.analyze_formal_v3_scale_replication import (
    _cluster_bootstrap,
    exact_cluster_signflip,
)


def test_eight_positive_clusters_can_reach_point_zero_zero_four() -> None:
    result = exact_cluster_signflip([1.0] * 8)
    assert result["permutations"] == 256
    assert result["p_value_one_sided"] == 1 / 256
    assert result["minimum_attainable_p"] == 1 / 256


def test_scale_bootstrap_weights_datasets_equally_when_pair_counts_differ() -> None:
    rows = [
        {"dataset_id": "a", "interaction": 1.0},
        {"dataset_id": "a", "interaction": 1.0},
        {"dataset_id": "b", "interaction": -1.0},
    ]
    result = _cluster_bootstrap(rows, samples=1000, seed=7)
    assert result["estimate"] == 0.0
    assert result["clusters"] == 2
    assert result["cluster_weighting"] == "equal_dataset"
