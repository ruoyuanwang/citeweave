from __future__ import annotations

import pandas as pd

from citeweave.models import VisualizationPolicy
from citeweave.visual_selection import adaptive_occurrence_threshold, select_network


def _fixture_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "id": [f"node-{index:03d}" for index in range(60)],
            "label": [f"Term {index}" for index in range(60)],
            "occurrences": [max(5, 100 - index) for index in range(60)],
        }
    )
    edges = []
    for index in range(60):
        for offset in (1, 2, 3):
            target = index + offset
            if target < 60:
                edges.append(
                    {
                        "source_id": f"node-{index:03d}",
                        "target_id": f"node-{target:03d}",
                        "weight": 8 - offset,
                    }
                )
    return nodes, pd.DataFrame(edges)


def test_adaptive_occurrence_threshold_bounds_candidate_pool() -> None:
    occurrences = pd.Series(range(1, 1_001))
    threshold = adaptive_occurrence_threshold(
        occurrences,
        network_name="keyword_cooccurrence",
        target_candidates=120,
    )
    assert threshold == 881
    assert int((occurrences >= threshold).sum()) == 120


def test_candidate_first_selection_is_bounded_sparse_and_reproducible() -> None:
    nodes, edges = _fixture_graph()
    policy = VisualizationPolicy(
        max_display_nodes=30,
        label_budget=12,
        max_edges_per_node=3,
        layout_restarts=2,
        layout_iterations=120,
    )
    first = select_network("keyword_cooccurrence", nodes, edges, policy, seed=19)
    second = select_network("keyword_cooccurrence", nodes, edges, policy, seed=19)
    assert first is not None and second is not None
    assert len(first.nodes) <= 30
    assert len(first.edges) <= 3 * len(first.nodes)
    assert first.disclosure["matrix_materialized"] is False
    assert first.disclosure["normalization"].startswith("association strength")
    assert first.positions.keys() == second.positions.keys()
    for node in first.positions:
        assert first.positions[node].tolist() == second.positions[node].tolist()
    assert first.layout_qa["overlap_ratio"] <= 0.03


def test_sparse_map_keeps_each_display_component_connected() -> None:
    nodes, edges = _fixture_graph()
    policy = VisualizationPolicy(
        max_display_nodes=25,
        max_edges_per_node=2,
        layout_restarts=1,
        layout_iterations=100,
    )
    result = select_network("coauthorship", nodes, edges, policy, seed=7)
    assert result is not None
    degrees = (
        pd.concat(
            [
                result.edges["source"].value_counts(),
                result.edges["target"].value_counts(),
            ],
            axis=1,
        )
        .fillna(0)
        .sum(axis=1)
    )
    assert set(result.nodes["id"]).issubset(set(degrees.index))
