import pandas as pd
import pytest

from citeweave.author_graph_sensitivity import SparseAuthorGraph


def test_independent_sparse_distance_and_deletions():
    graph = SparseAuthorGraph(
        pd.DataFrame(
            {
                "source_id": ["a", "b", "a"],
                "target_id": ["b", "c", "c"],
                "weight": [2, 2, 0.5],
            }
        )
    )
    assert graph.summary() == {"nodes": 3, "edges": 3, "components": 1, "largest_component": 3}
    assert graph.path_distance("a", "c") == 1.0
    assert graph.edge_removal("a", "c") == {
        "component_increase": 0,
        "alternate_hops_after_deletion": 2,
    }
    assert graph.node_removal("b") == {
        "components_after": 1,
        "largest_component_fraction_of_original_nodes": 2 / 3,
    }
    assert graph.node_removal("missing") is None
    assert graph.edge_removal("a", "missing") is None
    with pytest.raises(ValueError, match="Duplicate"):
        SparseAuthorGraph(
            pd.DataFrame({"source_id": ["a", "a"], "target_id": ["b", "b"], "weight": [1, 1]})
        )


def test_disconnection_is_not_a_zero_hop_path():
    graph = SparseAuthorGraph(
        pd.DataFrame({"source_id": ["a", "b"], "target_id": ["b", "c"], "weight": [1, 1]})
    )
    assert graph.edge_removal("a", "b") == {
        "component_increase": 1,
        "alternate_hops_after_deletion": None,
    }
    assert graph.node_removal("b")["components_after"] == 2
