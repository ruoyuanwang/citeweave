from dataclasses import asdict

import networkx as nx
import pandas as pd
import pytest

from citeweave.graph_discovery import _community_task, _resilience_task, _temporal_structure_task
from citeweave.graph_robustness import (
    baseline_checks,
    community_roles,
    compare_variants,
    load_verified_trends,
    measure_fixed_edge,
    measure_fixed_hub,
    measure_fixed_path,
    measure_temporal,
    measure_variant,
    partition,
    perturb_graph,
    registered_variants,
)


def toy_graph():
    graph = nx.Graph()
    for n, importance in (("a", 5), ("b", 6), ("c", 4), ("d", 3), ("e", 2), ("f", 1)):
        graph.add_node(n, label=n, importance=importance)
    for n, (u, v, w) in enumerate(
        (
            ("a", "b", 3),
            ("b", "c", 3),
            ("a", "c", 1),
            ("c", "d", 1),
            ("d", "e", 3),
            ("e", "f", 3),
            ("d", "f", 1),
        )
    ):
        graph.add_edge(u, v, weight=w, distance=1 / w, evidence_id=f"e{n}")
    return graph


def spec(**changes):
    return {**registered_variants()[0], **changes}


def test_design_is_fixed_one_factor_and_has_18_cases():
    variants = registered_variants()
    assert len(variants) == len({v["variant_id"] for v in variants}) == 18
    keys = set(variants[0]) - {"family", "variant_id"}
    for variant in variants[1:]:
        changed = {key for key in keys if variant[key] != variants[0][key]}
        assert 1 <= len(changed) <= 2  # dropout fraction plus its randomization seed
        if len(changed) == 2:
            assert changed == {"drop_fraction", "drop_seed"}


def test_threshold_and_dropout_preserve_original_vertices_and_graph():
    graph = toy_graph()
    changed = perturb_graph(graph, spec(minimum_weight=4))
    assert set(changed) == set(graph)
    assert changed.number_of_edges() == 0
    assert nx.number_connected_components(changed) == 6
    assert graph.number_of_edges() == 7
    dropped = perturb_graph(graph, spec(drop_fraction=0.5))
    assert dropped.number_of_edges() == 4


def test_dropout_is_independent_of_input_edge_iteration_order():
    graph = toy_graph()
    reverse = nx.Graph()
    reverse.add_nodes_from(reversed(list(graph.nodes(data=True))))
    reverse.add_edges_from(reversed(list(graph.edges(data=True))))

    def edges(g):
        return {frozenset(edge) for edge in g.edges}

    assert edges(perturb_graph(graph, spec(drop_fraction=0.5))) == edges(
        perturb_graph(reverse, spec(drop_fraction=0.5))
    )


def test_original_shortest_path_cost_and_disconnection_are_measured():
    graph = toy_graph()
    groups = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1, "f": 1}
    task = {"operator_trace": [{"operator": "shortest_path", "path": ["a", "b", "c"]}]}
    result = measure_fixed_path(graph, groups, task)
    assert result["total_inverse_weight_distance"] == pytest.approx(2 / 3)
    assert result["original_path_still_optimal"]
    graph.remove_edges_from([("a", "b"), ("a", "c")])
    assert measure_fixed_path(graph, groups, task)["status"] == "disconnected"


def test_edge_anchor_absence_is_not_counted_as_robust_redundancy():
    graph = toy_graph()
    task = {"operator_trace": [{"operator": "edge_betweenness", "selected_edge": ["c", "d"]}]}
    groups = {node: int(node > "c") for node in graph}
    result = measure_fixed_edge(graph, groups, task)
    assert result["component_increase"] == 1
    assert result["redundant_connector"] is False
    assert result["highest_betweenness_selection_retested"] is False
    graph.remove_edge("c", "d")
    assert measure_fixed_edge(graph, groups, task) == {"status": "anchor_edge_absent"}


def test_hub_denominator_counts_induced_isolates():
    graph = toy_graph()
    task = {"operator_trace": [{"operator": "delete_node", "node": "c"}]}
    changed = perturb_graph(graph, spec(minimum_weight=4))
    result = measure_fixed_hub(changed, task)
    assert result["components_after"] == 5
    assert result["largest_component_fraction_of_original_nodes"] == pytest.approx(1 / 6)


def test_community_memberships_not_arbitrary_labels_determine_stability():
    graph = toy_graph()
    groups = {node: int(node > "c") for node in graph}
    relabeled = {node: 100 + group for node, group in groups.items()}
    old = {"measurements": {"community_role_contrast": community_roles(graph, groups)}}
    new = {"measurements": {"community_role_contrast": community_roles(graph, relabeled)}}
    result = compare_variants(old, new, groups, relabeled)
    assert result["partition_adjusted_rand_index"] == 1
    assert result["community_role_contrast"]["dominant_membership_jaccard"] == 1
    assert result["community_role_contrast"]["outward_membership_jaccard"] == 1


def test_new_baseline_aggregates_match_original_builders():
    graph = toy_graph()
    groups = partition(graph, spec())
    common = {
        "dataset_id": "toy",
        "network": "keyword_cooccurrence",
        "scale": "large",
        "communities": groups,
    }
    tasks = [asdict(builder(graph, **common)) for builder in (_community_task, _resilience_task)]
    measured, _ = measure_variant(
        graph, tasks, pd.DataFrame(), spec(), precomputed_partition=groups
    )
    assert all(row["passed"] for row in baseline_checks(measured["measurements"], tasks))


def test_temporal_nonoverlap_and_baseline_fidelity(tmp_path):
    graph = toy_graph()
    groups = partition(graph, spec())
    trends = pd.DataFrame(
        [
            {"keyword": node, "year": year, "documents": year if node == "a" else 2}
            for node in graph
            for year in range(2018, 2026)
        ]
    )
    target = tmp_path / "analyses"
    target.mkdir()
    trends.to_parquet(target / "keyword_trends.parquet", index=False)
    task = asdict(
        _temporal_structure_task(
            graph,
            workspace=tmp_path,
            dataset_id="toy",
            network="keyword_cooccurrence",
            scale="large",
            communities=groups,
        )
    )
    measured = measure_temporal(graph, groups, trends, 3)
    assert baseline_checks({"temporal_structural_shift": measured}, [task])[0]["passed"]
    four = measure_temporal(graph, groups, trends, 4)
    assert set(four["early_years"]).isdisjoint(four["recent_years"])
    assert (
        measure_temporal(graph, groups, trends, 5)["status"] == "insufficient_nonoverlapping_years"
    )


def test_all_isolate_partition_is_well_defined():
    graph = perturb_graph(toy_graph(), spec(minimum_weight=4))
    groups = partition(graph, spec())
    assert len(set(groups.values())) == len(graph)
    assert community_roles(graph, groups)["status"] == "insufficient_communities"


def test_temporal_table_is_recomputed_and_drift_rejected(tmp_path):
    canonical = tmp_path / "canonical"
    (canonical / "visualization").mkdir(parents=True)
    pd.DataFrame([{"keyword": "a", "occurrences": 2}]).to_parquet(
        canonical / "visualization/keyword_occurrences.parquet", index=False
    )
    pd.DataFrame(
        [
            {"work_id": "w1", "keyword": "a"},
            {"work_id": "w1", "keyword": "a"},
            {"work_id": "w2", "keyword": "a"},
        ]
    ).to_parquet(canonical / "keywords.parquet", index=False)
    pd.DataFrame([{"work_id": "w1", "year": 2020}, {"work_id": "w2", "year": 2020}]).to_parquet(
        canonical / "works.parquet", index=False
    )
    original = tmp_path / "trends.parquet"
    expected = pd.DataFrame([{"keyword": "a", "year": 2020, "documents": 2, "global_documents": 2}])
    expected.to_parquet(original, index=False)
    assert load_verified_trends(tmp_path, original).iloc[0].documents == 2
    expected.loc[0, "documents"] = 3
    expected.to_parquet(original, index=False)
    with pytest.raises(AssertionError):
        load_verified_trends(tmp_path, original)
