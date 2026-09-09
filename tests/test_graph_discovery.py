from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.citeweave.graph_discovery import (
    build_discovery_benchmark,
    build_simple_control_benchmark,
    score_discovery_response,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "topic"
    root = workspace / "canonical" / "visualization"
    root.mkdir(parents=True)
    nodes = pd.DataFrame(
        {
            "keyword": [f"k{index}" for index in range(8)],
            "keyword_type": ["test"] * 8,
            "occurrences": [100, 90, 80, 70, 60, 50, 40, 30],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": ["k0", "k1", "k2", "k0", "k4", "k5", "k6", "k3"],
            "target_id": ["k1", "k2", "k3", "k2", "k5", "k6", "k7", "k4"],
            "weight": [9, 8, 7, 6, 9, 8, 7, 1],
        }
    )
    nodes.to_parquet(root / "keyword_occurrences.parquet", index=False)
    edges.to_parquet(root / "keyword_cooccurrence_edges.parquet", index=False)
    return workspace


def test_builds_multiscale_complex_tasks_and_budgeted_contexts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    benchmark = build_discovery_benchmark(
        workspace,
        tmp_path / "out",
        scales=("small",),
        networks=("keyword_cooccurrence",),
        record_budget=12,
    )
    assert benchmark["tasks"]
    task_types = {item["task_type"] for item in benchmark["tasks"]}
    assert "bridge_counterfactual" in task_types
    assert "hub_removal_resilience" in task_types
    assert "direct_edge_lookup" in task_types
    assert "node_attribute_lookup" in task_types
    for item in benchmark["tasks"]:
        assert item["complexity"] >= 1
        assert set(item["contexts"]) == {
            "flat_retrieval",
            "flat_tfidf",
            "flat_bm25",
            "flat_lsa",
            "flat_hybrid",
            "flat_program",
            "graph_query_retrieval",
            "graph_hierarchical_retrieval",
            "graph_hierarchical_retrieval_v2",
            "graph_retrieval",
            "operator_only",
            "graph_program",
        }
        assert item["contexts"]["flat_retrieval"]["record_budget"] == 12
        assert len(item["contexts"]["flat_lsa"]["rows"]) == 12
        assert len(item["contexts"]["flat_hybrid"]["rows"]) == 12
        assert item["contexts"]["flat_lsa"]["parameters"]["dimensions"] >= 2
        flat_program = item["contexts"]["flat_program"]
        graph_program = item["contexts"]["graph_program"]
        assert len(flat_program["rows"]) == len(graph_program["nodes"]) + len(
            graph_program["edges"]
        )
        assert flat_program["derived_rows"] == graph_program["operator_trace"]
        assert item["contexts"]["operator_only"]["operator_trace"] == graph_program[
            "operator_trace"
        ]
        query_context = item["contexts"]["graph_query_retrieval"]
        hierarchical_context = item["contexts"]["graph_hierarchical_retrieval"]
        hierarchical_context_v2 = item["contexts"]["graph_hierarchical_retrieval_v2"]
        assert hierarchical_context["summaries"][0]["summary_type"] == "global_graph"
        assert (
            len(hierarchical_context["summaries"])
            + len(hierarchical_context["nodes"])
            + len(hierarchical_context["edges"])
            <= 12
        )
        assert hierarchical_context_v2["summaries"][0]["summary_type"] == "global_graph"
        community_summaries = [
            row
            for row in hierarchical_context_v2["summaries"]
            if row["summary_type"] == "community"
        ]
        assert community_summaries
        assert all("external_edge_count_share" in row for row in community_summaries)
        assert all("external_edge_share_definition" in row for row in community_summaries)
        if item["task_type"] == "community_role_contrast":
            outward = next(
                row
                for row in community_summaries
                if row["community"] == item["answer"]["outward_community"]
            )
            assert (
                outward["external_edge_share"]
                == item["answer"]["outward_external_share"]
            )
        if item["task_type"] == "multi_hop_connector":
            shortest_path = next(
                row for row in item["operator_trace"] if row["operator"] == "shortest_path"
            )
            assert shortest_path["weight"] == "inverse_edge_weight"
            assert item["answer"]["total_inverse_weight_distance"] > 0
            assert len(query_context["retrieval_anchors"]) == 2
            query_ids = {
                row["evidence_id"]
                for row in [*query_context["nodes"], *query_context["edges"]]
            }
            assert set(item["evidence_ids"]).issubset(query_ids)
        elif item["complexity"] == 1:
            assert len(query_context["retrieval_anchors"]) in {1, 2}
            query_ids = {
                row["evidence_id"]
                for row in [*query_context["nodes"], *query_context["edges"]]
            }
            assert set(item["evidence_ids"]).issubset(query_ids)
        else:
            assert query_context["retrieval_anchors"] == []

    simple = [item for item in benchmark["tasks"] if item["complexity"] == 1]
    assert simple
    assert all(item["matched_complex_item_id"] for item in simple)


def test_scores_answer_and_evidence_separately(tmp_path: Path) -> None:
    benchmark = build_discovery_benchmark(
        _workspace(tmp_path),
        tmp_path / "out",
        scales=("small",),
        networks=("keyword_cooccurrence",),
    )
    task = benchmark["tasks"][0]
    score = score_discovery_response(
        task,
        {
            "answer": task["answer"],
            "evidence_ids": task["evidence_ids"],
            "limitation": "This depends on the constructed graph.",
            "abstain": False,
        },
    )
    assert score["answer_exact"] is True
    assert score["evidence_f1"] == 1.0
    assert score["has_required_limitation"] is True


def test_builds_simple_controls_from_frozen_parent_anchors(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source = build_discovery_benchmark(
        workspace,
        tmp_path / "source",
        scales=("small",),
        networks=("keyword_cooccurrence",),
        task_types=("bridge_counterfactual", "hub_removal_resilience"),
        record_budget=12,
    )
    controls = build_simple_control_benchmark(
        workspace,
        tmp_path / "controls",
        parent_tasks=source["tasks"],
        record_budget=12,
    )
    assert len(controls["tasks"]) == 2
    parents = {task["item_id"]: task for task in source["tasks"]}
    for control in controls["tasks"]:
        parent = parents[control["matched_complex_item_id"]]
        assert control["evidence_ids"][0] == parent["evidence_ids"][0]


def test_score_accepts_equivalent_four_decimal_rounding(tmp_path: Path) -> None:
    benchmark = build_discovery_benchmark(
        _workspace(tmp_path),
        tmp_path / "out",
        scales=("small",),
        networks=("keyword_cooccurrence",),
    )
    task = benchmark["tasks"][0]
    task["answer"] = {"fraction": 0.997831}
    score = score_discovery_response(
        task,
        {
            "answer": {"fraction": 0.9978},
            "evidence_ids": task["evidence_ids"],
            "limitation": "Rounded to four decimals.",
        },
    )
    assert score["answer_exact"] is True
