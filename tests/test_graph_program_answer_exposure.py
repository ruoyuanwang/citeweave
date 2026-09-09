from __future__ import annotations

from pathlib import Path

from citeweave.io import write_json
from scripts.audit_graph_program_answer_exposure import (
    audit_graph_program_answer_exposure,
)


def test_audit_distinguishes_partial_and_full_answer_exposure(tmp_path: Path) -> None:
    benchmark = tmp_path / "topic" / "benchmark.json"
    trace = [
        {"operator": "community_boundary_edges", "candidate_count": 4},
        {"operator": "edge_betweenness", "selected_edge": ["A", "B"]},
        {
            "operator": "delete_edge_counterfactual",
            "components_before": 1,
            "components_after": 1,
            "alternate_hops": 2,
        },
    ]
    task = {
        "item_id": "T1",
        "task_type": "bridge_counterfactual",
        "scale": "large",
        "complexity": 3,
        "answer": {
            "source_label": "A",
            "target_label": "B",
            "edge_weight": 8.0,
            "alternate_hops_after_deletion": 2,
            "component_increase": 0,
        },
        "contexts": {
            "graph_program": {"operator_trace": trace, "edges": [{
                "source_label": "A", "target_label": "B", "weight": 8.0
            }]},
            "operator_only": {"operator_trace": trace},
            "flat_program": {
                "derived_rows": trace,
                "rows": [{
                    "source_label": "A", "target_label": "B", "weight": 8.0
                }],
            },
        },
    }
    write_json(
        benchmark,
        {"dataset_id": "D1", "tasks": [task]},
    )
    result = audit_graph_program_answer_exposure([benchmark])
    assert result["exposure_summaries"]["operator_trace"][
        "fully_exposed_tasks"
    ] == 0
    assert result["exposure_summaries"]["flat_program_derived_rows"][
        "fully_exposed_tasks"
    ] == 0
    assert result["records"][0]["exposure"]["flat_program_derived_rows"][
        "coverage"
    ] == 3 / 5
    assert result["derivability_summaries"]["operator_trace"][
        "fully_derivable_tasks"
    ] == 0
    assert result["derivability_summaries"]["flat_program_full_context"][
        "fully_derivable_tasks"
    ] == 1
