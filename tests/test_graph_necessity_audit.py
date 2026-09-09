from __future__ import annotations

from citeweave.graph_necessity_audit import audit_graph_necessity_task


def _task(task_type: str, answer: dict, trace: list[dict]) -> dict:
    return {
        "item_id": f"dataset:small:{task_type}",
        "dataset_id": "dataset",
        "scale": "small",
        "task_type": task_type,
        "complexity": 1 if "lookup" in task_type else 4,
        "answer": answer,
        "evidence_ids": ["e1"],
        "operator_trace": trace,
        "contexts": {
            "flat_hybrid": {"rows": [{"evidence_id": "e1", "weight": 5.0}]},
            "graph_hierarchical_retrieval_v2": {"rows": []},
            "graph_program": {"rows": [{"evidence_id": "e1"}]},
        },
    }


def test_simple_lookup_requires_one_visible_row() -> None:
    task = _task(
        "direct_edge_lookup",
        {"source_label": "A", "target_label": "B", "edge_weight": 5.0},
        [{"operator": "edge_attribute_lookup"}],
    )
    task["contexts"]["flat_hybrid"]["rows"][0].update(
        {"source_label": "A", "target_label": "B"}
    )
    audit = audit_graph_necessity_task(task)
    assert audit["decisive_answer_single_flat_row"] is True
    assert audit["necessity_certificate_passed"] is True


def test_simple_lookup_is_intrinsically_simple_even_when_retrieval_misses() -> None:
    task = _task(
        "node_attribute_lookup",
        {"node_label": "A", "importance": 7.0, "weighted_degree": 11.0},
        [{"operator": "node_attribute_lookup"}],
    )
    audit = audit_graph_necessity_task(task)
    assert audit["decisive_answer_single_flat_row"] is False
    assert audit["intrinsic_single_record_lookup"] is True
    assert audit["necessity_certificate_passed"] is True


def test_counterfactual_rejects_single_row_copy() -> None:
    task = _task(
        "bridge_counterfactual",
        {
            "alternate_hops_after_deletion": 2,
            "component_increase": 0,
        },
        [
            {"operator": "edge_betweenness"},
            {"operator": "delete_edge_counterfactual"},
        ],
    )
    audit = audit_graph_necessity_task(task)
    assert audit["decisive_answer_single_flat_row"] is False
    assert audit["necessity_certificate_passed"] is True


def test_complex_certificate_rejects_unrelated_two_step_trace() -> None:
    task = _task(
        "bridge_counterfactual",
        {
            "alternate_hops_after_deletion": 2,
            "component_increase": 0,
        },
        [
            {"operator": "shortest_path"},
            {"operator": "path_projection"},
        ],
    )

    audit = audit_graph_necessity_task(task)

    assert audit["necessity_certificate_passed"] is False
    assert audit["missing_operator_classes"] == ["global_selection", "intervention"]
