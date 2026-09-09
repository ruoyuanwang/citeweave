from __future__ import annotations

from citeweave.article_generation_v3 import (
    SECTION_BUDGETS,
    build_budgeted_article_request,
    compact_graph_argument_plan,
)


def _writer_input() -> dict:
    task_types = (
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    phenomena = []
    sources = [{"reference_id": f"REF-{index}"} for index in range(10)]
    for index, task_type in enumerate(task_types):
        phenomena.append(
            {
                "phenomenon_id": f"PH-{index}",
                "task_type": task_type,
                "question": f"Question {index}",
                "verified_answer": {"value": index},
                "operator_trace": [{"operator": f"op-{index}"}],
                "graph_evidence_ids": [f"GE-{index}"],
                "reference_ids": [f"REF-{2 * index}", f"REF-{2 * index + 1}"],
                "interpretation_contract": {
                    "allowed": "bounded",
                    "forbidden": "causal",
                    "required_limitation": "construction dependent",
                },
            }
        )
    return {
        "dataset_id": "dataset",
        "writing_brief": {"permitted_range": [2700, 3300]},
        "graph_phenomena": phenomena,
        "representative_sources": sources,
    }


def test_compact_plan_keeps_dependencies_without_verified_answer_duplication() -> None:
    plan = compact_graph_argument_plan(_writer_input())
    assert len(plan["phenomenon_order"]) == 5
    assert len(plan["cross_phenomenon_synthesis"]) == 3
    assert len(plan["dependency_edges"]) >= 16
    assert "verified_answer" not in str(plan)


def test_both_conditions_share_hard_budget_contract() -> None:
    writer_input = _writer_input()
    one_shot, one_plan = build_budgeted_article_request(
        writer_input, condition="budgeted_one_shot"
    )
    graph, graph_plan = build_budgeted_article_request(
        writer_input, condition="budgeted_graph_plan"
    )
    assert one_plan is None
    assert graph_plan is not None
    for request in (one_shot, graph):
        prompt = request["messages"][1]["content"]
        assert request["max_tokens"] == 5000
        assert "2700-3100 words" in prompt
        assert "Do not use level-three" in prompt
        assert "never omit Limitations or Conclusion" in prompt
        assert all(section in prompt for section in SECTION_BUDGETS)
    assert "ARGUMENT_PLAN_JSON" not in one_shot["messages"][1]["content"]
    assert "ARGUMENT_PLAN_JSON" in graph["messages"][1]["content"]
