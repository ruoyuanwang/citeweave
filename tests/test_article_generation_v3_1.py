from __future__ import annotations

from citeweave.article_generation_v3_1 import build_checkpoint_article_request


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


def test_checkpoint_prompt_binds_discussion_pairs_and_completion() -> None:
    request, plan = build_checkpoint_article_request(
        _writer_input(), condition="checkpoint_graph_plan"
    )
    prompt = request["messages"][1]["content"]
    assert plan is not None
    assert "3000 words is an absolute ceiling" in prompt
    assert "Start Limitations before 2700" in prompt
    assert "Every Discussion paragraph must contain literal PH identifiers" in prompt
    assert "PH-0 and PH-1" in prompt
    assert "PH-2 and PH-3" in prompt
    assert "PH-4 and PH-2" in prompt
    assert request["max_tokens"] == 5000


def test_checkpoint_one_shot_does_not_receive_argument_plan() -> None:
    request, plan = build_checkpoint_article_request(
        _writer_input(), condition="checkpoint_one_shot"
    )
    assert plan is None
    assert "ARGUMENT_PLAN_JSON" not in request["messages"][1]["content"]
