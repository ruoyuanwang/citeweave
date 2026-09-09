from __future__ import annotations

from citeweave.article_section_repair import build_section_repair_request


def _writer_input() -> dict:
    task_types = (
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    return {
        "dataset_id": "dataset",
        "graph_phenomena": [
            {
                "phenomenon_id": f"PH-{index}",
                "task_type": task_type,
                "question": "question",
                "verified_answer": {"value": index},
                "operator_trace": [{"operator": "operator"}],
                "graph_evidence_ids": [f"GE-{index}"],
                "reference_ids": [f"REF-{2 * index}", f"REF-{2 * index + 1}"],
                "interpretation_contract": {
                    "allowed": "bounded",
                    "forbidden": "causal",
                    "required_limitation": "limited",
                },
            }
            for index, task_type in enumerate(task_types)
        ],
        "representative_sources": [
            {"reference_id": f"REF-{index}"} for index in range(10)
        ],
    }


def test_results_repair_has_room_to_finish_and_exact_structure() -> None:
    request = build_section_repair_request(
        _writer_input(), section="Results", original_body="PH-0 REF-0"
    )
    prompt = request["messages"][1]["content"]
    assert request["max_tokens"] == 2000
    assert "exactly eight paragraphs" in prompt
    assert "never emit a shortened identifier" in prompt
    assert "PH-0" in prompt and "PH-4" in prompt


def test_discussion_repair_receives_repaired_results() -> None:
    request = build_section_repair_request(
        _writer_input(),
        section="Discussion",
        original_body="Old discussion",
        repaired_results="Repaired PH-0 result",
    )
    prompt = request["messages"][1]["content"]
    assert request["max_tokens"] == 1600
    assert "Repaired PH-0 result" in prompt
    assert "Every paragraph's first sentence" in prompt
