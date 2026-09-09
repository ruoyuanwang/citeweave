from __future__ import annotations

from citeweave.article_compiler import (
    GENERATION_ORDER,
    SECTION_ORDER,
    assemble_article,
    build_section_request,
    normalize_section_body,
)


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
        "writing_brief": {"permitted_range": [2700, 3300]},
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


def test_generation_order_respects_section_dependencies() -> None:
    assert GENERATION_ORDER.index("Results") < GENERATION_ORDER.index("Discussion")
    assert GENERATION_ORDER.index("Discussion") < GENERATION_ORDER.index("Conclusion")
    assert GENERATION_ORDER[-1] == "Abstract"


def test_discussion_request_contains_results_and_exact_pairs() -> None:
    request = build_section_request(
        _writer_input(), section="Discussion", compiled_sections={"Results": "PH-0"}
    )
    prompt = request["messages"][1]["content"]
    assert '"Results":"PH-0"' in prompt
    assert "PH-0 with PH-1" in prompt
    assert "PH-2 with PH-3" in prompt
    assert "PH-4 with PH-2" in prompt
    assert request["max_tokens"] == 1250


def test_assembly_has_exact_top_level_sections_and_strips_duplicate_heading() -> None:
    bodies = {section: f"## {section}\nBody {section}." for section in SECTION_ORDER}
    article = assemble_article(bodies)
    assert article.count("## ") == 7
    assert normalize_section_body("```markdown\n## Results\nBody.\n```", "Results") == "Body."
