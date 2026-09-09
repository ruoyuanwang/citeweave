from __future__ import annotations

from citeweave.article_compiler_v2 import (
    argument_payload,
    build_compiler_repair_request,
    build_compiler_section_request,
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


def test_flat_and_graph_payloads_have_same_nodes_and_syntheses() -> None:
    writer = _writer_input()
    flat = argument_payload(writer, condition="flat_article_compiler")
    graph = argument_payload(writer, condition="graph_dependency_compiler")
    assert {row["phenomenon_id"] for row in flat["records"]} == {
        row["phenomenon_id"] for row in graph["phenomenon_order"]
    }
    assert flat["syntheses"] == graph["cross_phenomenon_synthesis"]
    assert "dependency_edges" not in flat
    assert len(graph["dependency_edges"]) > 0


def test_results_and_discussion_require_twenty_six_claim_sentences() -> None:
    writer = _writer_input()
    results = build_compiler_section_request(
        writer,
        condition="graph_dependency_compiler",
        section="Results",
        compiled_sections={},
    )
    discussion = build_compiler_section_request(
        writer,
        condition="flat_article_compiler",
        section="Discussion",
        compiled_sections={"Results": "result"},
    )
    assert "at least sixteen distinct PH-bearing claim sentences" in results["messages"][1]["content"]
    assert "at least ten distinct PH-bearing" in discussion["messages"][1]["content"]
    assert results["max_tokens"] == 1800
    assert discussion["max_tokens"] == 1450


def test_repair_is_precommitted_for_results_and_discussion() -> None:
    request = build_compiler_repair_request(
        _writer_input(),
        condition="graph_dependency_compiler",
        section="Results",
        original_body="PH-0 REF-0",
        compiled_sections={},
    )
    assert request["max_tokens"] == 2100
    assert "ORIGINAL_SECTION" in request["messages"][1]["content"]
