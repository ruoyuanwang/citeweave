from __future__ import annotations

from scripts.run_machine_article_generation import assess_generated_article


def _writer_input() -> dict[str, object]:
    task_types = (
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    phenomena = [
        {"phenomenon_id": f"PH-{index:016d}", "task_type": task_type}
        for index, task_type in enumerate(task_types)
    ]
    sources = [
        {"reference_id": f"REF-{index:016d}"} for index in range(10)
    ]
    return {
        "writing_brief": {"permitted_range": [1, 10_000]},
        "graph_phenomena": phenomena,
        "representative_sources": sources,
    }


def _article() -> str:
    tokens = " ".join(
        [f"PH-{index:016d}" for index in range(5)]
        + [f"REF-{index:016d}" for index in range(10)]
    )
    ordinary = ("Abstract", "Introduction", "Methods", "Limitations")
    parts = [f"## {section}\n{tokens}" for section in ordinary]
    parts.extend(
        [
            (
                "## Results\n"
                "The path and deletion results jointly qualify connectivity "
                "PH-0000000000000000 PH-0000000000000001.\n\n"
                "The community and hub-removal results jointly test dependence "
                "PH-0000000000000002 PH-0000000000000003 PH-0000000000000004."
            ),
            (
                "## Discussion\n"
                "Connectivity is not necessity PH-0000000000000000 "
                "PH-0000000000000001.\n\n"
                "Prominence is not dependence PH-0000000000000002 "
                "PH-0000000000000003 PH-0000000000000004."
            ),
            (
                "## Conclusion\n"
                "The combined evidence remains construction-dependent "
                "PH-0000000000000000 PH-0000000000000002 PH-0000000000000004."
            ),
        ]
    )
    return "\n\n".join(parts)


def test_generated_article_gate_accepts_registered_evidence() -> None:
    report = assess_generated_article(_article(), writer_input=_writer_input())
    assert report["passed"]
    assert len(report["phenomena_in_results"]) == 5
    assert len(report["phenomena_in_discussion"]) == 5
    assert len(report["phenomena_in_conclusion"]) == 3
    assert len(report["registered_synthesis_hits"]) >= 2
    assert len(report["cross_phenomenon_paragraphs"]) >= 2


def test_generated_article_gate_rejects_unregistered_reference() -> None:
    report = assess_generated_article(
        _article() + "\nREF-UNREGISTERED0000",
        writer_input=_writer_input(),
    )
    assert not report["passed"]
    assert report["unexpected_evidence_tokens"] == ["REF-UNREGISTERED0000"]


def test_generated_article_gate_rejects_token_pile_without_distributed_synthesis() -> None:
    tokens = " ".join(f"PH-{index:016d}" for index in range(5))
    article = "\n\n".join(
        f"## {section}\n{tokens}"
        for section in (
            "Abstract",
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
            "Limitations",
            "Conclusion",
        )
    )

    report = assess_generated_article(article, writer_input=_writer_input())

    assert report["passed"] is False
    assert report["quality_gates"][
        "cross_phenomenon_reasoning_spans_at_least_two_paragraphs"
    ] is False
