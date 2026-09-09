from __future__ import annotations

from citeweave.article_generation_diagnostic import (
    assess_generated_article_hierarchically,
    hierarchical_section_text,
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
        "writing_brief": {"permitted_range": [1, 1000]},
        "graph_phenomena": [
            {"phenomenon_id": f"PH-{index}", "task_type": task_type}
            for index, task_type in enumerate(task_types)
        ],
        "representative_sources": [
            {"reference_id": f"REF-{index}"} for index in range(5)
        ],
    }


def test_hierarchical_section_keeps_nested_subsections() -> None:
    article = """## Results
### Finding A
PH-0 and PH-1 with REF-0.

### Finding B
PH-2.

## Discussion
PH-3 and PH-4.
"""
    results = hierarchical_section_text(article, "Results")
    assert "### Finding A" in results
    assert "PH-2" in results
    assert "PH-3" not in results


def test_secondary_assessment_recovers_nested_phenomenon_mentions() -> None:
    ph = " ".join(f"PH-{index}" for index in range(5))
    sections = [
        "## Abstract\nSummary.",
        "## Introduction\nContext.",
        "## Methods\nMethod.",
        f"## Results\n### Findings\n{ph}.\n\nPH-0 PH-1.",
        f"## Discussion\n### Synthesis\n{ph}.\n\nPH-0 PH-1.\n\nPH-2 PH-3.",
        "## Limitations\nLimited.",
        "## Conclusion\nPH-0 PH-1 PH-2.",
    ]
    report = assess_generated_article_hierarchically(
        "\n\n".join(sections), writer_input=_writer_input()
    )
    assert report["missing_sections"] == []
    assert len(report["phenomena_in_results"]) == 5
    assert len(report["phenomena_in_discussion"]) == 5
    assert report["quality_gates"]["all_five_phenomena_in_results"] is True
