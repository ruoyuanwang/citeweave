from __future__ import annotations

from citeweave.article_context_compaction import (
    audit_section_writer_view,
    section_writer_view,
)


def _writer_input() -> dict:
    return {
        "dataset_id": "topic",
        "writing_brief": {"permitted_range": [2700, 3300]},
        "condition_contract": {"same_evidence": True},
        "graph_phenomena": [
            {
                "phenomenon_id": "PH-1",
                "task_type": "multi_hop_connector",
                "reference_ids": ["REF-1"],
                "verified_answer": {"path": ["a", "b"]},
            }
        ],
        "representative_sources": [
            {
                "reference_id": "REF-1",
                "title": "Title",
                "year": 2025,
                "matched_keyword": "keyword",
                "abstract_excerpt": "Exact evidence excerpt.",
                "retrieval_query": "large duplicated retrieval metadata",
                "cited_by_count": 42,
            }
        ],
        "nonvisual_figure_metadata": {"full_graph_nodes": 100},
    }


def test_full_evidence_sections_preserve_exact_excerpts_and_phenomena() -> None:
    writer = _writer_input()
    for section in ("Introduction", "Results", "Discussion"):
        view = section_writer_view(writer, section=section)
        audit = audit_section_writer_view(writer, view)
        assert audit["passed"] is True
        assert view["graph_phenomena"] == writer["graph_phenomena"]
        assert view["representative_sources"][0]["abstract_excerpt"] == (
            "Exact evidence excerpt."
        )
        assert "retrieval_query" not in view["representative_sources"][0]


def test_structural_sections_keep_registry_but_omit_excerpts() -> None:
    writer = _writer_input()
    for section in ("Abstract", "Methods", "Limitations", "Conclusion"):
        view = section_writer_view(writer, section=section)
        audit = audit_section_writer_view(writer, view)
        assert audit["passed"] is True
        assert set(view["representative_sources"][0]) == {
            "reference_id",
            "title",
            "year",
            "matched_keyword",
        }
        assert audit["view_source_chars"] < audit["original_source_chars"]
