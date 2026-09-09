from __future__ import annotations

import re
from typing import Any

from .article_generation import REQUIRED_SECTIONS

TOKEN_PATTERN = re.compile(r"\b(?:PH|REF)-[A-Za-z0-9_-]+\b")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def hierarchical_section_text(article: str, section: str) -> str:
    """Return a Markdown section while retaining nested subsections.

    The frozen generation gate stopped at every subsequent Markdown heading.  That
    made the first ``###`` subsection terminate a containing ``## Results`` or
    ``## Discussion`` section.  This diagnostic parser stops only at a heading of
    the same or a higher level and deliberately leaves the frozen gate untouched.
    """

    lines = article.splitlines()
    target = section.strip().casefold()
    start: int | None = None
    target_level: int | None = None
    for index, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        if match.group(2).strip().casefold() == target:
            start = index + 1
            target_level = len(match.group(1))
            break
    if start is None or target_level is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        match = HEADING_PATTERN.match(lines[index])
        if match and len(match.group(1)) <= target_level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _prose_paragraphs(section: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section)
        if paragraph.strip()
        and not paragraph.lstrip().startswith(("#", "|", "```"))
    ]


def assess_generated_article_hierarchically(
    article: str,
    *,
    writer_input: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the frozen gates with only section-boundary parsing corrected."""

    permitted_low, permitted_high = writer_input["writing_brief"]["permitted_range"]
    words = len(WORD_PATTERN.findall(article))
    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if not hierarchical_section_text(article, section)
    ]
    allowed_ph = {row["phenomenon_id"] for row in writer_input["graph_phenomena"]}
    phenomenon_by_type = {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }
    required_types = {
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    if set(phenomenon_by_type) != required_types:
        raise ValueError("Article assessment requires all five registered phenomenon types")
    registered_synthesis_pairs = {
        "connectivity_redundancy": frozenset(
            {
                phenomenon_by_type["multi_hop_connector"],
                phenomenon_by_type["bridge_counterfactual"],
            }
        ),
        "centrality_resilience": frozenset(
            {
                phenomenon_by_type["community_role_contrast"],
                phenomenon_by_type["hub_removal_resilience"],
            }
        ),
        "temporal_topology": frozenset(
            {
                phenomenon_by_type["temporal_structural_shift"],
                phenomenon_by_type["community_role_contrast"],
            }
        ),
    }
    allowed_refs = {
        row["reference_id"] for row in writer_input["representative_sources"]
    }
    observed_tokens = set(TOKEN_PATTERN.findall(article))
    unexpected_tokens = sorted(observed_tokens - allowed_ph - allowed_refs)
    results_tokens = set(
        TOKEN_PATTERN.findall(hierarchical_section_text(article, "Results"))
    )
    discussion_tokens = set(
        TOKEN_PATTERN.findall(hierarchical_section_text(article, "Discussion"))
    )
    conclusion_tokens = set(
        TOKEN_PATTERN.findall(hierarchical_section_text(article, "Conclusion"))
    )
    phenomena_in_results = allowed_ph & results_tokens
    phenomena_in_discussion = allowed_ph & discussion_tokens
    phenomena_in_conclusion = allowed_ph & conclusion_tokens
    synthesis_hits: set[str] = set()
    synthesis_paragraphs: list[dict[str, Any]] = []
    for section_name in ("Results", "Discussion"):
        for paragraph_index, paragraph in enumerate(
            _prose_paragraphs(hierarchical_section_text(article, section_name)), start=1
        ):
            paragraph_phenomena = allowed_ph & set(TOKEN_PATTERN.findall(paragraph))
            matched = sorted(
                synthesis_id
                for synthesis_id, pair in registered_synthesis_pairs.items()
                if pair.issubset(paragraph_phenomena)
            )
            if matched and 2 <= len(paragraph_phenomena) < len(allowed_ph):
                synthesis_hits.update(matched)
                synthesis_paragraphs.append(
                    {
                        "section": section_name,
                        "paragraph_index": paragraph_index,
                        "synthesis_ids": matched,
                        "phenomenon_ids": sorted(paragraph_phenomena),
                    }
                )
    gates = {
        "word_range": permitted_low <= words <= permitted_high,
        "required_sections": not missing_sections,
        "no_unregistered_evidence_tokens": not unexpected_tokens,
        "all_phenomena_used": allowed_ph.issubset(observed_tokens),
        "all_five_phenomena_in_results": allowed_ph.issubset(phenomena_in_results),
        "at_least_four_phenomena_in_discussion": len(phenomena_in_discussion) >= 4,
        "at_least_three_phenomena_in_conclusion": len(phenomena_in_conclusion) >= 3,
        "at_least_two_registered_cross_phenomenon_syntheses": len(synthesis_hits) >= 2,
        "cross_phenomenon_reasoning_spans_at_least_two_paragraphs": (
            len(synthesis_paragraphs) >= 2
        ),
        "no_rendered_image_markup": "![" not in article,
    }
    return {
        "schema_version": 1,
        "assessment_role": "post_generation_secondary_parser_diagnostic",
        "primary_gate_reclassified": False,
        "passed_all_recomputed_gates": all(gates.values()),
        "word_count": words,
        "permitted_word_range": [permitted_low, permitted_high],
        "missing_sections": missing_sections,
        "unexpected_evidence_tokens": unexpected_tokens,
        "phenomena_in_results": sorted(phenomena_in_results),
        "phenomena_in_discussion": sorted(phenomena_in_discussion),
        "phenomena_in_conclusion": sorted(phenomena_in_conclusion),
        "registered_synthesis_hits": sorted(synthesis_hits),
        "cross_phenomenon_paragraphs": synthesis_paragraphs,
        "quality_gates": gates,
    }
