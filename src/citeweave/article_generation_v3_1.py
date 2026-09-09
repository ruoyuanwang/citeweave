from __future__ import annotations

import json
from typing import Any

from .article_generation_v3 import compact_graph_argument_plan

PILOT_CONDITIONS = ("checkpoint_one_shot", "checkpoint_graph_plan")
PROMPT_VERSION = "same-evidence-budgeted-article-v3.1-development-20260908"


def _phenomenon_ids(writer_input: dict[str, Any]) -> dict[str, str]:
    by_type = {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }
    required = {
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    if set(by_type) != required:
        raise ValueError("Article-v3.1 requires the five registered phenomenon types")
    return by_type


def _checkpoint_contract(writer_input: dict[str, Any]) -> str:
    ph = _phenomenon_ids(writer_input)
    all_ids = ", ".join(row["phenomenon_id"] for row in writer_input["graph_phenomena"])
    return f"""LENGTH AND COMPLETION CHECKPOINTS
- Target 2750-2900 words; 3000 words is an absolute ceiling.
- Abstract: 140-160 words, one paragraph.
- Introduction: 300-330 words, three paragraphs.
- Methods: 380-410 words, four paragraphs.
- Results: 820-880 words, eight paragraphs. The draft must still be below 1800 cumulative words when Results ends.
- Discussion: 620-680 words, five paragraphs. No Discussion paragraph may exceed 140 words.
- Limitations: 180-210 words, one paragraph.
- Conclusion: 120-140 words, one paragraph.
- Use exactly seven level-two headings in this order: Abstract, Introduction, Methods, Results, Discussion, Limitations, Conclusion.
- Do not use any other heading, table, list, appendix, provenance map, or references section.
- Start Limitations before 2700 cumulative words. Never omit or truncate Limitations or Conclusion.

DISCUSSION CITATION CHECKPOINTS
- Every Discussion paragraph must contain literal PH identifiers in its first sentence.
- Paragraph 1 first sentence must contain {ph['multi_hop_connector']} and {ph['bridge_counterfactual']}.
- Paragraph 2 first sentence must contain {ph['community_role_contrast']} and {ph['hub_removal_resilience']}.
- Paragraph 3 first sentence must contain {ph['temporal_structural_shift']} and {ph['community_role_contrast']}.
- Paragraph 4 first sentence must contain all five identifiers: {all_ids}.
- Paragraph 5 first sentence must contain at least three of those identifiers.
- Identifiers must support a relational argument, not appear as a detached list."""


def build_checkpoint_article_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    model: str = "deepseek-v4-pro",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if condition not in PILOT_CONDITIONS:
        raise ValueError(f"Unsupported article-v3.1 pilot condition: {condition}")
    argument_plan = (
        compact_graph_argument_plan(writer_input)
        if condition == "checkpoint_graph_plan"
        else None
    )
    instructions = [
        f"PROMPT_VERSION: {PROMPT_VERSION}",
        _checkpoint_contract(writer_input),
        (
            "Use all five PH phenomena in Results and Discussion and at least three in "
            "Conclusion. Results paragraphs 1-5 each analyze one phenomenon; paragraphs "
            "6-8 synthesize connectivity-redundancy, centrality-resilience, and "
            "temporal-topology. For each synthesis distinguish observation, possible "
            "explanation, alternative explanation, and failure condition. Use only exact "
            "PH/REF identifiers supplied below. Do not repeat numeric details outside "
            "Results. Finish all sections before adding elaboration."
        ),
    ]
    if argument_plan is not None:
        instructions.extend(
            [
                (
                    "Use this compact deterministic dependency plan for ordering and "
                    "provenance. It is not additional evidence."
                ),
                "ARGUMENT_PLAN_JSON:\n"
                + json.dumps(argument_plan, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    instructions.append(
        "WRITER_INPUT_JSON:\n"
        + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Produce a complete, concise confirmatory scientific field-analysis "
                        "article using only supplied structured graph results and source "
                        "excerpts. Never invent evidence, mechanisms, causality, or visual "
                        "observations. Every numeric or graph-derived claim must cite exact "
                        "PH and relevant REF identifiers. Return only English Markdown."
                    ),
                },
                {"role": "user", "content": "\n\n".join(instructions)},
            ],
            "temperature": 0,
            "max_tokens": 5000,
            "thinking": {"type": "disabled"},
            "stream": False,
        },
        argument_plan,
    )
