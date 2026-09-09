from __future__ import annotations

import json
from typing import Any

from .article_generation import build_graph_article_blueprint

PILOT_CONDITIONS = ("budgeted_one_shot", "budgeted_graph_plan")
PROMPT_VERSION = "same-evidence-budgeted-article-v3-development-20260908"
SECTION_BUDGETS = {
    "Abstract": "150-180 words, exactly 1 paragraph",
    "Introduction": "300-360 words, exactly 3 paragraphs",
    "Methods": "380-450 words, exactly 4 paragraphs",
    "Results": "850-950 words, exactly 8 paragraphs",
    "Discussion": "650-750 words, exactly 5 paragraphs",
    "Limitations": "180-220 words, exactly 1 paragraph",
    "Conclusion": "120-150 words, exactly 1 paragraph",
}


def compact_graph_argument_plan(writer_input: dict[str, Any]) -> dict[str, Any]:
    blueprint = build_graph_article_blueprint(writer_input)
    return {
        "schema_version": 1,
        "role": "argument_dependency_plan_not_additional_evidence",
        "phenomenon_order": [
            {
                "phenomenon_id": card["phenomenon_id"],
                "task_type": card["task_type"],
                "reference_ids": card["reference_ids"],
            }
            for card in blueprint["phenomenon_cards"]
        ],
        "cross_phenomenon_synthesis": blueprint["cross_phenomenon_synthesis"],
        "dependency_edges": blueprint["dependency_edges"],
        "results_slots": [
            "paragraphs 1-5: one registered phenomenon per paragraph",
            "paragraph 6: connectivity-redundancy synthesis",
            "paragraph 7: centrality-resilience synthesis",
            "paragraph 8: temporal-topology synthesis",
        ],
        "discussion_slots": [
            "paragraphs 1-3: interpret the three registered syntheses",
            "paragraph 4: compare alternative explanations and failure conditions",
            "paragraph 5: bounded implications without causal or field-stage claims",
        ],
    }


def _system_prompt() -> str:
    return (
        "Write a complete, concise, confirmatory scientific field-analysis article. "
        "Use only supplied structured graph results and source excerpts. Never invent "
        "a source, experiment, mechanism, causal relation, or visual observation. Treat "
        "graph structure as corpus- and construction-dependent. Every numeric or "
        "graph-derived statement must cite exact PH and relevant REF identifiers. Return "
        "only the English Markdown article."
    )


def _budget_contract() -> str:
    lines = [f"- {section}: {budget}" for section, budget in SECTION_BUDGETS.items()]
    return (
        "The finished article must be 2700-3100 words. Obey every section budget below.\n"
        + "\n".join(lines)
        + "\nUse exactly seven level-two headings in this exact order: Abstract, "
        "Introduction, Methods, Results, Discussion, Limitations, Conclusion. Do not "
        "use level-three or deeper headings, tables, appendices, provenance maps, "
        "bullet lists, or a references section. Finish all seven sections before "
        "expanding prose. If space is tight, compress wording and repeated numbers; "
        "never omit Limitations or Conclusion."
    )


def _reasoning_contract() -> str:
    return (
        "Use all five registered PH phenomena in Results and Discussion. Results must "
        "contain five focused phenomenon paragraphs followed by three focused synthesis "
        "paragraphs: connectivity-redundancy, centrality-resilience, and temporal-topology. "
        "Discussion must interpret all three syntheses and give alternative explanations "
        "and failure conditions. Conclusion must cite at least three PH identifiers. "
        "Do not merely list identifiers: each synthesis paragraph must explicitly relate "
        "the paired observations. Avoid repeating the same numeric result outside Results. "
        "Use inline PH/REF identifiers only; cite no identifier absent from the input."
    )


def build_budgeted_article_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    model: str = "deepseek-v4-pro",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if condition not in PILOT_CONDITIONS:
        raise ValueError(f"Unsupported article-v3 pilot condition: {condition}")
    plan = (
        compact_graph_argument_plan(writer_input)
        if condition == "budgeted_graph_plan"
        else None
    )
    parts = [
        f"PROMPT_VERSION: {PROMPT_VERSION}",
        _budget_contract(),
        _reasoning_contract(),
    ]
    if plan is not None:
        parts.extend(
            [
                (
                    "Follow this deterministic argument-dependency plan. It is an "
                    "ordering aid, not new evidence; verify every statement against "
                    "WRITER_INPUT_JSON."
                ),
                "ARGUMENT_PLAN_JSON:\n"
                + json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    parts.append(
        "WRITER_INPUT_JSON:\n"
        + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
    )
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        "temperature": 0,
        "max_tokens": 5000,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    return request, plan
