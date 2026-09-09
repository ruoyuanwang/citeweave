from __future__ import annotations

import json
import re
from typing import Any

from .article_generation_v3 import compact_graph_argument_plan

SECTION_ORDER = (
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
)
GENERATION_ORDER = (
    "Methods",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
    "Introduction",
    "Abstract",
)
SECTION_LIMITS = {
    "Abstract": (170, 190, 400),
    "Introduction": (330, 360, 650),
    "Methods": (410, 450, 800),
    "Results": (900, 980, 1600),
    "Discussion": (700, 760, 1250),
    "Limitations": (220, 250, 500),
    "Conclusion": (150, 170, 400),
}
PROMPT_VERSION = "graph-dependency-article-compiler-development-20260908"


def _phenomenon_ids(writer_input: dict[str, Any]) -> dict[str, str]:
    return {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }


def _section_specific_contract(section: str, writer_input: dict[str, Any]) -> str:
    ph = _phenomenon_ids(writer_input)
    if section == "Results":
        return (
            "Write exactly eight paragraphs. Paragraphs 1-5 each analyze one of the "
            "five supplied PH phenomena and cite its exact PH plus relevant REF IDs. "
            f"Paragraph 6 must relate {ph['multi_hop_connector']} and "
            f"{ph['bridge_counterfactual']}; paragraph 7 must relate "
            f"{ph['community_role_contrast']} and {ph['hub_removal_resilience']}; "
            f"paragraph 8 must relate {ph['temporal_structural_shift']} and "
            f"{ph['community_role_contrast']}. Report observations without causal or "
            "field-stage claims."
        )
    if section == "Discussion":
        return (
            "Write exactly five paragraphs. Each paragraph's first sentence must cite "
            "literal PH identifiers. Paragraphs 1-3 interpret respectively: "
            f"{ph['multi_hop_connector']} with {ph['bridge_counterfactual']}; "
            f"{ph['community_role_contrast']} with {ph['hub_removal_resilience']}; "
            f"{ph['temporal_structural_shift']} with {ph['community_role_contrast']}. "
            "Paragraph 4 compares alternative explanations and failure conditions using "
            "all five PH IDs. Paragraph 5 states bounded research implications using at "
            "least three PH IDs. Cite relevant REF IDs and do not repeat numeric detail."
        )
    if section == "Methods":
        return (
            "Explain corpus scope, graph construction, five registered operator families, "
            "cross-phenomenon synthesis, PH/REF provenance, and the interpretation guard. "
            "Do not report findings."
        )
    if section == "Limitations":
        return (
            "Cover corpus selection, graph construction, community/centrality algorithm, "
            "temporal windows, source-excerpt sufficiency, and non-causal interpretation. "
            "Relate these limitations to supplied PH evidence without adding findings."
        )
    if section == "Conclusion":
        ids = [row["phenomenon_id"] for row in writer_input["graph_phenomena"]]
        return (
            "Write one bounded synthesis paragraph citing at least these three literal "
            f"identifiers: {ids[0]}, {ids[2]}, {ids[4]}. Do not introduce new claims."
        )
    if section == "Introduction":
        return (
            "Motivate the field-analysis question using supplied sources, state why local "
            "frequency summaries are insufficient, and preview the registered structural "
            "questions. Do not reveal detailed result values."
        )
    if section == "Abstract":
        return (
            "Summarize objective, evidence, deterministic graph methods, principal bounded "
            "findings, and limitations in one paragraph. Cite at least three PH and three "
            "REF identifiers."
        )
    raise ValueError(f"Unsupported article section: {section}")


def _dependency_context(section: str, compiled: dict[str, str]) -> dict[str, str]:
    dependencies = {
        "Discussion": ("Results",),
        "Limitations": ("Methods", "Results"),
        "Conclusion": ("Results", "Discussion", "Limitations"),
        "Introduction": ("Methods",),
        "Abstract": (
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
            "Limitations",
            "Conclusion",
        ),
    }
    return {name: compiled[name] for name in dependencies.get(section, ())}


def build_section_request(
    writer_input: dict[str, Any],
    *,
    section: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    if section not in SECTION_ORDER:
        raise ValueError(f"Unsupported article section: {section}")
    low, high, max_tokens = SECTION_LIMITS[section]
    argument_plan = compact_graph_argument_plan(writer_input)
    dependency_context = _dependency_context(section, compiled_sections)
    prompt = (
        f"PROMPT_VERSION: {PROMPT_VERSION}\n"
        f"Generate only the body of the {section} section, with no heading. Write "
        f"{low}-{high} words; never exceed {high} words. "
        "Use prose paragraphs only: no headings, lists, tables, appendices, references "
        "section, or prefatory commentary. "
        + _section_specific_contract(section, writer_input)
        + "\nUse only evidence in WRITER_INPUT_JSON and claims already present in "
        "DEPENDENCY_CONTEXT_JSON. The argument plan is an ordering and provenance aid, "
        "not additional evidence.\n\nARGUMENT_PLAN_JSON:\n"
        + json.dumps(argument_plan, ensure_ascii=False, separators=(",", ":"))
        + "\n\nDEPENDENCY_CONTEXT_JSON:\n"
        + json.dumps(dependency_context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nWRITER_INPUT_JSON:\n"
        + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are one node in a dependency-aware scientific article compiler. "
                    "Never invent evidence, mechanisms, causality, or visual observations. "
                    "Every graph-derived or numeric claim must cite exact supplied PH and "
                    "relevant REF identifiers. Return only the requested section body."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }


def normalize_section_body(body: str, section: str) -> str:
    text = body.strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        rf"^#+\s+{re.escape(section)}\s*\n+", "", text, flags=re.IGNORECASE
    )
    return text.strip()


def assemble_article(compiled_sections: dict[str, str]) -> str:
    if set(compiled_sections) != set(SECTION_ORDER):
        raise ValueError("Article compiler requires all seven sections")
    return (
        "\n\n".join(
            f"## {section}\n\n{normalize_section_body(compiled_sections[section], section)}"
            for section in SECTION_ORDER
        )
        + "\n"
    )
