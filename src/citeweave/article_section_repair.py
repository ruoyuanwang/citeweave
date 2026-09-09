from __future__ import annotations

import json
from typing import Any

from .article_generation_v3 import compact_graph_argument_plan

REPAIR_SECTIONS = ("Results", "Discussion")
PROMPT_VERSION = "article-compiler-section-repair-development-20260908"


def _phenomenon_ids(writer_input: dict[str, Any]) -> dict[str, str]:
    return {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }


def build_section_repair_request(
    writer_input: dict[str, Any],
    *,
    section: str,
    original_body: str,
    repaired_results: str | None = None,
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    if section not in REPAIR_SECTIONS:
        raise ValueError(f"Unsupported repair section: {section}")
    ph = _phenomenon_ids(writer_input)
    if section == "Results":
        target = "880-950 words"
        max_tokens = 2000
        structure = (
            "Write exactly eight paragraphs of 4-6 sentences each; no paragraph may "
            "exceed 125 words. Paragraphs 1-5 each cover one registered phenomenon. "
            f"Paragraph 6 relates {ph['multi_hop_connector']} and "
            f"{ph['bridge_counterfactual']}; paragraph 7 relates "
            f"{ph['community_role_contrast']} and {ph['hub_removal_resilience']}; "
            f"paragraph 8 relates {ph['temporal_structural_shift']} and "
            f"{ph['community_role_contrast']}."
        )
        dependency = {}
    else:
        target = "650-720 words"
        max_tokens = 1600
        structure = (
            "Write exactly five paragraphs of 4-6 sentences each; no paragraph may "
            "exceed 145 words. Every paragraph's first sentence must contain literal PH "
            f"identifiers. Paragraphs 1-3 use respectively {ph['multi_hop_connector']} "
            f"with {ph['bridge_counterfactual']}; {ph['community_role_contrast']} with "
            f"{ph['hub_removal_resilience']}; and {ph['temporal_structural_shift']} with "
            f"{ph['community_role_contrast']}. Paragraph 4 uses all five PH IDs for "
            "alternative explanations and failure conditions. Paragraph 5 uses at least "
            "three PH IDs for bounded implications."
        )
        dependency = {"repaired_results": repaired_results or ""}
    prompt = (
        f"PROMPT_VERSION: {PROMPT_VERSION}\nRepair only the {section} section below. "
        f"Return only its body with no heading. Write {target} and finish naturally. "
        "Preserve supported claims while removing repetition. Complete every PH/REF token; "
        "never emit a shortened identifier. Do not add a claim, number, PH, or REF absent "
        "from the original body or writer input. Use prose only: no headings, lists, tables, "
        "or commentary. "
        + structure
        + "\n\nARGUMENT_PLAN_JSON:\n"
        + json.dumps(
            compact_graph_argument_plan(writer_input),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\nDEPENDENCY_CONTEXT_JSON:\n"
        + json.dumps(dependency, ensure_ascii=False, separators=(",", ":"))
        + "\n\nORIGINAL_SECTION:\n"
        + original_body
        + "\n\nWRITER_INPUT_JSON:\n"
        + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fail-closed scientific section repair compiler. Preserve "
                    "meaning and provenance, remove repetition, and never invent evidence "
                    "or causal interpretation. Return only the repaired section body."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
