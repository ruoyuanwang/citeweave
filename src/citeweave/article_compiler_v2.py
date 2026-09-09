from __future__ import annotations

import json
from typing import Any

from .article_compiler import SECTION_LIMITS
from .article_generation_v3 import compact_graph_argument_plan

COMPILER_CONDITIONS = ("flat_article_compiler", "graph_dependency_compiler")
PROMPT_VERSION = "matched-article-compiler-claim-ready-development-20260908"


def _phenomenon_by_type(writer_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_type = {row["task_type"]: row for row in writer_input["graph_phenomena"]}
    required = {
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    if set(by_type) != required:
        raise ValueError("Matched compiler requires the five registered phenomenon types")
    return by_type


def flat_argument_records(writer_input: dict[str, Any]) -> dict[str, Any]:
    graph_plan = compact_graph_argument_plan(writer_input)
    synthesis_membership: dict[str, list[str]] = {}
    for synthesis in graph_plan["cross_phenomenon_synthesis"]:
        for phenomenon_id in synthesis["phenomenon_ids"]:
            synthesis_membership.setdefault(phenomenon_id, []).append(
                synthesis["synthesis_id"]
            )
    return {
        "schema_version": 1,
        "role": "flat_same_information_argument_records",
        "records": [
            {
                "phenomenon_id": row["phenomenon_id"],
                "task_type": row["task_type"],
                "reference_ids": row["reference_ids"],
                "synthesis_ids": sorted(synthesis_membership[row["phenomenon_id"]]),
            }
            for row in graph_plan["phenomenon_order"]
        ],
        "syntheses": graph_plan["cross_phenomenon_synthesis"],
        "results_slots": graph_plan["results_slots"],
        "discussion_slots": graph_plan["discussion_slots"],
    }


def argument_payload(writer_input: dict[str, Any], *, condition: str) -> dict[str, Any]:
    if condition == "flat_article_compiler":
        return flat_argument_records(writer_input)
    if condition == "graph_dependency_compiler":
        return compact_graph_argument_plan(writer_input)
    raise ValueError(f"Unsupported compiler condition: {condition}")


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


def _claim_density_contract(section: str, writer_input: dict[str, Any]) -> str:
    ph = _phenomenon_by_type(writer_input)
    ids = {task_type: row["phenomenon_id"] for task_type, row in ph.items()}
    if section == "Results":
        return (
            "Write exactly eight paragraphs. The first two sentences of every paragraph "
            "must each be a self-contained atomic reviewable claim containing at least "
            "one literal PH identifier and a relevant REF identifier; this creates at "
            "least sixteen distinct PH-bearing claim sentences. Paragraphs 1-5 each "
            "analyze one registered phenomenon. Paragraph 6 relates "
            f"{ids['multi_hop_connector']} with {ids['bridge_counterfactual']}; paragraph "
            f"7 relates {ids['community_role_contrast']} with "
            f"{ids['hub_removal_resilience']}; paragraph 8 relates "
            f"{ids['temporal_structural_shift']} with "
            f"{ids['community_role_contrast']}. Use 4-6 sentences per paragraph."
        )
    if section == "Discussion":
        all_ids = ", ".join(row["phenomenon_id"] for row in writer_input["graph_phenomena"])
        return (
            "Write exactly five paragraphs. The first two sentences of every paragraph "
            "must each be a self-contained atomic reviewable claim containing literal PH "
            "and relevant REF identifiers; this creates at least ten distinct PH-bearing "
            "claim sentences. Paragraphs 1-3 interpret respectively "
            f"{ids['multi_hop_connector']} with {ids['bridge_counterfactual']}; "
            f"{ids['community_role_contrast']} with {ids['hub_removal_resilience']}; "
            f"and {ids['temporal_structural_shift']} with "
            f"{ids['community_role_contrast']}. Paragraph 4 compares alternatives and "
            f"failure conditions using all five IDs ({all_ids}). Paragraph 5 states "
            "bounded implications using at least three PH IDs. Use 4-6 sentences per "
            "paragraph and do not repeat numeric detail."
        )
    return ""


def _section_contract(section: str, writer_input: dict[str, Any]) -> str:
    if section in {"Results", "Discussion"}:
        return _claim_density_contract(section, writer_input)
    if section == "Methods":
        return (
            "Explain corpus scope, graph construction, the five registered operator "
            "families, synthesis procedure, PH/REF provenance, and interpretation guard. "
            "Do not report findings."
        )
    if section == "Limitations":
        return (
            "Cover corpus selection, graph construction, algorithm sensitivity, temporal "
            "windows, source-excerpt sufficiency, and non-causal interpretation."
        )
    if section == "Conclusion":
        ids = [row["phenomenon_id"] for row in writer_input["graph_phenomena"]]
        return (
            "Write one bounded synthesis paragraph citing at least these three literal "
            f"identifiers: {ids[0]}, {ids[2]}, {ids[4]}. Do not add a new claim."
        )
    if section == "Introduction":
        return (
            "Motivate why local frequency summaries are insufficient and preview the "
            "registered structural questions without revealing detailed result values."
        )
    if section == "Abstract":
        return (
            "Summarize objective, evidence, deterministic graph methods, bounded findings, "
            "and limitations in one paragraph with at least three PH and three REF IDs."
        )
    raise ValueError(f"Unsupported compiler section: {section}")


def build_compiler_section_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    section: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    if section not in SECTION_LIMITS:
        raise ValueError(f"Unsupported compiler section: {section}")
    low, high, max_tokens = SECTION_LIMITS[section]
    if section == "Results":
        max_tokens = 1800
    elif section == "Discussion":
        max_tokens = 1450
    payload = argument_payload(writer_input, condition=condition)
    prompt = (
        f"PROMPT_VERSION: {PROMPT_VERSION}\nCONDITION: {condition}\nGenerate only "
        f"the body of the {section} section, without a heading. Write {low}-{high} "
        f"words and never exceed {high}. Use prose only. "
        + _section_contract(section, writer_input)
        + "\nUse only WRITER_INPUT_JSON and claims already present in "
        "DEPENDENCY_CONTEXT_JSON. ARGUMENT_PAYLOAD_JSON is an ordering aid, not new "
        "evidence. Never invent an identifier, number, mechanism, causal relation, or "
        "visual observation.\n\nARGUMENT_PAYLOAD_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nDEPENDENCY_CONTEXT_JSON:\n"
        + json.dumps(
            _dependency_context(section, compiled_sections),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\nWRITER_INPUT_JSON:\n"
        + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are one node in a matched scientific article compiler. Every "
                    "numeric or graph-derived claim must cite exact supplied PH and "
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


def build_compiler_repair_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    section: str,
    original_body: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    if section not in {"Results", "Discussion"}:
        raise ValueError("Matched compiler repairs only Results and Discussion")
    low, high, _ = SECTION_LIMITS[section]
    max_tokens = 2100 if section == "Results" else 1700
    payload = argument_payload(writer_input, condition=condition)
    prompt = (
        f"PROMPT_VERSION: {PROMPT_VERSION}-repair\nCONDITION: {condition}\nRepair "
        f"only the {section} body below. Write {low}-{high} words and finish naturally. "
        "Return prose only with no heading. Preserve supported meaning and complete every "
        "PH/REF token; do not add evidence. "
        + _section_contract(section, writer_input)
        + "\n\nARGUMENT_PAYLOAD_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\nDEPENDENCY_CONTEXT_JSON:\n"
        + json.dumps(
            _dependency_context(section, compiled_sections),
            ensure_ascii=False,
            separators=(",", ":"),
        )
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
                    "You are the precommitted second pass of a scientific article compiler. "
                    "Preserve evidence and meaning, enforce atomic claim density, and never "
                    "invent content. Return only the repaired section body."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
