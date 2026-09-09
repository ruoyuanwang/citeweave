from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json

MACHINE_ARTICLE_CONDITIONS = ("one_shot_llm", "citeweave_graph_review")
ARTICLE_GENERATION_PROMPT_VERSION = "same-evidence-article-v1-20260827"
REQUIRED_SECTIONS = (
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_graph_article_blueprint(writer_input: dict[str, Any]) -> dict[str, Any]:
    phenomena = writer_input.get("graph_phenomena") or []
    sources = writer_input.get("representative_sources") or []
    if len(phenomena) != 5:
        raise ValueError("CiteWeave article blueprint requires exactly five phenomena")
    source_ids = {row["reference_id"] for row in sources}
    if len(source_ids) < 8:
        raise ValueError("Article blueprint requires at least eight distinct sources")
    phenomenon_by_type = {row["task_type"]: row for row in phenomena}
    required_types = {
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    if set(phenomenon_by_type) != required_types:
        raise ValueError("Writer input does not contain the five registered task types")
    cards = []
    for phenomenon in phenomena:
        references = list(phenomenon.get("reference_ids") or [])
        if len(references) < 2 or not set(references).issubset(source_ids):
            raise ValueError(
                f"Phenomenon source binding is incomplete: {phenomenon['phenomenon_id']}"
            )
        cards.append(
            {
                "phenomenon_id": phenomenon["phenomenon_id"],
                "task_type": phenomenon["task_type"],
                "structural_question": phenomenon["question"],
                "verified_answer": phenomenon["verified_answer"],
                "operator_sequence": [
                    row["operator"] for row in phenomenon["operator_trace"]
                ],
                "operator_trace": phenomenon["operator_trace"],
                "graph_evidence_ids": phenomenon["graph_evidence_ids"],
                "reference_ids": references,
                "allowed_interpretation": phenomenon["interpretation_contract"][
                    "allowed"
                ],
                "forbidden_interpretation": phenomenon["interpretation_contract"][
                    "forbidden"
                ],
                "required_limitation": phenomenon["interpretation_contract"][
                    "required_limitation"
                ],
            }
        )
    type_to_id = {
        card["task_type"]: card["phenomenon_id"] for card in cards
    }
    synthesis_claims = [
        {
            "synthesis_id": "SYN-CONNECTIVITY-REDUNDANCY",
            "phenomenon_ids": [
                type_to_id["multi_hop_connector"],
                type_to_id["bridge_counterfactual"],
            ],
            "question": (
                "Does the observed cross-community connection depend on one "
                "irreplaceable relation, or is it structurally redundant?"
            ),
            "required_reasoning": (
                "Combine the path result with edge-deletion behavior; do not infer "
                "information flow or causality."
            ),
        },
        {
            "synthesis_id": "SYN-CENTRALITY-RESILIENCE",
            "phenomenon_ids": [
                type_to_id["community_role_contrast"],
                type_to_id["hub_removal_resilience"],
            ],
            "question": (
                "Does structural prominence imply system-level dependence on the "
                "prominent community or hub?"
            ),
            "required_reasoning": (
                "Contrast aggregate role with the node-deletion counterfactual and "
                "name the replacement structure."
            ),
        },
        {
            "synthesis_id": "SYN-TEMPORAL-STRUCTURAL",
            "phenomenon_ids": [
                type_to_id["temporal_structural_shift"],
                type_to_id["community_role_contrast"],
            ],
            "question": (
                "How does recent growth relate to the established topology without "
                "equating growth with scientific importance?"
            ),
            "required_reasoning": (
                "Join temporal aggregation and topology, then provide at least one "
                "indexing- or window-based alternative explanation."
            ),
        },
    ]
    blueprint = {
        "schema_version": 1,
        "status": "deterministic_graph_article_blueprint",
        "dataset_id": writer_input["dataset_id"],
        "writer_input_content_sha256": _canonical_hash(writer_input),
        "phenomenon_cards": cards,
        "cross_phenomenon_synthesis": synthesis_claims,
        "section_plan": {
            "Results": [
                "Report all five structural observations with exact PH/REF provenance.",
                "Develop the three registered cross-phenomenon syntheses.",
            ],
            "Discussion": [
                "Interpret each synthesis using only supplied source excerpts.",
                "For every synthesis provide an alternative explanation and failure condition.",
            ],
            "Limitations": [
                "Separate corpus, graph-construction, algorithm, temporal-window, and source-excerpt limitations."
            ],
        },
        "dependency_edges": [
            *[
                {
                    "parent": reference_id,
                    "child": card["phenomenon_id"],
                    "relation": "literature_context_for",
                }
                for card in cards
                for reference_id in card["reference_ids"]
            ],
            *[
                {
                    "parent": phenomenon_id,
                    "child": synthesis["synthesis_id"],
                    "relation": "required_for_synthesis",
                }
                for synthesis in synthesis_claims
                for phenomenon_id in synthesis["phenomenon_ids"]
            ],
        ],
    }
    blueprint["blueprint_sha256"] = _canonical_hash(blueprint)
    return blueprint


def _shared_system_prompt() -> str:
    return (
        "You are producing a confirmatory same-evidence scientific field-analysis "
        "article. Use only the supplied structured graph results and source excerpts. "
        "Never invent a source, experiment, mechanism, causal relation, or visual "
        "observation. Treat graph structure as corpus- and construction-dependent. "
        "Every numeric or graph-derived statement must cite its exact PH identifier "
        "and relevant REF identifiers. Return only the English Markdown article."
    )


def _shared_writing_instruction(writer_input: dict[str, Any]) -> str:
    brief = writer_input["writing_brief"]
    return (
        f"Write {brief['permitted_range'][0]}-{brief['permitted_range'][1]} words. "
        f"Use exactly these top-level sections: {', '.join(REQUIRED_SECTIONS)}. "
        "Use all five PH phenomena across Results and Discussion, not in an isolated "
        "figure-caption paragraph. Develop at least two cross-phenomenon arguments. "
        "For every major interpretation distinguish observation, possible explanation, "
        "alternative explanation, and limitation. Cite only supplied PH-* and REF-* "
        "identifiers. Do not include a provenance appendix outside the registered sections."
    )


def build_article_generation_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    if condition not in MACHINE_ARTICLE_CONDITIONS:
        raise ValueError(f"Unsupported machine article condition: {condition}")
    shared = _shared_writing_instruction(writer_input)
    if condition == "one_shot_llm":
        user_content = (
            f"{shared}\n\nWRITER_INPUT_JSON:\n"
            + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
        )
    else:
        blueprint = build_graph_article_blueprint(writer_input)
        user_content = (
            f"{shared}\n\nFollow the deterministic graph-analysis blueprint. "
            "Use its dependency structure to synthesize phenomena, but verify every "
            "statement against WRITER_INPUT_JSON. This is a pre-review draft: do not "
            "claim that human review has occurred.\n\nGRAPH_BLUEPRINT_JSON:\n"
            + json.dumps(blueprint, ensure_ascii=False, separators=(",", ":"))
            + "\n\nWRITER_INPUT_JSON:\n"
            + json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
        )
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": _shared_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 8000,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    return request


def build_machine_article_generation_plan(
    writer_input_manifest_path: Path,
    *,
    output_dir: Path,
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    manifest = read_json(writer_input_manifest_path)
    if manifest.get("status") != "text_only_same_evidence_writer_inputs_ready":
        raise ValueError("Machine generation requires audited text-only writer inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for record in sorted(manifest["records"], key=lambda row: row["dataset_id"]):
        writer_input_path = Path(record["pack"])
        if sha256_file(writer_input_path) != record["pack_sha256"]:
            raise ValueError(f"Writer input hash mismatch: {record['dataset_id']}")
        writer_input = read_json(writer_input_path)
        topic_dir = output_dir / record["dataset_id"]
        blueprint = build_graph_article_blueprint(writer_input)
        blueprint_path = topic_dir / "graph_blueprint.json"
        write_json(blueprint_path, blueprint)
        for condition in MACHINE_ARTICLE_CONDITIONS:
            request = build_article_generation_request(
                writer_input,
                condition=condition,
                model=model,
            )
            request_path = topic_dir / condition / "request.json"
            write_json(request_path, request)
            cells.append(
                {
                    "cell_id": f"{record['dataset_id']}:{condition}",
                    "dataset_id": record["dataset_id"],
                    "condition": condition,
                    "writer_input": str(writer_input_path.resolve()),
                    "writer_input_sha256": record["pack_sha256"],
                    "blueprint": str(blueprint_path.resolve()),
                    "blueprint_sha256": sha256_file(blueprint_path),
                    "request": str(request_path.resolve()),
                    "request_sha256": sha256_file(request_path),
                    "output_dir": str(request_path.parent.resolve()),
                    "generation_requests_allowed": 1,
                    "rendered_figure_access": False,
                    "status": "planned_not_executed",
                }
            )
    if len(cells) != 16:
        raise ValueError(f"Machine article plan requires 16 cells, got {len(cells)}")
    plan = {
        "schema_version": 1,
        "status": "machine_article_generation_plan_frozen",
        "prompt_version": ARTICLE_GENERATION_PROMPT_VERSION,
        "writer_input_manifest": str(writer_input_manifest_path.resolve()),
        "writer_input_manifest_sha256": sha256_file(writer_input_manifest_path),
        "model": model,
        "topics": 8,
        "conditions": list(MACHINE_ARTICLE_CONDITIONS),
        "cells": cells,
    }
    write_json(output_dir / "plan.json", plan)
    return plan
