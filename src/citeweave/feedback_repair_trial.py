from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from .io import read_json, sha256_file, write_json

REPAIR_CONDITIONS = ("dependency_compiled", "raw_memory_prompt")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def assess_feedback_repair_trial_readiness(
    worklist_path: Path, *, cases_per_topic: int = 12
) -> dict[str, Any]:
    gaps = []
    if not worklist_path.is_file():
        return {
            "schema_version": 1,
            "status": "blocked",
            "topics": 0,
            "eligible_cases": 0,
            "cases_per_topic": cases_per_topic,
            "gaps": ["validated real-feedback dependency worklist is missing"],
        }
    worklist = read_json(worklist_path)
    if worklist.get("status") != "controlled_paragraph_revision_ready":
        gaps.append("dependency worklist is not ready for controlled revision")
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worklist.get("worklist") or []:
        if not row.get("directives"):
            continue
        by_topic[str(row["dataset_id"])].append(row)
    if len(by_topic) != 8:
        gaps.append("repair trial requires exactly eight registered topic clusters")
    for topic_id, rows in sorted(by_topic.items()):
        paragraph_ids = {str(row["paragraph_id"]) for row in rows}
        if len(paragraph_ids) < cases_per_topic:
            gaps.append(
                f"{topic_id} has {len(paragraph_ids)} eligible paragraphs; "
                f"requires {cases_per_topic}"
            )
    return {
        "schema_version": 1,
        "status": "ready" if not gaps else "blocked",
        "topics": len(by_topic),
        "eligible_cases": sum(len(rows) for rows in by_topic.values()),
        "cases_per_topic": cases_per_topic,
        "topic_case_counts": {
            topic: len({str(row["paragraph_id"]) for row in rows})
            for topic, rows in sorted(by_topic.items())
        },
        "worklist_sha256": sha256_file(worklist_path),
        "gaps": gaps,
    }


def build_feedback_repair_trial_plan(
    worklist_path: Path,
    *,
    output_path: Path,
    cases_per_topic: int = 12,
    seed: int = 20260901,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("Feedback-repair trial plan is already frozen")
    readiness = assess_feedback_repair_trial_readiness(
        worklist_path, cases_per_topic=cases_per_topic
    )
    if readiness["status"] != "ready":
        raise ValueError("; ".join(readiness["gaps"]))
    worklist = read_json(worklist_path)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worklist["worklist"]:
        if row.get("directives"):
            by_topic[str(row["dataset_id"])].append(row)

    cases = []
    cells = []
    for topic_id, rows in sorted(by_topic.items()):
        ordered = sorted(
            rows,
            key=lambda row: _canonical_hash(
                [seed, topic_id, row["paragraph_id"], row["paragraph_sha256"]]
            ),
        )
        selected = ordered[:cases_per_topic]
        if len({row["paragraph_id"] for row in selected}) != cases_per_topic:
            raise ValueError(f"Selected repair cases are not paragraph-distinct: {topic_id}")
        for row in selected:
            feedback_source = [
                {
                    "packet_id": directive["packet_id"],
                    "rationale": directive["rationale"],
                    "action": directive["action"],
                    "replacement": directive.get("replacement"),
                    "invalid_dependency_ids": sorted(
                        directive.get("invalid_dependency_ids") or []
                    ),
                    "guard": directive.get("guard") or {},
                }
                for directive in row["directives"]
            ]
            feedback_hash = _canonical_hash(feedback_source)
            case_id = "FR-" + _canonical_hash(
                [topic_id, row["paragraph_id"], row["paragraph_sha256"], seed]
            )[:16].upper()
            cases.append(
                {
                    "case_id": case_id,
                    "topic_id": topic_id,
                    "paragraph_id": row["paragraph_id"],
                    "original_paragraph_sha256": row["paragraph_sha256"],
                    "feedback_content_sha256": feedback_hash,
                }
            )
            for condition in REPAIR_CONDITIONS:
                if condition == "dependency_compiled":
                    treatment = {
                        "typed_directives": feedback_source,
                        "allowed_evidence_ids": row["allowed_evidence_ids"],
                        "locked_scope": {
                            "paragraph_id": row["paragraph_id"],
                            "original_paragraph_sha256": row["paragraph_sha256"],
                            "unaffected_paragraphs_locked": True,
                        },
                    }
                else:
                    treatment = {
                        "chronological_feedback": [
                            " | ".join(
                                [
                                    str(item["rationale"]),
                                    f"action={item['action']}",
                                    f"replacement={item['replacement'] or ''}",
                                    "invalid=" + ",".join(item["invalid_dependency_ids"]),
                                    "guard="
                                    + json.dumps(
                                        item["guard"], sort_keys=True, ensure_ascii=False
                                    ),
                                ]
                            )
                            for item in feedback_source
                        ],
                        "typed_dependency_graph": None,
                        "promotion_or_scope_guard": None,
                    }
                cells.append(
                    {
                        "case_id": case_id,
                        "topic_id": topic_id,
                        "condition": condition,
                        "original_text": row["original_text"],
                        "feedback_content_sha256": feedback_hash,
                        "treatment": treatment,
                        "generation_contract": {
                            "same_model_temperature_token_budget": True,
                            "one_provider_response": True,
                            "quality_resampling": False,
                        },
                    }
                )
    result = {
        "schema_version": 1,
        "status": "frozen_ready_to_generate",
        "seed": seed,
        "worklist_path": str(worklist_path.resolve()),
        "worklist_sha256": sha256_file(worklist_path),
        "topics": len(by_topic),
        "cases_per_topic": cases_per_topic,
        "cases": len(cases),
        "logical_cells": len(cells),
        "conditions": list(REPAIR_CONDITIONS),
        "cases_manifest": cases,
        "cells": cells,
        "interference_control": (
            "Each cell starts from an independent copy of one original paragraph; "
            "paragraph-distinct cases are never composed into one article during this trial."
        ),
    }
    write_json(output_path, result)
    write_json(
        output_path.with_suffix(".freeze.json"),
        {
            "schema_version": 1,
            "sha256": sha256_file(output_path),
            "provider_outcomes_present": False,
            "human_evaluation_outcomes_present": False,
        },
    )
    return result


@dataclass(frozen=True)
class RepairTrialOutcome:
    case_id: str
    topic_id: str
    condition: Literal["dependency_compiled", "raw_memory_prompt"]
    primary_reviewer_ids: tuple[str, str]
    resolution_method: Literal["agreement", "adjudication"]
    adjudicator_id: str | None
    targeted_defects_corrected: bool
    unaffected_claims_preserved: bool
    new_unsupported_claims: int
    domain_specificity_regression: bool
    revision_seconds: float
    input_tokens: int
    output_tokens: int
    evaluation_seconds: float


def _exact_signflip(effects: list[float]) -> dict[str, Any]:
    if len(effects) != 8 or any(not math.isfinite(value) for value in effects):
        raise ValueError("Primary repair analysis requires eight finite topic effects")
    observed = mean(effects)
    null = [
        mean(sign * value for sign, value in zip(signs, effects, strict=True))
        for signs in itertools.product((-1.0, 1.0), repeat=8)
    ]
    return {
        "estimate": observed,
        "topics": 8,
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in null)
        / len(null),
        "enumerated_sign_patterns": len(null),
        "minimum_attainable_p": 1 / len(null),
    }


def _holm(p_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    running = 0.0
    result = {}
    total = len(ordered)
    for index, name in enumerate(ordered):
        adjusted = min(1.0, (total - index) * p_values[name])
        running = max(running, adjusted)
        result[name] = {
            "raw_p": p_values[name],
            "holm_adjusted_p": running,
            "reject_at_0_05": running < 0.05,
        }
    return result


def analyze_feedback_repair_trial(
    outcomes: list[RepairTrialOutcome], *, cases_per_topic: int = 12
) -> dict[str, Any]:
    indexed = {(row.case_id, row.condition): row for row in outcomes}
    if len(indexed) != len(outcomes):
        raise ValueError("Duplicate repair case-condition outcome")
    case_ids = sorted({row.case_id for row in outcomes})
    if len(case_ids) != 8 * cases_per_topic:
        raise ValueError("Repair trial requires exactly eight topics × registered cases")
    expected = {
        (case_id, condition) for case_id in case_ids for condition in REPAIR_CONDITIONS
    }
    if set(indexed) != expected:
        raise ValueError("Every repair case requires both registered conditions")

    topic_cases: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        rows = [indexed[(case_id, condition)] for condition in REPAIR_CONDITIONS]
        if len({row.topic_id for row in rows}) != 1:
            raise ValueError(f"Paired repair topic mismatch: {case_id}")
        topic_cases[rows[0].topic_id].append(case_id)
    if len(topic_cases) != 8 or any(
        len(rows) != cases_per_topic for rows in topic_cases.values()
    ):
        raise ValueError("Repair trial must have equal case counts in eight topics")

    for row in outcomes:
        reviewers = tuple(row.primary_reviewer_ids)
        if len(reviewers) != 2 or len(set(reviewers)) != 2:
            raise ValueError("Each repair variant needs two independent primary reviewers")
        if row.resolution_method == "adjudication":
            if not row.adjudicator_id or row.adjudicator_id in reviewers:
                raise ValueError("Adjudication requires an independent third reviewer")
        elif row.adjudicator_id is not None:
            raise ValueError("Agreement-resolved variants must not name an adjudicator")
        if (
            row.new_unsupported_claims < 0
            or row.revision_seconds <= 0
            or row.input_tokens <= 0
            or row.output_tokens <= 0
            or row.evaluation_seconds <= 0
        ):
            raise ValueError("Repair counts and server-accounted times must be valid")

    def strict(row: RepairTrialOutcome) -> float:
        return float(
            row.targeted_defects_corrected
            and row.unaffected_claims_preserved
            and row.new_unsupported_claims == 0
            and not row.domain_specificity_regression
        )

    def collateral(row: RepairTrialOutcome) -> float:
        return float(
            not row.unaffected_claims_preserved
            or row.new_unsupported_claims > 0
            or row.domain_specificity_regression
        )

    strict_effects = {}
    collateral_effects = {}
    time_effects = {}
    token_effects = {}
    for topic_id, topic_case_ids in sorted(topic_cases.items()):
        paired = [
            (
                indexed[(case_id, "dependency_compiled")],
                indexed[(case_id, "raw_memory_prompt")],
            )
            for case_id in topic_case_ids
        ]
        strict_effects[topic_id] = mean(strict(left) - strict(right) for left, right in paired)
        collateral_effects[topic_id] = mean(
            collateral(right) - collateral(left) for left, right in paired
        )
        time_effects[topic_id] = mean(
            right.revision_seconds - left.revision_seconds for left, right in paired
        )
        token_effects[topic_id] = mean(
            (right.input_tokens + right.output_tokens)
            - (left.input_tokens + left.output_tokens)
            for left, right in paired
        )

    r1 = _exact_signflip(list(strict_effects.values()))
    r2 = _exact_signflip(list(collateral_effects.values()))
    multiplicity = _holm(
        {"R1_strict_repair_success": r1["p_value_one_sided"],
         "R2_collateral_damage": r2["p_value_one_sided"]}
    )
    r1["topic_effects"] = strict_effects
    r1["multiplicity"] = multiplicity["R1_strict_repair_success"]
    r2["topic_effects"] = collateral_effects
    r2["multiplicity"] = multiplicity["R2_collateral_damage"]
    joint = (
        r1["estimate"] > 0
        and r2["estimate"] > 0
        and r1["multiplicity"]["reject_at_0_05"]
        and r2["multiplicity"]["reject_at_0_05"]
    )
    return {
        "schema_version": 1,
        "status": "feedback_repair_mechanism_analysis_complete",
        "topics": 8,
        "cases": len(case_ids),
        "outcomes": len(outcomes),
        "registered_contrasts": {
            "R1_compiled_vs_raw_strict_success": r1,
            "R2_compiled_vs_raw_collateral_damage": r2,
        },
        "joint_mechanism_success": joint,
        "cost_descriptives": {
            "raw_minus_compiled_revision_seconds_per_case": mean(time_effects.values()),
            "raw_minus_compiled_total_tokens_per_case": mean(token_effects.values()),
            "topic_revision_time_effects": time_effects,
            "topic_token_effects": token_effects,
        },
        "interpretation_guard": (
            "The trial isolates repair representation for the same resolved human feedback. "
            "It does not estimate reviewer recruitment cost, generalize beyond the eight topic "
            "clusters, or permit a human-superiority claim without the separate article panel."
        ),
    }
