from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any

from .article_claim_review import EVIDENCE_TOKEN, _claim_candidates
from .io import read_json, sha256_file, write_json, write_jsonl

ACTIONS = ("accept", "qualify", "reject", "abstain")
SEVERITIES = ("minor", "major", "critical")
RISK_TYPES = (
    "numeric_error",
    "path_error",
    "missing_evidence",
    "scope_overclaim",
    "causal_overclaim",
    "semantic_reification",
    "parameter_sensitivity_omitted",
    "temporal_instability_omitted",
    "community_instability_omitted",
    "unsupported_domain_mechanism",
    "conflicting_evidence",
    "unanswerable",
)

CAUSAL = re.compile(
    r"\b(?:cause[sd]?|causal|drive[sn]?|driven by|lead(?:s|ing)? to|led to|"
    r"result(?:s|ed)? in|determin(?:e|es|ed)|prove[sd]?|demonstrate[sd]?)\b",
    re.IGNORECASE,
)
SCOPE = re.compile(
    r"\b(?:the entire field|the field as a whole|universally|in all research|"
    r"always|definitively|irrefutably)\b",
    re.IGNORECASE,
)
REIFICATION = re.compile(
    r"\b(?:the (?:cluster|community) (?:is|represents|defines)|"
    r"a core mechanism|research mechanism|scientific mechanism)\b",
    re.IGNORECASE,
)
ROBUSTNESS_ASSERTION = re.compile(
    r"\b(?:stable|robust|invariant|insensitive to|unaffected by)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _opaque(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest.upper()}"


def _safe_numbers(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            values.update(_safe_numbers(child))
    elif isinstance(value, list):
        for child in value:
            values.update(_safe_numbers(child))
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        values.add(str(value))
        if isinstance(value, float) and value.is_integer():
            values.add(str(int(value)))
    elif isinstance(value, str):
        values.update(match.rstrip("%").replace(",", "") for match in NUMBER.findall(value))
    return values


def _number_supported(observed: str, allowed: set[str]) -> bool:
    if observed in allowed:
        return True
    try:
        observed_value = float(observed)
    except ValueError:
        return False
    for candidate in allowed:
        try:
            candidate_value = float(candidate)
        except ValueError:
            continue
        if math.isclose(observed_value, candidate_value, rel_tol=5e-7, abs_tol=5e-7):
            return True
    return False


def _compact_measurement(value: dict[str, Any]) -> dict[str, Any]:
    excluded = {"members", "metrics", "path"}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in excluded:
            continue
        if isinstance(item, dict):
            result[key] = {
                child_key: child_value
                for child_key, child_value in item.items()
                if child_key not in excluded and not isinstance(child_value, (dict, list))
            }
        elif not isinstance(item, list):
            result[key] = item
    return result


def load_perturbation_profile(
    robustness_root: Path,
    *,
    dataset_id: str,
    task_type: str,
) -> dict[str, Any]:
    topic_dir = robustness_root / dataset_id
    if not topic_dir.is_dir():
        return {"available": False, "reason": "topic_robustness_missing", "variants": []}
    variants = []
    for path in sorted(topic_dir.glob("*.json")):
        payload = read_json(path)
        measurement = (payload.get("measurements") or {}).get(task_type)
        if not isinstance(measurement, dict):
            continue
        comparison = (payload.get("comparison") or {}).get(task_type)
        variants.append(
            {
                "variant_id": (payload.get("variant") or {}).get("variant_id", path.stem),
                "family": (payload.get("variant") or {}).get("family"),
                "parameters": {
                    key: value
                    for key, value in (payload.get("variant") or {}).items()
                    if key not in {"variant_id", "family"}
                },
                "measurement": _compact_measurement(measurement),
                "comparison_to_baseline": comparison,
                "artifact_sha256": sha256_file(path),
            }
        )
    return {
        "available": bool(variants),
        "source": str(topic_dir.resolve()),
        "variants": variants,
    }


def empty_review() -> dict[str, Any]:
    return {
        "factual_supported": None,
        "interpretation_calibrated": None,
        "alternative_adequate": None,
        "evidence_sufficient": None,
        "action": None,
        "risk_types": [],
        "severity": None,
        "unsupported_spans": [],
        "decisive_evidence_ids": [],
        "invalid_dependency_ids": [],
        "minimal_revision": None,
        "review_seconds": None,
    }


def _article_record_index(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["article_id"]): row for row in analysis.get("records", [])}


def _linked_material(
    claim: dict[str, Any], writer_input: dict[str, Any], robustness_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phenomena_by_id = {
        str(row["phenomenon_id"]): row for row in writer_input["graph_phenomena"]
    }
    sources_by_id = {
        str(row["reference_id"]): row for row in writer_input["representative_sources"]
    }
    phenomena = [
        phenomena_by_id[value]
        for value in claim["phenomenon_ids"]
        if value in phenomena_by_id
    ]
    reference_ids = {
        token for token in claim["evidence_tokens"] if token in sources_by_id
    }
    reference_ids.update(
        reference_id
        for phenomenon in phenomena
        for reference_id in phenomenon.get("reference_ids", [])
    )
    sources = [sources_by_id[value] for value in sorted(reference_ids)]
    perturbations = {
        phenomenon["phenomenon_id"]: load_perturbation_profile(
            robustness_root,
            dataset_id=writer_input["dataset_id"],
            task_type=phenomenon["task_type"],
        )
        for phenomenon in phenomena
    }
    return phenomena, sources, perturbations


def build_pilot_benchmark(
    *,
    plan_path: Path,
    analysis_path: Path,
    robustness_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize the inspected development articles as a non-confirmatory pilot bench."""
    plan = read_json(plan_path)
    analysis = read_json(analysis_path)
    records = _article_record_index(analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    source_artifacts = []
    for article in plan.get("articles", []):
        article_id = str(article["article_id"])
        if article_id not in records:
            raise ValueError(f"Article is absent from analysis: {article_id}")
        result = records[article_id]
        article_output_dir = Path(article["output_dir"])
        draft_path = article_output_dir / "draft.md"
        execution_record_path = article_output_dir / "execution_record.json"
        writer_input_path = Path(article["writer_input"])
        if (
            not draft_path.is_file()
            or not execution_record_path.is_file()
            or not writer_input_path.is_file()
        ):
            raise FileNotFoundError(f"Pilot article inputs are incomplete: {article_id}")
        if sha256_file(draft_path) != result["draft_sha256"]:
            raise ValueError(f"Draft hash mismatch: {article_id}")
        if sha256_file(writer_input_path) != article["writer_input_sha256"]:
            raise ValueError(f"Writer input hash mismatch: {article_id}")
        draft = draft_path.read_text(encoding="utf-8")
        execution_record = read_json(execution_record_path)
        writer_input = read_json(writer_input_path)
        repair_sections = []
        for section, call_dir in (
            ("Results", "03_results_repair"),
            ("Discussion", "05_discussion_repair"),
        ):
            repair_record_path = article_output_dir / "calls" / call_dir / "execution_record.json"
            if repair_record_path.is_file():
                repair_record = read_json(repair_record_path)
                if (
                    repair_record.get("pass_type") == "repair"
                    and repair_record.get("response_received") is True
                    and repair_record.get("finish_reason") == "stop"
                ):
                    repair_sections.append(section)
        extracted = _claim_candidates(draft)
        if len(extracted) != int(result["reviewable_claim_candidates"]):
            raise ValueError(f"Claim inventory changed for {article_id}")
        source_artifacts.append(
            {
                "article_id": article_id,
                "draft": str(draft_path.resolve()),
                "draft_sha256": sha256_file(draft_path),
                "writer_input": str(writer_input_path.resolve()),
                "writer_input_sha256": sha256_file(writer_input_path),
                "machine_gate_passed": bool(result["machine_gate_passed"]),
                "provider_calls": int(execution_record["provider_calls"]),
                "preexisting_section_repairs": repair_sections,
            }
        )
        for claim in extracted:
            phenomena, sources, perturbations = _linked_material(
                claim, writer_input, robustness_root
            )
            case_id = _opaque(
                "CCP",
                article_id,
                claim["candidate_id"],
                claim["text"],
            )
            case = {
                "schema_version": 1,
                "case_id": case_id,
                "benchmark_split": "pilot_development_only",
                "topic_id": writer_input["dataset_id"],
                "article_id": _opaque("ARTICLE", article_id),
                "section": claim["section"],
                "paragraph_id": claim["paragraph_id"],
                "parent_sentence": claim["text"],
                "atomic_claim": claim["text"],
                "local_context": claim["paragraph_context"],
                "claim_extraction": {
                    "method": "ph_bearing_sentence_candidate_v1",
                    "source_span": [claim["start_char"], claim["end_char"]],
                    "sampling_probability": 1.0,
                    "atomicity_requires_human_confirmation": True,
                },
                "risk_features": claim["risk_features"],
                "graph_scope": {
                    "dataset_id": writer_input["dataset_id"],
                    "writing_brief": writer_input.get("writing_brief"),
                    "nonvisual_figure_metadata": writer_input.get(
                        "nonvisual_figure_metadata"
                    ),
                },
                "phenomena": phenomena,
                "sources": sources,
                "perturbation_profiles": perturbations,
                "allowed_evidence_ids": sorted(
                    {
                        *claim["evidence_tokens"],
                        *(row["phenomenon_id"] for row in phenomena),
                        *(row["reference_id"] for row in sources),
                        *(
                            evidence_id
                            for row in phenomena
                            for evidence_id in row.get("graph_evidence_ids", [])
                        ),
                    }
                ),
                "eligibility": {
                    "machine_gate_passed": bool(result["machine_gate_passed"]),
                    "eligible_for_natural_prevalence": False,
                    "reason": "inspected_development_article",
                },
                "lineage": {
                    "draft_sha256": sha256_file(draft_path),
                    "paragraph_sha256": claim["paragraph_sha256"],
                    "writer_input_sha256": sha256_file(writer_input_path),
                },
                "review": empty_review(),
            }
            case["lineage"]["case_material_sha256"] = canonical_sha256(
                {key: value for key, value in case.items() if key != "review"}
            )
            cases.append(case)
            private_rows.append(
                {
                    "case_id": case_id,
                    "source_article_id": article_id,
                    "condition": article["condition"],
                    "candidate_id": claim["candidate_id"],
                    "machine_gate_passed": bool(result["machine_gate_passed"]),
                    "provider_calls_before_benchmark": int(
                        execution_record["provider_calls"]
                    ),
                    "section_had_preexisting_repair_call": claim["section"]
                    in set(repair_sections),
                }
            )
    cases.sort(key=lambda row: (row["topic_id"], row["article_id"], row["case_id"]))
    write_jsonl(output_dir / "cases.jsonl", cases)
    qualified_cases = [row for row in cases if row["eligibility"]["machine_gate_passed"]]
    failed_article_cases = [
        row for row in cases if not row["eligibility"]["machine_gate_passed"]
    ]
    write_jsonl(output_dir / "qualified_article_cases.jsonl", qualified_cases)
    write_jsonl(output_dir / "failed_article_cases.jsonl", failed_article_cases)
    write_json(output_dir / "private_condition_map.json", private_rows)
    packet_dir = output_dir / "review_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_records = []
    for case in cases:
        path = packet_dir / f"{case['case_id']}.json"
        write_json(path, case)
        packet_records.append(
            {"case_id": case["case_id"], "path": str(path.resolve()), "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": 1,
        "status": "citecalibrator_pilot_benchmark_ready_for_blinded_review",
        "scientific_role": "rubric_and_pipeline_development_only",
        "natural_prevalence_claims_prohibited": True,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "robustness_root": str(robustness_root.resolve()),
        "articles": len(source_artifacts),
        "machine_gate_passed_articles": sum(
            row["machine_gate_passed"] for row in source_artifacts
        ),
        "topics": len({row["topic_id"] for row in cases}),
        "cases": len(cases),
        "qualified_article_cases": len(qualified_cases),
        "failed_article_cases": len(failed_article_cases),
        "cases_file": str((output_dir / "cases.jsonl").resolve()),
        "cases_sha256": sha256_file(output_dir / "cases.jsonl"),
        "qualified_article_cases_file": str(
            (output_dir / "qualified_article_cases.jsonl").resolve()
        ),
        "qualified_article_cases_sha256": sha256_file(
            output_dir / "qualified_article_cases.jsonl"
        ),
        "failed_article_cases_file": str(
            (output_dir / "failed_article_cases.jsonl").resolve()
        ),
        "failed_article_cases_sha256": sha256_file(
            output_dir / "failed_article_cases.jsonl"
        ),
        "primary_reviews_per_case": 2,
        "primary_reviews_required": len(cases) * 2,
        "review_packets": packet_records,
        "source_artifacts": source_artifacts,
        "metrics": [
            "major_critical_defect_recall",
            "accept_false_intervention_rate",
            "action_macro_f1",
            "mean_latency_seconds",
        ],
    }
    review_template_path = output_dir / "primary_review_returns_template.jsonl"
    write_jsonl(
        review_template_path,
        [
            {
                "case_id": case["case_id"],
                "review_slot": review_slot,
                "reviewer_id": None,
                **empty_review(),
            }
            for case in cases
            for review_slot in ("primary_a", "primary_b")
        ],
    )
    manifest["primary_review_template"] = str(review_template_path.resolve())
    manifest["primary_review_template_sha256"] = sha256_file(review_template_path)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_controlled_challenge(
    *,
    pilot_cases_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a synthetic development challenge without implying natural prevalence."""
    pilot_cases = [
        json.loads(line)
        for line in pilot_cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phenomenon_cases: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for case in pilot_cases:
        for phenomenon in case.get("phenomena") or []:
            key = (str(case["topic_id"]), str(phenomenon["phenomenon_id"]))
            phenomenon_cases.setdefault(key, (case, phenomenon))
    if not phenomenon_cases:
        raise ValueError("Pilot cases contain no graph phenomena")

    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    interventions = (
        (
            "supported",
            (
                "For this benchmark item, the recorded verified answer is {answer}. "
                "{phenomenon_id} {reference_id}"
            ),
            "accept",
            "minor",
            [],
            None,
        ),
        (
            "causal",
            (
                "For this benchmark item, the recorded graph pattern causes the reported "
                "research outcome. {phenomenon_id} {reference_id}"
            ),
            "qualify",
            "major",
            ["causal_overclaim"],
            "State only that the graph pattern was observed; do not infer causality.",
        ),
        (
            "reification",
            (
                "The cluster is a scientific mechanism that defines the reported research "
                "area. {phenomenon_id} {reference_id}"
            ),
            "qualify",
            "major",
            ["semantic_reification"],
            "Describe the cluster as a corpus- and construction-dependent graph grouping.",
        ),
        (
            "fabricated_numeric",
            (
                "The verified answer reports exactly 999999 hops. "
                "{phenomenon_id} {reference_id}"
            ),
            "reject",
            "major",
            ["numeric_error"],
            None,
        ),
    )
    private_map = []
    for (topic_id, phenomenon_id), (source_case, phenomenon) in sorted(
        phenomenon_cases.items()
    ):
        reference_ids = list(phenomenon.get("reference_ids") or [])
        reference_id = reference_ids[0] if reference_ids else ""
        answer = json.dumps(
            phenomenon.get("verified_answer"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for intervention, template, action, severity, risks, revision in interventions:
            claim = template.format(
                answer=answer,
                phenomenon_id=phenomenon_id,
                reference_id=reference_id,
            ).strip()
            case_id = _opaque("CCC", topic_id, phenomenon_id, intervention)
            case = deepcopy(source_case)
            case.update(
                {
                    "case_id": case_id,
                    "benchmark_split": "controlled_challenge_development_only",
                    "article_id": _opaque("CHALLENGE", topic_id, phenomenon_id),
                    "section": "controlled_challenge",
                    "paragraph_id": _opaque("PARA", case_id),
                    "parent_sentence": claim,
                    "atomic_claim": claim,
                    "local_context": claim,
                    "claim_extraction": {
                        "method": "controlled_template_v1",
                        "source_span": [0, len(claim)],
                        "sampling_probability": 1.0,
                        "atomicity_requires_human_confirmation": False,
                    },
                    "risk_features": {
                        "contains_number": bool(NUMBER.search(claim)),
                        "contains_causal_language": bool(CAUSAL.search(claim)),
                    },
                    "phenomena": [phenomenon],
                    "sources": [
                        row
                        for row in source_case.get("sources") or []
                        if row.get("reference_id") == reference_id
                    ],
                    "perturbation_profiles": {
                        phenomenon_id: (source_case.get("perturbation_profiles") or {}).get(
                            phenomenon_id,
                            {"available": False, "variants": []},
                        )
                    },
                    "allowed_evidence_ids": sorted(
                        {
                            phenomenon_id,
                            *([reference_id] if reference_id else []),
                            *(phenomenon.get("graph_evidence_ids") or []),
                        }
                    ),
                    "eligibility": {
                        "eligible_for_natural_prevalence": False,
                        "reason": "controlled_synthetic_challenge",
                    },
                    "review": empty_review(),
                }
            )
            case["lineage"] = {
                "source_case_id": source_case["case_id"],
                "source_case_material_sha256": source_case["lineage"][
                    "case_material_sha256"
                ],
            }
            case["lineage"]["case_material_sha256"] = canonical_sha256(
                {key: value for key, value in case.items() if key != "review"}
            )
            cases.append(case)
            gold.append(
                {
                    "case_id": case_id,
                    "article_id": case["article_id"],
                    "action": action,
                    "severity": severity,
                    "risk_types": risks,
                    "unsupported_spans": [],
                    "decisive_evidence_ids": [
                        value for value in (phenomenon_id, reference_id) if value
                    ],
                    "minimal_revision": revision,
                }
            )
            private_map.append({"case_id": case_id, "intervention": intervention})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "cases.jsonl", cases)
    write_jsonl(output_dir / "gold.jsonl", gold)
    write_json(output_dir / "private_intervention_map.json", private_map)
    manifest = {
        "schema_version": 1,
        "status": "citecalibrator_controlled_challenge_complete",
        "scientific_role": "pipeline_and_error_sensitivity_validation_only",
        "natural_prevalence_claims_prohibited": True,
        "training_need_claims_prohibited": True,
        "source_pilot_cases": str(pilot_cases_path.resolve()),
        "source_pilot_cases_sha256": sha256_file(pilot_cases_path),
        "topics": len({row[0] for row in phenomenon_cases}),
        "phenomena": len(phenomenon_cases),
        "cases": len(cases),
        "action_counts": dict(Counter(row["action"] for row in gold)),
        "cases_sha256": sha256_file(output_dir / "cases.jsonl"),
        "gold_sha256": sha256_file(output_dir / "gold.jsonl"),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def deterministic_rule_judge(case: dict[str, Any]) -> dict[str, Any]:
    claim = str(case["atomic_claim"])
    tokens = set(EVIDENCE_TOKEN.findall(claim))
    allowed = set(case.get("allowed_evidence_ids") or [])
    risks: list[str] = []
    unsupported_spans: list[str] = []
    action = "accept"
    severity = "minor"

    if tokens - allowed or not any(token.startswith("PH-") for token in tokens):
        risks.append("missing_evidence")
        action = "reject"
        severity = "major"
    if not any(token.startswith("REF-") for token in tokens):
        risks.append("missing_evidence")
        action = "qualify" if action == "accept" else action
        severity = "major"
    for pattern, risk in (
        (CAUSAL, "causal_overclaim"),
        (SCOPE, "scope_overclaim"),
        (REIFICATION, "semantic_reification"),
    ):
        matches = [match.group(0) for match in pattern.finditer(claim)]
        if matches:
            risks.append(risk)
            unsupported_spans.extend(matches)
            action = "qualify" if action == "accept" else action
            severity = "major"

    allowed_numbers: set[str] = set()
    for phenomenon in case.get("phenomena") or []:
        allowed_numbers.update(_safe_numbers(phenomenon.get("verified_answer")))
        allowed_numbers.update(_safe_numbers(phenomenon.get("operator_trace")))
    for source in case.get("sources") or []:
        allowed_numbers.update(_safe_numbers(source.get("year")))
    observed_numbers = {
        value.rstrip("%").replace(",", "")
        for value in NUMBER.findall(EVIDENCE_TOKEN.sub("", claim))
    }
    unexpected_numbers = sorted(
        value for value in observed_numbers if not _number_supported(value, allowed_numbers)
    )
    if unexpected_numbers:
        risks.append("numeric_error")
        unsupported_spans.extend(unexpected_numbers)
        action = "reject"
        severity = "major"

    if ROBUSTNESS_ASSERTION.search(claim):
        changed = False
        for profile in (case.get("perturbation_profiles") or {}).values():
            for variant in profile.get("variants", []):
                comparison = variant.get("comparison_to_baseline") or {}
                for key, value in comparison.items():
                    if key == "comparable" or key.endswith(
                        ("_equal", "_still_optimal", "_still_argmax")
                    ):
                        changed = changed or value is False
                    elif key.endswith("_change") and isinstance(value, (int, float)):
                        changed = changed or not math.isclose(
                            float(value), 0.0, abs_tol=1e-12
                        )
                    elif key.endswith(("_jaccard", "_adjusted_rand_index")) and isinstance(
                        value, (int, float)
                    ):
                        changed = changed or float(value) < 1.0 - 1e-12
        if changed:
            risks.append("parameter_sensitivity_omitted")
            unsupported_spans.extend(
                match.group(0) for match in ROBUSTNESS_ASSERTION.finditer(claim)
            )
            action = "qualify" if action == "accept" else action
            severity = "major"

    return {
        "case_id": case["case_id"],
        "condition": "deterministic_rules_v1",
        "action": action,
        "risk_types": sorted(set(risks)),
        "severity": severity,
        "unsupported_spans": sorted(set(unsupported_spans)),
        "decisive_evidence_ids": sorted(tokens & allowed),
        "minimal_revision": None,
        "latency_seconds": 0.0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def validate_review(review: dict[str, Any]) -> None:
    if review.get("action") not in ACTIONS:
        raise ValueError(f"Invalid review action: {review.get('action')}")
    if review.get("severity") not in SEVERITIES:
        raise ValueError(f"Invalid review severity: {review.get('severity')}")
    unknown = set(review.get("risk_types") or []) - set(RISK_TYPES)
    if unknown:
        raise ValueError(f"Unknown risk types: {sorted(unknown)}")
    if review["action"] == "accept" and review.get("risk_types"):
        raise ValueError("Accept judgments cannot carry defect risk types")
    if review["action"] == "qualify" and not review.get("minimal_revision"):
        raise ValueError("Qualify judgments require a minimal revision")


HUMAN_DECISION_FIELDS = (
    "factual_supported",
    "interpretation_calibrated",
    "alternative_adequate",
    "evidence_sufficient",
    "action",
    "risk_types",
    "severity",
)


def _validate_human_review(review: dict[str, Any]) -> None:
    validate_review(review)
    if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
        raise ValueError("Human reviews require a nonempty reviewer_id")
    for field in (
        "factual_supported",
        "interpretation_calibrated",
        "alternative_adequate",
        "evidence_sufficient",
    ):
        if not isinstance(review.get(field), bool):
            raise TypeError(f"Human review field must be boolean: {field}")


def _decision_signature(review: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        tuple(sorted(review.get(field) or []))
        if field == "risk_types"
        else review.get(field)
        for field in HUMAN_DECISION_FIELDS
    )


def resolve_human_reviews(
    *,
    cases: Iterable[dict[str, Any]],
    primary_reviews: Iterable[dict[str, Any]],
    adjudications: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve two independent reviews, requiring a blind third decision on conflicts."""
    case_by_id = {str(row["case_id"]): row for row in cases}
    primary_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in primary_reviews:
        _validate_human_review(review)
        case_id = str(review["case_id"])
        if case_id not in case_by_id:
            raise ValueError(f"Primary review references an unknown case: {case_id}")
        primary_by_id[case_id].append(review)
    adjudication_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in adjudications:
        _validate_human_review(review)
        case_id = str(review["case_id"])
        if case_id not in case_by_id:
            raise ValueError(f"Adjudication references an unknown case: {case_id}")
        adjudication_by_id[case_id].append(review)

    missing_primary = []
    conflicts = []
    gold = []
    for case_id, case in sorted(case_by_id.items()):
        reviews = primary_by_id.get(case_id, [])
        reviewer_ids = [row["reviewer_id"] for row in reviews]
        if len(reviews) != 2 or len(set(reviewer_ids)) != 2:
            missing_primary.append(case_id)
            continue
        conflict = _decision_signature(reviews[0]) != _decision_signature(reviews[1])
        adjudication_rows = adjudication_by_id.get(case_id, [])
        if conflict:
            if len(adjudication_rows) != 1:
                conflicts.append(
                    {
                        **case,
                        "review": empty_review(),
                    }
                )
                continue
            adjudication = adjudication_rows[0]
            if adjudication["reviewer_id"] in set(reviewer_ids):
                raise ValueError("Adjudicator must differ from both primary reviewers")
            resolved = adjudication
            source = "adjudicated"
        else:
            if adjudication_rows:
                raise ValueError("Consensus cases must not receive adjudication")
            resolved = reviews[0]
            source = "primary_consensus"
        gold.append(
            {
                "case_id": case_id,
                "topic_id": case["topic_id"],
                "article_id": case["article_id"],
                **{field: resolved.get(field) for field in HUMAN_DECISION_FIELDS},
                "unsupported_spans": resolved.get("unsupported_spans") or [],
                "decisive_evidence_ids": resolved.get("decisive_evidence_ids") or [],
                "invalid_dependency_ids": resolved.get("invalid_dependency_ids") or [],
                "minimal_revision": resolved.get("minimal_revision"),
                "resolution_source": source,
            }
        )
    return {
        "status": (
            "resolved"
            if not missing_primary and not conflicts and len(gold) == len(case_by_id)
            else "awaiting_human_reviews"
        ),
        "cases": len(case_by_id),
        "resolved": len(gold),
        "missing_primary_case_ids": missing_primary,
        "adjudication_packets": conflicts,
        "gold": gold,
    }


def _macro_f1(gold: list[str], predicted: list[str]) -> float:
    scores = []
    observed_labels = [label for label in ACTIONS if label in set(gold)]
    for label in observed_labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def cohen_kappa(left: list[str], right: list[str], *, labels: Iterable[str]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Kappa requires two nonempty rating lists of equal length")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score_predictions(
    gold_rows: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    prediction_by_id = {str(row["case_id"]): row for row in predictions}
    shared = sorted(set(gold_by_id) & set(prediction_by_id))
    if not shared:
        raise ValueError("No predictions match the gold judgments")
    gold_actions = [str(gold_by_id[key]["action"]) for key in shared]
    predicted_actions = [str(prediction_by_id[key]["action"]) for key in shared]
    major = [
        key
        for key in shared
        if gold_by_id[key]["action"] in {"qualify", "reject"}
        and gold_by_id[key]["severity"] in {"major", "critical"}
    ]
    accepted = [key for key in shared if gold_by_id[key]["action"] == "accept"]
    major_recall = (
        sum(prediction_by_id[key]["action"] != "accept" for key in major) / len(major)
        if major
        else None
    )
    false_intervention = (
        sum(prediction_by_id[key]["action"] != "accept" for key in accepted)
        / len(accepted)
        if accepted
        else None
    )
    latencies = [float(prediction_by_id[key].get("latency_seconds") or 0.0) for key in shared]
    return {
        "cases": len(shared),
        "major_critical_gold_cases": len(major),
        "accepted_gold_cases": len(accepted),
        "metrics": {
            "major_critical_defect_recall": major_recall,
            "accept_false_intervention_rate": false_intervention,
            "action_macro_f1": _macro_f1(gold_actions, predicted_actions),
            "mean_latency_seconds": sum(latencies) / len(latencies),
        },
    }


def summarize_gold(gold_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(gold_rows)
    for row in rows:
        validate_review(row)
    major = [
        row
        for row in rows
        if row["action"] in {"qualify", "reject"}
        and row["severity"] in {"major", "critical"}
    ]
    article_ids = {str(row["article_id"]) for row in rows}
    affected_articles = {str(row["article_id"]) for row in major}
    major_rate = len(major) / len(rows) if rows else None
    article_rate = len(affected_articles) / len(article_ids) if article_ids else None
    return {
        "cases": len(rows),
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "major_critical_defects": len(major),
        "major_critical_defect_rate": major_rate,
        "articles": len(article_ids),
        "articles_with_major_critical_defect_rate": article_rate,
        "topic_cluster_bootstrap_95ci": _topic_cluster_bootstrap(rows),
    }


def _topic_cluster_bootstrap(
    rows: list[dict[str, Any]], *, samples: int = 5000, seed: int = 20260908
) -> dict[str, list[float] | None]:
    if not rows or any(not row.get("topic_id") for row in rows):
        return {
            "major_critical_defect_rate": None,
            "articles_with_major_critical_defect_rate": None,
        }
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[str(row["topic_id"])].append(row)
    topics = sorted(by_topic)
    generator = Random(seed)
    claim_rates = []
    article_rates = []
    for _ in range(samples):
        sampled_rows = []
        for draw_index in range(len(topics)):
            topic = generator.choice(topics)
            sampled_rows.extend((draw_index, row) for row in by_topic[topic])
        sampled_major = [
            (draw_index, row)
            for draw_index, row in sampled_rows
            if row["action"] in {"qualify", "reject"}
            and row["severity"] in {"major", "critical"}
        ]
        claim_rates.append(len(sampled_major) / len(sampled_rows))
        sampled_articles = {
            (draw_index, str(row["article_id"])) for draw_index, row in sampled_rows
        }
        sampled_affected = {
            (draw_index, str(row["article_id"])) for draw_index, row in sampled_major
        }
        article_rates.append(len(sampled_affected) / len(sampled_articles))

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    return {
        "major_critical_defect_rate": interval(claim_rates),
        "articles_with_major_critical_defect_rate": interval(article_rates),
    }
