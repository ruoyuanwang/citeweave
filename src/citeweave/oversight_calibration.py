from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json

CALIBRATION_ISSUES = (
    "graph_answer_consistency",
    "causal_overreach",
    "counterevidence_coverage",
    "revision_safety",
)


class OversightCalibrationError(ValueError):
    pass


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]
    return f"OC-{digest}"


def _candidate_answer(packet: dict[str, Any]) -> dict[str, Any]:
    marker = "Candidate structural answer:"
    claim = str(packet.get("claim") or "")
    if marker not in claim:
        raise OversightCalibrationError("Prototype claim lacks a candidate answer")
    try:
        value = json.loads(claim.split(marker, 1)[1].strip())
    except json.JSONDecodeError as exc:
        raise OversightCalibrationError("Prototype candidate answer is not JSON") from exc
    if not isinstance(value, dict) or not value:
        raise OversightCalibrationError("Prototype candidate answer must be an object")
    return value


def _mutate_answer(answer: dict[str, Any]) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(answer)
    if isinstance(mutated.get("hops"), int):
        mutated["hops"] += 1
        return mutated, "incremented_hop_count"
    for field in (
        "total_inverse_weight_distance",
        "weighted_cross_share",
        "reduction_fraction",
        "growth_ratio",
    ):
        value = mutated.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mutated[field] = round(float(value) + max(abs(float(value)) * 0.25, 0.1), 6)
            return mutated, f"perturbed_{field}"
    for field, value in mutated.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delta = 1 if isinstance(value, int) else max(abs(float(value)) * 0.25, 0.1)
            mutated[field] = type(value)(value + delta)
            return mutated, f"perturbed_{field}"
    for field, value in mutated.items():
        if isinstance(value, list) and value:
            mutated[field] = list(reversed(value))
            return mutated, f"reversed_{field}"
    raise OversightCalibrationError("No auditable answer field can be perturbed")


def _common_public(
    *,
    packet_id: str,
    base_case_id: str,
    source: dict[str, Any],
    issue_type: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packet_type": "multidimensional_oversight_calibration",
        "packet_id": packet_id,
        "base_case_id": base_case_id,
        "dataset_id": source["dataset_id"],
        "domain": source["domain"],
        "issue_type": issue_type,
        "training_only": True,
        "confirmatory_exclusion": True,
        "generation_condition_hidden": True,
        "gold_label_hidden": True,
        "response_schema": {
            "verdict": ["valid", "invalid", "abstain"],
            "error_type": "string|null",
            "decisive_evidence_ids": "list[string]",
            "rationale": "string",
        },
    }


def _build_issue_packet(
    *,
    source: dict[str, Any],
    base_case_id: str,
    issue_type: str,
    valid: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_id = _stable_id(base_case_id, issue_type)
    public = _common_public(
        packet_id=packet_id,
        base_case_id=base_case_id,
        source=source,
        issue_type=issue_type,
    )
    forbidden = [str(value) for value in source.get("forbidden_inferences") or []]
    limitation = str((source.get("alternative_explanations") or ["corpus dependence"])[0])
    internal: dict[str, Any] = {
        "schema_version": 1,
        "packet_id": packet_id,
        "base_case_id": base_case_id,
        "dataset_id": source["dataset_id"],
        "issue_type": issue_type,
        "expected_verdict": "valid" if valid else "invalid",
        "confirmatory_exclusion": True,
    }

    if issue_type == "graph_answer_consistency":
        answer = _candidate_answer(source)
        shown_answer = answer
        mutation = "none"
        if not valid:
            shown_answer, mutation = _mutate_answer(answer)
        public.update(
            {
                "task": "Check whether the candidate answer is exactly consistent with the visible operator trace.",
                "candidate_answer": shown_answer,
                "operator_trace": source["operator_trace"],
                "review_instructions": [
                    "Recompute or cross-check every answer field supported by the trace.",
                    "Mark invalid if any reported path, count, label, or value conflicts with the trace.",
                ],
            }
        )
        internal["mutation"] = mutation
    elif issue_type == "causal_overreach":
        target = forbidden[0] if forbidden else "causality"
        conclusion = (
            f"The computation is descriptive and does not establish {target}."
            if valid
            else f"The graph pattern demonstrates {target}."
        )
        public.update(
            {
                "task": "Audit whether the proposed conclusion stays within the registered interpretation contract.",
                "candidate_conclusion": conclusion,
                "operator_trace": source["operator_trace"],
                "forbidden_inferences": forbidden,
                "review_instructions": [
                    "Treat a forbidden inference as invalid even when the numeric graph result is correct.",
                    "Judge the conclusion, not the fluency of its wording.",
                ],
            }
        )
        internal["mutation"] = "none" if valid else "inserted_forbidden_inference"
    elif issue_type == "counterevidence_coverage":
        conclusion = (
            f"The pattern is conditional on the corpus and construction; {limitation}"
            if valid
            else "The observed graph pattern is robust and has no material alternative explanation."
        )
        public.update(
            {
                "task": "Decide whether the conclusion adequately accounts for both blinded evidence sets.",
                "candidate_conclusion": conclusion,
                "evidence_set_a": source["evidence_set_a"],
                "evidence_set_b": source["evidence_set_b"],
                "review_instructions": [
                    "Inspect both sets independently; do not assume which one supports the claim.",
                    "Mark invalid when decisive counterevidence or a required limitation is ignored.",
                ],
            }
        )
        internal["mutation"] = "none" if valid else "removed_counterevidence_and_limitation"
    elif issue_type == "revision_safety":
        target = forbidden[0] if forbidden else "causality"
        original = f"The graph records a descriptive association and does not establish {target}."
        revision = (
            f"The graph records a descriptive pattern; it remains insufficient to establish {target}."
            if valid
            else f"The graph records a pattern that confirms {target}."
        )
        public.update(
            {
                "task": "Audit whether the proposed revision fixes wording without introducing a new unsupported claim.",
                "original_text": original,
                "proposed_revision": revision,
                "forbidden_inferences": forbidden,
                "review_instructions": [
                    "Compare original and revision at claim level.",
                    "Mark invalid if the revision introduces any unsupported or forbidden inference.",
                ],
            }
        )
        internal["mutation"] = "none" if valid else "revision_introduced_forbidden_inference"
    else:
        raise OversightCalibrationError(f"Unknown calibration issue: {issue_type}")
    return public, internal


def build_multidimensional_calibration_packets(
    *,
    prototype_manifest_path: Path,
    output_root: Path,
    cases_per_issue_per_dataset: int = 3,
) -> dict[str, Any]:
    """Build training-only, injected-error packets without touching held-out outputs."""
    if cases_per_issue_per_dataset < 2:
        raise OversightCalibrationError("At least two cases per issue and dataset are required")
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightCalibrationError("Refusing to overwrite calibration packets")
    source_manifest = read_json(prototype_manifest_path)
    if source_manifest.get("human_outcomes_inspected") is not False:
        raise OversightCalibrationError("Prototype manifest is not pre-outcome")
    prototype_root = prototype_manifest_path.resolve().parent
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_manifest.get("records") or []:
        dataset_id = str(row.get("dataset_id") or "")
        public_path = prototype_root / str(row.get("public_path") or "")
        if not dataset_id or not public_path.is_file():
            raise OversightCalibrationError("Prototype record is incomplete")
        if sha256_file(public_path) != row.get("public_sha256"):
            raise OversightCalibrationError("Prototype public packet hash mismatch")
        by_dataset[dataset_id].append({**row, "source_public_path": public_path})
    if len(by_dataset) < 8:
        raise OversightCalibrationError("Calibration requires at least eight datasets")

    records = []
    validity: Counter[bool] = Counter()
    issue_counts: Counter[str] = Counter()
    for dataset_index, (dataset_id, candidates) in enumerate(sorted(by_dataset.items())):
        candidates = sorted(candidates, key=lambda row: str(row["case_id"]))
        if len(candidates) < cases_per_issue_per_dataset:
            raise OversightCalibrationError(f"Too few prototypes for {dataset_id}")
        for issue_index, issue_type in enumerate(CALIBRATION_ISSUES):
            offset = issue_index % len(candidates)
            selected = [
                candidates[(offset + index) % len(candidates)]
                for index in range(cases_per_issue_per_dataset)
            ]
            for slot, row in enumerate(selected):
                source = read_json(row["source_public_path"])
                valid = (dataset_index + issue_index + slot) % 2 == 0
                public, internal = _build_issue_packet(
                    source=source,
                    base_case_id=str(row["case_id"]),
                    issue_type=issue_type,
                    valid=valid,
                )
                packet_id = public["packet_id"]
                public_path = output_root / "public" / f"{packet_id}.json"
                internal_path = output_root / "internal" / f"{packet_id}.json"
                write_json(public_path, public)
                write_json(internal_path, internal)
                records.append(
                    {
                        "packet_id": packet_id,
                        "base_case_id": row["case_id"],
                        "dataset_id": dataset_id,
                        "domain": dataset_id,
                        "issue_type": issue_type,
                        "source_public_sha256": row["public_sha256"],
                        "public_path": str(public_path.relative_to(output_root)),
                        "public_sha256": sha256_file(public_path),
                        "internal_path": str(internal_path.relative_to(output_root)),
                        "internal_sha256": sha256_file(internal_path),
                    }
                )
                validity[valid] += 1
                issue_counts[issue_type] += 1
    packet_ids = [row["packet_id"] for row in records]
    if len(packet_ids) != len(set(packet_ids)):
        raise OversightCalibrationError("Calibration packet IDs are not unique")
    manifest = {
        "schema_version": 1,
        "status": "multidimensional_calibration_packets_frozen_before_human_returns",
        "study_role": "training_only_reviewer_capability_calibration",
        "human_outcomes_inspected": False,
        "confirmatory_exclusion": True,
        "injected_errors": True,
        "gold_labels_public": False,
        "prototype_manifest_sha256": sha256_file(prototype_manifest_path),
        "datasets": len(by_dataset),
        "dataset_ids": sorted(by_dataset),
        "issues": list(CALIBRATION_ISSUES),
        "cases": len(records),
        "cases_per_issue_per_dataset": cases_per_issue_per_dataset,
        "issue_counts": dict(sorted(issue_counts.items())),
        "internal_validity_counts": {
            "valid": validity[True],
            "invalid": validity[False],
        },
        "records": records,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
