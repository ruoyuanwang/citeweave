from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

from .io import read_json, write_json
from .review_learning import StructuredFeedback


class HumanReviewValidationError(ValueError):
    pass


def validate_review_return(
    payload: dict[str, Any],
    *,
    internal_manifest: dict[str, Any],
    packet_root: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    reviewer = payload.get("reviewer_code")
    if reviewer not in internal_manifest["assignments"]:
        raise HumanReviewValidationError(f"Unknown reviewer code: {reviewer!r}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise HumanReviewValidationError("results must be a list")
    expected = {
        packet_id: layer
        for layer in ("factual", "semantic")
        for packet_id in internal_manifest["assignments"][reviewer][layer]
    }
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for result in results:
        packet_id = result.get("packet_id")
        if packet_id not in expected:
            raise HumanReviewValidationError(f"Unassigned packet: {packet_id!r}")
        if packet_id in seen:
            raise HumanReviewValidationError(f"Duplicate packet result: {packet_id}")
        seen.add(packet_id)
        layer = expected[packet_id]
        packet_path = packet_root / "packets" / layer / f"{packet_id}.json"
        if not packet_path.is_file():
            raise HumanReviewValidationError(f"Missing packet file: {packet_id}")
        seconds = result.get("review_seconds")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise HumanReviewValidationError(f"Positive review_seconds required: {packet_id}")
        if result.get("reviewer_code") != reviewer:
            raise HumanReviewValidationError(f"Reviewer mismatch: {packet_id}")
        if not isinstance(result.get("rationale"), str) or not result["rationale"].strip():
            raise HumanReviewValidationError(f"Nonempty rationale required: {packet_id}")
        validated.append({**result, "review_layer": layer})
    missing = sorted(set(expected) - seen)
    if require_complete and missing:
        raise HumanReviewValidationError(f"Incomplete review return: {len(missing)} missing")
    return {
        "schema_version": 1,
        "reviewer_code": reviewer,
        "validated_results": validated,
        "completed": len(validated),
        "expected": len(expected),
        "missing": missing,
        "total_review_seconds": sum(float(result["review_seconds"]) for result in validated),
    }


def build_adjudication_worklist(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    left_records = {result["packet_id"]: result for result in left["validated_results"]}
    right_records = {result["packet_id"]: result for result in right["validated_results"]}
    common = sorted(set(left_records) & set(right_records))
    disagreements = []
    agreement_fields = {
        "factual": ("answer_correct", "evidence_sufficient", "action"),
        "semantic": (
            "calibrated",
            "alternative_adequate",
            "limitation_adequate",
            "action",
        ),
    }
    exact_agreement = 0
    binary_pairs: dict[str, tuple[list[int], list[int]]] = {
        "answer_correct": ([], []),
        "evidence_sufficient": ([], []),
        "calibrated": ([], []),
        "alternative_adequate": ([], []),
        "limitation_adequate": ([], []),
    }
    for packet_id in common:
        left_result = left_records[packet_id]
        right_result = right_records[packet_id]
        layer = left_result["review_layer"]
        if layer != right_result["review_layer"]:
            raise HumanReviewValidationError(f"Layer mismatch: {packet_id}")
        fields = agreement_fields[layer]
        differences = {
            field: {"left": left_result.get(field), "right": right_result.get(field)}
            for field in fields
            if left_result.get(field) != right_result.get(field)
        }
        if differences:
            disagreements.append(
                {
                    "packet_id": packet_id,
                    "review_layer": layer,
                    "differences": differences,
                    "left": left_result,
                    "right": right_result,
                }
            )
        else:
            exact_agreement += 1
        for field in binary_pairs:
            if field in fields:
                binary_pairs[field][0].append(int(bool(left_result.get(field))))
                binary_pairs[field][1].append(int(bool(right_result.get(field))))
    kappas = {}
    for field, (left_values, right_values) in binary_pairs.items():
        if not left_values:
            continue
        if len(set(left_values + right_values)) < 2:
            kappas[field] = None
        else:
            kappas[field] = float(cohen_kappa_score(left_values, right_values))
    worklist = {
        "schema_version": 1,
        "reviewers": [left["reviewer_code"], right["reviewer_code"]],
        "common_packets": len(common),
        "exact_agreement_rate": exact_agreement / len(common) if common else None,
        "cohen_kappa": kappas,
        "disagreements": disagreements,
        "adjudication_required": len(disagreements),
    }
    if output_path:
        write_json(output_path, worklist)
    return worklist


def adjudicated_semantic_feedback(
    adjudicated_results: list[dict[str, Any]],
    *,
    packet_root: Path,
) -> list[StructuredFeedback]:
    feedback = []
    for result in adjudicated_results:
        packet_id = result["packet_id"]
        packet = read_json(packet_root / "packets" / "semantic" / f"{packet_id}.json")
        action = result["action"]
        feedback.append(
            StructuredFeedback(
                feedback_id=f"ADJ-{packet_id}",
                reviewer_id=str(result["adjudicator_code"]),
                dataset_id=str(packet["dataset_id"]),
                target_id=packet_id,
                target_type="claim",
                issue_type=f"semantic:{packet['task_type']}",
                action=action,
                rationale=str(result["rationale"]),
                replacement=result.get("replacement"),
                guard=dict(result.get("guard") or {}),
                review_seconds=float(result["review_seconds"]),
            )
        )
    return feedback
