from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .io import read_json, sha256_file


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _load_optional_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML object: {path}")
    return value


def assess_complementary_oversight_readiness(
    *,
    packet_manifest_path: Path,
    reviewer_registry_path: Path | None = None,
    capability_freeze_path: Path | None = None,
    assignment_manifest_path: Path | None = None,
    complementary_amendment_path: Path | None = None,
    cluster_inference_amendment_path: Path | None = None,
    factorial_analysis_amendment_path: Path | None = None,
    factorial_analysis_freeze_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed until every prospective human-oversight prerequisite is frozen."""
    packet_manifest = read_json(packet_manifest_path)
    records = packet_manifest.get("records") or []
    case_ids = [str(row.get("case_id") or "") for row in records]
    prospective_cases = (
        packet_manifest.get("study_role")
        == "prospective_heldout_candidate_outputs"
    )
    candidate_outputs_present = bool(records) and all(
        row.get("candidate_output_sha256")
        and row.get("generation_record_sha256")
        and row.get("issue_type")
        and row.get("severity") in {"low", "medium", "high", "critical"}
        for row in records
    )

    registry = _load_optional(reviewer_registry_path)
    reviewers = (registry or {}).get("reviewers") or []
    reviewer_ids = [str(row.get("reviewer_id") or "") for row in reviewers]
    reviewer_calibration_valid = len(reviewers) >= 6 and all(
        int(row.get("warmup_cases_completed") or 0) >= 12
        and len(set(row.get("warmup_domains") or [])) >= 2
        and row.get("warmup_outcomes_adjudicated") is True
        and row.get("warmup_excluded_from_confirmatory") is True
        for row in reviewers
    )

    capability = _load_optional(capability_freeze_path)
    capability_frozen = bool(
        capability
        and capability.get("status") == "frozen_before_heldout_assignment"
        and capability.get("heldout_outcomes_inspected") is False
        and capability.get("predecision_sha256")
        and set(capability.get("reviewer_ids") or []) == set(reviewer_ids)
    )

    assignment = _load_optional(assignment_manifest_path)
    assignments = (assignment or {}).get("records") or []
    arm_counts: Counter[str] = Counter()
    seen_by_reviewer: set[tuple[str, str]] = set()
    duplicate_reviewer_case = False
    first_pass_reviewers: dict[str, set[str]] = defaultdict(set)
    reviewer_arms: dict[str, set[str]] = defaultdict(set)
    assignment_references_valid = bool(assignments)
    for row in assignments:
        case_id = str(row.get("case_id") or "")
        reviewer_id = str(row.get("reviewer_id") or "")
        packet_arm = str(row.get("packet_arm") or "")
        assignment_arm = str(row.get("assignment_arm") or "")
        arm = f"{packet_arm}__{assignment_arm}"
        arm_counts[arm] += 1
        reviewer_arms[reviewer_id].add(arm)
        key = (reviewer_id, case_id)
        duplicate_reviewer_case |= key in seen_by_reviewer
        seen_by_reviewer.add(key)
        if row.get("pass") == "first":
            first_pass_reviewers[case_id].add(reviewer_id)
        assignment_references_valid &= (
            case_id in set(case_ids) and reviewer_id in set(reviewer_ids)
        )
    required_arms = {
        "standard_claim_first__qualified_random",
        "standard_claim_first__capability_cost_router",
        "adversarial_two_sided__qualified_random",
        "adversarial_two_sided__capability_cost_router",
    }
    four_arms_present = all(arm_counts[arm] > 0 for arm in required_arms)
    reviewer_crossover = bool(reviewer_ids) and all(
        required_arms <= reviewer_arms[reviewer_id] for reviewer_id in reviewer_ids
    )
    heldout_case_set = set(case_ids)
    double_reviewed = sum(
        len(first_pass_reviewers[case_id]) >= 2 for case_id in heldout_case_set
    )
    overlap_fraction = (
        double_reviewed / len(heldout_case_set) if heldout_case_set else 0.0
    )
    assignment_valid = bool(
        assignment
        and assignment.get("status") == "frozen_before_heldout_review"
        and assignment.get("heldout_outcomes_inspected") is False
        and assignment_references_valid
        and not duplicate_reviewer_case
        and four_arms_present
        and reviewer_crossover
        and overlap_fraction >= 0.5
    )

    factorial_amendment = _load_optional_yaml(factorial_analysis_amendment_path)
    factorial_freeze = _load_optional(factorial_analysis_freeze_path)
    factorial_hash = (
        sha256_file(factorial_analysis_amendment_path)
        if factorial_analysis_amendment_path
        and factorial_analysis_amendment_path.is_file()
        else None
    )
    factorial_frozen = bool(
        factorial_amendment
        and factorial_freeze
        and factorial_freeze.get("sha256") == factorial_hash
        and factorial_freeze.get("status")
        == "frozen_before_real_reviewer_and_factorial_outcomes"
        and factorial_freeze.get("human_outcomes_inspected") is False
        and factorial_freeze.get("factorial_outcomes_inspected") is False
    )
    complementary_hash = (
        sha256_file(complementary_amendment_path)
        if complementary_amendment_path and complementary_amendment_path.is_file()
        else None
    )
    cluster_inference_hash = (
        sha256_file(cluster_inference_amendment_path)
        if cluster_inference_amendment_path
        and cluster_inference_amendment_path.is_file()
        else None
    )
    factorial_chain_valid = bool(
        factorial_amendment
        and factorial_amendment.get("complementary_oversight_amendment_sha256")
        == complementary_hash
        and factorial_amendment.get("cluster_inference_amendment_sha256")
        == cluster_inference_hash
    )
    implementation_paths = {
        "oversight_factorial_analysis.py": Path(
            "src/citeweave/oversight_factorial_analysis.py"
        ),
        "analyze_complementary_oversight_factorial.py": Path(
            "scripts/analyze_complementary_oversight_factorial.py"
        ),
        "oversight_readiness.py": Path("src/citeweave/oversight_readiness.py"),
        "audit_complementary_oversight_readiness.py": Path(
            "scripts/audit_complementary_oversight_readiness.py"
        ),
    }
    implementation_hashes = (factorial_amendment or {}).get("implementations") or {}
    factorial_implementations_valid = bool(
        factorial_amendment
        and set(implementation_hashes) == set(implementation_paths)
        and all(
            path.is_file() and sha256_file(path) == implementation_hashes[name]
            for name, path in implementation_paths.items()
        )
    )
    registered_test = (factorial_amendment or {}).get("test") or {}
    registered_test_path = Path(str(registered_test.get("path") or ""))
    factorial_implementations_valid &= bool(
        registered_test_path.is_file()
        and registered_test.get("sha256") == sha256_file(registered_test_path)
    )
    registered_readiness_test = (factorial_amendment or {}).get(
        "readiness_test"
    ) or {}
    registered_readiness_test_path = Path(
        str(registered_readiness_test.get("path") or "")
    )
    factorial_implementations_valid &= bool(
        registered_readiness_test_path.is_file()
        and registered_readiness_test.get("sha256")
        == sha256_file(registered_readiness_test_path)
    )

    checks = {
        "packet_manifest_pre_outcome": packet_manifest.get(
            "human_outcomes_inspected"
        )
        is False,
        "unique_nonempty_case_ids": bool(case_ids)
        and all(case_ids)
        and len(case_ids) == len(set(case_ids)),
        "minimum_96_heldout_cases": len(case_ids) >= 96,
        "prospective_candidate_outputs_not_prototypes": prospective_cases,
        "candidate_generation_records_bound": candidate_outputs_present,
        "minimum_6_reviewers": len(set(reviewer_ids)) >= 6,
        "reviewer_ids_unique_nonempty": bool(reviewer_ids)
        and all(reviewer_ids)
        and len(reviewer_ids) == len(set(reviewer_ids)),
        "reviewer_warmup_and_exclusion_valid": reviewer_calibration_valid,
        "capability_model_frozen_without_heldout_leakage": capability_frozen,
        "assignment_references_valid": assignment_references_valid,
        "four_factorial_arms_present": four_arms_present,
        "reviewer_crossover_valid": reviewer_crossover,
        "no_reviewer_sees_same_case_twice": not duplicate_reviewer_case,
        "double_review_overlap_at_least_50_percent": overlap_fraction >= 0.5,
        "assignment_frozen_without_heldout_leakage": assignment_valid,
        "factorial_analysis_amendment_frozen_pre_outcome": factorial_frozen,
        "factorial_analysis_amendment_chain_valid": factorial_chain_valid,
        "factorial_analysis_implementation_hashes_valid": (
            factorial_implementations_valid
        ),
    }
    blocking_reasons = [name for name, passed in checks.items() if not passed]
    artifact_hashes = {"packet_manifest_sha256": sha256_file(packet_manifest_path)}
    for name, path in (
        ("reviewer_registry_sha256", reviewer_registry_path),
        ("capability_freeze_sha256", capability_freeze_path),
        ("assignment_manifest_sha256", assignment_manifest_path),
        ("complementary_amendment_sha256", complementary_amendment_path),
        ("cluster_inference_amendment_sha256", cluster_inference_amendment_path),
        ("factorial_analysis_amendment_sha256", factorial_analysis_amendment_path),
        ("factorial_analysis_freeze_sha256", factorial_analysis_freeze_path),
    ):
        artifact_hashes[name] = sha256_file(path) if path and path.is_file() else None
    return {
        "schema_version": 1,
        "status": "ready" if not blocking_reasons else "blocked",
        "human_outcomes_inspected": False,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "counts": {
            "heldout_cases": len(case_ids),
            "reviewers": len(set(reviewer_ids)),
            "assignments": len(assignments),
            "double_reviewed_cases": double_reviewed,
            "double_review_overlap_fraction": overlap_fraction,
            "factorial_arm_counts": dict(sorted(arm_counts.items())),
        },
        "artifact_hashes": artifact_hashes,
    }
