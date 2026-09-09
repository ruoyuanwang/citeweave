from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .io import read_json, sha256_file

_ARMS = {
    ("standard_claim_first", "qualified_random"),
    ("standard_claim_first", "capability_cost_router"),
    ("adversarial_two_sided", "qualified_random"),
    ("adversarial_two_sided", "capability_cost_router"),
}
_PERSONALIZED_TIERS = {"exact", "issue", "domain"}
_REQUIRED_IMPLEMENTATIONS = {
    "capability_freeze",
    "heldout_packet_builder",
    "factorial_assignment",
    "factorial_outcomes",
    "factorial_analysis",
    "server_timed_review_ui",
    "readiness_v2",
    "capability_freeze_cli",
    "heldout_packet_cli",
    "factorial_assignment_cli",
    "factorial_outcomes_cli",
    "factorial_analysis_cli",
    "readiness_v2_cli",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_file_hash(path: Path, expected: Any) -> bool:
    return path.is_file() and isinstance(expected, str) and sha256_file(path) == expected


def assess_factorial_oversight_readiness_v2(
    *,
    packet_manifest_path: Path,
    reviewer_registry_path: Path,
    capability_freeze_path: Path,
    combined_observations_path: Path,
    assignment_root: Path,
    amendment_path: Path,
    amendment_freeze_path: Path,
    previous_amendment_path: Path,
) -> dict[str, Any]:
    """Fail closed on the exact prospective 2x2 human-oversight execution."""
    packet = read_json(packet_manifest_path)
    registry = read_json(reviewer_registry_path)
    capability = read_json(capability_freeze_path)
    observations = read_json(combined_observations_path)
    assignment_path = assignment_root / "assignment_manifest.json"
    internal_path = assignment_root / "internal_manifest.json"
    assignment = read_json(assignment_path)
    internal = read_json(internal_path)
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    freeze = read_json(amendment_freeze_path)

    packet_records = packet.get("records") or []
    packet_index = {str(row.get("case_id")): row for row in packet_records}
    reviewer_rows = registry.get("reviewers") or []
    reviewer_ids = {str(row.get("reviewer_id")) for row in reviewer_rows}
    routes = assignment.get("case_routes") or []
    route_index = {str(row.get("case_id")): row for row in routes}
    flat = assignment.get("records") or []

    dataset_cases: dict[str, list[str]] = defaultdict(list)
    dataset_arms: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    reviewer_arms: dict[str, set[tuple[str, str]]] = defaultdict(set)
    expected_flat = set()
    route_hashes_valid = True
    routes_valid = True
    critical_dual = True
    capability_qualified = True
    conflicts_respected = True
    no_role_reuse = True
    roster_index = {str(row.get("reviewer_id")): row for row in reviewer_rows}
    threshold = float(assignment.get("minimum_capability_lower_95") or 0.0)
    for route in routes:
        case_id = str(route.get("case_id") or "")
        dataset_id = str(route.get("dataset_id") or "")
        arm = (str(route.get("packet_arm")), str(route.get("assignment_arm")))
        primaries = list(map(str, route.get("primary_reviewers") or []))
        adjudicator = route.get("adjudicator_id")
        profiles = route.get("capability_profiles") or []
        profile_index = {str(row.get("reviewer_id")): row for row in profiles}
        hash_payload = {key: value for key, value in route.items() if key != "predecision_sha256"}
        route_hashes_valid &= route.get("predecision_sha256") == _canonical_hash(hash_payload)
        dataset_cases[dataset_id].append(case_id)
        dataset_arms[dataset_id][arm] += 1
        routes_valid &= bool(case_id and dataset_id and arm in _ARMS)
        routes_valid &= route.get("route") in {
            "single_review",
            "dual_review_with_adjudication",
        }
        routes_valid &= len(primaries) in {1, 2} and len(primaries) == len(set(primaries))
        routes_valid &= route.get("design_double_review") is not True or len(primaries) == 2
        routes_valid &= route.get("route") == (
            "dual_review_with_adjudication" if len(primaries) == 2 else "single_review"
        )
        routes_valid &= set(primaries) <= reviewer_ids
        critical_dual &= route.get("severity") != "critical" or len(primaries) == 2
        if len(primaries) == 2:
            routes_valid &= isinstance(adjudicator, str) and adjudicator in reviewer_ids
            no_role_reuse &= adjudicator not in primaries
        else:
            routes_valid &= adjudicator is None
        role_ids = [*primaries, *((str(adjudicator),) if adjudicator else ())]
        no_role_reuse &= len(role_ids) == len(set(role_ids))
        capability_qualified &= set(profile_index) == set(role_ids)
        for reviewer in role_ids:
            profile = profile_index.get(reviewer) or {}
            capability_qualified &= (
                profile.get("evidence_tier") in _PERSONALIZED_TIERS
                and float(profile.get("posterior_lower_95") or -1.0) >= threshold
            )
            roster_row = roster_index.get(reviewer) or {}
            eligible = set(map(str, roster_row.get("eligible_domains") or []))
            conflicts = set(map(str, roster_row.get("conflicted_domains") or []))
            conflicts_respected &= route.get("domain") in eligible - conflicts
        for reviewer in primaries:
            expected_flat.add((case_id, reviewer, *arm))
            reviewer_arms[reviewer].add(arm)

    actual_flat = {
        (
            str(row.get("case_id")),
            str(row.get("reviewer_id")),
            str(row.get("packet_arm")),
            str(row.get("assignment_arm")),
        )
        for row in flat
    }
    packet_files_valid = True
    assignment_packet_root = assignment_root / "packets" / "adversarial"
    for row in packet_records:
        case_id = str(row.get("case_id") or "")
        route = route_index.get(case_id) or {}
        field = (
            "standard_public_sha256"
            if route.get("packet_arm") == "standard_claim_first"
            else "adversarial_public_sha256"
        )
        assigned = assignment_packet_root / f"{case_id}.json"
        packet_files_valid &= _is_file_hash(assigned, row.get(field))

    double_count = sum(
        len(route.get("primary_reviewers") or []) == 2 for route in routes
    )
    overlap = double_count / len(routes) if routes else 0.0
    amendment_hash = sha256_file(amendment_path)
    implementation = amendment.get("implementation") or {}
    implementation_valid = _REQUIRED_IMPLEMENTATIONS <= set(implementation)
    for name in _REQUIRED_IMPLEMENTATIONS:
        receipt = implementation.get(name) or {}
        path = Path(str(receipt.get("path") or ""))
        implementation_valid &= _is_file_hash(path, receipt.get("sha256"))

    no_return_files = not any(
        path.is_file()
        for root in (assignment_root / "returns", assignment_root / "adjudication")
        if root.exists()
        for path in root.rglob("*")
    )
    checks = {
        "prospective_packet_manifest_exact_96": (
            packet.get("status") == "prospective_factorial_packets_frozen_before_review"
            and packet.get("human_outcomes_inspected") is False
            and len(packet_records) == 96
            and len(packet_index) == 96
            and all(packet_index)
        ),
        "eight_datasets_exactly_12_cases": (
            len(dataset_cases) == 8
            and all(len(case_ids) == 12 for case_ids in dataset_cases.values())
        ),
        "every_dataset_has_three_cases_per_factorial_arm": (
            bool(dataset_arms)
            and all(counts == Counter({arm: 3 for arm in _ARMS}) for counts in dataset_arms.values())
        ),
        "integrated_registry_exactly_six_real_reviewer_slots": (
            registry.get("status") == "integrated_reviewer_registry_frozen_before_heldout"
            and len(reviewer_rows) == 6
            and len(reviewer_ids) == 6
            and all(row.get("conflicts_declared") is True for row in reviewer_rows)
        ),
        "capability_inputs_bound_and_predecision_frozen": (
            capability.get("status") == "frozen_before_heldout_assignment"
            and capability.get("heldout_outcomes_inspected") is False
            and capability.get("reviewer_registry_sha256") == sha256_file(reviewer_registry_path)
            and capability.get("combined_observations_sha256") == sha256_file(combined_observations_path)
            and capability.get("predecision_sha256") == sha256_file(combined_observations_path)
            and observations.get("heldout_outcomes_inspected") is False
        ),
        "assignment_artifact_bindings_valid": (
            assignment.get("status") == "frozen_before_heldout_review"
            and assignment.get("heldout_outcomes_inspected") is False
            and assignment.get("packet_manifest_sha256") == sha256_file(packet_manifest_path)
            and assignment.get("reviewer_registry_sha256") == sha256_file(reviewer_registry_path)
            and assignment.get("capability_freeze_sha256") == sha256_file(capability_freeze_path)
            and assignment.get("combined_observations_sha256") == sha256_file(combined_observations_path)
            and assignment.get("amendment_sha256") == sha256_file(previous_amendment_path)
            and internal.get("packet_manifest_sha256") == sha256_file(packet_manifest_path)
        ),
        "exact_case_routes_and_primary_records": (
            len(routes) == 96
            and len(route_index) == 96
            and set(route_index) == set(packet_index)
            and expected_flat == actual_flat
            and len(actual_flat) == len(flat)
            and routes_valid
        ),
        "route_predecision_hashes_valid": route_hashes_valid,
        "packet_copies_match_frozen_arm_hashes": packet_files_valid,
        "personalized_capability_threshold_enforced": capability_qualified,
        "conflicts_and_role_separation_enforced": conflicts_respected and no_role_reuse,
        "all_critical_cases_dual_reviewed": critical_dual,
        "double_review_overlap_at_least_50_percent": overlap >= 0.5,
        "reviewer_crossover_all_four_arms": (
            reviewer_ids == set(reviewer_arms)
            and all(_ARMS <= reviewer_arms[reviewer] for reviewer in reviewer_ids)
        ),
        "no_human_return_files_before_readiness": no_return_files,
        "amendment_005_frozen_before_human_outcomes": (
            amendment.get("amendment_id")
            == "review_policy_amendment_005_integrated_capability_and_factorial_execution"
            and amendment.get("human_outcomes_inspected") is False
            and amendment.get("heldout_outcomes_inspected") is False
            and freeze.get("sha256") == amendment_hash
            and freeze.get("human_outcomes_inspected") is False
            and freeze.get("heldout_outcomes_inspected") is False
        ),
        "amendment_chain_to_004_valid": amendment.get(
            "previous_amendment_sha256"
        )
        == sha256_file(previous_amendment_path),
        "registered_implementation_hashes_valid": implementation_valid,
    }
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 2,
        "status": "ready" if not blocking else "blocked",
        "human_outcomes_inspected": False,
        "checks": checks,
        "blocking_reasons": blocking,
        "counts": {
            "cases": len(packet_records),
            "datasets": len(dataset_cases),
            "reviewers": len(reviewer_ids),
            "primary_decisions": len(flat),
            "double_reviewed_cases": double_count,
            "double_review_overlap_fraction": overlap,
        },
        "artifact_hashes": {
            "packet_manifest_sha256": sha256_file(packet_manifest_path),
            "reviewer_registry_sha256": sha256_file(reviewer_registry_path),
            "capability_freeze_sha256": sha256_file(capability_freeze_path),
            "combined_observations_sha256": sha256_file(combined_observations_path),
            "assignment_manifest_sha256": sha256_file(assignment_path),
            "assignment_internal_manifest_sha256": sha256_file(internal_path),
            "amendment_sha256": amendment_hash,
            "amendment_freeze_sha256": sha256_file(amendment_freeze_path),
        },
    }
