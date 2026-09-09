from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .complementary_oversight import ReviewerCapabilityModel, ReviewerObservation
from .io import read_json, sha256_file, write_json
from .oversight_factorial_analysis import ASSIGNMENT_ARMS, PACKET_ARMS

FACTORIAL_ARMS = tuple(
    (packet_arm, assignment_arm)
    for packet_arm in PACKET_ARMS
    for assignment_arm in ASSIGNMENT_ARMS
)


class OversightFactorialAssignmentError(ValueError):
    pass


@dataclass(frozen=True)
class _TeamCandidate:
    case_id: str
    primaries: tuple[str, ...]
    adjudicator: str | None
    objective: float
    expected_final_accuracy: float
    expected_review_seconds: float
    capability_profiles: tuple[dict[str, Any], ...]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_amendment(amendment_path: Path, freeze_path: Path) -> str:
    amendment_hash = sha256_file(amendment_path)
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    if (
        amendment.get("amendment_id")
        != "review_policy_amendment_004_multidimensional_calibration"
        or freeze.get("amendment_id") != amendment.get("amendment_id")
        or freeze.get("sha256") != amendment_hash
        or amendment.get("human_outcomes_inspected") is not False
        or amendment.get("heldout_outcomes_inspected") is not False
    ):
        raise OversightFactorialAssignmentError("Oversight amendment chain is not frozen")
    return amendment_hash


def _load_capability_inputs(
    *,
    reviewer_registry_path: Path,
    capability_freeze_path: Path,
    combined_observations_path: Path,
) -> tuple[list[str], dict[str, set[str]], ReviewerCapabilityModel, dict[str, Any]]:
    registry = read_json(reviewer_registry_path)
    capability = read_json(capability_freeze_path)
    observations_payload = read_json(combined_observations_path)
    reviewers = registry.get("reviewers") or []
    reviewer_ids = sorted(str(row.get("reviewer_id") or "") for row in reviewers)
    if len(reviewer_ids) != 6 or any(not value for value in reviewer_ids):
        raise OversightFactorialAssignmentError("Exactly six calibrated reviewers are required")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise OversightFactorialAssignmentError("Reviewer identities are duplicated")
    if (
        registry.get("status") != "integrated_reviewer_registry_frozen_before_heldout"
        or capability.get("status") != "frozen_before_heldout_assignment"
        or capability.get("heldout_outcomes_inspected") is not False
        or observations_payload.get("status")
        != "integrated_warmup_observations_frozen_before_heldout"
        or observations_payload.get("heldout_outcomes_inspected") is not False
        or capability.get("reviewer_registry_sha256")
        != sha256_file(reviewer_registry_path)
        or capability.get("combined_observations_sha256")
        != sha256_file(combined_observations_path)
        or capability.get("predecision_sha256")
        != sha256_file(combined_observations_path)
        or set(capability.get("reviewer_ids") or []) != set(reviewer_ids)
    ):
        raise OversightFactorialAssignmentError("Capability inputs are not a bound predecision freeze")
    eligibility = {}
    for row in reviewers:
        reviewer_id = str(row["reviewer_id"])
        if (
            row.get("conflicts_declared") is not True
            or row.get("warmup_outcomes_adjudicated") is not True
            or row.get("warmup_excluded_from_confirmatory") is not True
            or int(row.get("warmup_cases_completed") or 0) < 12
        ):
            raise OversightFactorialAssignmentError(
                f"Reviewer {reviewer_id} is not confirmatory-eligible"
            )
        eligibility[reviewer_id] = set(map(str, row.get("eligible_domains") or [])) - set(
            map(str, row.get("conflicted_domains") or [])
        )
    try:
        observations = [
            ReviewerObservation(**row)
            for row in observations_payload.get("observations") or []
        ]
    except (TypeError, ValueError) as exc:
        raise OversightFactorialAssignmentError("Capability observation schema is invalid") from exc
    model = ReviewerCapabilityModel(
        observations,
        prior_alpha=float(capability.get("prior_alpha")),
        prior_beta=float(capability.get("prior_beta")),
        minimum_exact_observations=int(capability.get("minimum_exact_observations")),
        minimum_issue_observations=int(capability.get("minimum_issue_observations")),
        minimum_domain_observations=int(capability.get("minimum_domain_observations")),
    )
    return reviewer_ids, eligibility, model, capability


def _validate_packet_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(path)
    records = manifest.get("records") or []
    required = {
        "case_id",
        "dataset_id",
        "domain",
        "issue_type",
        "severity",
        "predecision_error_risk",
        "estimated_default_review_seconds",
        "candidate_output_sha256",
        "generation_record_sha256",
        "standard_public_path",
        "standard_public_sha256",
        "adversarial_public_path",
        "adversarial_public_sha256",
    }
    if (
        manifest.get("study_role") != "prospective_heldout_candidate_outputs"
        or manifest.get("status") != "prospective_factorial_packets_frozen_before_review"
        or manifest.get("human_outcomes_inspected") is not False
        or len(records) < 96
        or any(not required <= set(row) for row in records)
    ):
        raise OversightFactorialAssignmentError("Held-out packet manifest is incomplete or not prospective")
    case_ids = [str(row["case_id"]) for row in records]
    if any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        raise OversightFactorialAssignmentError("Held-out case identities are invalid")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["severity"] not in {"low", "medium", "high", "critical"}:
            raise OversightFactorialAssignmentError("Held-out severity is invalid")
        risk = row["predecision_error_risk"]
        seconds = row["estimated_default_review_seconds"]
        if (
            isinstance(risk, bool)
            or not isinstance(risk, (int, float))
            or not 0 <= float(risk) <= 1
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or float(seconds) <= 0
        ):
            raise OversightFactorialAssignmentError("Held-out routing features are invalid")
        grouped[str(row["dataset_id"])].append(row)
    if len(grouped) < 8 or any(len(rows) < 12 for rows in grouped.values()):
        raise OversightFactorialAssignmentError(
            "Held-out panel needs at least eight datasets and 12 cases per dataset"
        )
    return manifest, records


def _allocate_arms(records: list[dict[str, Any]], *, seed: int) -> dict[str, tuple[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["dataset_id"])].append(row)
    assignments = {}
    for dataset_id, rows in sorted(grouped.items()):
        rng = Random(f"{seed}:arms:{dataset_id}")
        strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            strata[(str(row["domain"]), str(row["issue_type"]), str(row["severity"]))].append(row)
        dataset_counts: Counter[tuple[str, str]] = Counter()
        stratum_counts: dict[tuple[str, str, str], Counter[tuple[str, str]]] = defaultdict(
            Counter
        )
        ordered = []
        for stratum, stratum_rows in sorted(strata.items()):
            shuffled = sorted(stratum_rows, key=lambda row: str(row["case_id"]))
            rng.shuffle(shuffled)
            ordered.extend((stratum, row) for row in shuffled)
        for stratum, row in ordered:
            tie = {arm: rng.random() for arm in FACTORIAL_ARMS}
            arm = min(
                FACTORIAL_ARMS,
                key=lambda value: (
                    dataset_counts[value],
                    stratum_counts[stratum][value],
                    tie[value],
                    value,
                ),
            )
            assignments[str(row["case_id"])] = arm
            dataset_counts[arm] += 1
            stratum_counts[stratum][arm] += 1
        if any(dataset_counts[arm] < 3 for arm in FACTORIAL_ARMS):
            raise OversightFactorialAssignmentError(
                f"Dataset {dataset_id} cannot support three cases in every arm"
            )
    return assignments


def _design_double_review(
    records: list[dict[str, Any]],
    arms: dict[str, tuple[str, str]],
    *,
    seed: int,
) -> set[str]:
    grouped: dict[tuple[str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["dataset_id"]), arms[str(row["case_id"])])].append(row)
    selected = {
        str(row["case_id"])
        for key, rows in sorted(grouped.items())
        for row in sorted(
            rows,
            key=lambda item: hashlib.sha256(
                f"{seed}:overlap:{key}:{item['case_id']}".encode()
            ).hexdigest(),
        )[: len(rows) // 2]
    }
    selected |= {
        str(row["case_id"]) for row in records if row["severity"] == "critical"
    }
    if len(selected) / len(records) < 0.5:
        remaining = sorted(
            (row for row in records if str(row["case_id"]) not in selected),
            key=lambda row: hashlib.sha256(
                f"{seed}:overlap-fill:{row['case_id']}".encode()
            ).hexdigest(),
        )
        needed = math.ceil(len(records) * 0.5) - len(selected)
        selected |= {str(row["case_id"]) for row in remaining[:needed]}
    return selected


def _qualified_profiles(
    row: dict[str, Any],
    reviewer_ids: list[str],
    eligibility: dict[str, set[str]],
    model: ReviewerCapabilityModel,
    *,
    minimum_lower_95: float,
) -> dict[str, dict[str, Any]]:
    domain = str(row["domain"])
    issue_type = str(row["issue_type"])
    profiles = {
        reviewer: asdict(model.estimate(reviewer, domain=domain, issue_type=issue_type))
        for reviewer in reviewer_ids
        if domain in eligibility[reviewer]
    }
    return {
        reviewer: profile
        for reviewer, profile in profiles.items()
        if profile["posterior_lower_95"] >= minimum_lower_95
        and profile["evidence_tier"] in {"exact", "issue", "domain"}
    }


def _team_candidates(
    row: dict[str, Any],
    arm: tuple[str, str],
    profiles: dict[str, dict[str, Any]],
    *,
    require_double: bool,
    reviewer_second_cost: float,
    seed: int,
) -> list[_TeamCandidate]:
    case_id = str(row["case_id"])
    minimum = 2 if require_double else 1
    if len(profiles) < (3 if minimum == 2 else 1):
        return []
    sizes = [minimum]
    if arm[1] == "capability_cost_router" and minimum == 1 and len(profiles) >= 3:
        sizes.append(2)
    candidates = []
    harm_scale = {"low": 1.0, "medium": 4.0, "high": 12.0, "critical": 40.0}
    harm = harm_scale[row["severity"]] * (
        1.0 + math.log1p(max(0, int(row.get("descendants") or 0)))
    )
    for size in sizes:
        for primaries in itertools.combinations(sorted(profiles), size):
            adjudicators = [None]
            if size == 2:
                adjudicators = sorted(set(profiles) - set(primaries))
            for adjudicator in adjudicators:
                if size == 1:
                    profile = profiles[primaries[0]]
                    accuracy = float(profile["posterior_mean_accuracy"])
                    seconds = float(profile["expected_review_seconds"])
                else:
                    first, second = (profiles[reviewer] for reviewer in primaries)
                    adjudicator_profile = profiles[str(adjudicator)]
                    p1 = float(first["posterior_mean_accuracy"])
                    p2 = float(second["posterior_mean_accuracy"])
                    pa = float(adjudicator_profile["posterior_mean_accuracy"])
                    disagreement = p1 * (1 - p2) + (1 - p1) * p2
                    accuracy = p1 * p2 + disagreement * pa
                    seconds = (
                        float(first["expected_review_seconds"])
                        + float(second["expected_review_seconds"])
                        + disagreement * float(adjudicator_profile["expected_review_seconds"])
                    )
                if arm[1] == "capability_cost_router":
                    objective = harm * (1 - accuracy) + reviewer_second_cost * seconds
                else:
                    objective = int(
                        hashlib.sha256(
                            f"{seed}:random:{case_id}:{primaries}:{adjudicator}".encode()
                        ).hexdigest()[:12],
                        16,
                    ) / float(16**12)
                audit_profiles = tuple(
                    {"reviewer_id": reviewer, **profiles[reviewer]}
                    for reviewer in (*primaries, *((adjudicator,) if adjudicator else ()))
                )
                candidates.append(
                    _TeamCandidate(
                        case_id=case_id,
                        primaries=primaries,
                        adjudicator=adjudicator,
                        objective=objective,
                        expected_final_accuracy=accuracy,
                        expected_review_seconds=seconds,
                        capability_profiles=audit_profiles,
                    )
                )
    return candidates


def _solve_global_assignment(
    records: list[dict[str, Any]],
    arms: dict[str, tuple[str, str]],
    double_ids: set[str],
    reviewer_ids: list[str],
    candidates_by_case: dict[str, list[_TeamCandidate]],
) -> dict[str, _TeamCandidate]:
    variables = [
        candidate
        for row in sorted(records, key=lambda item: str(item["case_id"]))
        for candidate in candidates_by_case[str(row["case_id"])]
    ]
    if not variables:
        raise OversightFactorialAssignmentError("No feasible reviewer teams")
    case_variable_indices: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(variables):
        case_variable_indices[candidate.case_id].append(index)
    rows: list[dict[int, float]] = []
    lower = []
    upper = []
    for case_id in sorted(case_variable_indices):
        rows.append({index: 1.0 for index in case_variable_indices[case_id]})
        lower.append(1.0)
        upper.append(1.0)
    for reviewer in reviewer_ids:
        for arm in FACTORIAL_ARMS:
            indices = {
                index: 1.0
                for index, candidate in enumerate(variables)
                if reviewer in candidate.primaries and arms[candidate.case_id] == arm
            }
            if not indices:
                raise OversightFactorialAssignmentError(
                    f"Reviewer {reviewer} cannot receive crossover arm {arm}"
                )
            rows.append(indices)
            lower.append(1.0)
            upper.append(np.inf)
    workload_cap = math.ceil(2 * len(records) / len(reviewer_ids))
    for reviewer in reviewer_ids:
        rows.append(
            {
                index: float(reviewer in candidate.primaries)
                for index, candidate in enumerate(variables)
                if reviewer in candidate.primaries
            }
        )
        lower.append(0.0)
        upper.append(float(workload_cap))
    matrix = lil_matrix((len(rows), len(variables)), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    objective = np.array(
        [candidate.objective + index * 1e-12 for index, candidate in enumerate(variables)],
        dtype=float,
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lower), np.array(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise OversightFactorialAssignmentError(
            f"Global crossover assignment is infeasible: {result.message}"
        )
    selected = {
        candidate.case_id: candidate
        for candidate, value in zip(variables, result.x, strict=True)
        if value > 0.5
    }
    if len(selected) != len(records):
        raise OversightFactorialAssignmentError("MILP did not select exactly one team per case")
    if sum(len(team.primaries) >= 2 for team in selected.values()) < len(double_ids):
        raise OversightFactorialAssignmentError("Double-review overlap constraint was lost")
    return selected


def build_factorial_oversight_assignment(
    *,
    packet_manifest_path: Path,
    reviewer_registry_path: Path,
    capability_freeze_path: Path,
    combined_observations_path: Path,
    amendment_path: Path,
    amendment_freeze_path: Path,
    output_root: Path,
    seed: int = 20260907,
    minimum_capability_lower_95: float = 0.5,
    reviewer_second_cost: float = 0.002,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightFactorialAssignmentError("Refusing to overwrite held-out assignment")
    amendment_hash = _validate_amendment(amendment_path, amendment_freeze_path)
    _, records = _validate_packet_manifest(packet_manifest_path)
    reviewer_ids, eligibility, model, capability = _load_capability_inputs(
        reviewer_registry_path=reviewer_registry_path,
        capability_freeze_path=capability_freeze_path,
        combined_observations_path=combined_observations_path,
    )
    missing_issues = sorted(
        {str(row["issue_type"]) for row in records}
        - set(capability.get("required_issue_types") or [])
    )
    if missing_issues:
        raise OversightFactorialAssignmentError(
            f"Held-out issues lack frozen warmup coverage: {missing_issues}"
        )
    arms = _allocate_arms(records, seed=seed)
    double_ids = _design_double_review(records, arms, seed=seed)
    profiles_by_case = {
        str(row["case_id"]): _qualified_profiles(
            row,
            reviewer_ids,
            eligibility,
            model,
            minimum_lower_95=minimum_capability_lower_95,
        )
        for row in records
    }
    candidates_by_case = {}
    for row in records:
        case_id = str(row["case_id"])
        candidates_by_case[case_id] = _team_candidates(
            row,
            arms[case_id],
            profiles_by_case[case_id],
            require_double=case_id in double_ids,
            reviewer_second_cost=reviewer_second_cost,
            seed=seed,
        )
        if not candidates_by_case[case_id]:
            raise OversightFactorialAssignmentError(
                f"No capability-qualified team for held-out case {case_id}"
            )
    selected = _solve_global_assignment(
        records, arms, double_ids, reviewer_ids, candidates_by_case
    )

    manifest_root = packet_manifest_path.resolve().parent
    assignments: dict[str, dict[str, list[str]]] = {
        reviewer: {"adversarial": []} for reviewer in reviewer_ids
    }
    flat_records = []
    case_routes = []
    for row in sorted(records, key=lambda item: str(item["case_id"])):
        case_id = str(row["case_id"])
        packet_arm, assignment_arm = arms[case_id]
        path_field = (
            "standard_public_path"
            if packet_arm == "standard_claim_first"
            else "adversarial_public_path"
        )
        hash_field = (
            "standard_public_sha256"
            if packet_arm == "standard_claim_first"
            else "adversarial_public_sha256"
        )
        source = manifest_root / str(row[path_field])
        if not source.is_file() or sha256_file(source) != row[hash_field]:
            raise OversightFactorialAssignmentError(f"Held-out packet changed: {case_id}")
        target = output_root / "packets" / "adversarial" / f"{case_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        team = selected[case_id]
        for reviewer in team.primaries:
            assignments[reviewer]["adversarial"].append(case_id)
            flat_records.append(
                {
                    "case_id": case_id,
                    "dataset_id": row["dataset_id"],
                    "reviewer_id": reviewer,
                    "packet_arm": packet_arm,
                    "assignment_arm": assignment_arm,
                    "pass": "first",
                    "packet_sha256": sha256_file(target),
                }
            )
        route_payload = {
            "case_id": case_id,
            "dataset_id": row["dataset_id"],
            "domain": row["domain"],
            "issue_type": row["issue_type"],
            "severity": row["severity"],
            "packet_arm": packet_arm,
            "assignment_arm": assignment_arm,
            "design_double_review": case_id in double_ids,
            "route": (
                "dual_review_with_adjudication"
                if len(team.primaries) == 2
                else "single_review"
            ),
            "primary_reviewers": list(team.primaries),
            "adjudicator_id": team.adjudicator,
            "expected_final_accuracy": team.expected_final_accuracy,
            "expected_review_seconds": team.expected_review_seconds,
            "capability_profiles": list(team.capability_profiles),
            "candidate_output_sha256": row["candidate_output_sha256"],
            "generation_record_sha256": row["generation_record_sha256"],
        }
        case_routes.append(
            {**route_payload, "predecision_sha256": _canonical_hash(route_payload)}
        )
    frozen_assignments = {
        reviewer: {"adversarial": sorted(layers["adversarial"])}
        for reviewer, layers in sorted(assignments.items())
    }
    internal = {
        "schema_version": 1,
        "status": "factorial_oversight_assignments_frozen_before_review",
        "reviewers": reviewer_ids,
        "assignments": frozen_assignments,
        "adjudicator_commitments": {
            row["case_id"]: row["adjudicator_id"]
            for row in case_routes
            if row["adjudicator_id"]
        },
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "reviewer_registry_sha256": sha256_file(reviewer_registry_path),
        "capability_freeze_sha256": sha256_file(capability_freeze_path),
        "combined_observations_sha256": sha256_file(combined_observations_path),
        "amendment_sha256": amendment_hash,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "internal_manifest.json", internal)
    arm_counts = Counter(
        f"{arms[str(row['case_id'])][0]}__{arms[str(row['case_id'])][1]}"
        for row in records
    )
    primary_load = Counter(
        reviewer for route in case_routes for reviewer in route["primary_reviewers"]
    )
    result = {
        "schema_version": 2,
        "status": "frozen_before_heldout_review",
        "heldout_outcomes_inspected": False,
        "seed": seed,
        "global_assignment_method": "binary_milp_with_crossover_and_capacity_constraints",
        "qualified_random_method": "frozen_hash_objective_over_capability_qualified_teams",
        "capability_router_method": "minimum_expected_severity_weighted_error_plus_server_time_cost",
        "minimum_capability_lower_95": minimum_capability_lower_95,
        "personalized_evidence_tiers": ["exact", "issue", "domain"],
        "reviewer_second_cost": reviewer_second_cost,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "reviewer_registry_sha256": sha256_file(reviewer_registry_path),
        "capability_freeze_sha256": sha256_file(capability_freeze_path),
        "combined_observations_sha256": sha256_file(combined_observations_path),
        "amendment_sha256": amendment_hash,
        "cases": len(records),
        "datasets": len({row["dataset_id"] for row in records}),
        "double_reviewed_cases": sum(
            len(route["primary_reviewers"]) == 2 for route in case_routes
        ),
        "double_review_overlap_fraction": sum(
            len(route["primary_reviewers"]) == 2 for route in case_routes
        )
        / len(records),
        "factorial_arm_case_counts": dict(sorted(arm_counts.items())),
        "primary_assignments_per_reviewer": dict(sorted(primary_load.items())),
        "records": flat_records,
        "case_routes": case_routes,
    }
    write_json(output_root / "assignment_manifest.json", result)
    return result
