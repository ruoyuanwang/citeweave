from __future__ import annotations

import itertools
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from .io import read_json, sha256_file, write_json
from .oversight_calibration import CALIBRATION_ISSUES


class OversightCalibrationAssignmentError(ValueError):
    pass


def _reviewer_eligibility(roster: dict[str, Any]) -> tuple[list[str], dict[str, set[str]]]:
    rows = roster.get("reviewers") or []
    reviewer_ids = [str(row.get("reviewer_id") or "") for row in rows]
    if len(reviewer_ids) != 6 or any(not value for value in reviewer_ids):
        raise OversightCalibrationAssignmentError("Exactly six named reviewers are required")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise OversightCalibrationAssignmentError("Reviewer IDs must be unique")
    eligible = {
        str(row["reviewer_id"]): set(map(str, row.get("eligible_domains") or []))
        - set(map(str, row.get("conflicted_domains") or []))
        for row in rows
    }
    return sorted(reviewer_ids), eligible


def _select_double_review(
    records: list[dict[str, Any]], *, dataset_index: int, seed: int
) -> set[str]:
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_counts: Counter[str] = Counter(str(row["base_case_id"]) for row in records)
    for row in records:
        by_issue[str(row["issue_type"])].append(row)
    if set(by_issue) != set(CALIBRATION_ISSUES) or any(
        len(by_issue[issue]) != 3 for issue in CALIBRATION_ISSUES
    ):
        raise OversightCalibrationAssignmentError(
            "Every dataset needs three calibration cases for each issue"
        )
    target_two = {
        CALIBRATION_ISSUES[dataset_index % len(CALIBRATION_ISSUES)],
        CALIBRATION_ISSUES[(dataset_index + 1) % len(CALIBRATION_ISSUES)],
    }
    choices = []
    for issue in CALIBRATION_ISSUES:
        size = 2 if issue in target_two else 1
        choices.append(list(itertools.combinations(by_issue[issue], size)))
    feasible: list[tuple[float, set[str]]] = []
    for selection in itertools.product(*choices):
        selected = [row for group in selection for row in group]
        additions = Counter(str(row["base_case_id"]) for row in selected)
        if any(base_counts[base] + 2 * count > 6 for base, count in additions.items()):
            continue
        rng = Random(f"{seed}:double:{records[0]['dataset_id']}")
        tie = sum(rng.random() for _ in selected)
        feasible.append((tie, {str(row["packet_id"]) for row in selected}))
    if not feasible:
        raise OversightCalibrationAssignmentError("No feasible balanced double-review subset")
    return min(feasible, key=lambda row: (row[0], sorted(row[1])))[1]


def _attempt_role_assignment(
    records: list[dict[str, Any]],
    double_ids: set[str],
    reviewer_ids: list[str],
    eligible: dict[str, set[str]],
    *,
    seed: int,
    attempt: int,
) -> list[dict[str, Any]] | None:
    role_rows = []
    for row in records:
        role_rows.append((row, "primary"))
        if row["packet_id"] in double_ids:
            role_rows.extend(((row, "primary"), (row, "adjudicator")))
    rng = Random(f"{seed}:roles:{attempt}")
    rng.shuffle(role_rows)
    role_rows.sort(key=lambda item: item[1] == "primary", reverse=True)
    seen_bases: dict[str, set[str]] = defaultdict(set)
    case_reviewers: dict[str, set[str]] = defaultdict(set)
    workload: Counter[str] = Counter()
    issue_load: dict[str, Counter[str]] = defaultdict(Counter)
    adjudication_load: Counter[str] = Counter()
    assigned = []
    for row, role in role_rows:
        domain = str(row["domain"])
        base_case_id = str(row["base_case_id"])
        packet_id = str(row["packet_id"])
        issue_type = str(row["issue_type"])
        candidates = [
            reviewer_id
            for reviewer_id in reviewer_ids
            if domain in eligible[reviewer_id]
            and base_case_id not in seen_bases[reviewer_id]
            and reviewer_id not in case_reviewers[packet_id]
        ]
        if not candidates:
            return None
        tie_break = {reviewer_id: rng.random() for reviewer_id in candidates}
        selected = min(
            candidates,
            key=lambda reviewer_id: (
                issue_load[reviewer_id][issue_type],
                workload[reviewer_id],
                adjudication_load[reviewer_id] if role == "adjudicator" else 0,
                tie_break[reviewer_id],
                reviewer_id,
            ),
        )
        seen_bases[selected].add(base_case_id)
        case_reviewers[packet_id].add(selected)
        workload[selected] += 1
        issue_load[selected][issue_type] += 1
        adjudication_load[selected] += role == "adjudicator"
        assigned.append({**row, "role": role, "reviewer_id": selected})
    if any(
        issue_load[reviewer_id][issue] < 3
        for reviewer_id in reviewer_ids
        for issue in CALIBRATION_ISSUES
    ):
        return None
    if any(len(seen_bases[reviewer_id]) != workload[reviewer_id] for reviewer_id in reviewer_ids):
        return None
    return assigned


def build_calibration_assignment(
    *,
    packet_root: Path,
    reviewer_roster_path: Path,
    output_root: Path,
    seed: int = 20260907,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightCalibrationAssignmentError("Refusing to overwrite calibration assignment")
    packet_manifest_path = packet_root / "manifest.json"
    manifest = read_json(packet_manifest_path)
    if (
        manifest.get("status")
        != "multidimensional_calibration_packets_frozen_before_human_returns"
        or manifest.get("study_role")
        != "training_only_reviewer_capability_calibration"
        or manifest.get("human_outcomes_inspected") is not False
        or manifest.get("confirmatory_exclusion") is not True
        or manifest.get("cases") != 96
    ):
        raise OversightCalibrationAssignmentError("Packet panel is not the frozen 96-case warmup")
    records = manifest.get("records") or []
    if len(records) != 96 or len({row.get("packet_id") for row in records}) != 96:
        raise OversightCalibrationAssignmentError("Calibration packet identities are invalid")
    roster = read_json(reviewer_roster_path)
    reviewer_ids, eligible = _reviewer_eligibility(roster)
    datasets = sorted({str(row["dataset_id"]) for row in records})
    if len(datasets) != 8:
        raise OversightCalibrationAssignmentError("Calibration assignment needs eight datasets")
    for dataset in datasets:
        if sum(dataset in eligible[reviewer] for reviewer in reviewer_ids) < 3:
            raise OversightCalibrationAssignmentError(
                f"Dataset {dataset} lacks three conflict-free reviewers"
            )
    double_ids: set[str] = set()
    for dataset_index, dataset in enumerate(datasets):
        dataset_rows = [row for row in records if row["dataset_id"] == dataset]
        if len(dataset_rows) != 12:
            raise OversightCalibrationAssignmentError(
                f"Dataset {dataset} does not have 12 calibration cases"
            )
        double_ids |= _select_double_review(
            dataset_rows, dataset_index=dataset_index, seed=seed
        )
    if len(double_ids) != 48:
        raise OversightCalibrationAssignmentError("Exactly half the panel must be double reviewed")

    assigned = None
    for attempt in range(5000):
        assigned = _attempt_role_assignment(
            records,
            double_ids,
            reviewer_ids,
            eligible,
            seed=seed,
            attempt=attempt,
        )
        if assigned is not None:
            break
    if assigned is None:
        raise OversightCalibrationAssignmentError(
            "Could not satisfy crossover, conflict, and base-case separation constraints"
        )

    by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignments: dict[str, dict[str, list[str]]] = {
        reviewer: {"calibration": []} for reviewer in reviewer_ids
    }
    for row in assigned:
        by_packet[str(row["packet_id"])].append(row)
        if row["role"] == "primary":
            assignments[row["reviewer_id"]]["calibration"].append(row["packet_id"])
    output_records = []
    for row in sorted(records, key=lambda item: str(item["packet_id"])):
        packet_id = str(row["packet_id"])
        roles = by_packet[packet_id]
        primaries = sorted(
            item["reviewer_id"] for item in roles if item["role"] == "primary"
        )
        adjudicators = [
            item["reviewer_id"] for item in roles if item["role"] == "adjudicator"
        ]
        if len(primaries) != (2 if packet_id in double_ids else 1):
            raise OversightCalibrationAssignmentError("Primary reviewer count mismatch")
        if len(adjudicators) != (1 if packet_id in double_ids else 0):
            raise OversightCalibrationAssignmentError("Adjudicator commitment mismatch")
        source_path = packet_root / str(row["public_path"])
        if sha256_file(source_path) != row["public_sha256"]:
            raise OversightCalibrationAssignmentError(f"Packet changed: {packet_id}")
        target_path = output_root / "packets" / "calibration" / f"{packet_id}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        output_records.append(
            {
                "packet_id": packet_id,
                "base_case_id": row["base_case_id"],
                "dataset_id": row["dataset_id"],
                "domain": row["domain"],
                "issue_type": row["issue_type"],
                "primary_reviewers": primaries,
                "double_review": packet_id in double_ids,
                "adjudicator_id": adjudicators[0] if adjudicators else None,
                "packet_sha256": sha256_file(target_path),
            }
        )
    frozen_assignments = {
        reviewer: {"calibration": sorted(layers["calibration"])}
        for reviewer, layers in sorted(assignments.items())
    }
    internal = {
        "schema_version": 1,
        "status": "calibration_assignments_frozen_before_returns",
        "reviewers": reviewer_ids,
        "assignments": frozen_assignments,
        "adjudicator_commitments": {
            row["packet_id"]: row["adjudicator_id"]
            for row in output_records
            if row["double_review"]
        },
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "internal_manifest.json", internal)
    primary_counts = Counter(
        reviewer
        for row in output_records
        for reviewer in row["primary_reviewers"]
    )
    adjudication_counts = Counter(
        row["adjudicator_id"] for row in output_records if row["adjudicator_id"]
    )
    result = {
        "schema_version": 1,
        "status": "multidimensional_calibration_assignment_ready_before_returns",
        "human_returns_present": False,
        "confirmatory_exclusion": True,
        "seed": seed,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
        "datasets": 8,
        "packets": 96,
        "primary_decisions": 144,
        "double_reviewed_packets": 48,
        "potential_adjudications": 48,
        "primary_assignments_per_reviewer": dict(sorted(primary_counts.items())),
        "potential_adjudications_per_reviewer": dict(
            sorted(adjudication_counts.items())
        ),
        "no_reviewer_repeats_base_case_across_any_committed_role": True,
        "minimum_issue_cases_per_reviewer": 3,
        "records": output_records,
    }
    write_json(output_root / "assignment_manifest.json", result)
    return result
