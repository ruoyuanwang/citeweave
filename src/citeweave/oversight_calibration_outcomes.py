from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .complementary_oversight import ReviewerObservation
from .io import read_json, sha256_file, write_json
from .review_ui import ReviewStore


class OversightCalibrationOutcomeError(ValueError):
    pass


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OversightCalibrationOutcomeError(f"{label} must be a positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise OversightCalibrationOutcomeError(f"{label} must be a positive number")
    return converted


def _assignment_index(assignment_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(assignment_root / "assignment_manifest.json")
    internal = read_json(assignment_root / "internal_manifest.json")
    if (
        manifest.get("status")
        != "multidimensional_calibration_assignment_ready_before_returns"
        or manifest.get("human_returns_present") is not False
        or manifest.get("confirmatory_exclusion") is not True
        or manifest.get("packets") != 96
        or manifest.get("primary_decisions") != 144
        or manifest.get("double_reviewed_packets") != 48
    ):
        raise OversightCalibrationOutcomeError("Assignment is not the frozen calibration panel")
    records = manifest.get("records") or []
    if len(records) != 96 or len({row.get("packet_id") for row in records}) != 96:
        raise OversightCalibrationOutcomeError("Calibration assignment records are invalid")
    assigned = {
        (reviewer, packet_id)
        for reviewer, layers in internal.get("assignments", {}).items()
        for packet_id in layers.get("calibration", [])
    }
    expected = {
        (reviewer, row["packet_id"])
        for row in records
        for reviewer in row["primary_reviewers"]
    }
    if assigned != expected or len(assigned) != 144:
        raise OversightCalibrationOutcomeError("Internal primary assignments do not match manifest")
    commitments = internal.get("adjudicator_commitments") or {}
    expected_commitments = {
        row["packet_id"]: row["adjudicator_id"]
        for row in records
        if row["double_review"]
    }
    if commitments != expected_commitments or len(commitments) != 48:
        raise OversightCalibrationOutcomeError("Adjudicator commitments do not match manifest")
    return manifest, internal


def _validate_result(
    row: dict[str, Any], *, reviewer_id: str, packet_id: str
) -> dict[str, Any]:
    if row.get("reviewer_code") != reviewer_id or row.get("packet_id") != packet_id:
        raise OversightCalibrationOutcomeError(f"Reviewer or packet mismatch: {packet_id}")
    if row.get("timing_method") != "visibility_heartbeat_server_accounted":
        raise OversightCalibrationOutcomeError(f"Untrusted timing method: {packet_id}")
    review_seconds = _positive_number(
        row.get("review_seconds"), label=f"review_seconds for {packet_id}"
    )
    elapsed = _positive_number(
        row.get("server_elapsed_seconds"),
        label=f"server_elapsed_seconds for {packet_id}",
    )
    _positive_number(
        row.get("submitted_at_unix"), label=f"submitted_at_unix for {packet_id}"
    )
    if review_seconds > elapsed + 0.002:
        raise OversightCalibrationOutcomeError(
            f"Active review time exceeds elapsed time: {packet_id}"
        )
    answers = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "packet_id",
            "reviewer_code",
            "session_id",
            "review_seconds",
            "server_elapsed_seconds",
            "timing_method",
            "submitted_at_unix",
        }
    }
    try:
        ReviewStore._validate_answers("calibration", answers)
    except (TypeError, ValueError) as exc:
        raise OversightCalibrationOutcomeError(
            f"Invalid calibration response for {packet_id}: {exc}"
        ) from exc
    return {**row, "review_seconds": review_seconds, "server_elapsed_seconds": elapsed}


def _return_payload(
    *, root: Path, reviewer_id: str, expected_packets: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "returns" / f"{reviewer_id}.json"
    if not path.is_file():
        raise OversightCalibrationOutcomeError(f"Missing return for {reviewer_id}")
    payload = read_json(path)
    rows = payload.get("results")
    if payload.get("reviewer_code") != reviewer_id or not isinstance(rows, list):
        raise OversightCalibrationOutcomeError(f"Invalid return envelope for {reviewer_id}")
    indexed = {str(row.get("packet_id")): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected_packets:
        raise OversightCalibrationOutcomeError(
            f"Return is incomplete, duplicated, or unexpected for {reviewer_id}"
        )
    validated = [
        _validate_result(indexed[packet_id], reviewer_id=reviewer_id, packet_id=packet_id)
        for packet_id in sorted(expected_packets)
    ]
    return validated, {
        "reviewer_id": reviewer_id,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def validate_calibration_primary_returns(
    assignment_root: Path, *, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise OversightCalibrationOutcomeError("Refusing to overwrite primary validation")
    manifest, internal = _assignment_index(assignment_root)
    validated = []
    return_files = []
    for reviewer_id, layers in sorted(internal["assignments"].items()):
        rows, receipt = _return_payload(
            root=assignment_root,
            reviewer_id=reviewer_id,
            expected_packets=set(layers["calibration"]),
        )
        validated.extend(rows)
        return_files.append(receipt)
    if len(validated) != 144:
        raise OversightCalibrationOutcomeError("Exactly 144 primary decisions are required")
    result = {
        "schema_version": 1,
        "status": "calibration_primary_returns_validated",
        "assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "internal_manifest_sha256": sha256_file(
            assignment_root / "internal_manifest.json"
        ),
        "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "reviewers": sorted(internal["reviewers"]),
        "decisions": len(validated),
        "return_files": return_files,
        "validated_results": validated,
    }
    write_json(output_path, result)
    return result


def prepare_calibration_adjudication(
    assignment_root: Path,
    primary_validation_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightCalibrationOutcomeError("Refusing to overwrite adjudication panel")
    manifest, internal = _assignment_index(assignment_root)
    primary = read_json(primary_validation_path)
    if (
        primary.get("status") != "calibration_primary_returns_validated"
        or primary.get("assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
        or primary.get("internal_manifest_sha256")
        != sha256_file(assignment_root / "internal_manifest.json")
    ):
        raise OversightCalibrationOutcomeError("Primary validation is not bound to assignment")
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary["validated_results"]:
        indexed[row["packet_id"]].append(row)
    assignments: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"calibration": []}
    )
    records = []
    for assignment in manifest["records"]:
        if not assignment["double_review"]:
            continue
        packet_id = assignment["packet_id"]
        rows = indexed[packet_id]
        if len(rows) != 2:
            raise OversightCalibrationOutcomeError(f"Double-review missing: {packet_id}")
        if rows[0]["verdict"] == rows[1]["verdict"]:
            continue
        adjudicator = internal["adjudicator_commitments"][packet_id]
        assignments[adjudicator]["calibration"].append(packet_id)
        source = assignment_root / "packets" / "calibration" / f"{packet_id}.json"
        target = output_root / "packets" / "calibration" / f"{packet_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "packet_id": packet_id,
                "adjudicator_id": adjudicator,
                "packet_sha256": sha256_file(target),
            }
        )
    frozen_assignments = {
        reviewer: {"calibration": sorted(layers["calibration"])}
        for reviewer, layers in sorted(assignments.items())
    }
    adjudication_internal = {
        "schema_version": 1,
        "status": "calibration_adjudication_assignments_frozen",
        "reviewers": sorted(frozen_assignments),
        "assignments": frozen_assignments,
        "primary_assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "primary_validation_sha256": sha256_file(primary_validation_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "internal_manifest.json", adjudication_internal)
    result = {
        "schema_version": 1,
        "status": "calibration_adjudication_ready",
        "primary_judgments_hidden": True,
        "precommitted_adjudicators_only": True,
        "primary_assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudications_required": len(records),
        "records": records,
    }
    write_json(output_root / "manifest.json", result)
    return result


def _adjudication_results(adjudication_root: Path) -> dict[str, dict[str, Any]]:
    internal = read_json(adjudication_root / "internal_manifest.json")
    validated = {}
    for reviewer_id, layers in sorted(internal.get("assignments", {}).items()):
        rows, _ = _return_payload(
            root=adjudication_root,
            reviewer_id=reviewer_id,
            expected_packets=set(layers["calibration"]),
        )
        for row in rows:
            validated[row["packet_id"]] = row
    return validated


def finalize_calibration_observations(
    assignment_root: Path,
    packet_root: Path,
    reviewer_roster_path: Path,
    primary_validation_path: Path,
    adjudication_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightCalibrationOutcomeError("Refusing to overwrite final calibration")
    assignment, internal = _assignment_index(assignment_root)
    if internal.get("reviewer_roster_sha256") != sha256_file(reviewer_roster_path):
        raise OversightCalibrationOutcomeError("Reviewer roster changed after assignment")
    primary = read_json(primary_validation_path)
    adjudication_manifest = read_json(adjudication_root / "manifest.json")
    if (
        adjudication_manifest.get("primary_assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
        or adjudication_manifest.get("primary_validation_sha256")
        != sha256_file(primary_validation_path)
    ):
        raise OversightCalibrationOutcomeError("Adjudication is not bound to primary inputs")
    adjudicated = _adjudication_results(adjudication_root)
    required = {row["packet_id"] for row in adjudication_manifest["records"]}
    if set(adjudicated) != required:
        raise OversightCalibrationOutcomeError("Adjudication returns are incomplete or unexpected")

    packet_manifest = read_json(packet_root / "manifest.json")
    if sha256_file(packet_root / "manifest.json") != assignment["packet_manifest_sha256"]:
        raise OversightCalibrationOutcomeError("Private gold manifest changed after assignment")
    packet_index = {row["packet_id"]: row for row in packet_manifest["records"]}
    assignment_index = {row["packet_id"]: row for row in assignment["records"]}
    observations = []
    correctness: dict[str, Counter[str]] = defaultdict(Counter)
    for result in primary["validated_results"]:
        packet_id = result["packet_id"]
        packet_row = packet_index[packet_id]
        internal_path = packet_root / packet_row["internal_path"]
        if sha256_file(internal_path) != packet_row["internal_sha256"]:
            raise OversightCalibrationOutcomeError(f"Private gold changed: {packet_id}")
        gold = read_json(internal_path)
        correct = result["verdict"] == gold["expected_verdict"]
        row = assignment_index[packet_id]
        observations.append(
            ReviewerObservation(
                observation_id=f"CAL-{packet_id}-{result['reviewer_code']}",
                reviewer_id=result["reviewer_code"],
                dataset_id=row["dataset_id"],
                domain=row["domain"],
                issue_type=row["issue_type"],
                correct_after_adjudication=correct,
                review_seconds=float(result["review_seconds"]),
            )
        )
        correctness[result["reviewer_code"]]["correct" if correct else "incorrect"] += 1
    if len(observations) != 144:
        raise OversightCalibrationOutcomeError("Final calibration requires 144 observations")
    by_reviewer_issue: dict[str, Counter[str]] = defaultdict(Counter)
    by_reviewer_domains: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        by_reviewer_issue[row.reviewer_id][row.issue_type] += 1
        by_reviewer_domains[row.reviewer_id].add(row.domain)
    if any(
        by_reviewer_issue[reviewer][issue] < 3
        for reviewer in internal["reviewers"]
        for issue in packet_manifest["issues"]
    ):
        raise OversightCalibrationOutcomeError("Reviewer issue coverage is insufficient")

    roster = read_json(reviewer_roster_path)
    roster_index = {row["reviewer_id"]: row for row in roster["reviewers"]}
    registry_rows = []
    for reviewer in sorted(internal["reviewers"]):
        source = roster_index[reviewer]
        registry_rows.append(
            {
                "reviewer_id": reviewer,
                "eligible_domains": sorted(set(source.get("eligible_domains") or [])),
                "conflicted_domains": sorted(set(source.get("conflicted_domains") or [])),
                "warmup_cases_completed": sum(by_reviewer_issue[reviewer].values()),
                "warmup_domains": sorted(by_reviewer_domains[reviewer]),
                "warmup_issue_types": dict(sorted(by_reviewer_issue[reviewer].items())),
                "warmup_outcomes_objectively_scored": True,
                "warmup_disagreements_blind_adjudicated": True,
                "warmup_excluded_from_confirmatory": True,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    observations_path = output_root / "calibration_observations.json"
    registry_path = output_root / "calibration_reviewer_registry.json"
    write_json(
        observations_path,
        {
            "schema_version": 1,
            "status": "multidimensional_calibration_observations_resolved",
            "private_injected_gold": True,
            "confirmatory_exclusion": True,
            "assignment_manifest_sha256": sha256_file(
                assignment_root / "assignment_manifest.json"
            ),
            "primary_validation_sha256": sha256_file(primary_validation_path),
            "adjudication_manifest_sha256": sha256_file(
                adjudication_root / "manifest.json"
            ),
            "observations": [asdict(row) for row in observations],
        },
    )
    write_json(
        registry_path,
        {
            "schema_version": 1,
            "status": "multidimensional_calibration_registry_resolved",
            "reviewers": registry_rows,
        },
    )
    result = {
        "schema_version": 1,
        "status": "multidimensional_calibration_finalized",
        "confirmatory_exclusion": True,
        "primary_observations": len(observations),
        "double_reviewed_packets": 48,
        "blind_adjudications": len(adjudicated),
        "reviewers": len(registry_rows),
        "calibration_observations_sha256": sha256_file(observations_path),
        "calibration_reviewer_registry_sha256": sha256_file(registry_path),
        "reviewer_correctness_counts": {
            reviewer: dict(sorted(counts.items()))
            for reviewer, counts in sorted(correctness.items())
        },
    }
    write_json(output_root / "finalization_receipt.json", result)
    return result
