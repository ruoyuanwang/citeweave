from __future__ import annotations

import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json
from .review_ui import ReviewStore


class OversightFactorialOutcomeError(ValueError):
    pass


_SERVER_TIMING = "visibility_heartbeat_server_accounted"
_SERVER_FIELDS = {
    "packet_id",
    "reviewer_code",
    "session_id",
    "review_seconds",
    "server_elapsed_seconds",
    "timing_method",
    "submitted_at_unix",
}


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OversightFactorialOutcomeError(f"{label} must be a positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise OversightFactorialOutcomeError(f"{label} must be a positive number")
    return converted


def _assignment_index(
    assignment_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = assignment_root / "assignment_manifest.json"
    internal_path = assignment_root / "internal_manifest.json"
    manifest = read_json(manifest_path)
    internal = read_json(internal_path)
    routes = manifest.get("case_routes") or []
    if (
        manifest.get("status") != "frozen_before_heldout_review"
        or manifest.get("heldout_outcomes_inspected") is not False
        or manifest.get("cases") != 96
        or manifest.get("datasets") != 8
        or len(routes) != 96
        or len({row.get("case_id") for row in routes}) != 96
        or internal.get("status")
        != "factorial_oversight_assignments_frozen_before_review"
        or internal.get("packet_manifest_sha256")
        != manifest.get("packet_manifest_sha256")
    ):
        raise OversightFactorialOutcomeError("Assignment is not the frozen factorial panel")
    expected = {
        (reviewer, case_id)
        for reviewer, layers in internal.get("assignments", {}).items()
        for case_id in layers.get("adversarial", [])
    }
    routed = {
        (reviewer, str(route["case_id"]))
        for route in routes
        for reviewer in route.get("primary_reviewers") or []
    }
    flat = {
        (str(row.get("reviewer_id")), str(row.get("case_id")))
        for row in manifest.get("records") or []
    }
    if expected != routed or expected != flat:
        raise OversightFactorialOutcomeError("Primary assignments are inconsistent")
    commitments = internal.get("adjudicator_commitments") or {}
    expected_commitments = {
        str(route["case_id"]): route["adjudicator_id"]
        for route in routes
        if len(route.get("primary_reviewers") or []) == 2
    }
    if commitments != expected_commitments:
        raise OversightFactorialOutcomeError("Adjudicator commitments are inconsistent")
    return manifest, internal, {str(row["case_id"]): row for row in routes}


def _validate_result(
    row: dict[str, Any],
    *,
    reviewer_id: str,
    case_id: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    if row.get("reviewer_code") != reviewer_id or row.get("packet_id") != case_id:
        raise OversightFactorialOutcomeError(f"Reviewer or case mismatch: {case_id}")
    if row.get("timing_method") != _SERVER_TIMING:
        raise OversightFactorialOutcomeError(f"Untrusted timing method: {case_id}")
    review_seconds = _positive_number(
        row.get("review_seconds"), label=f"review_seconds for {case_id}"
    )
    elapsed = _positive_number(
        row.get("server_elapsed_seconds"),
        label=f"server_elapsed_seconds for {case_id}",
    )
    _positive_number(
        row.get("submitted_at_unix"), label=f"submitted_at_unix for {case_id}"
    )
    if review_seconds > elapsed + 0.002:
        raise OversightFactorialOutcomeError(
            f"Active review time exceeds elapsed time: {case_id}"
        )
    answers = {key: value for key, value in row.items() if key not in _SERVER_FIELDS}
    try:
        ReviewStore._validate_answers("adversarial", answers, packet)
    except (TypeError, ValueError) as exc:
        raise OversightFactorialOutcomeError(
            f"Invalid held-out response for {case_id}: {exc}"
        ) from exc
    return {**row, "review_seconds": review_seconds, "server_elapsed_seconds": elapsed}


def _return_payload(
    *, root: Path, reviewer_id: str, expected_cases: set[str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = root / "returns" / f"{reviewer_id}.json"
    if not path.is_file():
        raise OversightFactorialOutcomeError(f"Missing return for {reviewer_id}")
    payload = read_json(path)
    rows = payload.get("results")
    if payload.get("reviewer_code") != reviewer_id or not isinstance(rows, list):
        raise OversightFactorialOutcomeError(f"Invalid return envelope for {reviewer_id}")
    indexed = {str(row.get("packet_id")): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected_cases:
        raise OversightFactorialOutcomeError(
            f"Return is incomplete, duplicated, or unexpected for {reviewer_id}"
        )
    validated = []
    for case_id in sorted(expected_cases):
        packet_path = root / "packets" / "adversarial" / f"{case_id}.json"
        if not packet_path.is_file():
            raise OversightFactorialOutcomeError(f"Missing assigned packet: {case_id}")
        validated.append(
            _validate_result(
                indexed[case_id],
                reviewer_id=reviewer_id,
                case_id=case_id,
                packet=read_json(packet_path),
            )
        )
    return validated, {
        "reviewer_id": reviewer_id,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def validate_factorial_primary_returns(
    assignment_root: Path, *, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise OversightFactorialOutcomeError("Refusing to overwrite primary validation")
    manifest, internal, _ = _assignment_index(assignment_root)
    validated = []
    receipts = []
    for reviewer_id, layers in sorted(internal["assignments"].items()):
        rows, receipt = _return_payload(
            root=assignment_root,
            reviewer_id=reviewer_id,
            expected_cases=set(layers["adversarial"]),
        )
        validated.extend(rows)
        receipts.append(receipt)
    if len(validated) != len(manifest.get("records") or []):
        raise OversightFactorialOutcomeError("Primary decision count is incomplete")
    result = {
        "schema_version": 1,
        "status": "factorial_primary_returns_validated",
        "assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "internal_manifest_sha256": sha256_file(
            assignment_root / "internal_manifest.json"
        ),
        "packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "decisions": len(validated),
        "return_files": receipts,
        "validated_results": validated,
    }
    write_json(output_path, result)
    return result


def prepare_factorial_adjudication(
    assignment_root: Path,
    primary_validation_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightFactorialOutcomeError("Refusing to overwrite adjudication panel")
    _, internal, routes = _assignment_index(assignment_root)
    primary = read_json(primary_validation_path)
    if (
        primary.get("status") != "factorial_primary_returns_validated"
        or primary.get("assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
        or primary.get("internal_manifest_sha256")
        != sha256_file(assignment_root / "internal_manifest.json")
    ):
        raise OversightFactorialOutcomeError("Primary validation is not bound to assignment")
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary.get("validated_results") or []:
        indexed[str(row["packet_id"])].append(row)
    assignments: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"adversarial": []}
    )
    records = []
    for case_id, route in sorted(routes.items()):
        if len(route.get("primary_reviewers") or []) != 2:
            continue
        rows = indexed[case_id]
        if len(rows) != 2:
            raise OversightFactorialOutcomeError(f"Double-review missing: {case_id}")
        if rows[0]["verdict"] == rows[1]["verdict"]:
            continue
        adjudicator = internal["adjudicator_commitments"][case_id]
        assignments[adjudicator]["adversarial"].append(case_id)
        source = assignment_root / "packets" / "adversarial" / f"{case_id}.json"
        target = output_root / "packets" / "adversarial" / f"{case_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "case_id": case_id,
                "adjudicator_id": adjudicator,
                "packet_sha256": sha256_file(target),
            }
        )
    frozen = {
        reviewer: {"adversarial": sorted(layers["adversarial"])}
        for reviewer, layers in sorted(assignments.items())
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "internal_manifest.json",
        {
            "schema_version": 1,
            "status": "factorial_adjudication_assignments_frozen",
            "assignments": frozen,
            "primary_assignment_manifest_sha256": sha256_file(
                assignment_root / "assignment_manifest.json"
            ),
            "primary_validation_sha256": sha256_file(primary_validation_path),
        },
    )
    result = {
        "schema_version": 1,
        "status": "factorial_adjudication_ready",
        "primary_identities_hidden": True,
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
            expected_cases=set(layers["adversarial"]),
        )
        for row in rows:
            if row["packet_id"] in validated:
                raise OversightFactorialOutcomeError("Duplicate adjudication result")
            validated[row["packet_id"]] = row
    return validated


def finalize_factorial_outcomes(
    assignment_root: Path,
    packet_manifest_path: Path,
    primary_validation_path: Path,
    adjudication_root: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise OversightFactorialOutcomeError("Refusing to overwrite final outcomes")
    assignment, _, routes = _assignment_index(assignment_root)
    if assignment.get("packet_manifest_sha256") != sha256_file(packet_manifest_path):
        raise OversightFactorialOutcomeError("Held-out packet manifest changed")
    primary = read_json(primary_validation_path)
    adjudication_manifest = read_json(adjudication_root / "manifest.json")
    if (
        primary.get("assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
        or adjudication_manifest.get("primary_assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
        or adjudication_manifest.get("primary_validation_sha256")
        != sha256_file(primary_validation_path)
    ):
        raise OversightFactorialOutcomeError("Outcome inputs are not cryptographically bound")
    primary_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary.get("validated_results") or []:
        primary_by_case[str(row["packet_id"])].append(row)
    adjudicated = _adjudication_results(adjudication_root)
    required = {str(row["case_id"]) for row in adjudication_manifest.get("records") or []}
    if set(adjudicated) != required:
        raise OversightFactorialOutcomeError(
            "Adjudication returns are incomplete or unexpected"
        )
    packet_manifest = read_json(packet_manifest_path)
    packet_root = packet_manifest_path.resolve().parent
    packets = {str(row["case_id"]): row for row in packet_manifest.get("records") or []}
    if set(packets) != set(routes):
        raise OversightFactorialOutcomeError("Packet and assignment cases differ")
    records = []
    diagnostics = []
    for case_id, route in sorted(routes.items()):
        primary_rows = primary_by_case[case_id]
        expected_count = len(route["primary_reviewers"])
        if len(primary_rows) != expected_count:
            raise OversightFactorialOutcomeError(f"Primary decisions missing: {case_id}")
        verdicts = [str(row["verdict"]) for row in primary_rows]
        disagreement = len(set(verdicts)) > 1
        if disagreement:
            if case_id not in adjudicated:
                raise OversightFactorialOutcomeError(f"Missing adjudication: {case_id}")
            final_verdict = str(adjudicated[case_id]["verdict"])
        else:
            if case_id in adjudicated:
                raise OversightFactorialOutcomeError(
                    f"Unexpected adjudication without disagreement: {case_id}"
                )
            final_verdict = verdicts[0]
        packet_row = packets[case_id]
        internal_path = packet_root / str(packet_row["internal_path"])
        if (
            not internal_path.is_file()
            or sha256_file(internal_path) != packet_row["internal_sha256"]
        ):
            raise OversightFactorialOutcomeError(f"Private gold changed: {case_id}")
        gold_verdict = str(read_json(internal_path)["gold_verdict"])
        total_seconds = sum(float(row["review_seconds"]) for row in primary_rows)
        if case_id in adjudicated:
            total_seconds += float(adjudicated[case_id]["review_seconds"])
        records.append(
            {
                "case_id": case_id,
                "dataset_id": route["dataset_id"],
                "packet_arm": route["packet_arm"],
                "assignment_arm": route["assignment_arm"],
                "adjudicated_final_correct": final_verdict == gold_verdict,
                "review_seconds": total_seconds,
            }
        )
        diagnostics.append(
            {
                "case_id": case_id,
                "primary_decisions": expected_count,
                "primary_disagreement": disagreement,
                "blind_adjudication_used": case_id in adjudicated,
                "final_verdict": final_verdict,
                "gold_verdict": gold_verdict,
            }
        )
    if len(records) != 96:
        raise OversightFactorialOutcomeError("Exactly 96 final outcomes are required")
    result = {
        "schema_version": 1,
        "status": "factorial_oversight_outcomes_finalized",
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_manifest_sha256": sha256_file(
            adjudication_root / "manifest.json"
        ),
        "server_timing_only": True,
        "precommitted_blind_adjudication": True,
        "cases": len(records),
        "primary_decisions": len(primary.get("validated_results") or []),
        "blind_adjudications": len(adjudicated),
        "records": records,
        "diagnostics": diagnostics,
    }
    write_json(output_path, result)
    return result
