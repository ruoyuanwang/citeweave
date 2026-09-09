from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.io import read_json, write_json
from citeweave.oversight_calibration_assignment import build_calibration_assignment
from citeweave.oversight_calibration_outcomes import (
    OversightCalibrationOutcomeError,
    finalize_calibration_observations,
    prepare_calibration_adjudication,
    validate_calibration_primary_returns,
)
from tests.test_oversight_calibration_assignment import _packets_and_roster


def _result(reviewer: str, packet_id: str, verdict: str) -> dict[str, object]:
    return {
        "packet_id": packet_id,
        "reviewer_code": reviewer,
        "verdict": verdict,
        "error_type": None if verdict == "valid" else "controlled_error",
        "decisive_evidence_ids": [],
        "rationale": "The visible candidate was checked against the supplied evidence.",
        "review_seconds": 10.0,
        "server_elapsed_seconds": 12.0,
        "timing_method": "visibility_heartbeat_server_accounted",
        "submitted_at_unix": 1_800_000_000.0,
    }


def _gold(packet_root: Path) -> dict[str, str]:
    manifest = read_json(packet_root / "manifest.json")
    return {
        row["packet_id"]: read_json(packet_root / row["internal_path"])[
            "expected_verdict"
        ]
        for row in manifest["records"]
    }


def _write_primary_returns(
    assignment_root: Path, packet_root: Path, *, create_disagreement: bool
) -> str:
    internal = read_json(assignment_root / "internal_manifest.json")
    assignment = read_json(assignment_root / "assignment_manifest.json")
    gold = _gold(packet_root)
    disagreement_packet = next(
        row["packet_id"] for row in assignment["records"] if row["double_review"]
    )
    disagreement_reviewer = next(
        row["primary_reviewers"][1]
        for row in assignment["records"]
        if row["packet_id"] == disagreement_packet
    )
    for reviewer, layers in internal["assignments"].items():
        results = []
        for packet_id in layers["calibration"]:
            verdict = gold[packet_id]
            if (
                create_disagreement
                and packet_id == disagreement_packet
                and reviewer == disagreement_reviewer
            ):
                verdict = "invalid" if verdict == "valid" else "valid"
            results.append(_result(reviewer, packet_id, verdict))
        write_json(
            assignment_root / "returns" / f"{reviewer}.json",
            {"schema_version": 1, "reviewer_code": reviewer, "results": results},
        )
    return disagreement_packet


def test_calibration_returns_close_blind_adjudication_and_observation_loop(
    tmp_path: Path,
) -> None:
    packet_root, roster = _packets_and_roster(tmp_path)
    assignment_root = tmp_path / "assignment"
    build_calibration_assignment(
        packet_root=packet_root,
        reviewer_roster_path=roster,
        output_root=assignment_root,
    )
    disagreement_packet = _write_primary_returns(
        assignment_root, packet_root, create_disagreement=True
    )
    primary_path = tmp_path / "primary.json"
    primary = validate_calibration_primary_returns(
        assignment_root, output_path=primary_path
    )
    assert primary["decisions"] == 144
    adjudication_root = tmp_path / "adjudication"
    adjudication = prepare_calibration_adjudication(
        assignment_root,
        primary_path,
        output_root=adjudication_root,
    )
    assert adjudication["adjudications_required"] == 1
    assert adjudication["records"][0]["packet_id"] == disagreement_packet
    gold = _gold(packet_root)
    internal = read_json(adjudication_root / "internal_manifest.json")
    for reviewer, layers in internal["assignments"].items():
        write_json(
            adjudication_root / "returns" / f"{reviewer}.json",
            {
                "schema_version": 1,
                "reviewer_code": reviewer,
                "results": [
                    _result(reviewer, packet_id, gold[packet_id])
                    for packet_id in layers["calibration"]
                ],
            },
        )
    final_root = tmp_path / "final"
    result = finalize_calibration_observations(
        assignment_root,
        packet_root,
        roster,
        primary_path,
        adjudication_root,
        output_root=final_root,
    )
    assert result["primary_observations"] == 144
    assert result["blind_adjudications"] == 1
    observations = read_json(final_root / "calibration_observations.json")
    registry = read_json(final_root / "calibration_reviewer_registry.json")
    assert len(observations["observations"]) == 144
    assert len(registry["reviewers"]) == 6
    assert all(
        len(row["warmup_issue_types"]) == 4 for row in registry["reviewers"]
    )


def test_primary_return_rejects_client_timing(tmp_path: Path) -> None:
    packet_root, roster = _packets_and_roster(tmp_path)
    assignment_root = tmp_path / "assignment"
    build_calibration_assignment(
        packet_root=packet_root,
        reviewer_roster_path=roster,
        output_root=assignment_root,
    )
    _write_primary_returns(assignment_root, packet_root, create_disagreement=False)
    return_path = next((assignment_root / "returns").glob("*.json"))
    payload = read_json(return_path)
    payload["results"][0]["timing_method"] = "client_reported"
    write_json(return_path, payload)
    with pytest.raises(OversightCalibrationOutcomeError, match="Untrusted timing"):
        validate_calibration_primary_returns(
            assignment_root, output_path=tmp_path / "must_not_exist.json"
        )
