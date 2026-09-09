from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.io import read_json, write_json
from citeweave.oversight_factorial_assignment import build_factorial_oversight_assignment
from citeweave.oversight_factorial_outcomes import (
    OversightFactorialOutcomeError,
    finalize_factorial_outcomes,
    prepare_factorial_adjudication,
    validate_factorial_primary_returns,
)
from tests.test_oversight_factorial_assignment import (
    AMENDMENT,
    AMENDMENT_FREEZE,
    _factorial_inputs,
)


def _result(reviewer: str, case_id: str, verdict: str) -> dict[str, object]:
    return {
        "packet_id": case_id,
        "reviewer_code": reviewer,
        "verdict": verdict,
        "decisive_evidence_ids": ["E1"] if verdict != "abstain" else [],
        "invalid_operator_steps": [],
        "failure_mode": "No material failure found.",
        "minimal_rewrite": None,
        "rationale": "The claim was checked against both visible evidence sets.",
        "review_seconds": 10.0,
        "server_elapsed_seconds": 12.0,
        "timing_method": "visibility_heartbeat_server_accounted",
        "submitted_at_unix": 1_800_000_000.0,
    }


def _assignment(tmp_path: Path) -> tuple[Path, Path]:
    manifest, registry, capability, observations = _factorial_inputs(tmp_path)
    root = tmp_path / "assignment"
    build_factorial_oversight_assignment(
        packet_manifest_path=manifest,
        reviewer_registry_path=registry,
        capability_freeze_path=capability,
        combined_observations_path=observations,
        amendment_path=AMENDMENT,
        amendment_freeze_path=AMENDMENT_FREEZE,
        output_root=root,
        minimum_capability_lower_95=0.4,
    )
    return root, manifest


def _write_primary_returns(root: Path, *, disagreement: bool) -> str:
    internal = read_json(root / "internal_manifest.json")
    routes = read_json(root / "assignment_manifest.json")["case_routes"]
    double = next(row for row in routes if len(row["primary_reviewers"]) == 2)
    disagree_case = double["case_id"]
    disagree_reviewer = double["primary_reviewers"][1]
    for reviewer, layers in internal["assignments"].items():
        results = []
        for case_id in layers["adversarial"]:
            verdict = (
                "reject"
                if disagreement
                and case_id == disagree_case
                and reviewer == disagree_reviewer
                else "supported"
            )
            results.append(_result(reviewer, case_id, verdict))
        write_json(
            root / "returns" / f"{reviewer}.json",
            {"schema_version": 1, "reviewer_code": reviewer, "results": results},
        )
    return disagree_case


def test_factorial_returns_close_blind_adjudication_and_analysis_input(
    tmp_path: Path,
) -> None:
    assignment_root, packet_manifest = _assignment(tmp_path)
    disagreement_case = _write_primary_returns(assignment_root, disagreement=True)
    primary_path = tmp_path / "primary.json"
    primary = validate_factorial_primary_returns(
        assignment_root, output_path=primary_path
    )
    assert primary["decisions"] >= 96
    adjudication_root = tmp_path / "adjudication"
    adjudication = prepare_factorial_adjudication(
        assignment_root, primary_path, output_root=adjudication_root
    )
    assert adjudication["adjudications_required"] == 1
    assert adjudication["records"][0]["case_id"] == disagreement_case
    internal = read_json(adjudication_root / "internal_manifest.json")
    for reviewer, layers in internal["assignments"].items():
        write_json(
            adjudication_root / "returns" / f"{reviewer}.json",
            {
                "schema_version": 1,
                "reviewer_code": reviewer,
                "results": [
                    _result(reviewer, case_id, "supported")
                    for case_id in layers["adversarial"]
                ],
            },
        )
    output = tmp_path / "outcomes.json"
    result = finalize_factorial_outcomes(
        assignment_root,
        packet_manifest,
        primary_path,
        adjudication_root,
        output_path=output,
    )
    assert result["cases"] == 96
    assert result["blind_adjudications"] == 1
    assert all(row["adjudicated_final_correct"] for row in result["records"])
    assert all(row["review_seconds"] > 0 for row in result["records"])


def test_factorial_primary_rejects_client_timing(tmp_path: Path) -> None:
    assignment_root, _ = _assignment(tmp_path)
    _write_primary_returns(assignment_root, disagreement=False)
    path = next((assignment_root / "returns").glob("*.json"))
    payload = read_json(path)
    payload["results"][0]["timing_method"] = "client_reported"
    write_json(path, payload)
    with pytest.raises(OversightFactorialOutcomeError, match="Untrusted timing"):
        validate_factorial_primary_returns(
            assignment_root, output_path=tmp_path / "must_not_exist.json"
        )
