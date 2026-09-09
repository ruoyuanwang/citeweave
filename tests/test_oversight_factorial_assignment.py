from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_capability_freeze import (
    freeze_integrated_reviewer_capabilities,
)
from citeweave.oversight_factorial_assignment import (
    FACTORIAL_ARMS,
    OversightFactorialAssignmentError,
    build_factorial_oversight_assignment,
)
from tests.test_oversight_capability_freeze import _inputs

AMENDMENT = Path(
    "experiments/human_review_v2/"
    "review_policy_amendment_004_multidimensional_calibration.yml"
)
AMENDMENT_FREEZE = Path(
    "experiments/human_review_v2/"
    "review_policy_amendment_004_multidimensional_calibration_freeze.json"
)


def _factorial_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source, calibration, roster = _inputs(tmp_path / "warmup")
    capability_root = tmp_path / "capability"
    freeze_integrated_reviewer_capabilities(
        source_observations_path=source,
        calibration_observations_path=calibration,
        reviewer_roster_path=roster,
        output_root=capability_root,
    )
    packet_root = tmp_path / "heldout"
    records = []
    issues = [
        "graph_answer_consistency",
        "causal_overreach",
        "counterevidence_coverage",
        "revision_safety",
    ]
    for dataset_index in range(8):
        dataset_id = f"domain_{dataset_index}"
        for case_index in range(12):
            case_id = f"H-{dataset_index}-{case_index}"
            public = {
                "schema_version": 1,
                "case_id": case_id,
                "claim": "Audit this held-out graph-derived claim.",
                "evidence_set_a": [{"evidence_id": "E1"}],
                "evidence_set_b": [{"evidence_id": "E2"}],
                "operator_trace": [{"operator": "test"}],
            }
            standard = packet_root / "standard" / f"{case_id}.json"
            adversarial = packet_root / "adversarial" / f"{case_id}.json"
            internal = packet_root / "internal" / f"{case_id}.json"
            write_json(standard, {**public, "packet_arm": "standard_claim_first"})
            write_json(adversarial, {**public, "packet_arm": "adversarial_two_sided"})
            write_json(internal, {"case_id": case_id, "gold_verdict": "supported"})
            records.append(
                {
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "domain": dataset_id,
                    "issue_type": issues[case_index % len(issues)],
                    "severity": ["low", "medium", "high"][case_index % 3],
                    "predecision_error_risk": 0.2 + case_index / 100,
                    "estimated_default_review_seconds": 30.0,
                    "descendants": case_index % 4,
                    "candidate_output_sha256": f"candidate-{case_id}",
                    "generation_record_sha256": f"generation-{case_id}",
                    "standard_public_path": str(standard.relative_to(packet_root)),
                    "standard_public_sha256": sha256_file(standard),
                    "adversarial_public_path": str(adversarial.relative_to(packet_root)),
                    "adversarial_public_sha256": sha256_file(adversarial),
                    "internal_path": str(internal.relative_to(packet_root)),
                    "internal_sha256": sha256_file(internal),
                }
            )
    manifest = packet_root / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "status": "prospective_factorial_packets_frozen_before_review",
            "study_role": "prospective_heldout_candidate_outputs",
            "human_outcomes_inspected": False,
            "records": records,
        },
    )
    return (
        manifest,
        capability_root / "reviewer_registry.json",
        capability_root / "capability_freeze.json",
        capability_root / "combined_reviewer_observations.json",
    )


def test_builds_global_constrained_factorial_assignment(tmp_path: Path) -> None:
    manifest, registry, capability, observations = _factorial_inputs(tmp_path)
    output = tmp_path / "assignment"
    result = build_factorial_oversight_assignment(
        packet_manifest_path=manifest,
        reviewer_registry_path=registry,
        capability_freeze_path=capability,
        combined_observations_path=observations,
        amendment_path=AMENDMENT,
        amendment_freeze_path=AMENDMENT_FREEZE,
        output_root=output,
        minimum_capability_lower_95=0.4,
    )
    assert result["cases"] == 96
    assert result["datasets"] == 8
    assert result["double_review_overlap_fraction"] >= 0.5
    assert set(result["factorial_arm_case_counts"].values()) == {24}
    assert result["heldout_outcomes_inspected"] is False
    reviewer_arms: dict[str, set[tuple[str, str]]] = defaultdict(set)
    seen = set()
    for row in result["records"]:
        key = (row["reviewer_id"], row["case_id"])
        assert key not in seen
        seen.add(key)
        reviewer_arms[row["reviewer_id"]].add(
            (row["packet_arm"], row["assignment_arm"])
        )
    assert all(set(FACTORIAL_ARMS) <= arms for arms in reviewer_arms.values())
    internal = read_json(output / "internal_manifest.json")
    assert internal["status"] == "factorial_oversight_assignments_frozen_before_review"
    assert len(result["case_routes"]) == 96
    assert all(len(row["predecision_sha256"]) == 64 for row in result["case_routes"])


def test_blocks_heldout_issue_missing_from_warmup(tmp_path: Path) -> None:
    manifest, registry, capability, observations = _factorial_inputs(tmp_path)
    payload = read_json(manifest)
    payload["records"][0]["issue_type"] = "unseen_issue"
    write_json(manifest, payload)
    with pytest.raises(OversightFactorialAssignmentError, match="lack frozen warmup"):
        build_factorial_oversight_assignment(
            packet_manifest_path=manifest,
            reviewer_registry_path=registry,
            capability_freeze_path=capability,
            combined_observations_path=observations,
            amendment_path=AMENDMENT,
            amendment_freeze_path=AMENDMENT_FREEZE,
            output_root=tmp_path / "assignment",
            minimum_capability_lower_95=0.4,
        )
