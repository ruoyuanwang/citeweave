from __future__ import annotations

from pathlib import Path

from src.citeweave.io import write_json
from src.citeweave.oversight_readiness import (
    assess_complementary_oversight_readiness,
)


def test_prototype_packets_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "status": "packets_built_before_real_review",
            "human_outcomes_inspected": False,
            "records": [{"case_id": f"P{i}"} for i in range(40)],
        },
    )
    result = assess_complementary_oversight_readiness(
        packet_manifest_path=manifest
    )
    assert result["status"] == "blocked"
    assert result["counts"]["heldout_cases"] == 40
    assert "minimum_96_heldout_cases" in result["blocking_reasons"]
    assert (
        "prospective_candidate_outputs_not_prototypes"
        in result["blocking_reasons"]
    )
    assert "minimum_6_reviewers" in result["blocking_reasons"]


def test_fully_frozen_factorial_panel_is_ready(tmp_path: Path) -> None:
    case_ids = [f"C{i:03d}" for i in range(96)]
    reviewers = [f"R{i}" for i in range(6)]
    packet_manifest = tmp_path / "packets.json"
    write_json(
        packet_manifest,
        {
            "study_role": "prospective_heldout_candidate_outputs",
            "human_outcomes_inspected": False,
            "records": [
                {
                    "case_id": case_id,
                    "candidate_output_sha256": f"candidate-{case_id}",
                    "generation_record_sha256": f"generation-{case_id}",
                    "issue_type": "operator",
                    "severity": "high",
                }
                for case_id in case_ids
            ],
        },
    )
    registry = tmp_path / "reviewers.json"
    write_json(
        registry,
        {
            "reviewers": [
                {
                    "reviewer_id": reviewer,
                    "warmup_cases_completed": 12,
                    "warmup_domains": ["domain-a", "domain-b"],
                    "warmup_outcomes_adjudicated": True,
                    "warmup_excluded_from_confirmatory": True,
                }
                for reviewer in reviewers
            ]
        },
    )
    capability = tmp_path / "capability.json"
    write_json(
        capability,
        {
            "status": "frozen_before_heldout_assignment",
            "heldout_outcomes_inspected": False,
            "predecision_sha256": "frozen-capabilities",
            "reviewer_ids": reviewers,
        },
    )
    arms = [
        ("standard_claim_first", "qualified_random"),
        ("standard_claim_first", "capability_cost_router"),
        ("adversarial_two_sided", "qualified_random"),
        ("adversarial_two_sided", "capability_cost_router"),
    ]
    assignments = []
    for index, case_id in enumerate(case_ids):
        first = index % len(reviewers)
        second = (first + 1) % len(reviewers)
        for offset, reviewer_index in enumerate((first, second)):
            arm_index = (index // len(reviewers) + offset) % len(arms)
            packet_arm, assignment_arm = arms[arm_index]
            assignments.append(
                {
                    "case_id": case_id,
                    "reviewer_id": reviewers[reviewer_index],
                    "packet_arm": packet_arm,
                    "assignment_arm": assignment_arm,
                    "pass": "first",
                }
            )
    assignment_manifest = tmp_path / "assignments.json"
    write_json(
        assignment_manifest,
        {
            "status": "frozen_before_heldout_review",
            "heldout_outcomes_inspected": False,
            "records": assignments,
        },
    )
    result = assess_complementary_oversight_readiness(
        packet_manifest_path=packet_manifest,
        reviewer_registry_path=registry,
        capability_freeze_path=capability,
        assignment_manifest_path=assignment_manifest,
        complementary_amendment_path=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_001_complementary_oversight.yml"
        ),
        cluster_inference_amendment_path=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_002_cluster_inference.yml"
        ),
        factorial_analysis_amendment_path=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_003_factorial_cluster_inference.yml"
        ),
        factorial_analysis_freeze_path=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_003_factorial_cluster_inference_freeze.json"
        ),
    )
    assert result["status"] == "ready"
    assert not result["blocking_reasons"]
    assert result["counts"]["double_review_overlap_fraction"] == 1.0


def test_complete_panel_without_frozen_factorial_analysis_is_blocked(
    tmp_path: Path,
) -> None:
    packet_manifest = tmp_path / "packets.json"
    write_json(
        packet_manifest,
        {
            "study_role": "prospective_heldout_candidate_outputs",
            "human_outcomes_inspected": False,
            "records": [
                {
                    "case_id": f"C{i:03d}",
                    "candidate_output_sha256": f"candidate-{i}",
                    "generation_record_sha256": f"generation-{i}",
                    "issue_type": "operator",
                    "severity": "high",
                }
                for i in range(96)
            ],
        },
    )
    result = assess_complementary_oversight_readiness(
        packet_manifest_path=packet_manifest
    )
    assert result["status"] == "blocked"
    assert (
        "factorial_analysis_amendment_frozen_pre_outcome"
        in result["blocking_reasons"]
    )
    assert (
        "factorial_analysis_implementation_hashes_valid"
        in result["blocking_reasons"]
    )
