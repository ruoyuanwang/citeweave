from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.io import read_json, write_json
from citeweave.oversight_calibration import CALIBRATION_ISSUES
from citeweave.oversight_capability_freeze import (
    OversightCapabilityFreezeError,
    freeze_integrated_reviewer_capabilities,
)


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    reviewers = [f"R{index}" for index in range(6)]
    domains = [f"domain_{index}" for index in range(8)]
    source_rows = []
    calibration_rows = []
    for reviewer_index, reviewer in enumerate(reviewers):
        for index in range(20):
            source_rows.append(
                {
                    "observation_id": f"SRC-{reviewer}-{index}",
                    "reviewer_id": reviewer,
                    "dataset_id": domains[index % len(domains)],
                    "domain": domains[index % len(domains)],
                    "issue_type": "evidence_relevance",
                    "correct_after_adjudication": index % 5 != reviewer_index % 5,
                    "review_seconds": 20.0 + index,
                }
            )
        for issue_index, issue in enumerate(CALIBRATION_ISSUES):
            for index in range(6):
                calibration_rows.append(
                    {
                        "observation_id": f"CAL-{reviewer}-{issue}-{index}",
                        "reviewer_id": reviewer,
                        "dataset_id": domains[(issue_index + index) % len(domains)],
                        "domain": domains[(issue_index + index) % len(domains)],
                        "issue_type": issue,
                        "correct_after_adjudication": index != reviewer_index % 6,
                        "review_seconds": 15.0 + index,
                    }
                )
    source = root / "source.json"
    calibration = root / "calibration.json"
    roster = root / "roster.json"
    write_json(
        source,
        {
            "status": "warmup_capability_observations_resolved",
            "observations": source_rows,
        },
    )
    write_json(
        calibration,
        {
            "status": "multidimensional_calibration_observations_resolved",
            "confirmatory_exclusion": True,
            "observations": calibration_rows,
        },
    )
    write_json(
        roster,
        {
            "reviewers": [
                {
                    "reviewer_id": reviewer,
                    "eligible_domains": domains,
                    "conflicted_domains": [],
                    "conflicts_declared": True,
                }
                for reviewer in reviewers
            ]
        },
    )
    return source, calibration, roster


def test_freezes_integrated_issue_specific_capabilities(tmp_path: Path) -> None:
    source, calibration, roster = _inputs(tmp_path)
    output = tmp_path / "output"
    receipt = freeze_integrated_reviewer_capabilities(
        source_observations_path=source,
        calibration_observations_path=calibration,
        reviewer_roster_path=roster,
        output_root=output,
    )
    assert receipt["observations"] == 264
    assert receipt["reviewers"] == 6
    capability = read_json(output / "capability_freeze.json")
    registry = read_json(output / "reviewer_registry.json")
    assert capability["status"] == "frozen_before_heldout_assignment"
    assert capability["fallback_order"] == ["exact", "issue", "domain", "global", "prior"]
    assert all(row["personalized_evidence"] for row in capability["profiles"])
    assert all(len(row["warmup_issue_types"]) == 5 for row in registry["reviewers"])


def test_freeze_rejects_missing_issue_coverage(tmp_path: Path) -> None:
    source, calibration, roster = _inputs(tmp_path)
    payload = read_json(calibration)
    payload["observations"] = [
        row
        for row in payload["observations"]
        if not (
            row["reviewer_id"] == "R0"
            and row["issue_type"] == "revision_safety"
        )
    ]
    write_json(calibration, payload)
    with pytest.raises(OversightCapabilityFreezeError, match="144 observations"):
        freeze_integrated_reviewer_capabilities(
            source_observations_path=source,
            calibration_observations_path=calibration,
            reviewer_roster_path=roster,
            output_root=tmp_path / "output",
        )
