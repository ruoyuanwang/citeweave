from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from citeweave.io import read_json, write_json
from citeweave.oversight_calibration import build_multidimensional_calibration_packets
from citeweave.oversight_calibration_assignment import (
    OversightCalibrationAssignmentError,
    build_calibration_assignment,
)
from tests.test_oversight_calibration import _prototype_panel


def _packets_and_roster(tmp_path: Path) -> tuple[Path, Path]:
    prototype = _prototype_panel(tmp_path / "prototype")
    packet_root = tmp_path / "packets"
    build_multidimensional_calibration_packets(
        prototype_manifest_path=prototype,
        output_root=packet_root,
    )
    domains = [f"dataset_{index}" for index in range(8)]
    roster = tmp_path / "roster.json"
    write_json(
        roster,
        {
            "reviewers": [
                {
                    "reviewer_id": f"R{index}",
                    "eligible_domains": domains,
                    "conflicted_domains": [],
                }
                for index in range(6)
            ]
        },
    )
    return packet_root, roster


def test_assignment_is_balanced_blind_and_base_case_separated(tmp_path: Path) -> None:
    packet_root, roster = _packets_and_roster(tmp_path)
    output_root = tmp_path / "assignment"
    manifest = build_calibration_assignment(
        packet_root=packet_root,
        reviewer_roster_path=roster,
        output_root=output_root,
    )
    assert manifest["packets"] == 96
    assert manifest["primary_decisions"] == 144
    assert manifest["double_reviewed_packets"] == 48
    internal = read_json(output_root / "internal_manifest.json")
    assert sum(
        len(row["calibration"]) for row in internal["assignments"].values()
    ) == 144
    by_reviewer_issue: dict[str, Counter[str]] = defaultdict(Counter)
    seen_bases: dict[str, set[str]] = defaultdict(set)
    for row in manifest["records"]:
        participants = [*row["primary_reviewers"]]
        if row["adjudicator_id"]:
            participants.append(row["adjudicator_id"])
        assert len(participants) == len(set(participants))
        for reviewer in participants:
            assert row["base_case_id"] not in seen_bases[reviewer]
            seen_bases[reviewer].add(row["base_case_id"])
        for reviewer in row["primary_reviewers"]:
            by_reviewer_issue[reviewer][row["issue_type"]] += 1
    assert all(
        by_reviewer_issue[reviewer][issue] >= 3
        for reviewer in internal["reviewers"]
        for issue in (
            "graph_answer_consistency",
            "causal_overreach",
            "counterevidence_coverage",
            "revision_safety",
        )
    )


def test_assignment_fails_when_a_domain_lacks_three_reviewers(tmp_path: Path) -> None:
    packet_root, roster_path = _packets_and_roster(tmp_path)
    roster = read_json(roster_path)
    for row in roster["reviewers"][2:]:
        row["conflicted_domains"] = ["dataset_0"]
    write_json(roster_path, roster)
    with pytest.raises(OversightCalibrationAssignmentError, match="lacks three"):
        build_calibration_assignment(
            packet_root=packet_root,
            reviewer_roster_path=roster_path,
            output_root=tmp_path / "assignment",
        )
