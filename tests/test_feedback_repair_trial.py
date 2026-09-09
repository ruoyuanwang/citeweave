from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.feedback_repair_trial import (
    REPAIR_CONDITIONS,
    RepairTrialOutcome,
    analyze_feedback_repair_trial,
    assess_feedback_repair_trial_readiness,
    build_feedback_repair_trial_plan,
)
from citeweave.io import write_json


def _worklist(path: Path) -> Path:
    rows = []
    for topic_index in range(8):
        for case_index in range(12):
            rows.append(
                {
                    "dataset_id": f"topic-{topic_index}",
                    "paragraph_id": f"PAR-{case_index:04d}",
                    "paragraph_sha256": f"{topic_index:02x}{case_index:02x}".ljust(
                        64, "a"
                    ),
                    "original_text": "Original claim PH-1 with REF-1.",
                    "allowed_evidence_ids": ["PH-1", "REF-1"],
                    "directives": [
                        {
                            "packet_id": f"AR-{topic_index}-{case_index}",
                            "rationale": "The claim overstates the evidence.",
                            "action": "rewrite",
                            "replacement": "Calibrated claim PH-1 with REF-1.",
                            "invalid_dependency_ids": [],
                            "guard": {"scope": "corpus"},
                        }
                    ],
                }
            )
    write_json(
        path,
        {
            "status": "controlled_paragraph_revision_ready",
            "worklist": rows,
        },
    )
    return path


def test_builds_same_feedback_two_condition_plan(tmp_path: Path) -> None:
    worklist = _worklist(tmp_path / "worklist.json")
    readiness = assess_feedback_repair_trial_readiness(worklist)
    assert readiness["status"] == "ready"
    output = tmp_path / "plan.json"
    plan = build_feedback_repair_trial_plan(worklist, output_path=output)
    assert plan["cases"] == 96
    assert plan["logical_cells"] == 192
    by_case = {}
    for row in plan["cells"]:
        by_case.setdefault(row["case_id"], []).append(row)
    assert all(
        {row["condition"] for row in rows} == set(REPAIR_CONDITIONS)
        and len({row["feedback_content_sha256"] for row in rows}) == 1
        for rows in by_case.values()
    )
    assert output.with_suffix(".freeze.json").is_file()


def test_missing_real_worklist_is_truthfully_blocked(tmp_path: Path) -> None:
    readiness = assess_feedback_repair_trial_readiness(tmp_path / "missing.json")
    assert readiness["status"] == "blocked"
    assert "real-feedback" in readiness["gaps"][0]


def _outcomes() -> list[RepairTrialOutcome]:
    rows = []
    for topic_index in range(8):
        for case_index in range(12):
            case_id = f"case-{topic_index}-{case_index}"
            for condition in REPAIR_CONDITIONS:
                compiled = condition == "dependency_compiled"
                rows.append(
                    RepairTrialOutcome(
                        case_id=case_id,
                        topic_id=f"topic-{topic_index}",
                        condition=condition,
                        primary_reviewer_ids=("E1", "E2"),
                        resolution_method="agreement",
                        adjudicator_id=None,
                        targeted_defects_corrected=compiled,
                        unaffected_claims_preserved=compiled,
                        new_unsupported_claims=0 if compiled else 1,
                        domain_specificity_regression=not compiled,
                        revision_seconds=8.0 if compiled else 12.0,
                        input_tokens=800 if compiled else 900,
                        output_tokens=100 if compiled else 140,
                        evaluation_seconds=20.0,
                    )
                )
    return rows


def test_analysis_uses_eight_topic_clusters_and_joint_gate() -> None:
    result = analyze_feedback_repair_trial(_outcomes())
    assert result["topics"] == 8
    assert result["cases"] == 96
    assert result["joint_mechanism_success"]
    for contrast in result["registered_contrasts"].values():
        assert contrast["topics"] == 8
        assert contrast["multiplicity"]["reject_at_0_05"]


def test_analysis_rejects_nonindependent_adjudicator() -> None:
    rows = _outcomes()
    first = rows[0]
    rows[0] = RepairTrialOutcome(
        **{
            **first.__dict__,
            "resolution_method": "adjudication",
            "adjudicator_id": "E1",
        }
    )
    with pytest.raises(ValueError, match="independent third reviewer"):
        analyze_feedback_repair_trial(rows)
