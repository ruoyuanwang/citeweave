from __future__ import annotations

import pytest

from scripts.export_adaptive_topic_counts import (
    _select_topics,
    _validate_baseline_contract,
)


def _references() -> dict:
    return {
        "references": [
            {"id": "dev-a", "role": "development"},
            {"id": "dev-b", "role": "development"},
            *[
                {"id": f"locked-{index}", "role": "locked"}
                for index in range(6)
            ],
        ]
    }


def _baseline_manifest(*, formal_results_used: bool) -> dict:
    return {
        "evaluation_target": "untouched_pre_intervention_original_candidate",
        "judge_may_modify_artifacts": False,
        "evaluation_updates_feedback_memory": False,
        "formal_results_used": formal_results_used,
    }


def test_formal_export_selects_only_six_locked_topics() -> None:
    assert _select_topics(
        _references(),
        development_calibration=False,
    ) == [f"locked-{index}" for index in range(6)]
    _validate_baseline_contract(
        _baseline_manifest(formal_results_used=True),
        development_calibration=False,
    )


def test_development_export_selects_only_two_nonformal_topics() -> None:
    assert _select_topics(
        _references(),
        development_calibration=True,
    ) == ["dev-a", "dev-b"]
    _validate_baseline_contract(
        _baseline_manifest(formal_results_used=False),
        development_calibration=True,
    )


def test_export_refuses_role_count_or_formal_flag_mismatch() -> None:
    references = _references()
    references["references"].pop()
    with pytest.raises(ValueError, match="exactly 6"):
        _select_topics(references, development_calibration=False)
    with pytest.raises(ValueError, match="violates"):
        _validate_baseline_contract(
            _baseline_manifest(formal_results_used=False),
            development_calibration=False,
        )
