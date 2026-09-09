from __future__ import annotations

from pathlib import Path

import pytest

from src.citeweave.human_review_v2 import (
    HumanReviewValidationError,
    build_adjudication_worklist,
    validate_review_return,
)
from src.citeweave.io import write_json


def _manifest() -> dict:
    return {
        "assignments": {
            "A": {"factual": ["p1"], "semantic": []},
            "B": {"factual": ["p1"], "semantic": []},
        }
    }


def _return(reviewer: str, correct: bool) -> dict:
    return {
        "reviewer_code": reviewer,
        "results": [
            {
                "packet_id": "p1",
                "reviewer_code": reviewer,
                "answer_correct": correct,
                "evidence_sufficient": True,
                "action": "accept" if correct else "rewrite",
                "rationale": "Checked the visible evidence.",
                "review_seconds": 12.0,
            }
        ],
    }


def test_validates_time_and_builds_disagreement_worklist(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packets" / "factual"
    packet_dir.mkdir(parents=True)
    write_json(packet_dir / "p1.json", {"packet_id": "p1"})
    left = validate_review_return(
        _return("A", True), internal_manifest=_manifest(), packet_root=tmp_path
    )
    right = validate_review_return(
        _return("B", False), internal_manifest=_manifest(), packet_root=tmp_path
    )
    worklist = build_adjudication_worklist(left, right)
    assert worklist["adjudication_required"] == 1
    assert worklist["exact_agreement_rate"] == 0.0


def test_rejects_zero_second_review(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packets" / "factual"
    packet_dir.mkdir(parents=True)
    write_json(packet_dir / "p1.json", {"packet_id": "p1"})
    payload = _return("A", True)
    payload["results"][0]["review_seconds"] = 0
    with pytest.raises(HumanReviewValidationError, match="Positive review_seconds"):
        validate_review_return(payload, internal_manifest=_manifest(), packet_root=tmp_path)
