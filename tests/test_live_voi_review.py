from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from citeweave.io import sha256_file, write_json
from citeweave.live_voi_review import create_live_voi_review_app


def _feature(packet_id: str, dependencies: list[str], severity: str = "medium") -> dict:
    return {
        "packet_id": packet_id,
        "routing_scope_id": "ARTICLE-1",
        "dependency_ids": dependencies,
        "severity": severity,
        "risk_features": {
            "causal_language": severity == "critical",
            "numeric": False,
            "multiple_phenomena": severity == "critical",
            "discussion_interpretation": severity in {"high", "critical"},
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "live"
    (root / "packets" / "article").mkdir(parents=True)
    reviewer = "REVIEWER-LIVE"
    token = "live-secret"
    packet_ids = ["A", "B", "D"]
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"article": packet_ids}},
        },
    )
    for packet_id in packet_ids:
        write_json(
            root / "packets" / "article" / f"{packet_id}.json",
            {
                "packet_id": packet_id,
                "claim": f"Claim {packet_id}",
                "allowed_decisive_evidence_ids": ["REF-X"],
            },
        )
    write_json(
        root / "live_voi_config.json",
        {
            "schema_version": 1,
            "status": "frozen_live_sequential_voi_development",
            "development_only": True,
            "formal_outcome_collection": False,
            "human_outcomes_before_freeze": 0,
            "policy": "sequential_dependency_value_of_information",
            "seed": 20260827,
            "features": [
                _feature("A", ["REF-X"], "critical"),
                _feature("B", ["REF-X"], "high"),
                _feature("D", ["REF-X"]),
            ],
            "estimated_review_seconds": {"A": 1, "B": 2, "D": 100},
            "reviewer_budget_seconds": {reviewer: 5},
        },
    )
    write_json(
        root / "live_voi_config_freeze.json",
        {"sha256": sha256_file(root / "live_voi_config.json")},
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    return root, access, token


def _answers(*, invalid: bool) -> dict:
    return {
        "factual_supported": not invalid,
        "interpretation_calibrated": not invalid,
        "alternative_adequate": True,
        "evidence_sufficient": not invalid,
        "action": "rewrite" if invalid else "accept",
        "decisive_evidence_ids": ["REF-X"],
        "invalid_dependency_ids": ["REF-X"] if invalid else [],
        "replacement": "Qualified claim." if invalid else None,
        "guard": {},
        "rationale": "Visible evidence was checked against the claim.",
    }


def test_live_voi_routes_propagates_and_reopens_with_hash_chain(tmp_path: Path) -> None:
    root, access, token = _fixture(tmp_path)
    client = TestClient(create_live_voi_review_app(root, access))
    query = "reviewer=REVIEWER-LIVE&layer=article"
    headers = {"X-Review-Token": token}

    first = client.get(f"/api/next?{query}", headers=headers).json()
    assert first["packet"]["packet_id"] == "A"
    assert first["routing"]["policy"] == "sequential_dependency_value_of_information"
    accepted = client.post(
        f"/api/submit?{query}",
        headers=headers,
        json={
            "packet_id": "A",
            "session_id": first["session_id"],
            "answers": _answers(invalid=False),
        },
    )
    assert accepted.status_code == 200

    second = client.get(f"/api/next?{query}", headers=headers).json()
    assert second["packet"]["packet_id"] == "B"
    corrected = client.post(
        f"/api/submit?{query}",
        headers=headers,
        json={
            "packet_id": "B",
            "session_id": second["session_id"],
            "answers": _answers(invalid=True),
        },
    )
    assert corrected.status_code == 200
    event = corrected.json()["result"]["routing_event"]
    assert event["newly_propagated_unreviewed_claim_ids"] == ["D"]
    assert event["newly_reopened_reviewed_claim_ids"] == ["A"]
    assert len(event["event_sha256"]) == 64

    finished = client.get(f"/api/next?{query}", headers=headers).json()
    assert finished["complete"] is True
    assert finished["completion_reason"] == "routing_complete"
    assert finished["revision_target_packet_ids"] == ["A", "B", "D"]
    progress = finished["progress"]["layers"]["article"]
    assert progress == {
        "completed": 2,
        "propagated": 1,
        "reopened": 1,
        "remaining": 0,
        "total": 3,
        "routing_status": "routing_complete",
        "predicted_seconds_scheduled": 3.0,
        "budget_seconds": 5.0,
    }


def test_live_voi_rejects_config_changed_after_freeze(tmp_path: Path) -> None:
    root, access, _ = _fixture(tmp_path)
    config = root / "live_voi_config.json"
    config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        create_live_voi_review_app(root, access)
    except ValueError as exc:
        assert "differs" in str(exc)
    else:
        raise AssertionError("Mutated live VOI config must fail closed")
