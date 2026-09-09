from __future__ import annotations

import pytest

from citeweave.review_voi import (
    replay_fixed_policy_with_propagation,
    replay_sequential_dependency_voi,
)


def _feature(packet: str, dependencies: list[str], *, causal: bool = False) -> dict:
    return {
        "packet_id": packet,
        "routing_scope_id": "article",
        "dependency_ids": dependencies,
        "severity": "critical" if causal else "medium",
        "risk_features": {
            "causal_language": causal,
            "numeric": False,
            "multiple_phenomena": causal,
            "discussion_interpretation": False,
        },
    }


def test_invalid_shared_dependency_propagates_to_unreviewed_descendant() -> None:
    features = [
        _feature("A", ["REF-X"], causal=True),
        _feature("B", ["REF-X"]),
        _feature("C", ["REF-Y"]),
    ]
    outcomes = [
        {
            "packet_id": "A",
            "action": "rewrite",
            "invalid_dependency_ids": ["REF-X"],
            "evidence_sufficient": False,
            "review_seconds": 9,
        },
        {
            "packet_id": "B",
            "action": "reject_claim",
            "invalid_dependency_ids": ["REF-X"],
            "evidence_sufficient": False,
            "review_seconds": 10,
        },
        {
            "packet_id": "C",
            "action": "accept",
            "invalid_dependency_ids": [],
            "evidence_sufficient": True,
            "review_seconds": 10,
        },
    ]
    replay = replay_sequential_dependency_voi(
        features,
        outcomes,
        estimated_review_seconds={"A": 10, "B": 10, "C": 10},
        budget_seconds=10,
    )
    assert replay["reviewed_claims"] == 1
    assert replay["propagated_claims"] == 1
    assert replay["captured_actionable_claims"] == 2
    assert replay["trace"][0]["selected_packet_id"] == "A"
    assert replay["trace"][0]["newly_propagated_claim_ids"] == ["B"]


def test_replay_rejects_invisible_invalid_dependency() -> None:
    with pytest.raises(ValueError, match="invisible dependency"):
        replay_sequential_dependency_voi(
            [_feature("A", ["REF-X"], causal=True)],
            [
                {
                    "packet_id": "A",
                    "action": "rewrite",
                    "invalid_dependency_ids": ["REF-HIDDEN"],
                    "evidence_sufficient": False,
                    "review_seconds": 9,
                }
            ],
            estimated_review_seconds={"A": 10},
            budget_seconds=10,
        )


def test_fixed_baselines_share_propagation_and_second_budget() -> None:
    features = [
        _feature("A", ["REF-X"], causal=True),
        _feature("B", ["REF-X"]),
        _feature("C", ["REF-Y"]),
    ]
    for row in features:
        row["reachable_claim_ids"] = (
            ["A", "B"] if "REF-X" in row["dependency_ids"] else ["C"]
        )
    outcomes = [
        {
            "packet_id": "A",
            "action": "rewrite",
            "invalid_dependency_ids": ["REF-X"],
            "evidence_sufficient": False,
            "review_seconds": 8,
        },
        {
            "packet_id": "B",
            "action": "reject_claim",
            "invalid_dependency_ids": ["REF-X"],
            "evidence_sufficient": False,
            "review_seconds": 8,
        },
        {
            "packet_id": "C",
            "action": "accept",
            "invalid_dependency_ids": [],
            "evidence_sufficient": True,
            "review_seconds": 8,
        },
    ]
    replay = replay_fixed_policy_with_propagation(
        features,
        outcomes,
        policy="static_dependency",
        estimated_review_seconds={"A": 10, "B": 10, "C": 10},
        budget_seconds=10,
    )
    assert replay["reviewed_claims"] == 1
    assert replay["captured_actionable_claims"] == 2
    assert replay["predicted_seconds_scheduled"] == 10
