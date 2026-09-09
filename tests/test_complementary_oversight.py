from __future__ import annotations

import json

from citeweave.complementary_oversight import (
    ComplementaryOversightRouter,
    OversightCase,
    ReviewerCapabilityModel,
    ReviewerObservation,
    build_adversarial_review_packet,
)


def _observations() -> list[ReviewerObservation]:
    rows = []
    for reviewer, domain, successes in (
        ("graph-expert", "graphs", 8),
        ("domain-expert", "medicine", 4),
        ("generalist", "graphs", 7),
        ("adjudicator", "graphs", 7),
    ):
        for index in range(8):
            rows.append(
                ReviewerObservation(
                    observation_id=f"{reviewer}-{domain}-{index}",
                    reviewer_id=reviewer,
                    dataset_id=f"d{index % 3}",
                    domain=domain,
                    issue_type="invalid_operator",
                    correct_after_adjudication=index < successes,
                    review_seconds=20.0 + index,
                )
            )
    return rows


def test_capability_model_uses_issue_specific_expertise() -> None:
    model = ReviewerCapabilityModel(_observations())
    graph = model.estimate(
        "graph-expert", domain="graphs", issue_type="invalid_operator"
    )
    off_domain = model.estimate(
        "domain-expert", domain="graphs", issue_type="invalid_operator"
    )
    assert graph.evidence_tier == "exact"
    assert off_domain.evidence_tier == "issue"
    assert graph.posterior_mean_accuracy > off_domain.posterior_mean_accuracy


def test_capability_model_transfers_issue_skill_across_domains() -> None:
    model = ReviewerCapabilityModel(_observations())
    transferred = model.estimate(
        "graph-expert", domain="unseen-domain", issue_type="invalid_operator"
    )
    unrelated = model.estimate(
        "graph-expert", domain="unseen-domain", issue_type="causal_overreach"
    )
    assert transferred.evidence_tier == "issue"
    assert transferred.observations == 8
    assert unrelated.evidence_tier == "global"


def test_router_selects_specialist_and_never_autoaccepts_high_risk() -> None:
    router = ComplementaryOversightRouter(
        ReviewerCapabilityModel(_observations()),
        minimum_capability_lower_95=0.45,
    )
    decision = router.route(
        OversightCase(
            "case-high",
            "heldout",
            "graphs",
            "invalid_operator",
            "high",
            0.4,
            30,
        )
    )
    assert decision["decision"]["route"] != "auto_accept"
    assert "graph-expert" in (
        decision["decision"]["reviewer_ids"]
        + [decision["decision"].get("adjudicator_id")]
    )
    assert len(decision["predecision_sha256"]) == 64


def test_critical_case_requires_independent_dual_review_and_adjudication() -> None:
    router = ComplementaryOversightRouter(
        ReviewerCapabilityModel(_observations()),
        minimum_capability_lower_95=0.45,
    )
    decision = router.route(
        OversightCase(
            "case-critical",
            "heldout",
            "graphs",
            "invalid_operator",
            "critical",
            0.8,
            30,
        )
    )["decision"]
    assert decision["route"] == "dual_review_with_adjudication"
    assert len({*decision["reviewer_ids"], decision["adjudicator_id"]}) == 3


def test_adversarial_packet_blinds_support_and_challenge_roles() -> None:
    public, internal = build_adversarial_review_packet(
        case_id="case-1",
        dataset_id="d1",
        domain="graphs",
        claim="The bridge is structurally necessary.",
        supporting_evidence=[{"evidence_id": "e-support"}],
        challenging_evidence=[{"evidence_id": "e-challenge"}],
        operator_trace=[{"operator": "delete_edge"}],
        alternative_explanations=["Threshold artifact"],
        forbidden_inferences=["causality"],
    )
    serialized = json.dumps(public, sort_keys=True)
    assert "role_map" not in serialized
    assert set(internal["role_map"].values()) == {"support", "challenge"}
    assert len(internal["public_packet_canonical_sha256"]) == 64
