from __future__ import annotations

from citeweave.article_oversight_readiness import audit_article_oversight_readiness


def _article(claims: int) -> str:
    phenomena = [f"PH-{index}" for index in range(5)]
    results = []
    discussion = []
    for index in range(claims):
        token = phenomena[index % len(phenomena)]
        sentence = f"Supported claim {index} uses {token} and REF-{index}."
        (results if index % 2 == 0 else discussion).append(sentence)
    return (
        "## Results\n\n"
        + "\n\n".join(results)
        + "\n\n## Discussion\n\n"
        + "\n\n".join(discussion)
    )


def test_readiness_blocks_fewer_than_twenty_claims() -> None:
    report = audit_article_oversight_readiness(
        _article(15), phenomenon_ids=[f"PH-{index}" for index in range(5)]
    )
    assert report["status"] == "blocked_before_claim_review_packetization"
    assert report["candidate_claims"] == 15
    assert "reviewable_claim_candidates_below_20:15" in report["blocking_reasons"]


def test_readiness_selects_twenty_distributed_claims() -> None:
    report = audit_article_oversight_readiness(
        _article(22), phenomenon_ids=[f"PH-{index}" for index in range(5)]
    )
    assert report["status"] == "ready_for_claim_review_packetization"
    assert report["selected_claims"] == 20
    assert report["candidate_claims_by_section"]["results"] == 11
    assert report["candidate_claims_by_section"]["discussion"] == 11
