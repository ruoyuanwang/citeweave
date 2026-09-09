from __future__ import annotations

from collections import Counter
from typing import Any

from .article_claim_review import _claim_candidates, select_article_review_claims


def audit_article_oversight_readiness(
    article: str,
    *,
    phenomenon_ids: list[str],
    required_claims: int = 20,
    minimum_per_phenomenon: int = 2,
) -> dict[str, Any]:
    candidates = _claim_candidates(article)
    by_section = Counter(row["section"].casefold() for row in candidates)
    by_phenomenon = {
        phenomenon_id: sum(
            phenomenon_id in row["phenomenon_ids"] for row in candidates
        )
        for phenomenon_id in phenomenon_ids
    }
    unique_paragraphs = len({row["paragraph_id"] for row in candidates})
    blockers = []
    if len(candidates) < required_claims:
        blockers.append(
            f"reviewable_claim_candidates_below_{required_claims}:{len(candidates)}"
        )
    undercovered = sorted(
        phenomenon_id
        for phenomenon_id, count in by_phenomenon.items()
        if count < minimum_per_phenomenon
    )
    if undercovered:
        blockers.append("phenomenon_claim_coverage_incomplete:" + ",".join(undercovered))
    if by_section.get("results", 0) < 8:
        blockers.append("results_claim_candidates_below_8")
    if by_section.get("discussion", 0) < 6:
        blockers.append("discussion_claim_candidates_below_6")
    selection = []
    if not blockers:
        selection = select_article_review_claims(
            article,
            phenomenon_ids=phenomenon_ids,
            claims=required_claims,
            minimum_per_phenomenon=minimum_per_phenomenon,
        )
    return {
        "schema_version": 1,
        "status": "ready_for_claim_review_packetization"
        if not blockers
        else "blocked_before_claim_review_packetization",
        "blocking_reasons": blockers,
        "required_claims": required_claims,
        "candidate_claims": len(candidates),
        "candidate_claims_by_section": dict(sorted(by_section.items())),
        "candidate_claims_by_phenomenon": by_phenomenon,
        "candidate_paragraphs": unique_paragraphs,
        "multiple_phenomenon_candidates": sum(
            row["risk_features"]["multiple_phenomena"] for row in candidates
        ),
        "discussion_interpretation_candidates": sum(
            row["risk_features"]["discussion_interpretation"] for row in candidates
        ),
        "selected_claims": len(selection),
    }
