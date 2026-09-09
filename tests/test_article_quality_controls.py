from __future__ import annotations

import pytest

from citeweave.article_expert_evaluation import ARTICLE_CONDITIONS, ClaimExpertRating
from citeweave.article_quality_controls import analyze_length_normalized_claim_quality


def _panel() -> tuple[list[ClaimExpertRating], list[dict[str, object]]]:
    claims = []
    registry = []
    for topic_index in range(8):
        topic_id = f"topic-{topic_index}"
        for condition in ARTICLE_CONDITIONS:
            article_id = f"{topic_id}-{condition}"
            registry.append(
                {
                    "topic_id": topic_id,
                    "condition": condition,
                    "article_code": article_id,
                    "word_count": 3000,
                    "duplicate_candidate_claim_rate": 0.0,
                }
            )
            for claim_index in range(20):
                good = condition == "citeweave_graph_review" or claim_index < 10
                for evaluator_index in range(2):
                    claims.append(
                        ClaimExpertRating(
                            topic_id=topic_id,
                            article_id=article_id,
                            condition=condition,
                            claim_id=f"{article_id}-claim-{claim_index}",
                            evaluator_id=f"{topic_id}-expert-{evaluator_index}",
                            stratum="results",
                            supported=good,
                            correct=good,
                            overclaim=not good,
                            evidence_sufficient=good,
                            cannot_assess=False,
                            evaluation_seconds=10.0,
                        )
                    )
    return claims, registry


def test_reports_length_normalized_claim_quality() -> None:
    claims, registry = _panel()
    result = analyze_length_normalized_claim_quality(claims, registry)
    assert result["status"] == "length_normalized_quality_descriptives_complete"
    assert len(result["articles"]) == 24
    graph = result["condition_summaries"]["citeweave_graph_review"]
    one_shot = result["condition_summaries"]["one_shot_llm"]
    assert graph["mean_correct_and_supported_claims_per_1000_words"] == pytest.approx(
        20 / 3
    )
    assert one_shot["mean_correct_and_supported_claims_per_1000_words"] == pytest.approx(
        10 / 3
    )


def test_rejects_rating_identity_not_in_registry() -> None:
    claims, registry = _panel()
    for index in (0, 1):
        row = claims[index]
        claims[index] = ClaimExpertRating(
            **{**row.__dict__, "article_id": "unknown"}
        )
    with pytest.raises(ValueError, match="unregistered article"):
        analyze_length_normalized_claim_quality(claims, registry)
