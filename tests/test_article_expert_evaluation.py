from __future__ import annotations

import pytest

from citeweave.article_expert_evaluation import (
    ARTICLE_CONDITIONS,
    ClaimExpertRating,
    HolisticExpertRating,
    PairwiseExpertPreference,
    analyze_article_expert_panel,
)


def _valid_panel() -> tuple[
    list[HolisticExpertRating],
    list[ClaimExpertRating],
    list[PairwiseExpertPreference],
]:
    holistic = []
    claims = []
    preferences = []
    strata = ["results", "discussion", "graph_derived", "numerical", "causal_risk"]
    for topic_index in range(8):
        topic = f"topic-{topic_index}"
        for condition_index, condition in enumerate(ARTICLE_CONDITIONS):
            article = f"{topic}-{condition}"
            for evaluator_index in range(3):
                score = 5 if condition == "citeweave_graph_review" else 3
                holistic.append(
                    HolisticExpertRating(
                        topic_id=topic,
                        article_id=article,
                        condition=condition,
                        evaluator_id=f"{topic}-expert-{evaluator_index}",
                        evaluator_role="domain_expert" if evaluator_index < 2 else "methods_expert",
                        factual_accuracy=score,
                        evidence_traceability=score,
                        phenomenon_depth=score,
                        alternative_explanations=score,
                        epistemic_calibration=score,
                        domain_specificity=score,
                        argumentative_coherence=score,
                        research_utility=score,
                        evaluation_seconds=120.0,
                    )
                )
            for claim_index in range(20):
                for evaluator_index in range(2):
                    good = condition == "citeweave_graph_review"
                    claims.append(
                        ClaimExpertRating(
                            topic_id=topic,
                            article_id=article,
                            condition=condition,
                            claim_id=f"{article}-claim-{claim_index}",
                            evaluator_id=f"{topic}-expert-{evaluator_index}",
                            stratum=strata[claim_index % len(strata)],
                            supported=good,
                            correct=good,
                            overclaim=not good,
                            evidence_sufficient=good,
                            cannot_assess=False,
                            evaluation_seconds=25.0,
                        )
                    )
        for left_index, left in enumerate(ARTICLE_CONDITIONS):
            for right in ARTICLE_CONDITIONS[left_index + 1 :]:
                for evaluator_index in range(2):
                    preferences.append(
                        PairwiseExpertPreference(
                            topic_id=topic,
                            evaluator_id=f"{topic}-expert-{evaluator_index}",
                            left_condition=left,
                            right_condition=right,
                            preferred_condition=(
                                left if left == "citeweave_graph_review" else right
                            ),
                            evaluation_seconds=15.0,
                        )
                    )
    return holistic, claims, preferences


def test_analyze_complete_expert_panel() -> None:
    holistic, claims, preferences = _valid_panel()
    result = analyze_article_expert_panel(
        holistic,
        claims,
        preferences,
        bootstrap_samples=200,
    )
    assert result["status"] == "confirmatory_exact_cluster_analysis_complete"
    assert result["topics"] == 8
    assert result["articles"] == 24
    assert result["registered_contrasts"][
        "A1_claim_correct_and_supported_graph_vs_one_shot"
    ]["estimate"] == 1.0
    assert result["registered_contrasts"][
        "A1_claim_correct_and_supported_graph_vs_one_shot"
    ]["multiplicity"]["reject_at_0_05"]
    assert result["success_guard_diagnostics"]["machine_exceeds_human_claim_permitted"]


def test_rejects_claims_without_adjudication() -> None:
    holistic, claims, preferences = _valid_panel()
    first = claims[0]
    claims[0] = ClaimExpertRating(
        **{
            **first.__dict__,
            "supported": not first.supported,
        }
    )
    with pytest.raises(ValueError, match="lacks third-person adjudication"):
        analyze_article_expert_panel(
            holistic, claims, preferences, bootstrap_samples=20
        )


def test_disputed_claim_uses_one_independent_adjudicator() -> None:
    holistic, claims, preferences = _valid_panel()
    first = claims[0]
    claims[0] = ClaimExpertRating(**{**first.__dict__, "supported": False})
    claims.append(
        ClaimExpertRating(
            **{
                **first.__dict__,
                "evaluator_id": f"{first.topic_id}-expert-adjudicator",
                "adjudication": True,
            }
        )
    )
    result = analyze_article_expert_panel(
        holistic, claims, preferences, bootstrap_samples=20
    )
    assert result["registered_contrasts"][
        "A1_claim_correct_and_supported_graph_vs_one_shot"
    ]["estimate"] == 1.0


def test_rejects_unnecessary_adjudication() -> None:
    holistic, claims, preferences = _valid_panel()
    first = claims[0]
    claims.append(
        ClaimExpertRating(
            **{
                **first.__dict__,
                "evaluator_id": f"{first.topic_id}-expert-adjudicator",
                "adjudication": True,
            }
        )
    )
    with pytest.raises(ValueError, match="must not receive post-hoc adjudication"):
        analyze_article_expert_panel(
            holistic, claims, preferences, bootstrap_samples=20
        )


def test_rejects_cannot_assess_with_substantive_labels() -> None:
    holistic, claims, preferences = _valid_panel()
    first = claims[0]
    claims[0] = ClaimExpertRating(**{**first.__dict__, "cannot_assess": True})
    with pytest.raises(ValueError, match="leave substantive labels null"):
        analyze_article_expert_panel(
            holistic, claims, preferences, bootstrap_samples=20
        )
