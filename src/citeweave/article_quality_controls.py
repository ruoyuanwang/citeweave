from __future__ import annotations

from collections import defaultdict
from typing import Any

from .article_expert_evaluation import ARTICLE_CONDITIONS, ClaimExpertRating


def _resolve_claims(claims: list[ClaimExpertRating]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[ClaimExpertRating]] = defaultdict(list)
    for row in claims:
        grouped[(row.topic_id, row.article_id, row.claim_id)].append(row)

    outcomes = []
    for (topic_id, article_id, claim_id), rows in sorted(grouped.items()):
        primary = [row for row in rows if not row.adjudication]
        if len(primary) != 2 or len({row.evaluator_id for row in primary}) != 2:
            raise ValueError(
                f"Claim needs two independent primary ratings: {topic_id}/{claim_id}"
            )
        signatures = {
            (
                row.cannot_assess,
                row.supported,
                row.correct,
                row.overclaim,
                row.evidence_sufficient,
            )
            for row in primary
        }
        adjudicators = [row for row in rows if row.adjudication]
        if len(signatures) > 1:
            if len(adjudicators) != 1:
                raise ValueError(
                    f"Disputed claim requires one adjudicator: {topic_id}/{claim_id}"
                )
            resolved = adjudicators[0]
        else:
            if adjudicators:
                raise ValueError(
                    f"Undisputed claim must not be adjudicated: {topic_id}/{claim_id}"
                )
            resolved = primary[0]
        outcomes.append(
            {
                "topic_id": topic_id,
                "article_id": article_id,
                "claim_id": claim_id,
                "condition": resolved.condition,
                "cannot_assess": resolved.cannot_assess,
                "correct_and_supported": bool(
                    not resolved.cannot_assess
                    and resolved.correct
                    and resolved.supported
                ),
                "overclaim": bool(
                    not resolved.cannot_assess and resolved.overclaim
                ),
            }
        )
    return outcomes


def analyze_length_normalized_claim_quality(
    claims: list[ClaimExpertRating],
    article_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    registry = {}
    for row in article_registry:
        article_id = str(row["article_code"])
        if article_id in registry:
            raise ValueError(f"Duplicate article registry entry: {article_id}")
        registry[article_id] = row
    if len(registry) != 24:
        raise ValueError("Quality-control registry requires exactly 24 articles")
    expected = {
        (str(row["topic_id"]), str(row["condition"])) for row in registry.values()
    }
    topics = {topic for topic, _ in expected}
    if len(topics) != 8 or expected != {
        (topic, condition) for topic in topics for condition in ARTICLE_CONDITIONS
    }:
        raise ValueError("Quality-control registry requires exactly 8 topics × 3 conditions")

    resolved = _resolve_claims(claims)
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        article_id = row["article_id"]
        if article_id not in registry:
            raise ValueError(f"Rating refers to an unregistered article: {article_id}")
        registered = registry[article_id]
        if (
            row["topic_id"] != registered["topic_id"]
            or row["condition"] != registered["condition"]
        ):
            raise ValueError(f"Rating identity mismatch: {article_id}")
        by_article[article_id].append(row)
    if set(by_article) != set(registry):
        raise ValueError("Every registered article must have resolved claim ratings")

    article_rows = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article_id, registered in sorted(registry.items()):
        rows = by_article[article_id]
        if len(rows) != 20:
            raise ValueError(f"Article must have exactly 20 resolved claims: {article_id}")
        word_count = int(registered["word_count"])
        if word_count <= 0:
            raise ValueError(f"Invalid article word count: {article_id}")
        scale = 1000.0 / word_count
        correct_supported = sum(row["correct_and_supported"] for row in rows)
        overclaims = sum(row["overclaim"] for row in rows)
        cannot_assess = sum(row["cannot_assess"] for row in rows)
        summary = {
            "topic_id": registered["topic_id"],
            "condition": registered["condition"],
            "article_code": article_id,
            "word_count": word_count,
            "resolved_sampled_claims": len(rows),
            "correct_and_supported_claims": correct_supported,
            "overclaims": overclaims,
            "cannot_assess_claims": cannot_assess,
            "correct_and_supported_claims_per_1000_words": correct_supported * scale,
            "overclaims_per_1000_words": overclaims * scale,
            "cannot_assess_claims_per_1000_words": cannot_assess * scale,
            "duplicate_candidate_claim_rate": registered.get(
                "duplicate_candidate_claim_rate"
            ),
        }
        article_rows.append(summary)
        by_condition[str(registered["condition"])].append(summary)

    condition_summaries = {}
    for condition in ARTICLE_CONDITIONS:
        rows = by_condition[condition]
        if len(rows) != 8:
            raise AssertionError(f"Condition lacks eight article clusters: {condition}")
        condition_summaries[condition] = {
            "articles": len(rows),
            "mean_word_count": sum(row["word_count"] for row in rows) / len(rows),
            "mean_correct_and_supported_claims_per_1000_words": sum(
                row["correct_and_supported_claims_per_1000_words"] for row in rows
            )
            / len(rows),
            "mean_overclaims_per_1000_words": sum(
                row["overclaims_per_1000_words"] for row in rows
            )
            / len(rows),
            "mean_cannot_assess_claims_per_1000_words": sum(
                row["cannot_assess_claims_per_1000_words"] for row in rows
            )
            / len(rows),
        }

    return {
        "schema_version": 1,
        "status": "length_normalized_quality_descriptives_complete",
        "articles": article_rows,
        "condition_summaries": condition_summaries,
        "interpretation_guard": (
            "These word-normalized outcomes are descriptive sensitivity diagnostics. "
            "They do not replace the frozen topic-cluster A1/A2/A3 tests and cannot "
            "create an unregistered superiority claim."
        ),
    }
