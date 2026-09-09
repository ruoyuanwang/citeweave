from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

ARTICLE_CONDITIONS = (
    "citeweave_graph_review",
    "one_shot_llm",
    "human_same_evidence",
)
HOLISTIC_DIMENSIONS = (
    "factual_accuracy",
    "evidence_traceability",
    "phenomenon_depth",
    "alternative_explanations",
    "epistemic_calibration",
    "domain_specificity",
    "argumentative_coherence",
    "research_utility",
)


@dataclass(frozen=True)
class HolisticExpertRating:
    topic_id: str
    article_id: str
    condition: Literal[
        "citeweave_graph_review",
        "one_shot_llm",
        "human_same_evidence",
    ]
    evaluator_id: str
    evaluator_role: Literal["domain_expert", "methods_expert"]
    factual_accuracy: int
    evidence_traceability: int
    phenomenon_depth: int
    alternative_explanations: int
    epistemic_calibration: int
    domain_specificity: int
    argumentative_coherence: int
    research_utility: int
    evaluation_seconds: float


@dataclass(frozen=True)
class ClaimExpertRating:
    topic_id: str
    article_id: str
    condition: Literal[
        "citeweave_graph_review",
        "one_shot_llm",
        "human_same_evidence",
    ]
    claim_id: str
    evaluator_id: str
    stratum: Literal[
        "results",
        "discussion",
        "graph_derived",
        "numerical",
        "causal_risk",
    ]
    supported: bool | None
    correct: bool | None
    overclaim: bool | None
    evidence_sufficient: bool | None
    cannot_assess: bool
    evaluation_seconds: float
    adjudication: bool = False


@dataclass(frozen=True)
class PairwiseExpertPreference:
    topic_id: str
    evaluator_id: str
    left_condition: Literal[
        "citeweave_graph_review",
        "one_shot_llm",
        "human_same_evidence",
    ]
    right_condition: Literal[
        "citeweave_graph_review",
        "one_shot_llm",
        "human_same_evidence",
    ]
    preferred_condition: Literal[
        "citeweave_graph_review",
        "one_shot_llm",
        "human_same_evidence",
    ]
    evaluation_seconds: float


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = round(probability * (len(sorted_values) - 1))
    return sorted_values[index]


def _topic_cluster_contrast(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    topics = sorted(set(left) & set(right))
    if not topics:
        raise ValueError("No paired topics available for registered contrast")
    differences = {
        topic: float(_mean(left[topic])) - float(_mean(right[topic]))
        for topic in topics
    }
    generator = Random(seed)
    draws = []
    for _ in range(samples):
        selected = generator.choices(topics, k=len(topics))
        draws.append(sum(differences[topic] for topic in selected) / len(selected))
    draws.sort()
    return {
        "topics": len(topics),
        "estimate": sum(differences.values()) / len(differences),
        "ci_low": _percentile(draws, 0.025),
        "ci_high": _percentile(draws, 0.975),
        "topic_differences": differences,
    }


def _exact_signflip(values: list[float], *, null_offset: float = 0.0) -> dict[str, Any]:
    """One-sided exact paired test of mean(values) > null_offset."""
    shifted = [value - null_offset for value in values]
    observed = sum(shifted) / len(shifted)
    draws = [
        sum(sign * value for sign, value in zip(signs, shifted, strict=True))
        / len(shifted)
        for signs in itertools.product((-1.0, 1.0), repeat=len(shifted))
    ]
    return {
        "estimate": sum(values) / len(values),
        "null_offset": null_offset,
        "clusters": len(values),
        "cluster_effects": values,
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in draws)
        / len(draws),
        "minimum_attainable_p": 1.0 / len(draws),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, key in enumerate(ordered):
        running = max(
            running,
            min(1.0, (len(ordered) - index) * p_values[key]),
        )
        adjusted[key] = running
    return {
        key: {
            "raw_p_value_one_sided": p_values[key],
            "holm_adjusted_p_value": adjusted[key],
            "reject_at_0_05": adjusted[key] < 0.05,
        }
        for key in p_values
    }


def _paired_holistic_topic_effects(
    ratings: list[HolisticExpertRating],
    *,
    dimension: str,
    left_condition: str,
    right_condition: str,
) -> dict[str, float]:
    by_topic_evaluator: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in ratings:
        by_topic_evaluator[row.topic_id][row.evaluator_id][row.condition] = float(
            getattr(row, dimension)
        )
    effects = {}
    for topic, evaluator_rows in sorted(by_topic_evaluator.items()):
        paired = [
            conditions[left_condition] - conditions[right_condition]
            for conditions in evaluator_rows.values()
            if left_condition in conditions and right_condition in conditions
        ]
        if not paired:
            raise ValueError(
                f"No paired evaluator for {left_condition}/{right_condition} in {topic}"
            )
        effects[topic] = sum(paired) / len(paired)
    return effects


def _resolved_claim_outcomes(
    claims: list[ClaimExpertRating],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[ClaimExpertRating]] = defaultdict(list)
    for row in claims:
        grouped[(row.topic_id, row.article_id, row.claim_id)].append(row)
    outcomes = []
    for (topic_id, article_id, claim_id), rows in sorted(grouped.items()):
        primary = [row for row in rows if not row.adjudication]
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
        resolved = adjudicators[0] if len(signatures) > 1 else primary[0]
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
                "resolved_by_adjudication": len(signatures) > 1,
            }
        )
    return outcomes


def _krippendorff_alpha(
    ratings_by_unit: dict[str, list[int | bool]], *, ordinal: bool
) -> float | None:
    units = [ratings for ratings in ratings_by_unit.values() if len(ratings) >= 2]
    if not units:
        return None

    def distance(left: int | bool, right: int | bool) -> float:
        if ordinal:
            return float(int(left) - int(right)) ** 2
        return float(left != right)

    observed_sum = 0.0
    observed_pairs = 0
    pooled: list[int | bool] = []
    for ratings in units:
        pooled.extend(ratings)
        for index, left in enumerate(ratings):
            for right in ratings[index + 1 :]:
                observed_sum += distance(left, right)
                observed_pairs += 1
    if observed_pairs == 0 or len(pooled) < 2:
        return None
    expected_sum = 0.0
    expected_pairs = 0
    for index, left in enumerate(pooled):
        for right in pooled[index + 1 :]:
            expected_sum += distance(left, right)
            expected_pairs += 1
    expected = expected_sum / expected_pairs
    if expected == 0:
        return 1.0 if observed_sum == 0 else None
    return 1.0 - (observed_sum / observed_pairs) / expected


def _validate_panel(
    holistic: list[HolisticExpertRating],
    claims: list[ClaimExpertRating],
    preferences: list[PairwiseExpertPreference],
) -> tuple[list[str], dict[tuple[str, str], str]]:
    if not holistic or not claims or not preferences:
        raise ValueError("Holistic, claim-level, and forced-pairwise ratings are required")
    if len({
        (row.topic_id, row.article_id, row.evaluator_id) for row in holistic
    }) != len(holistic):
        raise ValueError("Duplicate holistic article-evaluator rating")
    if len({
        (row.topic_id, row.article_id, row.claim_id, row.evaluator_id, row.adjudication)
        for row in claims
    }) != len(claims):
        raise ValueError("Duplicate claim-evaluator rating")

    article_condition: dict[tuple[str, str], str] = {}
    for row in holistic:
        key = (row.topic_id, row.article_id)
        if key in article_condition and article_condition[key] != row.condition:
            raise ValueError(f"Article changes condition across ratings: {row.article_id}")
        article_condition[key] = row.condition
        for dimension in HOLISTIC_DIMENSIONS:
            score = getattr(row, dimension)
            if not 1 <= score <= 5:
                raise ValueError(f"Holistic score outside 1..5: {dimension}")
        if row.evaluation_seconds <= 0:
            raise ValueError("Holistic ratings require positive server-timed seconds")

    topics = sorted({row.topic_id for row in holistic})
    if len(topics) < 8:
        raise ValueError("Confirmatory panel requires at least eight independent topics")
    for topic in topics:
        by_condition: dict[str, set[str]] = defaultdict(set)
        domain_evaluators = set()
        evaluator_conditions: dict[str, set[str]] = defaultdict(set)
        for row in holistic:
            if row.topic_id != topic:
                continue
            by_condition[row.condition].add(row.article_id)
            evaluator_conditions[row.evaluator_id].add(row.condition)
            if row.evaluator_role == "domain_expert":
                domain_evaluators.add(row.evaluator_id)
        if set(by_condition) != set(ARTICLE_CONDITIONS):
            raise ValueError(f"Topic lacks exactly the three registered conditions: {topic}")
        if any(len(article_ids) != 1 for article_ids in by_condition.values()):
            raise ValueError(f"Topic-condition must identify exactly one article: {topic}")
        if len(domain_evaluators) < 2:
            raise ValueError(f"Topic needs at least two domain experts: {topic}")
        article_rows = [row for row in holistic if row.topic_id == topic]
        counts: dict[str, set[str]] = defaultdict(set)
        for row in article_rows:
            counts[row.article_id].add(row.evaluator_id)
        if any(len(evaluators) < 3 for evaluators in counts.values()):
            raise ValueError(f"Every article needs at least three holistic raters: {topic}")
        for left_index, left in enumerate(ARTICLE_CONDITIONS):
            for right in ARTICLE_CONDITIONS[left_index + 1 :]:
                if not any(
                    {left, right}.issubset(conditions)
                    for conditions in evaluator_conditions.values()
                ):
                    raise ValueError(
                        f"No evaluator co-rated condition pair {left}/{right} in {topic}"
                    )

    claims_by_article: dict[tuple[str, str], dict[str, list[ClaimExpertRating]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in claims:
        key = (row.topic_id, row.article_id)
        if key not in article_condition:
            raise ValueError(f"Claim rating references an unknown article: {row.article_id}")
        if row.condition != article_condition[key]:
            raise ValueError(f"Claim condition mismatch: {row.claim_id}")
        if row.evaluation_seconds <= 0:
            raise ValueError("Claim ratings require positive server-timed seconds")
        if row.cannot_assess:
            if any(
                value is not None
                for value in (
                    row.supported,
                    row.correct,
                    row.overclaim,
                    row.evidence_sufficient,
                )
            ):
                raise ValueError("cannot_assess rows must leave substantive labels null")
        elif any(
            value is None
            for value in (
                row.supported,
                row.correct,
                row.overclaim,
                row.evidence_sufficient,
            )
        ):
            raise ValueError("Assessable claim rows require all four substantive labels")
        claims_by_article[key][row.claim_id].append(row)

    if set(claims_by_article) != set(article_condition):
        raise ValueError("Every registered article must have claim-level ratings")
    required_strata = {
        "results",
        "discussion",
        "graph_derived",
        "numerical",
        "causal_risk",
    }
    for key, claim_groups in claims_by_article.items():
        if len(claim_groups) != 20:
            raise ValueError(f"Article must have exactly 20 sampled claims: {key}")
        observed_strata = set()
        for claim_id, rows in claim_groups.items():
            strata = {row.stratum for row in rows}
            if len(strata) != 1:
                raise ValueError(f"Claim changes stratum across raters: {claim_id}")
            observed_strata.update(strata)
            primary_raters = {row.evaluator_id for row in rows if not row.adjudication}
            if len(primary_raters) < 2:
                raise ValueError(f"Claim needs two independent primary ratings: {claim_id}")
            primary = [row for row in rows if not row.adjudication]
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
            if len(signatures) > 1 and not adjudicators:
                raise ValueError(f"Disputed claim lacks third-person adjudication: {claim_id}")
            if len(signatures) > 1 and len(adjudicators) != 1:
                raise ValueError(
                    f"Disputed claim requires exactly one adjudicator: {claim_id}"
                )
            if len(signatures) == 1 and adjudicators:
                raise ValueError(
                    f"Undisputed claim must not receive post-hoc adjudication: {claim_id}"
                )
            if len(signatures) > 1 and any(
                row.evaluator_id in primary_raters for row in adjudicators
            ):
                raise ValueError(f"Claim adjudicator must be a third person: {claim_id}")
        if observed_strata != required_strata:
            raise ValueError(f"Claim sample lacks a registered stratum: {key}")

    preference_keys = set()
    preferences_by_topic_pair: dict[tuple[str, frozenset[str]], set[str]] = defaultdict(set)
    for row in preferences:
        if row.topic_id not in topics:
            raise ValueError(f"Pairwise preference references unknown topic: {row.topic_id}")
        pair = frozenset((row.left_condition, row.right_condition))
        if len(pair) != 2:
            raise ValueError("Pairwise preference must compare two distinct conditions")
        if row.preferred_condition not in pair:
            raise ValueError("Forced preference must select one of the displayed conditions")
        if row.evaluation_seconds <= 0:
            raise ValueError("Pairwise preferences require positive server-timed seconds")
        key = (row.topic_id, row.evaluator_id, pair)
        if key in preference_keys:
            raise ValueError("Duplicate evaluator-topic-condition-pair preference")
        preference_keys.add(key)
        preferences_by_topic_pair[(row.topic_id, pair)].add(row.evaluator_id)
    for topic in topics:
        for left_index, left in enumerate(ARTICLE_CONDITIONS):
            for right in ARTICLE_CONDITIONS[left_index + 1 :]:
                evaluators = preferences_by_topic_pair.get(
                    (topic, frozenset((left, right))), set()
                )
                if len(evaluators) < 2:
                    raise ValueError(
                        f"Condition pair needs two independent preferences in {topic}: "
                        f"{left}/{right}"
                    )
    return topics, article_condition


def analyze_article_expert_panel(
    holistic: list[HolisticExpertRating],
    claims: list[ClaimExpertRating],
    preferences: list[PairwiseExpertPreference],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260824,
) -> dict[str, Any]:
    topics, article_condition = _validate_panel(holistic, claims, preferences)

    holistic_summary: dict[str, Any] = {}
    dimension_topic_values: dict[str, dict[str, dict[str, list[float]]]] = {
        dimension: {condition: defaultdict(list) for condition in ARTICLE_CONDITIONS}
        for dimension in HOLISTIC_DIMENSIONS
    }
    for condition in ARTICLE_CONDITIONS:
        rows = [row for row in holistic if row.condition == condition]
        holistic_summary[condition] = {
            "ratings": len(rows),
            "articles": len({row.article_id for row in rows}),
            "evaluators": len({row.evaluator_id for row in rows}),
            "evaluation_seconds": sum(row.evaluation_seconds for row in rows),
            "dimension_means": {
                dimension: _mean([float(getattr(row, dimension)) for row in rows])
                for dimension in HOLISTIC_DIMENSIONS
            },
        }
        for row in rows:
            for dimension in HOLISTIC_DIMENSIONS:
                dimension_topic_values[dimension][condition][row.topic_id].append(
                    float(getattr(row, dimension))
                )

    primary_claim_rows = [row for row in claims if not row.adjudication]
    claim_summary: dict[str, Any] = {}
    claim_topic_values: dict[str, dict[str, list[float]]] = {
        condition: defaultdict(list) for condition in ARTICLE_CONDITIONS
    }
    for condition in ARTICLE_CONDITIONS:
        rows = [row for row in primary_claim_rows if row.condition == condition]
        assessable = [row for row in rows if not row.cannot_assess]
        joint = [bool(row.correct and row.supported) for row in assessable]
        claim_summary[condition] = {
            "ratings": len(rows),
            "claims": len({(row.topic_id, row.article_id, row.claim_id) for row in rows}),
            "cannot_assess_rate": sum(row.cannot_assess for row in rows) / len(rows),
            "correct_and_supported_rate_assessable": _mean([float(value) for value in joint]),
            "correct_and_supported_rate_strict": sum(joint) / len(rows),
            "overclaim_rate_assessable": _mean(
                [float(bool(row.overclaim)) for row in assessable]
            ),
            "evidence_sufficient_rate_assessable": _mean(
                [float(bool(row.evidence_sufficient)) for row in assessable]
            ),
            "evaluation_seconds": sum(row.evaluation_seconds for row in rows),
        }
        by_topic: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_topic[row.topic_id].append(
                float(not row.cannot_assess and bool(row.correct and row.supported))
            )
        claim_topic_values[condition] = by_topic

    resolved_claims = _resolved_claim_outcomes(claims)
    resolved_by_topic_condition: dict[str, dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in resolved_claims:
        resolved_by_topic_condition[row["topic_id"]][row["condition"]].append(row)
    a1_effects = {}
    a1_worst_case_effects = {}
    a1_best_case_effects = {}
    for topic in topics:
        condition_rates = {}
        optimistic_rates = {}
        for condition in ("citeweave_graph_review", "one_shot_llm"):
            rows = resolved_by_topic_condition[topic][condition]
            if len(rows) != 20:
                raise ValueError(
                    f"Resolved claim panel must contain 20 claims for {topic}/{condition}"
                )
            condition_rates[condition] = sum(
                row["correct_and_supported"] for row in rows
            ) / len(rows)
            optimistic_rates[condition] = sum(
                row["correct_and_supported"] or row["cannot_assess"] for row in rows
            ) / len(rows)
        a1_effects[topic] = (
            condition_rates["citeweave_graph_review"]
            - condition_rates["one_shot_llm"]
        )
        a1_worst_case_effects[topic] = (
            condition_rates["citeweave_graph_review"]
            - optimistic_rates["one_shot_llm"]
        )
        a1_best_case_effects[topic] = (
            optimistic_rates["citeweave_graph_review"]
            - condition_rates["one_shot_llm"]
        )
    a1_exact = _exact_signflip(list(a1_effects.values()))

    research_utility_effects = _paired_holistic_topic_effects(
        holistic,
        dimension="research_utility",
        left_condition="citeweave_graph_review",
        right_condition="human_same_evidence",
    )
    evidence_traceability_effects = _paired_holistic_topic_effects(
        holistic,
        dimension="evidence_traceability",
        left_condition="citeweave_graph_review",
        right_condition="human_same_evidence",
    )
    domain_specificity_effects = _paired_holistic_topic_effects(
        holistic,
        dimension="domain_specificity",
        left_condition="citeweave_graph_review",
        right_condition="human_same_evidence",
    )
    a2_exact = _exact_signflip(
        list(research_utility_effects.values()), null_offset=-0.35
    )
    a3_exact = _exact_signflip(list(evidence_traceability_effects.values()))
    domain_guard = _exact_signflip(
        list(domain_specificity_effects.values()), null_offset=-0.35
    )
    multiplicity = _holm_adjust(
        {
            "A1": a1_exact["p_value_one_sided"],
            "A2": a2_exact["p_value_one_sided"],
            "A3": a3_exact["p_value_one_sided"],
        }
    )

    a1 = {
        **a1_exact,
        "topic_effects": a1_effects,
        "estimand": "adjudicated strict correct-and-supported claim rate",
        "multiplicity": multiplicity["A1"],
        "cannot_assess_sensitivity": {
            "worst_case_graph_effects": a1_worst_case_effects,
            "best_case_graph_effects": a1_best_case_effects,
            "worst_case_estimate": sum(a1_worst_case_effects.values())
            / len(a1_worst_case_effects),
            "best_case_estimate": sum(a1_best_case_effects.values())
            / len(a1_best_case_effects),
        },
        "descriptive_cluster_bootstrap": _topic_cluster_contrast(
            claim_topic_values["citeweave_graph_review"],
            claim_topic_values["one_shot_llm"],
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
    }
    a2 = {
        **a2_exact,
        "topic_effects": research_utility_effects,
        "noninferiority_margin": 0.35,
        "paired_within_evaluator": True,
        "multiplicity": multiplicity["A2"],
    }
    a3 = {
        **a3_exact,
        "topic_effects": evidence_traceability_effects,
        "paired_within_evaluator": True,
        "multiplicity": multiplicity["A3"],
    }
    domain_specificity = {
        **domain_guard,
        "topic_effects": domain_specificity_effects,
        "noninferiority_margin": 0.35,
        "paired_within_evaluator": True,
        "passes_no_material_regression_guard": domain_guard["p_value_one_sided"]
        < 0.05,
    }

    holistic_agreement = {}
    for dimension in HOLISTIC_DIMENSIONS:
        units: dict[str, list[int | bool]] = defaultdict(list)
        for row in holistic:
            units[f"{row.topic_id}:{row.article_id}"].append(getattr(row, dimension))
        holistic_agreement[dimension] = _krippendorff_alpha(units, ordinal=True)
    claim_agreement = {}
    for label in ("supported", "correct", "overclaim", "evidence_sufficient"):
        units = defaultdict(list)
        for row in primary_claim_rows:
            value = getattr(row, label)
            if not row.cannot_assess and value is not None:
                units[f"{row.topic_id}:{row.article_id}:{row.claim_id}"].append(value)
        claim_agreement[label] = _krippendorff_alpha(units, ordinal=False)

    pairwise_summary: dict[str, Any] = {}
    for left_index, left in enumerate(ARTICLE_CONDITIONS):
        for right in ARTICLE_CONDITIONS[left_index + 1 :]:
            rows = [
                row
                for row in preferences
                if {row.left_condition, row.right_condition} == {left, right}
            ]
            pairwise_summary[f"{left}_vs_{right}"] = {
                "ratings": len(rows),
                "topics": len({row.topic_id for row in rows}),
                "preferred_left_rate": sum(
                    row.preferred_condition == left for row in rows
                )
                / len(rows),
                "evaluation_seconds": sum(row.evaluation_seconds for row in rows),
            }

    a2_passes = multiplicity["A2"]["reject_at_0_05"]
    a3_passes = multiplicity["A3"]["reject_at_0_05"]
    domain_passes = domain_specificity["passes_no_material_regression_guard"]
    return {
        "schema_version": 1,
        "status": "confirmatory_exact_cluster_analysis_complete",
        "topics": len(topics),
        "articles": len(article_condition),
        "holistic_ratings": len(holistic),
        "claim_ratings": len(claims),
        "pairwise_preferences": len(preferences),
        "holistic_summary": holistic_summary,
        "claim_summary": claim_summary,
        "forced_pairwise_preference_summary": pairwise_summary,
        "registered_contrasts": {
            "A1_claim_correct_and_supported_graph_vs_one_shot": a1,
            "A2_research_utility_graph_vs_human": a2,
            "A3_evidence_traceability_graph_vs_human": a3,
        },
        "success_guard_diagnostics": {
            "domain_specificity_graph_vs_human": domain_specificity,
            "machine_exceeds_human_claim_permitted": (
                a2_passes and a3_passes and domain_passes
            ),
            "reason": (
                "Permission requires Holm-adjusted A2 research-utility noninferiority, "
                "Holm-adjusted A3 evidence-traceability superiority, and the prospective "
                "domain-specificity non-regression guard. Mixed models are diagnostics "
                "under the finite-cluster amendment and cannot overturn exact tests."
            ),
        },
        "agreement": {
            "holistic_ordinal_krippendorff_alpha": holistic_agreement,
            "claim_nominal_krippendorff_alpha": claim_agreement,
        },
        "missingness": {
            "cannot_assess_is_retained_in_strict_estimand": True,
            "complete_case_only_claims_prohibited": True,
            "a1_extreme_case_sensitivity_reported": True,
        },
        "inference": {
            "cluster_unit": "topic",
            "primary_test": "exact paired topic-level sign-flip",
            "holistic_pairing": "within evaluator then averaged within topic",
            "claim_resolution": "agreement or exactly one independent adjudicator",
            "multiplicity": "Holm across A1, A2, and A3",
            "mixed_models": "diagnostic sensitivity; not a replacement for exact tests",
        },
    }
