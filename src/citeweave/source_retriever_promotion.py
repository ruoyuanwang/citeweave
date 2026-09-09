from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SourceRetrievalReplayRow:
    query_id: str
    dataset_id: str
    reference_id: str
    adjudicated_label: Literal[
        "direct", "contextual", "irrelevant", "challenges", "cannot_assess"
    ]
    baseline_rank: int | None
    candidate_rank: int | None
    fold_held_out_dataset: str
    training_datasets: tuple[str, ...]
    unrelated_query: bool = False


def _gain(label: str) -> float:
    return {"direct": 3.0, "contextual": 1.0, "challenges": 0.5}.get(label, 0.0)


def _valid_rank(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _dcg(rows: list[SourceRetrievalReplayRow], rank_field: str, *, k: int) -> float:
    ranked = sorted(
        (
            (getattr(row, rank_field), _gain(row.adjudicated_label))
            for row in rows
            if _valid_rank(getattr(row, rank_field))
            and int(getattr(row, rank_field)) <= k
        ),
        key=lambda item: int(item[0]),  # type: ignore[arg-type]
    )
    return sum(
        gain / math.log2(int(rank) + 1)
        for rank, gain in ranked
        if rank is not None
    )


def _ndcg(rows: list[SourceRetrievalReplayRow], rank_field: str, *, k: int) -> float:
    ideal_gains = sorted((_gain(row.adjudicated_label) for row in rows), reverse=True)[:k]
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return _dcg(rows, rank_field, k=k) / ideal if ideal else 1.0


def audit_source_retriever_promotion(
    rows: list[SourceRetrievalReplayRow],
    *,
    k: int = 3,
    minimum_datasets: int = 4,
    minimum_ndcg_improvement: float = 0.02,
    positive_recall_noninferiority_margin: float = 0.0,
    challenge_recall_noninferiority_margin: float = 0.02,
    maximum_unrelated_regression_rate: float = 0.05,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    if minimum_datasets <= 0:
        raise ValueError("minimum_datasets must be positive")
    if not rows:
        return {"decision": "hold", "reason": "no_held_out_replay", "rows": 0}
    queries: dict[str, list[SourceRetrievalReplayRow]] = defaultdict(list)
    leakage = []
    row_identities: list[tuple[str, str]] = []
    invalid_rank_rows: list[str] = []
    for row in rows:
        queries[row.query_id].append(row)
        row_identities.append((row.query_id, row.reference_id))
        if (
            row.fold_held_out_dataset != row.dataset_id
            or row.dataset_id in row.training_datasets
        ):
            leakage.append(f"{row.query_id}:{row.reference_id}")
        for field in ("baseline_rank", "candidate_rank"):
            rank = getattr(row, field)
            if rank is not None and not _valid_rank(rank):
                invalid_rank_rows.append(f"{row.query_id}:{row.reference_id}:{field}")
    duplicate_rows = [
        f"{query_id}:{reference_id}"
        for query_id, reference_id in sorted(set(row_identities))
        if row_identities.count((query_id, reference_id)) > 1
    ]
    duplicate_ranks = []
    inconsistent_query_folds = []
    for query_id, query_rows in queries.items():
        datasets_for_query = {row.dataset_id for row in query_rows}
        held_out_for_query = {row.fold_held_out_dataset for row in query_rows}
        training_for_query = {tuple(sorted(row.training_datasets)) for row in query_rows}
        if (
            len(datasets_for_query) != 1
            or len(held_out_for_query) != 1
            or len(training_for_query) != 1
        ):
            inconsistent_query_folds.append(query_id)
        for field in ("baseline_rank", "candidate_rank"):
            ranks = [getattr(row, field) for row in query_rows if getattr(row, field) is not None]
            if len(ranks) != len(set(ranks)):
                duplicate_ranks.append(f"{query_id}:{field}")
    datasets = sorted({row.dataset_id for row in rows})
    evaluable = [row for row in rows if row.adjudicated_label != "cannot_assess"]
    positives = [row for row in evaluable if row.adjudicated_label in {"direct", "contextual"}]
    challenges = [row for row in evaluable if row.adjudicated_label == "challenges"]
    unrelated = [row for row in evaluable if row.unrelated_query]

    def recall_at(field: str, population: list[SourceRetrievalReplayRow]) -> float:
        return (
            sum(
                _valid_rank(getattr(row, field)) and int(getattr(row, field)) <= k
                for row in population
            )
            / len(population)
            if population
            else 0.0
        )

    def off_topic_rate(field: str) -> float:
        selected = [
            row
            for row in evaluable
            if _valid_rank(getattr(row, field)) and int(getattr(row, field)) <= k
        ]
        return (
            sum(row.adjudicated_label == "irrelevant" for row in selected) / len(selected)
            if selected
            else 1.0
        )

    baseline_ndcg = sum(_ndcg(group, "baseline_rank", k=k) for group in queries.values()) / len(queries)
    candidate_ndcg = sum(_ndcg(group, "candidate_rank", k=k) for group in queries.values()) / len(queries)
    baseline_positive = recall_at("baseline_rank", positives)
    candidate_positive = recall_at("candidate_rank", positives)
    baseline_challenge = recall_at("baseline_rank", challenges)
    candidate_challenge = recall_at("candidate_rank", challenges)
    baseline_off_topic = off_topic_rate("baseline_rank")
    candidate_off_topic = off_topic_rate("candidate_rank")
    unrelated_regressions = sum(
        (not _valid_rank(row.baseline_rank) or int(row.baseline_rank) > k)
        and _valid_rank(row.candidate_rank)
        and row.candidate_rank <= k
        for row in unrelated
    )
    unrelated_rate = unrelated_regressions / len(unrelated) if unrelated else 1.0
    checks = {
        "cross_dataset_gate": len(datasets) >= minimum_datasets,
        "leave_one_dataset_out_integrity_gate": not leakage,
        "row_identity_integrity_gate": not duplicate_rows,
        "query_fold_integrity_gate": not inconsistent_query_folds,
        "rank_value_integrity_gate": not invalid_rank_rows,
        "rank_integrity_gate": not duplicate_ranks,
        "evaluable_positive_gate": bool(positives),
        "evaluable_challenge_gate": bool(challenges),
        "evaluable_unrelated_gate": bool(unrelated),
        "ndcg_improvement_gate": candidate_ndcg - baseline_ndcg >= minimum_ndcg_improvement,
        "positive_recall_gate": candidate_positive - baseline_positive >= -positive_recall_noninferiority_margin,
        "challenge_retention_gate": candidate_challenge - baseline_challenge >= -challenge_recall_noninferiority_margin,
        "off_topic_gate": candidate_off_topic <= baseline_off_topic,
        "unrelated_regression_gate": unrelated_rate <= maximum_unrelated_regression_rate,
    }
    return {
        "decision": "promote" if all(checks.values()) else "hold",
        "rows": len(rows),
        "queries": len(queries),
        "datasets": datasets,
        "checks": checks,
        "leakage_reference_ids": sorted(leakage),
        "duplicate_query_reference_rows": duplicate_rows,
        "inconsistent_query_folds": sorted(inconsistent_query_folds),
        "invalid_rank_rows": sorted(invalid_rank_rows),
        "duplicate_rank_groups": sorted(duplicate_ranks),
        "metrics": {
            "k": k,
            "baseline_ndcg": baseline_ndcg,
            "candidate_ndcg": candidate_ndcg,
            "ndcg_difference": candidate_ndcg - baseline_ndcg,
            "baseline_positive_recall": baseline_positive,
            "candidate_positive_recall": candidate_positive,
            "baseline_challenge_recall": baseline_challenge,
            "candidate_challenge_recall": candidate_challenge,
            "baseline_off_topic_rate": baseline_off_topic,
            "candidate_off_topic_rate": candidate_off_topic,
            "unrelated_cases": len(unrelated),
            "unrelated_regressions": unrelated_regressions,
            "unrelated_regression_rate": unrelated_rate,
        },
    }
