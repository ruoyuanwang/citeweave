from __future__ import annotations

from citeweave.source_retriever_promotion import (
    SourceRetrievalReplayRow,
    audit_source_retriever_promotion,
)


def _row(
    reference_id: str,
    label: str,
    baseline_rank: int | None,
    candidate_rank: int | None,
    *,
    unrelated: bool = False,
) -> SourceRetrievalReplayRow:
    return SourceRetrievalReplayRow(
        query_id="q1",
        dataset_id="heldout",
        reference_id=reference_id,
        adjudicated_label=label,  # type: ignore[arg-type]
        baseline_rank=baseline_rank,
        candidate_rank=candidate_rank,
        fold_held_out_dataset="heldout",
        training_datasets=("train-a", "train-b"),
        unrelated_query=unrelated,
    )


def test_source_retriever_promotes_only_with_safe_heldout_gain() -> None:
    rows = [
        _row("direct", "direct", 2, 1),
        _row("context", "contextual", 3, 2),
        _row("challenge", "challenges", 1, 3),
        _row("irrelevant", "irrelevant", None, None, unrelated=True),
    ]
    report = audit_source_retriever_promotion(
        rows,
        minimum_datasets=1,
        minimum_ndcg_improvement=0.0,
        challenge_recall_noninferiority_margin=0.0,
    )
    assert report["decision"] == "promote"


def test_source_retriever_holds_on_training_fold_leakage() -> None:
    row = _row("direct", "direct", 2, 1)
    leaked = SourceRetrievalReplayRow(
        **{**row.__dict__, "training_datasets": ("heldout",)}
    )
    report = audit_source_retriever_promotion([leaked], minimum_datasets=1)
    assert report["decision"] == "hold"
    assert report["checks"]["leave_one_dataset_out_integrity_gate"] is False


def test_source_retriever_holds_on_duplicate_query_reference_row() -> None:
    row = _row("direct", "direct", 2, 1)

    report = audit_source_retriever_promotion(
        [row, row],
        minimum_datasets=1,
        minimum_ndcg_improvement=0.0,
    )

    assert report["decision"] == "hold"
    assert report["checks"]["row_identity_integrity_gate"] is False


def test_source_retriever_holds_on_nonpositive_rank() -> None:
    row = _row("direct", "direct", 2, 0)

    report = audit_source_retriever_promotion([row], minimum_datasets=1)

    assert report["decision"] == "hold"
    assert report["checks"]["rank_value_integrity_gate"] is False


def test_source_retriever_holds_when_one_query_mixes_folds() -> None:
    first = _row("direct", "direct", 2, 1)
    second = SourceRetrievalReplayRow(
        **{
            **_row("context", "contextual", 3, 2).__dict__,
            "dataset_id": "another-heldout",
            "fold_held_out_dataset": "another-heldout",
        }
    )

    report = audit_source_retriever_promotion(
        [first, second],
        minimum_datasets=1,
        minimum_ndcg_improvement=0.0,
    )

    assert report["decision"] == "hold"
    assert report["checks"]["query_fold_integrity_gate"] is False
