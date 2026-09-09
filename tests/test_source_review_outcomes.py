from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.io import read_json, sha256_file, write_json
from citeweave.source_relevance_packets import build_source_relevance_packets
from citeweave.source_review_assignment import build_source_warmup_assignment
from citeweave.source_review_outcomes import (
    SourceReviewOutcomeError,
    build_source_adjudicator_commitment,
    finalize_source_warmup,
    prepare_source_adjudication_panel,
    validate_source_outcome_protocol,
    validate_source_primary_returns,
)


def _build_assignment(root: Path) -> tuple[Path, Path]:
    records = []
    domains = [f"domain_{index}" for index in range(8)]
    for dataset_index, dataset_id in enumerate(domains):
        pack_path = root / "packs" / dataset_id / "evidence_pack.json"
        sources = []
        phenomena = []
        for phenomenon_index in range(5):
            reference_ids = []
            for source_index in range(3):
                reference_id = f"REF-{dataset_index}-{phenomenon_index}-{source_index}"
                reference_ids.append(reference_id)
                sources.append(
                    {
                        "reference_id": reference_id,
                        "title": f"Source {reference_id}",
                        "year": 2025,
                        "doi": f"10.x/{reference_id}",
                        "abstract_excerpt": "Visible decisive source evidence.",
                        "selection_basis": "topic_task_bm25",
                        "retrieval_query": "hidden query",
                        "retrieval_rank": source_index + 1,
                        "matched_keyword": "hidden",
                        "topic_terms": ["domain"],
                        "topic_term_matches": 1,
                    }
                )
            phenomena.append(
                {
                    "phenomenon_id": f"PH-{dataset_index}-{phenomenon_index}",
                    "task_type": "multi_hop_connector",
                    "question": "What is the registered relation?",
                    "verified_answer": {"path": ["a", "b"]},
                    "interpretation_contract": {
                        "allowed": ["descriptive relation"],
                        "required_limitation": "No causal inference.",
                    },
                    "reference_ids": reference_ids,
                }
            )
        write_json(
            pack_path,
            {
                "passed": True,
                "dataset_id": dataset_id,
                "source_selection_version": "topic-task-bm25-v4-title-deduplicated",
                "representative_sources": sources,
                "graph_phenomena": phenomena,
            },
        )
        records.append(
            {
                "dataset_id": dataset_id,
                "pack": str(pack_path),
                "pack_sha256": sha256_file(pack_path),
            }
        )
    pack_manifest = root / "pack_manifest.json"
    write_json(
        pack_manifest,
        {"status": "same_evidence_packs_ready", "records": records},
    )
    packet_root = root / "packets"
    build_source_relevance_packets(
        manifest_path=pack_manifest,
        output_root=packet_root,
    )
    roster_path = root / "reviewer_roster.json"
    write_json(
        roster_path,
        {
            "reviewers": [
                {
                    "reviewer_id": f"R{index}",
                    "eligible_domains": domains,
                    "conflicted_domains": [],
                }
                for index in range(6)
            ]
        },
    )
    assignment_root = root / "assignment"
    build_source_warmup_assignment(
        packet_root=packet_root,
        reviewer_roster_path=roster_path,
        output_root=assignment_root,
    )
    return assignment_root, roster_path


def _server_result(
    reviewer_id: str,
    packet_id: str,
    *,
    direct: bool = True,
) -> dict[str, object]:
    return {
        "packet_id": packet_id,
        "reviewer_code": reviewer_id,
        "topic_relevance": "direct" if direct else "irrelevant",
        "evidence_role": "supports_interpretation" if direct else "irrelevant",
        "decisive_span": "Visible decisive source evidence",
        "failure_type": "none" if direct else "off_topic",
        "suggested_query_terms": ["decisive evidence"],
        "rationale": "The visible title and abstract were checked.",
        "review_seconds": 10.0,
        "server_elapsed_seconds": 12.0,
        "timing_method": "visibility_heartbeat_server_accounted",
        "submitted_at_unix": 1_800_000_000.0,
    }


def _write_primary_returns(
    assignment_root: Path,
    *,
    disagreement_packet: str,
    disagreeing_reviewer: str,
) -> None:
    internal = read_json(assignment_root / "internal_manifest.json")
    for reviewer_id, layers in internal["assignments"].items():
        results = [
            _server_result(
                reviewer_id,
                packet_id,
                direct=not (
                    packet_id == disagreement_packet
                    and reviewer_id == disagreeing_reviewer
                ),
            )
            for packet_id in layers["source"]
        ]
        write_json(
            assignment_root / "returns" / f"{reviewer_id}.json",
            {
                "schema_version": 1,
                "reviewer_code": reviewer_id,
                "results": results,
            },
        )


def test_source_warmup_closes_adjudication_and_capability_loop(tmp_path: Path) -> None:
    assignment_root, roster_path = _build_assignment(tmp_path)
    commitment_path = tmp_path / "adjudicator_commitment.json"
    commitment = build_source_adjudicator_commitment(
        assignment_root,
        roster_path,
        output_path=commitment_path,
    )
    assert commitment["double_reviewed_cases"] == 60
    target = commitment["records"][0]
    _write_primary_returns(
        assignment_root,
        disagreement_packet=target["packet_id"],
        disagreeing_reviewer=target["primary_reviewers"][1],
    )
    primary_path = tmp_path / "primary_validation.json"
    primary = validate_source_primary_returns(
        assignment_root,
        commitment_path,
        output_path=primary_path,
    )
    assert primary["decisions"] == 180
    adjudication_root = tmp_path / "adjudication"
    adjudication = prepare_source_adjudication_panel(
        assignment_root,
        commitment_path,
        primary_path,
        output_root=adjudication_root,
    )
    assert adjudication["adjudications_required"] == 1
    adjudicator_id = target["adjudicator_id"]
    write_json(
        adjudication_root / "returns" / f"{adjudicator_id}.json",
        {
            "schema_version": 1,
            "reviewer_code": adjudicator_id,
            "results": [
                _server_result(adjudicator_id, target["packet_id"], direct=True)
            ],
        },
    )
    final_root = tmp_path / "final"
    receipt = finalize_source_warmup(
        assignment_root,
        commitment_path,
        primary_path,
        adjudication_root,
        output_root=final_root,
    )
    assert receipt["resolved_labels"] == 120
    assert receipt["double_resolved_labels"] == 60
    assert receipt["single_review_training_labels"] == 60
    assert receipt["capability_observations"] == 120
    capability = read_json(final_root / "capability_freeze.json")
    registry = read_json(final_root / "reviewer_registry.json")
    assert capability["status"] == "frozen_before_heldout_assignment"
    assert capability["heldout_outcomes_inspected"] is False
    assert len(capability["reviewer_ids"]) == 6
    assert all(row["warmup_cases_completed"] >= 12 for row in registry["reviewers"])
    assert all(len(row["warmup_domains"]) >= 2 for row in registry["reviewers"])


def test_primary_return_rejects_nonserver_timing(tmp_path: Path) -> None:
    assignment_root, roster_path = _build_assignment(tmp_path)
    commitment_path = tmp_path / "adjudicator_commitment.json"
    commitment = build_source_adjudicator_commitment(
        assignment_root,
        roster_path,
        output_path=commitment_path,
    )
    target = commitment["records"][0]
    _write_primary_returns(
        assignment_root,
        disagreement_packet=target["packet_id"],
        disagreeing_reviewer=target["primary_reviewers"][1],
    )
    return_path = assignment_root / "returns" / f"{target['primary_reviewers'][0]}.json"
    payload = read_json(return_path)
    payload["results"][0]["timing_method"] = "client_reported"
    write_json(return_path, payload)

    with pytest.raises(SourceReviewOutcomeError, match="Untrusted timing method"):
        validate_source_primary_returns(
            assignment_root,
            commitment_path,
            output_path=tmp_path / "must_not_exist.json",
        )


def test_frozen_source_outcome_protocol_chain_is_valid() -> None:
    amendment = validate_source_outcome_protocol(
        protocol_path=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol.yml"
        ),
        protocol_freeze_path=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol_freeze.json"
        ),
        execution_amendment_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity.yml"
        ),
        execution_amendment_freeze_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity_freeze.json"
        ),
        outcome_amendment_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_002_closed_loop.yml"
        ),
        outcome_amendment_freeze_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_002_closed_loop_freeze.json"
        ),
    )
    assert amendment["amendment_id"] == (
        "source_relevance_warmup_amendment_002_closed_loop"
    )
