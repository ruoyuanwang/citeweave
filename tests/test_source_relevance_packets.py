from __future__ import annotations

from pathlib import Path

from src.citeweave.io import sha256_file, write_json
from src.citeweave.source_relevance_packets import (
    audit_source_relevance_packets,
    build_source_relevance_packets,
)
from src.citeweave.source_review_assignment import (
    assess_source_warmup_panel_readiness,
    build_source_warmup_assignment,
    validate_source_warmup_protocol,
)


def test_builds_blinded_source_relevance_warmup(tmp_path: Path) -> None:
    records = []
    for dataset_index in range(8):
        dataset_id = f"domain_{dataset_index}"
        pack_path = tmp_path / "packs" / dataset_id / "evidence_pack.json"
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
                        "abstract_excerpt": "Visible and relevant abstract evidence.",
                        "selection_basis": "topic_task_bm25",
                        "retrieval_query": "hidden query",
                        "retrieval_rank": source_index + 1,
                        "bm25_rank": source_index + 2,
                        "matched_keyword": "generic",
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
    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {"status": "same_evidence_packs_ready", "records": records},
    )
    output = tmp_path / "warmup"
    manifest = build_source_relevance_packets(
        manifest_path=manifest_path, output_root=output
    )
    audit = audit_source_relevance_packets(output)
    assert manifest["packets"] == 120
    assert audit["status"] == "passed"
    public_text = next((output / "public").glob("*.json")).read_text(encoding="utf-8")
    assert "retrieval_query" not in public_text
    assert "matched_keyword" not in public_text
    assert "Visible and relevant abstract evidence" in public_text
    roster_path = tmp_path / "reviewer_roster.json"
    domains = [f"domain_{index}" for index in range(8)]
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
    assert (
        assess_source_warmup_panel_readiness(
            packet_root=output, reviewer_roster_path=roster_path
        )["status"]
        == "ready"
    )
    panel = build_source_warmup_assignment(
        packet_root=output,
        reviewer_roster_path=roster_path,
        output_root=tmp_path / "panel",
    )
    assert panel["decisions"] == 180
    assert panel["double_reviewed_packets"] == 60
    assert set(panel["decisions_per_reviewer"].values()) == {30}
    assert all(len(values) >= 2 for values in panel["domains_per_reviewer"].values())


def test_frozen_source_warmup_execution_chain_is_valid() -> None:
    protocol = validate_source_warmup_protocol(
        protocol_path=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol.yml"
        ),
        protocol_freeze_path=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol_freeze.json"
        ),
        packet_root=Path(
            "experiments/human_review_v2/source_relevance_warmup_v4"
        ),
        amendment_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity.yml"
        ),
        amendment_freeze_path=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity_freeze.json"
        ),
    )
    assert protocol["protocol_id"] == "source_relevance_warmup_v1"


def test_source_warmup_execution_chain_rejects_protocol_drift(tmp_path: Path) -> None:
    protocol_path = Path(
        "experiments/human_review_v2/source_relevance_warmup_protocol.yml"
    )
    changed_protocol = tmp_path / "changed_protocol.yml"
    changed_protocol.write_text(
        protocol_path.read_text(encoding="utf-8") + "\n# unauthorized drift\n",
        encoding="utf-8",
    )
    try:
        validate_source_warmup_protocol(
            protocol_path=changed_protocol,
            protocol_freeze_path=Path(
                "experiments/human_review_v2/source_relevance_warmup_protocol_freeze.json"
            ),
            packet_root=Path(
                "experiments/human_review_v2/source_relevance_warmup_v4"
            ),
            amendment_path=Path(
                "experiments/human_review_v2/"
                "source_relevance_warmup_amendment_001_execution_integrity.yml"
            ),
            amendment_freeze_path=Path(
                "experiments/human_review_v2/"
                "source_relevance_warmup_amendment_001_execution_integrity_freeze.json"
            ),
        )
    except RuntimeError as error:
        assert "differs from its freeze" in str(error)
    else:
        raise AssertionError("Protocol drift must fail before source-review assignment")
