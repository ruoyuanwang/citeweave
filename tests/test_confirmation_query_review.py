from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from citeweave.confirmation_query_review import (
    build_query_review_adjudication,
    build_query_review_collection,
    finalize_query_relevance_selection,
    validate_query_review_primary_returns,
)
from citeweave.io import read_json, sha256_file, write_json
from citeweave.query_relevance_review_ui import create_query_relevance_review_app
from scripts.prepare_review_access import prepare_access


def _inputs(tmp_path: Path, *, synthetic: bool = False) -> tuple[Path, ...]:
    protocol = tmp_path / "protocol.yml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "candidate_topics": [
                    {
                        "priority": 1,
                        "id": "secret_topic",
                        "domain": "domain_a",
                        "keywords": ["concept a", "concept b"],
                    }
                ],
                "selection": {"target_topics": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    write_json(freeze, {"sha256": sha256_file(protocol)})
    packet = tmp_path / "query_packet.json"
    write_json(
        packet,
        {
            "schema_version": 1,
            "candidate_code": "CAND-BLIND",
            "concepts": ["concept a", "concept b"],
            "instructions": "Both concepts must be central.",
            "items": [
                {
                    "item_code": "QR-1",
                    "title": "First record",
                    "abstract": "Both concepts are central.",
                    "year": 2020,
                },
                {
                    "item_code": "QR-2",
                    "title": "Second record",
                    "abstract": "Relevance requires judgment.",
                    "year": 2021,
                },
            ],
        },
    )
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "status": "awaiting_blind_query_relevance_review",
            "protocol_sha256": sha256_file(protocol),
            "records": [
                {
                    "dataset_id": "secret_topic",
                    "priority": 1,
                    "candidate_code": "CAND-BLIND",
                    "automatic_data_gate_passed": True,
                    "query_review_packet": str(packet.resolve()),
                    "query_review_packet_sha256": sha256_file(packet),
                }
            ],
        },
    )
    roster = tmp_path / "roster.json"
    write_json(
        roster,
        {
            "status": "real_query_reviewers_registered",
            "reviewers": [
                {
                    "reviewer_id": f"R{index}",
                    "real_human_attestation": True,
                    "synthetic_or_proxy_reviewer": synthetic if index == 1 else False,
                    "qualified_domains": ["domain_a"],
                    "conflicts_with_candidate_codes": [],
                }
                for index in range(1, 4)
            ],
        },
    )
    return protocol, freeze, audit, roster


def _complete_return(template_path: Path, labels: list[bool]) -> dict:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["submitted_at"] = "2026-09-08T09:00:00+08:00"
    for result, label in zip(payload["results"], labels, strict=True):
        result["relevant"] = label
        result["cannot_assess"] = False
        result["rationale_code"] = (
            "both_central" if label else "one_or_both_peripheral"
        )
        result["review_seconds"] = 12.5
    return payload


def test_double_review_blinding_adjudication_and_frozen_selection(tmp_path: Path) -> None:
    protocol, freeze, audit, roster = _inputs(tmp_path)
    collection_dir = tmp_path / "collection"
    collection = build_query_review_collection(
        protocol,
        freeze,
        audit,
        roster,
        output_dir=collection_dir,
        seed=9,
    )
    assert collection["primary_reviews_per_item"] == 2
    assert collection["reviewed_candidates"] == 1
    assert len(collection["primary_task_registry"]) == 2
    for record in collection["primary_packets"].values():
        serialized = Path(record["path"]).read_text(encoding="utf-8")
        assert "secret_topic" not in serialized
        assert "candidate_priority" in serialized
        assert '"priority"' not in serialized

    primary_ids = sorted(collection["primary_task_registry"])
    returns = collection_dir / "returns" / "primary"
    for index, reviewer_id in enumerate(primary_ids):
        labels = [True, True] if index == 0 else [True, False]
        write_json(
            returns / f"{reviewer_id}.json",
            _complete_return(
                collection_dir / "return_templates" / f"{reviewer_id}.json",
                labels,
            ),
        )
    primary_path = collection_dir / "primary_validation.json"
    primary = validate_query_review_primary_returns(
        collection_dir / "collection_manifest.json", output_path=primary_path
    )
    assert primary["primary_integrity"]["agreements"] == 1
    assert primary["primary_integrity"]["disagreements"] == 1

    adjudication_dir = tmp_path / "adjudication"
    adjudication = build_query_review_adjudication(
        primary_path,
        collection_dir / "collection_manifest.json",
        output_dir=adjudication_dir,
    )
    assert adjudication["primary_answers_hidden"] is True
    assert sum(
        row["tasks"] for row in adjudication["adjudication_packets"].values()
    ) == 1
    adjudicator_id = next(iter(adjudication["adjudication_task_registry"]))
    write_json(
        adjudication_dir / "returns" / "adjudication" / f"{adjudicator_id}.json",
        _complete_return(
            adjudication_dir / "return_templates" / f"{adjudicator_id}.json",
            [True],
        ),
    )
    output = tmp_path / "selection.json"
    result = finalize_query_relevance_selection(
        primary_path,
        collection_dir / "collection_manifest.json",
        adjudication_manifest_path=adjudication_dir / "adjudication_manifest.json",
        output_path=output,
    )
    assert result["status"] == "selected_topics_ready_for_post_selection_freeze"
    assert result["selected_topics"] == ["secret_topic"]
    assert result["candidate_summaries"][0]["resolved_relevance_rate"] == 1.0
    assert result["candidate_summaries"][0]["blind_adjudications"] == 1


def test_rejects_synthetic_or_proxy_reviewer(tmp_path: Path) -> None:
    protocol, freeze, audit, roster = _inputs(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="Synthetic or proxy"):
        build_query_review_collection(
            protocol,
            freeze,
            audit,
            roster,
            output_dir=tmp_path / "collection",
        )


def test_rejects_packet_drift_after_candidate_audit(tmp_path: Path) -> None:
    protocol, freeze, audit, roster = _inputs(tmp_path)
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    packet = Path(audit_payload["records"][0]["query_review_packet"])
    packet.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after candidate audit"):
        build_query_review_collection(
            protocol,
            freeze,
            audit,
            roster,
            output_dir=tmp_path / "collection",
        )


def test_web_review_is_authenticated_blind_timed_and_immutable(tmp_path: Path) -> None:
    protocol, freeze, audit, roster = _inputs(tmp_path)
    collection_dir = tmp_path / "collection"
    collection = build_query_review_collection(
        protocol,
        freeze,
        audit,
        roster,
        output_dir=collection_dir,
    )
    reviewer_id = next(iter(collection["primary_task_registry"]))
    token = "review-secret"
    access = tmp_path / "access.json"
    write_json(
        access,
        {
            "reviewer_token_sha256": {
                reviewer_id: hashlib.sha256(token.encode()).hexdigest()
            }
        },
    )
    client = TestClient(
        create_query_relevance_review_app(
            collection_dir / "collection_manifest.json", access
        )
    )
    assert client.get(f"/api/next?reviewer={reviewer_id}").status_code == 403
    headers = {"X-Review-Token": token}
    first = client.get(
        f"/api/next?reviewer={reviewer_id}", headers=headers
    ).json()
    serialized = json.dumps(first, ensure_ascii=False)
    assert "secret_topic" not in serialized
    assert "other_reviewer_answers" not in serialized
    submission = {
        "task_id": first["task"]["task_id"],
        "session_id": first["session_id"],
        "relevant": True,
        "cannot_assess": False,
    }
    response = client.post(
        f"/api/submit?reviewer={reviewer_id}",
        headers=headers,
        json=submission,
    )
    assert response.status_code == 200
    assert response.json()["result"]["review_seconds"] > 0
    repeated = client.post(
        f"/api/submit?reviewer={reviewer_id}",
        headers=headers,
        json=submission,
    )
    assert repeated.status_code == 422
    returned = read_json(
        collection_dir / "returns" / "primary" / f"{reviewer_id}.json"
    )
    assert returned["blind_attestation"] is True
    assert returned["independent_completion_attestation"] is True
    assert token not in json.dumps(returned)


def test_access_generator_accepts_structured_reviewer_roster(tmp_path: Path) -> None:
    manifest = tmp_path / "roster.json"
    write_json(
        manifest,
        {
            "reviewers": [
                {"reviewer_id": "R1", "qualified_domains": ["domain_a"]},
                {"reviewer_id": "R2", "qualified_domains": ["domain_a"]},
            ]
        },
    )
    output = tmp_path / "access.json"
    invitations = prepare_access(manifest, output)
    access = read_json(output)
    assert set(invitations) == {"R1", "R2"}
    assert set(access["reviewer_token_sha256"]) == {"R1", "R2"}
    assert all(
        hashlib.sha256(invitations[reviewer].encode()).hexdigest()
        == access["reviewer_token_sha256"][reviewer]
        for reviewer in invitations
    )
    assert not any(token in output.read_text() for token in invitations.values())
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_access(manifest, output)
