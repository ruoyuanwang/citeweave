from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.citeweave.io import write_json
from src.citeweave.review_ui import ReviewStore, create_review_app


def test_answers_cannot_replace_server_identity() -> None:
    with pytest.raises(ValueError, match="server-owned"):
        ReviewStore._validate_answers("article", {"reviewer_code": "OTHER"})


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "review"
    (root / "packets" / "factual").mkdir(parents=True)
    (root / "packets" / "semantic").mkdir(parents=True)
    reviewer = "REVIEWER-A"
    token = "secret-review-token"
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"factual": ["F1"], "semantic": ["S1"]}},
        },
    )
    write_json(
        root / "packets" / "factual" / "F1.json",
        {
            "packet_id": "F1",
            "review_layer": "factual",
            "question": "Is the answer supported?",
            "candidate": {"answer": 1},
            "visible_evidence": {"bundle": []},
        },
    )
    write_json(
        root / "packets" / "semantic" / "S1.json",
        {
            "packet_id": "S1",
            "review_layer": "semantic_interpretation",
            "task_type": "test",
            "verified_structured_answer": {"answer": 1},
            "candidate": {"phenomenon": "Descriptive."},
            "interpretation_contract": {},
        },
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    return root, access, token


def test_blind_delivery_server_timing_and_frozen_submission(tmp_path: Path) -> None:
    root, access, token = _fixture(tmp_path)
    client = TestClient(create_review_app(root, access))
    query = "reviewer=REVIEWER-A&layer=factual"
    assert client.get(f"/api/next?{query}").status_code == 403
    headers = {"X-Review-Token": token}
    next_response = client.get(f"/api/next?{query}", headers=headers)
    assert next_response.status_code == 200
    delivery = next_response.json()
    assert "condition" not in delivery["packet"]
    assert "gold" not in delivery["packet"]
    heartbeat = {
        "packet_id": "F1",
        "session_id": delivery["session_id"],
    }
    assert (
        client.post(f"/api/heartbeat?{query}", headers=headers, json=heartbeat).status_code == 200
    )
    submission = {
        **heartbeat,
        "answers": {
            "answer_correct": True,
            "evidence_sufficient": True,
            "unsupported_fields": [],
            "action": "accept",
            "correction": None,
            "rationale": "The visible record supports the field.",
        },
    }
    accepted = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert accepted.status_code == 200
    result = accepted.json()["result"]
    assert result["review_seconds"] > 0
    assert result["timing_method"] == "visibility_heartbeat_server_accounted"
    duplicate = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert duplicate.status_code in {409, 422}
    returned = (root / "returns" / "REVIEWER-A.json").read_text(encoding="utf-8")
    assert "secret-review-token" not in returned


def test_adversarial_delivery_requires_evidence_and_valid_operator_indices(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    (root / "packets" / "adversarial").mkdir(parents=True)
    reviewer = "REVIEWER-GRAPH"
    token = "graph-review-token"
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"adversarial": ["A1"]}},
        },
    )
    write_json(
        root / "packets" / "adversarial" / "A1.json",
        {
            "case_id": "A1",
            "packet_type": "adversarial_claim_audit",
            "claim": "Deleting node X leaves one component.",
            "evidence_set_a": [{"evidence_id": "E1", "text": "support"}],
            "evidence_set_b": [{"evidence_id": "E2", "text": "challenge"}],
            "operator_trace": [{"operator": "delete_node", "node": "X"}],
        },
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    client = TestClient(create_review_app(root, access))
    query = "reviewer=REVIEWER-GRAPH&layer=adversarial"
    headers = {"X-Review-Token": token}
    delivery = client.get(f"/api/next?{query}", headers=headers).json()
    assert delivery["packet"]["packet_id"] == "A1"
    base = {
        "packet_id": "A1",
        "session_id": delivery["session_id"],
        "answers": {
            "verdict": "reject",
            "decisive_evidence_ids": ["UNKNOWN"],
            "invalid_operator_steps": [1],
            "failure_mode": "operator mismatch",
            "minimal_rewrite": None,
            "rationale": "The trace does not establish the stated result.",
        },
    }
    invalid = client.post(f"/api/submit?{query}", headers=headers, json=base)
    assert invalid.status_code == 422
    base["answers"]["decisive_evidence_ids"] = ["E2"]
    base["answers"]["invalid_operator_steps"] = [0]
    accepted = client.post(f"/api/submit?{query}", headers=headers, json=base)
    assert accepted.status_code == 200
    result = accepted.json()["result"]
    assert result["verdict"] == "reject"
    assert result["decisive_evidence_ids"] == ["E2"]


def test_source_relevance_requires_exact_visible_span(tmp_path: Path) -> None:
    root = tmp_path / "review"
    (root / "packets" / "source").mkdir(parents=True)
    reviewer = "REVIEWER-SOURCE"
    token = "source-review-token"
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"source": ["SRC1"]}},
        },
    )
    write_json(
        root / "packets" / "source" / "SRC1.json",
        {
            "packet_id": "SRC1",
            "phenomenon": {"question": "Is this a relevant cathode source?"},
            "source": {
                "title": "Sodium-ion battery cathodes",
                "abstract_excerpt": "Layered oxides are evaluated as sodium cathodes.",
            },
        },
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    client = TestClient(create_review_app(root, access))
    query = "reviewer=REVIEWER-SOURCE&layer=source"
    headers = {"X-Review-Token": token}
    delivery = client.get(f"/api/next?{query}", headers=headers).json()
    submission = {
        "packet_id": "SRC1",
        "session_id": delivery["session_id"],
        "answers": {
            "topic_relevance": "direct",
            "evidence_role": "supports_interpretation",
            "decisive_span": "not shown to the reviewer",
            "failure_type": "none",
            "suggested_query_terms": ["layered oxide"],
            "rationale": "The source directly concerns the registered topic.",
        },
    }
    invalid = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert invalid.status_code == 422
    submission["answers"]["decisive_span"] = "Layered oxides are evaluated"
    accepted = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert accepted.status_code == 200


def test_calibration_delivery_keeps_gold_hidden_and_server_times(tmp_path: Path) -> None:
    root = tmp_path / "review"
    (root / "packets" / "calibration").mkdir(parents=True)
    reviewer = "REVIEWER-CAL"
    token = "calibration-review-token"
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"calibration": ["OC1"]}},
        },
    )
    write_json(
        root / "packets" / "calibration" / "OC1.json",
        {
            "packet_id": "OC1",
            "task": "Check the visible answer against the trace.",
            "candidate_answer": {"hops": 3},
            "operator_trace": [{"operator": "path_projection", "hops": 2}],
        },
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    client = TestClient(create_review_app(root, access))
    query = "reviewer=REVIEWER-CAL&layer=calibration"
    headers = {"X-Review-Token": token}
    delivery = client.get(f"/api/next?{query}", headers=headers).json()
    assert "expected_verdict" not in delivery["packet"]
    submission = {
        "packet_id": "OC1",
        "session_id": delivery["session_id"],
        "answers": {
            "verdict": "invalid",
            "error_type": "hop_count_mismatch",
            "decisive_evidence_ids": [],
            "rationale": "The reported hop count conflicts with the visible trace.",
        },
    }
    accepted = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert accepted.status_code == 200
    result = accepted.json()["result"]
    assert result["timing_method"] == "visibility_heartbeat_server_accounted"
    assert result["review_seconds"] > 0


def test_article_claim_review_restricts_decisive_evidence_to_packet(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    (root / "packets" / "article").mkdir(parents=True)
    reviewer = "REVIEWER-ARTICLE"
    token = "article-review-token"
    write_json(
        root / "internal_manifest.json",
        {
            "reviewers": [reviewer],
            "assignments": {reviewer: {"article": ["AR1"]}},
        },
    )
    write_json(
        root / "packets" / "article" / "AR1.json",
        {
            "packet_id": "AR1",
            "review_layer": "article",
            "claim": "The bridge pattern is visible in PH-1.",
            "phenomena": [{"phenomenon_id": "PH-1"}],
            "sources": [{"reference_id": "REF-1"}],
            "allowed_decisive_evidence_ids": ["PH-1", "REF-1", "graph:edge:1"],
        },
    )
    access = tmp_path / "access.json"
    write_json(
        access,
        {"reviewer_token_sha256": {reviewer: hashlib.sha256(token.encode()).hexdigest()}},
    )
    client = TestClient(create_review_app(root, access))
    query = "reviewer=REVIEWER-ARTICLE&layer=article"
    headers = {"X-Review-Token": token}
    delivery = client.get(f"/api/next?{query}", headers=headers).json()
    submission = {
        "packet_id": "AR1",
        "session_id": delivery["session_id"],
        "answers": {
            "factual_supported": False,
            "interpretation_calibrated": False,
            "alternative_adequate": True,
            "evidence_sufficient": False,
            "action": "rewrite",
            "decisive_evidence_ids": ["PH-1", "OUTSIDE"],
            "invalid_dependency_ids": ["graph:edge:1"],
            "replacement": "The observed bridge pattern is corpus-bounded.",
            "guard": {"scope": "same graph construction"},
            "rationale": "The original claim omits a required corpus limitation.",
        },
    }
    invalid = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert invalid.status_code == 422
    submission["answers"]["decisive_evidence_ids"] = ["PH-1", "REF-1"]
    accepted = client.post(f"/api/submit?{query}", headers=headers, json=submission)
    assert accepted.status_code == 200
    result = accepted.json()["result"]
    assert result["action"] == "rewrite"
    assert result["invalid_dependency_ids"] == ["graph:edge:1"]
