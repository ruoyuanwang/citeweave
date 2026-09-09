from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from citeweave.article_expert_collection import (
    _opaque_id,
    build_article_expert_collection,
    finalize_article_expert_adjudication,
    prepare_article_expert_adjudication,
    validate_article_expert_primary_returns,
)
from citeweave.article_expert_review_ui import create_article_expert_app
from citeweave.io import read_json, sha256_file, write_json


def test_amendment_freeze_and_implementation_chain() -> None:
    root = Path(__file__).resolve().parents[1]
    amendment_path = (
        root
        / "experiments"
        / "article_quality_v2"
        / "expert_evaluation_amendment_007_blinded_evidence_collection.yml"
    )
    freeze = read_json(amendment_path.with_name(amendment_path.stem + "_freeze.json"))
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    assert freeze["sha256"] == sha256_file(amendment_path)
    previous_freeze = read_json(
        amendment_path.with_name(
            "expert_evaluation_amendment_006_design_sensitivity_and_claim_scope_freeze.json"
        )
    )
    assert amendment["previous_amendment_sha256"] == previous_freeze["sha256"]
    for artifact in amendment["implementation"].values():
        path = root / artifact["path"]
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    seed = 20260824
    topic_id = "topic-a"
    topic_code = _opaque_id("TOP", topic_id, seed=seed)
    raw_ph = "PH-AAAAAAAAAAAAAAAA"
    raw_ref = "REF-BBBBBBBBBBBBBBBB"
    opaque_ph = _opaque_id("EV", topic_id, raw_ph, seed=seed)
    opaque_ref = _opaque_id("EV", topic_id, raw_ref, seed=seed)

    writer_pack = tmp_path / "writer_pack.json"
    write_json(
        writer_pack,
        {
            "graph_phenomena": [
                {
                    "phenomenon_id": raw_ph,
                    "task_type": "bridge_counterfactual",
                    "question": "Does deleting the edge disconnect the graph?",
                    "verified_answer": {"connected_after_deletion": True},
                    "operator_trace": [{"operator": "delete_edge"}],
                    "interpretation_contract": {
                        "allowed": "structural redundancy",
                        "forbidden": ["causality"],
                    },
                    "graph_evidence_ids": ["graph:edge:1"],
                    "reference_ids": [raw_ref],
                }
            ],
            "representative_sources": [
                {
                    "reference_id": raw_ref,
                    "title": "A relevant source",
                    "year": 2025,
                    "doi": "10.1000/example",
                    "abstract_excerpt": "The abstract supplies domain context.",
                    "retrieval_rank": 1,
                }
            ],
        },
    )
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"not-a-real-png-but-byte-bound")
    article = tmp_path / "article.md"
    article.write_text(
        f"# Article\n\nThe edge is redundant ({opaque_ph}; {opaque_ref}).\n",
        encoding="utf-8",
    )
    claim = {
        "claim_id": "CLM-1",
        "stratum": "graph_derived",
        "text": "The edge is redundant.",
        "section": "Results",
        "evidence_tokens": [opaque_ph, opaque_ref],
    }
    intake = tmp_path / "intake.json"
    write_json(
        intake,
        {
            "articles": [
                {
                    "topic_id": topic_id,
                    "condition": condition,
                    "writer_pack": str(writer_pack.resolve()),
                    "writer_pack_sha256": sha256_file(writer_pack),
                }
                for condition in (
                    "citeweave_graph_review",
                    "one_shot_llm",
                    "human_same_evidence",
                )
            ]
        },
    )
    evaluator_records = []
    for evaluator_id in ("E1", "E2"):
        packet_path = tmp_path / f"{evaluator_id}.json"
        write_json(
            packet_path,
            {
                "schema_version": 1,
                "evaluator_id": evaluator_id,
                "topics": [
                    {
                        "topic_code": topic_code,
                        "figure_path": str(figure.resolve()),
                        "figure_sha256": sha256_file(figure),
                        "articles": [
                            {
                                "article_code": "ART-1",
                                "article_path": str(article.resolve()),
                                "article_sha256": sha256_file(article),
                                "claims": [claim],
                            }
                        ],
                    }
                ],
            },
        )
        evaluator_records.append(
            {
                "evaluator_id": evaluator_id,
                "path": str(packet_path.resolve()),
                "sha256": sha256_file(packet_path),
            }
        )
    packet_manifest = tmp_path / "packet_manifest.json"
    write_json(
        packet_manifest,
        {
            "status": "expert_packets_ready",
            "seed": seed,
            "intake_sha256": sha256_file(intake),
            "evaluator_packets": evaluator_records,
            "article_quality_controls": {
                "articles": [
                    {
                        "topic_id": topic_id,
                        "condition": condition,
                        "article_code": f"ART-{index + 1}",
                    }
                    for index, condition in enumerate(
                        (
                            "citeweave_graph_review",
                            "one_shot_llm",
                            "human_same_evidence",
                        )
                    )
                ]
            },
            "assignments": [
                {
                    "topic_id": topic_id,
                    "evaluator_id": "E1",
                    "evaluator_role": "domain_expert",
                    "conditions": ["citeweave_graph_review"],
                    "article_codes": ["ART-1"],
                },
                {
                    "topic_id": topic_id,
                    "evaluator_id": "E2",
                    "evaluator_role": "methods_expert",
                    "conditions": ["citeweave_graph_review"],
                    "article_codes": ["ART-1"],
                },
                {
                    "topic_id": topic_id,
                    "evaluator_id": "E3",
                    "evaluator_role": "domain_expert",
                    "conditions": ["one_shot_llm"],
                    "article_codes": ["ART-2"],
                },
            ],
        },
    )
    collection_root = tmp_path / "collection"
    build_article_expert_collection(
        intake, packet_manifest, output_dir=collection_root
    )
    token = "expert-secret"
    access = tmp_path / "access.json"
    write_json(
        access,
        {
            "reviewer_token_sha256": {
                evaluator_id: hashlib.sha256(token.encode()).hexdigest()
                for evaluator_id in ("E1", "E2", "E3")
            }
        },
    )
    return collection_root / "collection_manifest.json", access, token


def test_builds_condition_blind_evidence_viewer(tmp_path: Path) -> None:
    manifest_path, _, _ = _fixture(tmp_path)
    manifest = read_json(manifest_path)
    assert manifest["status"] == "expert_collection_ready"
    assert manifest["task_counts"] == {"claim": 2, "holistic": 2, "pairwise": 0}
    viewer_record = next(iter(manifest["evidence_viewers"].values()))
    viewer_text = Path(viewer_record["path"]).read_text(encoding="utf-8")
    assert "PH-AAAAAAAAAAAAAAAA" not in viewer_text
    assert "REF-BBBBBBBBBBBBBBBB" not in viewer_text
    viewer = json.loads(viewer_text)
    assert {row["evidence_type"] for row in viewer["items"]} == {
        "graph_phenomenon",
        "bibliographic_source",
    }
    assert viewer["items"][0]["operator_trace"] == [{"operator": "delete_edge"}]
    assert all("retrieval_rank" not in row for row in viewer["items"])


def test_expert_service_enforces_blinding_evidence_and_immutable_timing(
    tmp_path: Path,
) -> None:
    manifest, access, token = _fixture(tmp_path)
    client = TestClient(create_article_expert_app(manifest, access))
    assert client.get("/api/next?evaluator=E1").status_code == 403
    headers = {"X-Review-Token": token}

    first = client.get("/api/next?evaluator=E1", headers=headers).json()
    assert first["task"]["task_type"] == "holistic"
    serialized = json.dumps(first, ensure_ascii=False)
    assert "citeweave_graph_review" not in serialized
    invalid_holistic = {
        "task_id": first["task"]["task_id"],
        "session_id": first["session_id"],
        "answers": {
            **{
                dimension: 3
                for dimension in (
                    "factual_accuracy",
                    "evidence_traceability",
                    "phenomenon_depth",
                    "alternative_explanations",
                    "epistemic_calibration",
                    "domain_specificity",
                    "argumentative_coherence",
                    "research_utility",
                )
            },
            "research_utility": 6,
            "rationale": "Independent assessment.",
        },
    }
    assert (
        client.post(
            "/api/submit?evaluator=E1",
            headers=headers,
            json=invalid_holistic,
        ).status_code
        == 422
    )
    invalid_holistic["answers"]["research_utility"] = 3
    accepted = client.post(
        "/api/submit?evaluator=E1", headers=headers, json=invalid_holistic
    )
    assert accepted.status_code == 200
    assert accepted.json()["result"]["review_seconds"] > 0

    second = client.get("/api/next?evaluator=E1", headers=headers).json()
    assert second["task"]["task_type"] == "claim"
    claim_submission = {
        "task_id": second["task"]["task_id"],
        "session_id": second["session_id"],
        "answers": {
            "supported": True,
            "correct": True,
            "overclaim": False,
            "evidence_sufficient": True,
            "cannot_assess": False,
            "decisive_evidence_ids": ["EV-UNKNOWN"],
            "rationale": "The visible operator trace supports the claim.",
        },
    }
    assert (
        client.post(
            "/api/submit?evaluator=E1",
            headers=headers,
            json=claim_submission,
        ).status_code
        == 422
    )
    claim_submission["answers"]["decisive_evidence_ids"] = second["task"][
        "allowed_decisive_evidence_ids"
    ]
    accepted = client.post(
        "/api/submit?evaluator=E1", headers=headers, json=claim_submission
    )
    assert accepted.status_code == 200
    completed = client.get("/api/next?evaluator=E1", headers=headers).json()
    assert completed["complete"] is True
    returned = (manifest.parent / "returns" / "E1.json").read_text(encoding="utf-8")
    assert token not in returned


def _submit_fixture_tasks(
    client: TestClient,
    *,
    evaluator_id: str,
    token: str,
    claim_correct: bool,
) -> None:
    headers = {"X-Review-Token": token}
    first = client.get(
        f"/api/next?evaluator={evaluator_id}", headers=headers
    ).json()
    scores = {
        dimension: 3
        for dimension in (
            "factual_accuracy",
            "evidence_traceability",
            "phenomenon_depth",
            "alternative_explanations",
            "epistemic_calibration",
            "domain_specificity",
            "argumentative_coherence",
            "research_utility",
        )
    }
    response = client.post(
        f"/api/submit?evaluator={evaluator_id}",
        headers=headers,
        json={
            "task_id": first["task"]["task_id"],
            "session_id": first["session_id"],
            "answers": {**scores, "rationale": "Holistic assessment."},
        },
    )
    assert response.status_code == 200
    second = client.get(
        f"/api/next?evaluator={evaluator_id}", headers=headers
    ).json()
    response = client.post(
        f"/api/submit?evaluator={evaluator_id}",
        headers=headers,
        json={
            "task_id": second["task"]["task_id"],
            "session_id": second["session_id"],
            "answers": {
                "supported": True,
                "correct": claim_correct,
                "overclaim": False,
                "evidence_sufficient": True,
                "cannot_assess": False,
                "decisive_evidence_ids": second["task"][
                    "allowed_decisive_evidence_ids"
                ],
                "rationale": "Claim assessment.",
            },
        },
    )
    assert response.status_code == 200


def test_primary_return_validator_restores_conditions_and_materializes_disputes(
    tmp_path: Path,
) -> None:
    manifest, access, token = _fixture(tmp_path)
    client = TestClient(create_article_expert_app(manifest, access))
    _submit_fixture_tasks(client, evaluator_id="E1", token=token, claim_correct=True)
    _submit_fixture_tasks(client, evaluator_id="E2", token=token, claim_correct=False)
    protocol = tmp_path / "protocol.yml"
    protocol.write_text("schema_version: 1\n", encoding="utf-8")
    output = tmp_path / "primary_validation.json"
    result = validate_article_expert_primary_returns(
        manifest, protocol, output_path=output
    )
    assert result["status"] == "awaiting_blind_claim_adjudication"
    assert result["primary_integrity"] == {
        "evaluators": 2,
        "tasks": 4,
        "task_counts": {"claim": 2, "holistic": 2},
        "claim_pairs": 1,
        "claim_disagreements": 1,
        "all_registered_tasks_returned_once": True,
        "all_server_owned_fields_match": True,
        "all_timings_server_accounted": True,
    }
    assert {row["condition"] for row in result["holistic_ratings"]} == {
        "citeweave_graph_review"
    }
    assert result["adjudication_worklist"][0]["excluded_primary_evaluators"] == [
        "E1",
        "E2",
    ]

    adjudication_root = tmp_path / "adjudication"
    adjudication = prepare_article_expert_adjudication(
        output,
        manifest,
        output_dir=adjudication_root,
    )
    assert adjudication["collection_mode"] == "blind_claim_adjudication"
    assert list(adjudication["evaluator_packets"]) == ["E3"]
    adjudication_manifest = adjudication_root / "collection_manifest.json"
    adjudication_client = TestClient(
        create_article_expert_app(adjudication_manifest, access)
    )
    task = adjudication_client.get(
        "/api/next?evaluator=E3", headers={"X-Review-Token": token}
    ).json()
    serialized = json.dumps(task, ensure_ascii=False)
    assert "E1" not in serialized and "E2" not in serialized
    assert "claim_correct" not in serialized
    response = adjudication_client.post(
        "/api/submit?evaluator=E3",
        headers={"X-Review-Token": token},
        json={
            "task_id": task["task"]["task_id"],
            "session_id": task["session_id"],
            "answers": {
                "supported": True,
                "correct": True,
                "overclaim": False,
                "evidence_sufficient": True,
                "cannot_assess": False,
                "decisive_evidence_ids": task["task"][
                    "allowed_decisive_evidence_ids"
                ],
                "rationale": "Independent adjudication from visible evidence.",
            },
        },
    )
    assert response.status_code == 200
    final_output = tmp_path / "analysis_input.json"
    finalized = finalize_article_expert_adjudication(
        output,
        adjudication_manifest,
        output_path=final_output,
    )
    assert finalized["status"] == "analysis_input_ready_after_blind_adjudication"
    assert finalized["adjudication_integrity"] == {
        "disputes": 1,
        "adjudications": 1,
        "one_independent_adjudicator_per_dispute": True,
        "primary_answers_hidden_from_adjudicator": True,
        "all_timings_server_accounted": True,
    }
    assert len(finalized["claim_ratings"]) == 3
    assert finalized["claim_ratings"][-1]["adjudication"] is True
