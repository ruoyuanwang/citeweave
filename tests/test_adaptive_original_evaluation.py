from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from citeweave.adaptive_original_evaluation import (
    BASELINE_CONDITION,
    build_original_evaluation_packet,
    export_blind_batch,
    finalize_original_evaluation,
    import_blind_results,
    prepare_original_evaluation,
)
from citeweave.adaptive_semantic_audit import (
    SemanticAuditResult,
    import_semantic_audit_exchange,
    prepare_semantic_audit_exchange,
)
from citeweave.formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    FormalAdaptiveCase,
)
from citeweave.io import sha256_file


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    topics = ["dev-a", "dev-b", *(f"locked-{index}" for index in range(6))]
    references = tmp_path / "references.yml"
    references.write_text(
        yaml.safe_dump(
            {
                "references": [
                    {
                        "id": topic,
                        "role": "development" if index < 2 else "locked",
                    }
                    for index, topic in enumerate(topics)
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows = []
    for index, topic in enumerate(topics):
        role = "development" if index < 2 else "locked"
        for artifact_type in ("report", "graph"):
            rows.append(
                {
                    "sample_id": f"{topic}:{artifact_type}",
                    "dataset_id": topic,
                    "topic_role": role,
                    "artifact_type": artifact_type,
                    "canonical_evidence": [
                        {
                            "evidence_id": f"E{index:02d}{artifact_type[0]}",
                            "statement": f"Evidence for {topic}.",
                        }
                    ],
                    "anonymous_candidate": f"Untouched {artifact_type} for {topic}.",
                    "stage": "formal_output",
                    "issue_signature": f"{artifact_type}_quality",
                    "severity": "low",
                    "detector_score": 0.1,
                    "auto_accept_context_ok": True,
                }
            )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return references, cases, rows


def _write_results(batch: Path, *, failing_samples: set[str]) -> None:
    for packet_path in (batch / "packets" / "evaluation").glob("*.json"):
        packet = AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        target = batch / "results" / "evaluation" / packet_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            AdaptiveEvaluationResult(
                packet_id=packet.packet_id,
                decision="fail" if packet.sample_id in failing_samples else "pass",
                confidence=0.95,
                reason="Independent evidence-grounded quality assessment.",
                evidence_ids=packet.allowed_evidence_ids,
            ).model_dump_json(),
            encoding="utf-8",
        )


def _write_semantic_audit(batch: Path, *, rejected_packet: str | None = None) -> None:
    packet_ids = [
        path.stem
        for path in sorted((batch / "packets" / "evaluation").glob("*.json"))
    ]
    items = [
        {
            "packet_id": packet_id,
            "semantic_consistent": packet_id != rejected_packet,
            "decision_defensible": packet_id != rejected_packet,
            "audit_reason": "Independent packet-result semantic verification.",
            "source_packet_sha256": sha256_file(
                batch / "packets" / "evaluation" / f"{packet_id}.json"
            ),
            "source_result_sha256": sha256_file(
                batch / "results" / "evaluation" / f"{packet_id}.json"
            ),
        }
        for packet_id in packet_ids
    ]
    (batch / "semantic_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_version": "adaptive-original-semantic-audit-v1",
                "auditor_role": "independent_evaluation_consistency_auditor",
                "packets": len(packet_ids),
                "all_consistent": rejected_packet is None,
                "read_only": True,
                "judge_modified_artifacts": False,
                "items": items,
            }
        ),
        encoding="utf-8",
    )


def test_prepares_one_neutral_immutable_packet_per_original_case(tmp_path: Path) -> None:
    references, cases, rows = _inputs(tmp_path)
    root = tmp_path / "original"
    manifest = prepare_original_evaluation(
        cases_path=cases,
        reference_registry=references,
        output_root=root,
    )
    second = prepare_original_evaluation(
        cases_path=cases,
        reference_registry=references,
        output_root=root,
    )

    assert manifest == second
    assert len(manifest["items"]) == len(rows)
    assert len({row["sample_id"] for row in manifest["items"]}) == len(rows)
    assert [row["packet_id"] for row in manifest["items"]] == sorted(
        row["packet_id"] for row in manifest["items"]
    )
    assert manifest["evaluation_target"] == (
        "untouched_pre_intervention_original_candidate"
    )
    assert manifest["judge_may_modify_artifacts"] is False
    assert manifest["evaluation_updates_feedback_memory"] is False

    packet_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "packets" / "evaluation").glob("*.json")
    ).casefold()
    for hidden_name in (
        BASELINE_CONDITION,
        "always_review",
        "static_review",
        "adaptive_review",
    ):
        assert hidden_name not in packet_text

    first_packet = root / manifest["items"][0]["packet_path"]
    first_packet.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different original evaluation packet"):
        prepare_original_evaluation(
            cases_path=cases,
            reference_registry=references,
            output_root=root,
        )


def test_blind_exchange_scores_original_quality_without_memory_or_edits(
    tmp_path: Path,
) -> None:
    references, cases, rows = _inputs(tmp_path)
    original_candidates = {
        row["sample_id"]: row["anonymous_candidate"] for row in rows
    }
    original_cases_bytes = cases.read_bytes()
    root = tmp_path / "original"
    memory = root / "feedback_memory.jsonl"
    memory.parent.mkdir(parents=True)
    memory.write_text('{"sentinel":"must-not-change"}\n', encoding="utf-8")
    memory_before = memory.read_bytes()
    prepare_original_evaluation(
        cases_path=cases,
        reference_registry=references,
        output_root=root,
    )
    batch = tmp_path / "blind"
    blind_manifest = export_blind_batch(root, batch)

    serialized_manifest = json.dumps(blind_manifest, sort_keys=True).casefold()
    assert "sample_id" not in serialized_manifest
    assert "dataset_id" not in serialized_manifest
    assert "topic_role" not in serialized_manifest
    assert "baseline_original" not in serialized_manifest

    failing = {"dev-a:report", "locked-0:graph"}
    _write_results(batch, failing_samples=failing)
    with pytest.raises(FileNotFoundError, match="semantic audit"):
        import_blind_results(root, batch)
    rejected_packet = next(
        (batch / "packets" / "evaluation").glob("*.json")
    ).stem
    _write_semantic_audit(batch, rejected_packet=rejected_packet)
    with pytest.raises(ValueError, match="did not accept"):
        import_blind_results(root, batch)
    _write_semantic_audit(batch)
    (batch / "semantic_audit.json").unlink()
    audit_exchange = tmp_path / "audit-exchange"
    controller = tmp_path / "audit-controller.json"
    prepare_semantic_audit_exchange(
        batch_root=batch,
        audit_root=audit_exchange,
        controller_manifest_path=controller,
    )
    worklist = json.loads(
        (audit_exchange / "worklist.json").read_text(encoding="utf-8")
    )
    for item in worklist["items"]:
        audit_result_path = audit_exchange / item["result_path"]
        audit_result_path.parent.mkdir(parents=True, exist_ok=True)
        audit_result_path.write_text(
            SemanticAuditResult(
                audit_packet_id=item["audit_packet_id"],
                semantic_consistent=True,
                decision_defensible=True,
                audit_reason="Independent packet-result semantic verification.",
            ).model_dump_json(),
            encoding="utf-8",
        )
    import_semantic_audit_exchange(
        batch_root=batch,
        audit_root=audit_exchange,
        controller_manifest_path=controller,
    )
    assert import_blind_results(root, batch) == {
        "expected": len(rows),
        "imported": len(rows),
    }
    assert import_blind_results(root, batch)["imported"] == len(rows)
    metrics = finalize_original_evaluation(root)
    assert finalize_original_evaluation(root) == metrics

    assert metrics["quality_errors"] == 2
    assert metrics["quality_error_rate"] == 2 / len(rows)
    locked = metrics["topics"]["locked-0"]["counts"]
    assert locked["items"] == 2
    assert locked["quality_errors"] == 1
    assert locked["quality_error_rate"] == 0.5
    assert locked["review_requests"] == 0
    assert locked["auto_accepts"] == 2
    assert locked["unsafe_auto_accepts"] == 1
    assert memory.read_bytes() == memory_before
    assert cases.read_bytes() == original_cases_bytes
    assert (
        root / "audits" / "semantic_audit.json"
    ).read_bytes() == (batch / "semantic_audit.json").read_bytes()

    for packet_path in (root / "packets" / "evaluation").glob("*.json"):
        packet = AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        assert packet.anonymous_candidate == original_candidates[packet.sample_id]

    result_path = next((root / "results" / "evaluation").glob("*.json"))
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["reason"] = "Tampered after immutable result manifest."
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="result manifest"):
        finalize_original_evaluation(root)


def test_evaluation_result_schema_cannot_carry_a_revision() -> None:
    case = FormalAdaptiveCase(
        sample_id="sample",
        dataset_id="topic",
        topic_role="locked",
        artifact_type="report",
        canonical_evidence=[{"evidence_id": "E001", "statement": "Fact."}],
        anonymous_candidate="Candidate.",
        stage="report",
        issue_signature="quality",
        severity="low",
        detector_score=0.1,
    )
    packet = build_original_evaluation_packet(
        case,
        rubric_version="evaluation-v1",
        seed=42,
    )
    with pytest.raises(ValidationError, match="suggested_revision"):
        AdaptiveEvaluationResult.model_validate(
            {
                "packet_id": packet.packet_id,
                "judge_id": "evaluation",
                "decision": "pass",
                "confidence": 0.9,
                "reason": "Supported.",
                "evidence_ids": ["E001"],
                "suggested_revision": "Forbidden edit.",
            }
        )


def test_formal_import_requires_semantic_audit_hash_binding(tmp_path: Path) -> None:
    references, cases, _rows = _inputs(tmp_path)
    root = tmp_path / "original"
    prepare_original_evaluation(
        cases_path=cases,
        reference_registry=references,
        output_root=root,
    )
    batch = tmp_path / "blind"
    export_blind_batch(root, batch)
    _write_results(batch, failing_samples=set())
    _write_semantic_audit(batch)
    audit_path = batch / "semantic_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["items"][0]["source_result_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="hash binding differs"):
        import_blind_results(root, batch)


def test_formal_preparation_refuses_any_split_other_than_two_plus_six(
    tmp_path: Path,
) -> None:
    references, cases, _rows = _inputs(tmp_path)
    registry = yaml.safe_load(references.read_text(encoding="utf-8"))
    registry["references"].pop()
    references.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"exactly 2 development \+ 6 locked"):
        prepare_original_evaluation(
            cases_path=cases,
            reference_registry=references,
            output_root=tmp_path / "original",
        )
