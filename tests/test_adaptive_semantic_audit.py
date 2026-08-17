from __future__ import annotations

import json
from pathlib import Path

import pytest

from citeweave.adaptive_semantic_audit import (
    SemanticAuditPacket,
    SemanticAuditResult,
    import_semantic_audit_exchange,
    prepare_semantic_audit_exchange,
    validate_semantic_audit_archive,
)
from citeweave.formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
)
from citeweave.io import sha256_file
from citeweave.judge_protocol import canonical_json


def _blind_batch(tmp_path: Path, *, count: int = 5) -> Path:
    root = tmp_path / "blind"
    rows = []
    for index in range(count):
        packet = AdaptiveEvaluationPacket(
            packet_id=f"EP{index:020x}",
            sample_id=f"secret-topic-{index}:report",
            rubric_version="adaptive-evaluation-v1",
            canonical_evidence=[
                {"evidence_id": f"E{index:03d}", "statement": f"Fact {index}."}
            ],
            anonymous_candidate=f"Candidate {index}.",
            allowed_evidence_ids=[f"E{index:03d}"],
            content_sha256=f"{index + 1:064x}",
        )
        packet_path = root / "packets" / "evaluation" / f"{packet.packet_id}.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(packet.model_dump_json(), encoding="utf-8")
        result = AdaptiveEvaluationResult(
            packet_id=packet.packet_id,
            decision="pass",
            confidence=0.9,
            reason=f"Candidate {index} is supported by the visible evidence.",
            evidence_ids=[f"E{index:03d}"],
        )
        result_path = root / "results" / "evaluation" / f"{packet.packet_id}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        rows.append(
            {
                "packet_id": packet.packet_id,
                "kind": "evaluation",
                "packet_sha256": sha256_file(packet_path),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "blind_exchange": True,
                "condition_identity_visible": False,
                "packets": rows,
                "result_directory": "results/evaluation",
            }
        ),
        encoding="utf-8",
    )
    return root


def _prepare(tmp_path: Path, *, count: int = 5) -> tuple[Path, Path, Path]:
    batch = _blind_batch(tmp_path, count=count)
    audit = tmp_path / "audit"
    controller = tmp_path / "private" / "controller.json"
    result = prepare_semantic_audit_exchange(
        batch_root=batch,
        audit_root=audit,
        controller_manifest_path=controller,
    )
    assert result["packets"] == count
    return batch, audit, controller


def _write_audit_results(audit: Path, *, rejected: str | None = None) -> None:
    worklist = json.loads((audit / "worklist.json").read_text(encoding="utf-8"))
    for row in worklist["items"]:
        audit_id = row["audit_packet_id"]
        path = audit / row["result_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        accepted = audit_id != rejected
        path.write_text(
            SemanticAuditResult(
                audit_packet_id=audit_id,
                semantic_consistent=accepted,
                decision_defensible=accepted,
                audit_reason="Independent read-only semantic comparison.",
            ).model_dump_json(),
            encoding="utf-8",
        )


def test_prepares_neutral_exact_coverage_packets_and_private_mapping(
    tmp_path: Path,
) -> None:
    batch, audit, controller = _prepare(tmp_path, count=7)
    del batch
    worklist = json.loads((audit / "worklist.json").read_text(encoding="utf-8"))
    assert len(worklist["items"]) == 7
    assert len({row["audit_packet_id"] for row in worklist["items"]}) == 7

    visible_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(audit.rglob("*.json"))
    ).casefold()
    for hidden in (
        "sample_id",
        "secret-topic",
        "dataset_id",
        "topic_role",
        "baseline_original",
        "always_review",
        "static_review",
        "adaptive_review",
        "feedback_memory",
        "source_packet_id",
    ):
        assert hidden not in visible_text
    assert "source_packet_id" in controller.read_text(encoding="utf-8")
    protocol = json.loads((audit / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["read_only"] is True
    assert protocol["may_edit_candidate_or_result"] is False

    for row in worklist["items"]:
        packet_path = audit / row["packet_path"]
        packet = SemanticAuditPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        visible = packet.model_dump(mode="json", exclude={"audit_packet_id", "content_sha256"})
        assert packet.content_sha256 == __import__("hashlib").sha256(
            canonical_json(visible).encode("utf-8")
        ).hexdigest()
        assert sha256_file(packet_path) == row["packet_sha256"]


def test_import_binds_every_pair_and_builds_revalidatable_immutable_archive(
    tmp_path: Path,
) -> None:
    batch, audit, controller = _prepare(tmp_path)
    _write_audit_results(audit)
    imported = import_semantic_audit_exchange(
        batch_root=batch,
        audit_root=audit,
        controller_manifest_path=controller,
    )
    assert imported["packets"] == 5
    assert imported["all_consistent"] is True
    validated = validate_semantic_audit_archive(batch)
    assert validated["packets"] == 5
    assert validated["all_consistent"] is True

    semantic = json.loads((batch / "semantic_audit.json").read_text(encoding="utf-8"))
    assert semantic["judge_modified_artifacts"] is False
    assert all(
        {
            "source_packet_sha256",
            "source_result_sha256",
            "audit_packet_sha256",
            "audit_result_sha256",
        }
        <= set(row)
        for row in semantic["items"]
    )

    first_result = next((batch / "semantic_audit_archive" / "results").glob("*.json"))
    first_result.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="archive artifact changed"):
        validate_semantic_audit_archive(batch)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_result", "exact packet and result coverage"),
        ("extra_result", "exact packet and result coverage"),
        ("extra_file", "missing or undeclared files"),
        ("tampered_source_result", "changed after audit preparation|Source packet/result pair changed"),
        ("tampered_packet", "packet binding differs"),
    ],
)
def test_import_fails_closed_on_incomplete_extra_or_changed_material(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    batch, audit, controller = _prepare(tmp_path)
    _write_audit_results(audit)
    if mutation == "missing_result":
        next((audit / "results").glob("*.json")).unlink()
    elif mutation == "extra_result":
        (audit / "results" / "AP00000000000000000000.json").write_text(
            "{}", encoding="utf-8"
        )
    elif mutation == "extra_file":
        (audit / "notes.txt").write_text("forbidden", encoding="utf-8")
    elif mutation == "tampered_source_result":
        path = next((batch / "results" / "evaluation").glob("*.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reason"] = "Changed after preparation."
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path = next((audit / "packets").glob("*.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["anonymous_candidate"] = "Changed."
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match=message):
        import_semantic_audit_exchange(
            batch_root=batch,
            audit_root=audit,
            controller_manifest_path=controller,
        )


def test_audit_result_schema_allows_only_verdict_flags_and_no_edits(
    tmp_path: Path,
) -> None:
    _batch, audit, _controller = _prepare(tmp_path, count=1)
    audit_id = json.loads(
        (audit / "worklist.json").read_text(encoding="utf-8")
    )["items"][0]["audit_packet_id"]
    with pytest.raises(Exception, match="edit|Extra inputs"):
        SemanticAuditResult.model_validate(
            {
                "audit_packet_id": audit_id,
                "semantic_consistent": True,
                "decision_defensible": True,
                "audit_reason": "Consistent.",
                "edit": "Rewrite the candidate.",
            }
        )


def test_rejected_audit_is_archived_but_existing_result_import_will_fail_closed(
    tmp_path: Path,
) -> None:
    batch, audit, controller = _prepare(tmp_path)
    rejected = json.loads(
        (audit / "worklist.json").read_text(encoding="utf-8")
    )["items"][0]["audit_packet_id"]
    _write_audit_results(audit, rejected=rejected)
    result = import_semantic_audit_exchange(
        batch_root=batch,
        audit_root=audit,
        controller_manifest_path=controller,
    )
    assert result["all_consistent"] is False
    assert validate_semantic_audit_archive(batch)["all_consistent"] is False
