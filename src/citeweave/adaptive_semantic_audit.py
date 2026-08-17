from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    validate_evaluation_result,
)
from .io import atomic_write_bytes, sha256_file
from .judge_protocol import canonical_json, scan_condition_leaks

AUDIT_PROTOCOL_VERSION = "adaptive-original-semantic-audit-v1"
HIDDEN_TERMS = (
    "baseline_original",
    "always_review",
    "static_review",
    "adaptive_review",
    "feedback_memory",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different {label}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _immutable_json(path: Path, value: Any, *, label: str) -> None:
    _immutable_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"),
        label=label,
    )


class SemanticAuditPacket(BaseModel):
    """Neutral packet-result pair visible to the independent audit role."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    audit_packet_id: str = Field(pattern=r"^AP[0-9a-f]{20}$")
    auditor_role: str = "independent_evaluation_consistency_auditor"
    canonical_evidence: Any
    anonymous_candidate: str
    evaluation_decision: str
    evaluation_confidence: float = Field(ge=0.0, le=1.0)
    evaluation_reason: str = Field(min_length=1)
    evaluation_evidence_ids: list[str]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SemanticAuditResult(BaseModel):
    """Read-only audit verdict. The schema intentionally has no edit field."""

    model_config = ConfigDict(extra="forbid")

    audit_packet_id: str = Field(pattern=r"^AP[0-9a-f]{20}$")
    auditor_role: str = "independent_evaluation_consistency_auditor"
    semantic_consistent: bool
    decision_defensible: bool
    audit_reason: str = Field(min_length=1)


def _load_batch_pairs(batch_root: Path) -> list[dict[str, Any]]:
    manifest_path = batch_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Blind evaluation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("packets")
    if (
        manifest.get("blind_exchange") is not True
        or manifest.get("condition_identity_visible") is not False
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Blind evaluation manifest violates the neutral protocol")
    packet_ids = [str(row.get("packet_id")) for row in rows if isinstance(row, dict)]
    if len(packet_ids) != len(rows) or len(set(packet_ids)) != len(packet_ids):
        raise ValueError("Blind evaluation manifest has invalid or duplicate packet IDs")

    packet_root = batch_root / "packets" / "evaluation"
    result_root = batch_root / "results" / "evaluation"
    expected_names = {f"{packet_id}.json" for packet_id in packet_ids}
    actual_packets = (
        {path.name for path in packet_root.iterdir() if path.is_file()}
        if packet_root.is_dir()
        else set()
    )
    actual_results = (
        {path.name for path in result_root.iterdir() if path.is_file()}
        if result_root.is_dir()
        else set()
    )
    if actual_packets != expected_names:
        raise ValueError(
            "Blind evaluation packet coverage differs: "
            f"missing={sorted(expected_names - actual_packets)}, "
            f"extra={sorted(actual_packets - expected_names)}"
        )
    if actual_results != expected_names:
        raise ValueError(
            "Blind evaluation result coverage differs: "
            f"missing={sorted(expected_names - actual_results)}, "
            f"extra={sorted(actual_results - expected_names)}"
        )

    manifest_by_id = {str(row["packet_id"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for packet_id in sorted(packet_ids):
        row = manifest_by_id[packet_id]
        if set(row) != {"packet_id", "kind", "packet_sha256"}:
            raise ValueError(f"Unexpected blind manifest fields for {packet_id}")
        if row["kind"] != "evaluation":
            raise ValueError(f"Non-evaluation item is prohibited: {packet_id}")
        packet_path = packet_root / f"{packet_id}.json"
        result_path = result_root / f"{packet_id}.json"
        if sha256_file(packet_path) != row["packet_sha256"]:
            raise ValueError(f"Blind evaluation packet hash differs: {packet_id}")
        packet = AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        result = AdaptiveEvaluationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        validate_evaluation_result(result, packet)
        pairs.append(
            {
                "source_packet_id": packet_id,
                "packet": packet,
                "result": result,
                "source_packet_sha256": sha256_file(packet_path),
                "source_result_sha256": sha256_file(result_path),
            }
        )
    return pairs


def _neutral_packet(pair: dict[str, Any]) -> SemanticAuditPacket:
    packet: AdaptiveEvaluationPacket = pair["packet"]
    result: AdaptiveEvaluationResult = pair["result"]
    visible = {
        "schema_version": 1,
        "auditor_role": "independent_evaluation_consistency_auditor",
        "canonical_evidence": packet.canonical_evidence,
        "anonymous_candidate": packet.anonymous_candidate,
        "evaluation_decision": result.decision,
        "evaluation_confidence": result.confidence,
        "evaluation_reason": result.reason,
        "evaluation_evidence_ids": result.evidence_ids,
    }
    identity = {
        "visible": visible,
        "packet_sha256": pair["source_packet_sha256"],
        "result_sha256": pair["source_result_sha256"],
    }
    audit_packet = SemanticAuditPacket(
        audit_packet_id=f"AP{_sha(identity)[:20]}",
        content_sha256=_sha(visible),
        **visible,
    )
    leaks = scan_condition_leaks(audit_packet.model_dump(mode="json"), HIDDEN_TERMS)
    if leaks:
        raise RuntimeError(f"Hidden role or condition leaked into audit packet: {leaks}")
    return audit_packet


def prepare_semantic_audit_exchange(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
) -> dict[str, Any]:
    """Create a neutral, exact-coverage worklist for a separate audit subagent."""

    batch_root = batch_root.resolve()
    audit_root = audit_root.resolve()
    controller_manifest_path = controller_manifest_path.resolve()
    if audit_root.exists():
        raise FileExistsError(f"Refusing to overwrite semantic audit exchange: {audit_root}")
    if controller_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite semantic audit controller manifest: "
            f"{controller_manifest_path}"
        )
    pairs = _load_batch_pairs(batch_root)
    work_items: list[dict[str, str]] = []
    controller_items: list[dict[str, str]] = []
    for pair in pairs:
        packet = _neutral_packet(pair)
        relative = Path("packets") / f"{packet.audit_packet_id}.json"
        target = audit_root / relative
        _immutable_json(
            target,
            packet.model_dump(mode="json"),
            label="semantic audit packet",
        )
        packet_hash = sha256_file(target)
        work_items.append(
            {
                "audit_packet_id": packet.audit_packet_id,
                "packet_path": relative.as_posix(),
                "packet_sha256": packet_hash,
                "result_path": (
                    Path("results") / f"{packet.audit_packet_id}.json"
                ).as_posix(),
            }
        )
        controller_items.append(
            {
                "audit_packet_id": packet.audit_packet_id,
                "source_packet_id": pair["source_packet_id"],
                "source_packet_sha256": pair["source_packet_sha256"],
                "source_result_sha256": pair["source_result_sha256"],
                "audit_packet_sha256": packet_hash,
            }
        )
    if len({row["audit_packet_id"] for row in work_items}) != len(work_items):
        raise RuntimeError("Semantic audit packet IDs are not unique")

    protocol = {
        "schema_version": 1,
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "auditor_role": "independent_evaluation_consistency_auditor",
        "read_only": True,
        "may_edit_candidate_or_result": False,
        "hidden_metadata_withheld": True,
        "allowed_result_fields": [
            "audit_packet_id",
            "auditor_role",
            "semantic_consistent",
            "decision_defensible",
            "audit_reason",
        ],
        "instructions": (
            "Inspect only whether the evaluation decision is semantically consistent "
            "with the visible candidate and evidence and whether its stated reason is "
            "defensible. Return flags and a reason only; do not edit either artifact."
        ),
    }
    worklist = {
        "schema_version": 1,
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "items": work_items,
    }
    _immutable_json(audit_root / "protocol.json", protocol, label="audit protocol")
    _immutable_json(audit_root / "worklist.json", worklist, label="audit worklist")
    controller = {
        "schema_version": 1,
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "source_batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "audit_protocol_sha256": sha256_file(audit_root / "protocol.json"),
        "audit_worklist_sha256": sha256_file(audit_root / "worklist.json"),
        "items": controller_items,
    }
    _immutable_json(
        controller_manifest_path,
        controller,
        label="semantic audit controller manifest",
    )
    return {
        "packets": len(work_items),
        "audit_root": str(audit_root),
        "worklist_sha256": controller["audit_worklist_sha256"],
    }


def import_semantic_audit_exchange(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
) -> dict[str, Any]:
    """Validate all audit verdicts and bind them immutably to source pair hashes."""

    batch_root = batch_root.resolve()
    audit_root = audit_root.resolve()
    controller_manifest_path = controller_manifest_path.resolve()
    controller = json.loads(controller_manifest_path.read_text(encoding="utf-8"))
    if controller.get("protocol_version") != AUDIT_PROTOCOL_VERSION:
        raise ValueError("Unknown semantic audit controller protocol")
    if sha256_file(batch_root / "manifest.json") != controller.get(
        "source_batch_manifest_sha256"
    ):
        raise RuntimeError("Blind evaluation manifest changed after audit preparation")
    if sha256_file(audit_root / "protocol.json") != controller.get(
        "audit_protocol_sha256"
    ):
        raise RuntimeError("Semantic audit protocol changed after preparation")
    if sha256_file(audit_root / "worklist.json") != controller.get(
        "audit_worklist_sha256"
    ):
        raise RuntimeError("Semantic audit worklist changed after preparation")

    pairs = _load_batch_pairs(batch_root)
    source_by_id = {pair["source_packet_id"]: pair for pair in pairs}
    controller_rows = controller.get("items")
    if not isinstance(controller_rows, list) or not controller_rows:
        raise ValueError("Semantic audit controller has no items")
    controller_by_audit = {
        str(row.get("audit_packet_id")): row
        for row in controller_rows
        if isinstance(row, dict)
    }
    if len(controller_by_audit) != len(controller_rows):
        raise ValueError("Semantic audit controller has duplicate or invalid IDs")
    if {str(row.get("source_packet_id")) for row in controller_rows} != set(
        source_by_id
    ):
        raise ValueError("Semantic audit controller source coverage differs")

    worklist = json.loads((audit_root / "worklist.json").read_text(encoding="utf-8"))
    work_rows = worklist.get("items")
    if not isinstance(work_rows, list):
        raise TypeError("Semantic audit worklist items are invalid")
    work_by_id = {
        str(row.get("audit_packet_id")): row
        for row in work_rows
        if isinstance(row, dict)
    }
    if len(work_by_id) != len(work_rows) or set(work_by_id) != set(
        controller_by_audit
    ):
        raise ValueError("Semantic audit worklist coverage differs")

    packet_root = audit_root / "packets"
    result_root = audit_root / "results"
    expected_names = {f"{audit_id}.json" for audit_id in controller_by_audit}
    actual_packets = (
        {path.name for path in packet_root.iterdir() if path.is_file()}
        if packet_root.is_dir()
        else set()
    )
    actual_results = (
        {path.name for path in result_root.iterdir() if path.is_file()}
        if result_root.is_dir()
        else set()
    )
    if actual_packets != expected_names or actual_results != expected_names:
        raise ValueError(
            "Semantic audit exchange requires exact packet and result coverage"
        )
    expected_exchange_files = {
        "protocol.json",
        "worklist.json",
        *(f"packets/{name}" for name in expected_names),
        *(f"results/{name}" for name in expected_names),
    }
    actual_exchange_files = {
        path.relative_to(audit_root).as_posix()
        for path in audit_root.rglob("*")
        if path.is_file()
    }
    if actual_exchange_files != expected_exchange_files:
        raise ValueError(
            "Semantic audit exchange contains missing or undeclared files"
        )

    audit_rows: list[dict[str, Any]] = []
    archive_sources: list[tuple[str, Path]] = [
        ("controller_manifest.json", controller_manifest_path),
        ("protocol.json", audit_root / "protocol.json"),
        ("worklist.json", audit_root / "worklist.json"),
    ]
    all_consistent = True
    for audit_id in sorted(controller_by_audit):
        control = controller_by_audit[audit_id]
        if set(control) != {
            "audit_packet_id",
            "source_packet_id",
            "source_packet_sha256",
            "source_result_sha256",
            "audit_packet_sha256",
        }:
            raise ValueError(f"Unexpected semantic audit controller fields: {audit_id}")
        pair = source_by_id[str(control["source_packet_id"])]
        if (
            pair["source_packet_sha256"] != control["source_packet_sha256"]
            or pair["source_result_sha256"] != control["source_result_sha256"]
        ):
            raise RuntimeError(f"Source packet/result pair changed: {audit_id}")
        work = work_by_id[audit_id]
        if set(work) != {
            "audit_packet_id",
            "packet_path",
            "packet_sha256",
            "result_path",
        }:
            raise ValueError(f"Unexpected semantic audit worklist fields: {audit_id}")
        packet_path = audit_root / str(work["packet_path"])
        result_path = audit_root / str(work["result_path"])
        if (
            packet_path != packet_root / f"{audit_id}.json"
            or result_path != result_root / f"{audit_id}.json"
            or sha256_file(packet_path) != work["packet_sha256"]
            or sha256_file(packet_path) != control["audit_packet_sha256"]
        ):
            raise RuntimeError(f"Semantic audit packet binding differs: {audit_id}")
        packet = SemanticAuditPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        result = SemanticAuditResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if result.audit_packet_id != packet.audit_packet_id:
            raise ValueError(f"Semantic audit result and packet IDs differ: {audit_id}")
        accepted = result.semantic_consistent and result.decision_defensible
        all_consistent = all_consistent and accepted
        audit_rows.append(
            {
                "packet_id": control["source_packet_id"],
                "semantic_consistent": result.semantic_consistent,
                "decision_defensible": result.decision_defensible,
                "audit_reason": result.audit_reason,
                "source_packet_sha256": control["source_packet_sha256"],
                "source_result_sha256": control["source_result_sha256"],
                "audit_packet_id": audit_id,
                "audit_packet_sha256": control["audit_packet_sha256"],
                "audit_result_sha256": sha256_file(result_path),
            }
        )
        archive_sources.extend(
            [
                (f"packets/{audit_id}.json", packet_path),
                (f"results/{audit_id}.json", result_path),
            ]
        )

    archive_binding = [
        {"path": relative, "sha256": sha256_file(source)}
        for relative, source in archive_sources
    ]
    semantic_audit = {
        "schema_version": 1,
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "auditor_role": "independent_evaluation_consistency_auditor",
        "packets": len(audit_rows),
        "all_consistent": all_consistent,
        "read_only": True,
        "judge_modified_artifacts": False,
        "archive_binding_sha256": _sha(archive_binding),
        "items": audit_rows,
    }
    semantic_path = batch_root / "semantic_audit.json"
    _immutable_json(
        semantic_path,
        semantic_audit,
        label="bound semantic audit decision",
    )

    archive_root = batch_root / "semantic_audit_archive"
    archived: list[dict[str, str]] = []
    for relative, source in archive_sources:
        target = archive_root / relative
        payload = source.read_bytes()
        _immutable_bytes(target, payload, label="semantic audit archive artifact")
        archived.append({"path": relative, "sha256": sha256_file(target)})
    semantic_target = archive_root / "semantic_audit.json"
    _immutable_bytes(
        semantic_target,
        semantic_path.read_bytes(),
        label="semantic audit archive decision",
    )
    archived.append(
        {"path": "semantic_audit.json", "sha256": sha256_file(semantic_target)}
    )
    archive_manifest = {
        "schema_version": 1,
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "source_batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "archive_binding_sha256": _sha(archive_binding),
        "artifacts": archived,
    }
    archive_manifest_path = archive_root / "archive_manifest.json"
    _immutable_json(
        archive_manifest_path,
        archive_manifest,
        label="semantic audit archive manifest",
    )
    return {
        "packets": len(audit_rows),
        "all_consistent": all_consistent,
        "semantic_audit_sha256": sha256_file(semantic_path),
        "archive_manifest_sha256": sha256_file(archive_manifest_path),
    }


def validate_semantic_audit_archive(batch_root: Path) -> dict[str, Any]:
    """Recompute every archived hash and its binding to the live audit decision."""

    batch_root = batch_root.resolve()
    archive_root = batch_root / "semantic_audit_archive"
    manifest_path = archive_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != AUDIT_PROTOCOL_VERSION:
        raise ValueError("Unknown semantic audit archive protocol")
    if manifest.get("source_batch_manifest_sha256") != sha256_file(
        batch_root / "manifest.json"
    ):
        raise RuntimeError("Semantic audit archive is bound to another blind batch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Semantic audit archive manifest has no artifacts")
    expected = {str(row.get("path")) for row in artifacts if isinstance(row, dict)}
    if len(expected) != len(artifacts):
        raise ValueError("Semantic audit archive has duplicate artifact paths")
    actual = {
        path.relative_to(archive_root).as_posix()
        for path in archive_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise ValueError("Semantic audit archive file coverage differs")
    for row in artifacts:
        if set(row) != {"path", "sha256"}:
            raise ValueError("Unexpected semantic audit archive artifact fields")
        path = archive_root / str(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Semantic audit archive artifact changed: {path}")
    live_semantic = batch_root / "semantic_audit.json"
    archived_semantic = archive_root / "semantic_audit.json"
    if live_semantic.read_bytes() != archived_semantic.read_bytes():
        raise RuntimeError("Live and archived semantic audits differ")
    semantic = json.loads(live_semantic.read_text(encoding="utf-8"))
    binding_rows = [
        row
        for row in artifacts
        if row["path"] != "semantic_audit.json"
    ]
    if (
        semantic.get("archive_binding_sha256") != _sha(binding_rows)
        or manifest.get("archive_binding_sha256") != _sha(binding_rows)
    ):
        raise RuntimeError("Semantic audit archive binding hash differs")
    return {
        "packets": semantic.get("packets"),
        "all_consistent": semantic.get("all_consistent"),
        "artifacts": len(artifacts),
        "archive_manifest_sha256": sha256_file(manifest_path),
    }
