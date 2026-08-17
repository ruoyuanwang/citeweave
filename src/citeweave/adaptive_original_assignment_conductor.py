from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adaptive_original_evaluation import import_blind_results
from .adaptive_semantic_audit import (
    AUDIT_PROTOCOL_VERSION,
    HIDDEN_TERMS,
    SemanticAuditPacket,
    SemanticAuditResult,
    import_semantic_audit_exchange,
    validate_semantic_audit_archive,
)
from .formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    validate_evaluation_result,
)
from .io import atomic_write_bytes, sha256_file
from .judge_protocol import canonical_json, scan_condition_leaks

ORIGINAL_ASSIGNMENT_PROTOCOL = "adaptive-original-evaluation-assignment-v1"
SEMANTIC_ASSIGNMENT_PROTOCOL = "adaptive-original-semantic-assignment-v1"


class AdaptiveOriginalEvaluationResult(AdaptiveEvaluationResult):
    """Strict result contract for the untouched-original evaluation assignment."""


class AdaptiveOriginalSemanticAuditResult(SemanticAuditResult):
    """Strict, read-only result contract for the independent semantic audit."""


class AdaptiveOriginalAssignmentError(ValueError):
    """Raised when either read-only assignment exchange must fail closed."""


@dataclass(frozen=True)
class _OriginalItem:
    packet_id: str
    packet_path: Path
    packet_sha256: str
    packet: AdaptiveEvaluationPacket


@dataclass(frozen=True)
class _SemanticItem:
    audit_packet_id: str
    packet_path: Path
    packet_sha256: str
    result_path: Path
    packet: SemanticAuditPacket


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveOriginalAssignmentError(f"Cannot read valid JSON: {path}") from exc


def _write_idempotent(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise AdaptiveOriginalAssignmentError(
                f"Refusing to overwrite different conductor artifact: {path}"
            )
        return
    atomic_write_bytes(path, payload)


def _copy_idempotent(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise AdaptiveOriginalAssignmentError(
                f"Refusing to overwrite different isolated packet: {target}"
            )
        return
    atomic_write_bytes(target, payload)


def _content_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_original_content_hash(packet: AdaptiveEvaluationPacket) -> None:
    visible = {
        "sample_id": packet.sample_id,
        "judge_id": packet.judge_id,
        "rubric_version": packet.rubric_version,
        "canonical_evidence": packet.canonical_evidence,
        "anonymous_candidate": packet.anonymous_candidate,
        "allowed_evidence_ids": packet.allowed_evidence_ids,
    }
    if packet.content_sha256 != _content_sha(visible):
        raise AdaptiveOriginalAssignmentError(
            f"Embedded original-evaluation content hash differs: {packet.packet_id}"
        )


def _validate_semantic_content_hash(packet: SemanticAuditPacket) -> None:
    visible = {
        "schema_version": packet.schema_version,
        "auditor_role": packet.auditor_role,
        "canonical_evidence": packet.canonical_evidence,
        "anonymous_candidate": packet.anonymous_candidate,
        "evaluation_decision": packet.evaluation_decision,
        "evaluation_confidence": packet.evaluation_confidence,
        "evaluation_reason": packet.evaluation_reason,
        "evaluation_evidence_ids": packet.evaluation_evidence_ids,
    }
    if packet.content_sha256 != _content_sha(visible):
        raise AdaptiveOriginalAssignmentError(
            f"Embedded semantic-audit content hash differs: {packet.audit_packet_id}"
        )


def _validate_original_batch(batch_root: Path) -> list[_OriginalItem]:
    manifest_path = batch_root / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_keys = {
        "schema_version",
        "blind_exchange",
        "condition_identity_visible",
        "packets",
        "result_directory",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise AdaptiveOriginalAssignmentError(
            "Original-evaluation blind manifest keys differ"
        )
    rows = manifest.get("packets")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("blind_exchange") is not True
        or manifest.get("condition_identity_visible") is not False
        or manifest.get("result_directory") != "results/evaluation"
        or not isinstance(rows, list)
        or not rows
    ):
        raise AdaptiveOriginalAssignmentError(
            "Original-evaluation blind manifest contract differs"
        )

    items: list[_OriginalItem] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "packet_id",
            "kind",
            "packet_sha256",
        }:
            raise AdaptiveOriginalAssignmentError(
                "Original-evaluation manifest row keys differ"
            )
        packet_id = row.get("packet_id")
        expected_sha = row.get("packet_sha256")
        if (
            not isinstance(packet_id, str)
            or packet_id in seen
            or row.get("kind") != "evaluation"
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise AdaptiveOriginalAssignmentError(
                "Malformed or duplicate original-evaluation manifest row"
            )
        seen.add(packet_id)
        packet_path = batch_root / "packets" / "evaluation" / f"{packet_id}.json"
        if not packet_path.is_file() or sha256_file(packet_path) != expected_sha:
            raise AdaptiveOriginalAssignmentError(
                f"Original-evaluation packet file/hash differs: {packet_id}"
            )
        try:
            packet = AdaptiveEvaluationPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise AdaptiveOriginalAssignmentError(
                f"Original-evaluation packet schema differs: {packet_id}"
            ) from exc
        if packet.packet_id != packet_id:
            raise AdaptiveOriginalAssignmentError(
                f"Original-evaluation packet identity differs: {packet_id}"
            )
        _validate_original_content_hash(packet)
        items.append(
            _OriginalItem(
                packet_id=packet_id,
                packet_path=packet_path,
                packet_sha256=expected_sha,
                packet=packet,
            )
        )

    packet_root = batch_root / "packets" / "evaluation"
    actual = (
        {path.name for path in packet_root.iterdir() if path.is_file()}
        if packet_root.is_dir()
        else set()
    )
    expected = {f"{item.packet_id}.json" for item in items}
    if actual != expected:
        raise AdaptiveOriginalAssignmentError(
            "Original-evaluation packet directory coverage differs"
        )
    if [item.packet_id for item in items] != sorted(seen):
        raise AdaptiveOriginalAssignmentError(
            "Original-evaluation manifest must use deterministic packet-ID order"
        )
    return items


def _validate_semantic_exchange(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
) -> list[_SemanticItem]:
    controller = _read_json(controller_manifest_path)
    protocol_path = audit_root / "protocol.json"
    worklist_path = audit_root / "worklist.json"
    protocol = _read_json(protocol_path)
    worklist = _read_json(worklist_path)

    expected_protocol_keys = {
        "schema_version",
        "protocol_version",
        "auditor_role",
        "read_only",
        "may_edit_candidate_or_result",
        "hidden_metadata_withheld",
        "allowed_result_fields",
        "instructions",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected_protocol_keys:
        raise AdaptiveOriginalAssignmentError("Semantic-audit protocol keys differ")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != AUDIT_PROTOCOL_VERSION
        or protocol.get("auditor_role")
        != "independent_evaluation_consistency_auditor"
        or protocol.get("read_only") is not True
        or protocol.get("may_edit_candidate_or_result") is not False
        or protocol.get("hidden_metadata_withheld") is not True
        or protocol.get("allowed_result_fields")
        != [
            "audit_packet_id",
            "auditor_role",
            "semantic_consistent",
            "decision_defensible",
            "audit_reason",
        ]
    ):
        raise AdaptiveOriginalAssignmentError("Semantic-audit protocol contract differs")

    if not isinstance(worklist, dict) or set(worklist) != {
        "schema_version",
        "protocol_version",
        "items",
    }:
        raise AdaptiveOriginalAssignmentError("Semantic-audit worklist keys differ")
    work_rows = worklist.get("items")
    if (
        worklist.get("schema_version") != 1
        or worklist.get("protocol_version") != AUDIT_PROTOCOL_VERSION
        or not isinstance(work_rows, list)
        or not work_rows
    ):
        raise AdaptiveOriginalAssignmentError("Semantic-audit worklist contract differs")

    if not isinstance(controller, dict) or set(controller) != {
        "schema_version",
        "protocol_version",
        "source_batch_manifest_sha256",
        "audit_protocol_sha256",
        "audit_worklist_sha256",
        "items",
    }:
        raise AdaptiveOriginalAssignmentError(
            "Semantic-audit controller manifest keys differ"
        )
    control_rows = controller.get("items")
    if (
        controller.get("schema_version") != 1
        or controller.get("protocol_version") != AUDIT_PROTOCOL_VERSION
        or controller.get("source_batch_manifest_sha256")
        != sha256_file(batch_root / "manifest.json")
        or controller.get("audit_protocol_sha256") != sha256_file(protocol_path)
        or controller.get("audit_worklist_sha256") != sha256_file(worklist_path)
        or not isinstance(control_rows, list)
        or not control_rows
    ):
        raise AdaptiveOriginalAssignmentError(
            "Semantic-audit controller manifest contract differs"
        )

    control_by_id: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    for row in control_rows:
        if not isinstance(row, dict) or set(row) != {
            "audit_packet_id",
            "source_packet_id",
            "source_packet_sha256",
            "source_result_sha256",
            "audit_packet_sha256",
        }:
            raise AdaptiveOriginalAssignmentError(
                "Semantic-audit controller row keys differ"
            )
        audit_id = row.get("audit_packet_id")
        source_id = row.get("source_packet_id")
        if (
            not isinstance(audit_id, str)
            or audit_id in control_by_id
            or not isinstance(source_id, str)
            or source_id in source_ids
        ):
            raise AdaptiveOriginalAssignmentError(
                "Semantic-audit controller contains duplicate identities"
            )
        control_by_id[audit_id] = row
        source_ids.add(source_id)

    items: list[_SemanticItem] = []
    seen: set[str] = set()
    for row in work_rows:
        if not isinstance(row, dict) or set(row) != {
            "audit_packet_id",
            "packet_path",
            "packet_sha256",
            "result_path",
        }:
            raise AdaptiveOriginalAssignmentError(
                "Semantic-audit worklist row keys differ"
            )
        audit_id = row.get("audit_packet_id")
        if not isinstance(audit_id, str) or audit_id in seen:
            raise AdaptiveOriginalAssignmentError(
                "Semantic-audit worklist contains duplicate identities"
            )
        seen.add(audit_id)
        if audit_id not in control_by_id:
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit worklist/controller coverage differs: {audit_id}"
            )
        packet_path = (audit_root / str(row.get("packet_path"))).resolve()
        result_path = (audit_root / str(row.get("result_path"))).resolve()
        expected_packet = (audit_root / "packets" / f"{audit_id}.json").resolve()
        expected_result = (audit_root / "results" / f"{audit_id}.json").resolve()
        control = control_by_id[audit_id]
        if (
            packet_path != expected_packet
            or result_path != expected_result
            or row.get("packet_sha256") != control.get("audit_packet_sha256")
            or not packet_path.is_file()
            or sha256_file(packet_path) != row.get("packet_sha256")
        ):
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit packet/result path binding differs: {audit_id}"
            )
        try:
            packet = SemanticAuditPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit packet schema differs: {audit_id}"
            ) from exc
        if packet.audit_packet_id != audit_id:
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit packet identity differs: {audit_id}"
            )
        _validate_semantic_content_hash(packet)
        if scan_condition_leaks(packet.model_dump(mode="json"), HIDDEN_TERMS):
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit packet leaks hidden condition metadata: {audit_id}"
            )
        items.append(
            _SemanticItem(
                audit_packet_id=audit_id,
                packet_path=packet_path,
                packet_sha256=str(row["packet_sha256"]),
                result_path=result_path,
                packet=packet,
            )
        )
    if set(seen) != set(control_by_id):
        raise AdaptiveOriginalAssignmentError(
            "Semantic-audit worklist/controller exact coverage differs"
        )
    packet_root = audit_root / "packets"
    actual_packets = (
        {path.name for path in packet_root.iterdir() if path.is_file()}
        if packet_root.is_dir()
        else set()
    )
    expected_packets = {f"{item.audit_packet_id}.json" for item in items}
    if actual_packets != expected_packets:
        raise AdaptiveOriginalAssignmentError(
            "Semantic-audit packet directory coverage differs"
        )
    return items


def _materialize_assignment(
    *,
    assignment_root: Path,
    protocol_version: str,
    role: str,
    items: list[_OriginalItem] | list[_SemanticItem],
    result_schema: dict[str, Any],
    instructions: list[str],
    source_ids_to_hide: set[str] | None = None,
) -> dict[str, Any]:
    work: list[dict[str, str]] = []
    for item in items:
        item_id = (
            item.packet_id
            if isinstance(item, _OriginalItem)
            else item.audit_packet_id
        )
        isolated_path = assignment_root / "packets" / f"{item_id}.json"
        result_path = (
            item.result_path
            if isinstance(item, _SemanticItem)
            else item.packet_path.parents[2]
            / "results"
            / "evaluation"
            / f"{item_id}.json"
        )
        work.append(
            {
                "packet_id": item_id,
                "packet_file": str(isolated_path.resolve()),
                "packet_sha256": item.packet_sha256,
                "result_file": str(result_path.resolve()),
            }
        )

    assignment = {
        "schema_version": 1,
        "protocol_version": protocol_version,
        "assignment_role": role,
        "read_only_judgment": True,
        "only_listed_packets_are_visible": True,
        "controller_metadata_in_scope": False,
        "condition_identity_visible": False,
        "topic_role_visible": False,
        "feedback_memory_visible": False,
        "may_read_unlisted_workspace_files": False,
        "may_modify_candidate_or_source_artifacts": False,
        "may_modify_query_corpus_or_pipeline": False,
        "may_update_feedback_memory": False,
        "instructions": instructions,
        "result_schema": result_schema,
        "output_contract": (
            "Return exactly one JSON object per listed packet at its unique "
            "result_file. Do not create or modify any other file."
        ),
        "write_scope": [row["result_file"] for row in work],
        "items": work,
    }
    serialized = json.dumps(assignment, ensure_ascii=False)
    forbidden = source_ids_to_hide or set()
    if any(source_id in serialized for source_id in forbidden):
        raise AdaptiveOriginalAssignmentError(
            "Assignment leaks a hidden source packet identity"
        )

    expected_isolated = {f"{row['packet_id']}.json" for row in work}
    actual_isolated = (
        {path.name for path in (assignment_root / "packets").iterdir() if path.is_file()}
        if (assignment_root / "packets").is_dir()
        else set()
    )
    if actual_isolated - expected_isolated:
        raise AdaptiveOriginalAssignmentError(
            "Assignment packet directory contains an undeclared packet"
        )
    for item in items:
        item_id = (
            item.packet_id
            if isinstance(item, _OriginalItem)
            else item.audit_packet_id
        )
        _copy_idempotent(
            item.packet_path,
            assignment_root / "packets" / f"{item_id}.json",
        )
    _write_idempotent(assignment_root / "assignment.json", assignment)
    final_isolated = {
        path.name for path in (assignment_root / "packets").iterdir() if path.is_file()
    }
    if final_isolated != expected_isolated:
        raise AdaptiveOriginalAssignmentError(
            "Assignment packet directory exact coverage differs"
        )
    return {
        "status": "assignment_ready",
        "assignment_path": str((assignment_root / "assignment.json").resolve()),
        "assignment_sha256": sha256_file(assignment_root / "assignment.json"),
        "packet_count": len(items),
    }


def prepare_original_evaluation_assignment(
    *,
    batch_root: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    batch_root = batch_root.resolve()
    assignment_root = assignment_root.resolve()
    items = _validate_original_batch(batch_root)
    return _materialize_assignment(
        assignment_root=assignment_root,
        protocol_version=ORIGINAL_ASSIGNMENT_PROTOCOL,
        role="independent_read_only_untouched_original_evaluation_judge",
        items=items,
        result_schema=AdaptiveOriginalEvaluationResult.model_json_schema(),
        instructions=[
            "Inspect only the supplied neutral evaluation packets.",
            (
                "Judge support and quality using only canonical_evidence, "
                "anonymous_candidate, and allowed_evidence_ids in each packet."
            ),
            (
                "Return only the strict evaluation result fields; do not propose or "
                "apply edits and do not update feedback memory."
            ),
            (
                "Do not browse, call APIs, use outside knowledge, or inspect any "
                "unlisted workspace file."
            ),
        ],
    )


def prepare_semantic_audit_assignment(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    batch_root = batch_root.resolve()
    audit_root = audit_root.resolve()
    controller_manifest_path = controller_manifest_path.resolve()
    assignment_root = assignment_root.resolve()
    items = _validate_semantic_exchange(
        batch_root=batch_root,
        audit_root=audit_root,
        controller_manifest_path=controller_manifest_path,
    )
    controller = _read_json(controller_manifest_path)
    source_ids = {
        str(row["source_packet_id"]) for row in controller["items"]
    }
    return _materialize_assignment(
        assignment_root=assignment_root,
        protocol_version=SEMANTIC_ASSIGNMENT_PROTOCOL,
        role="independent_read_only_original_evaluation_semantic_auditor",
        items=items,
        result_schema=AdaptiveOriginalSemanticAuditResult.model_json_schema(),
        instructions=[
            "Inspect only the supplied neutral semantic-audit packets.",
            (
                "Return only semantic_consistent, decision_defensible, and an "
                "audit_reason together with the packet identity and fixed auditor role."
            ),
            (
                "Do not propose, encode, or apply an edit to the candidate, evidence, "
                "evaluation result, query, corpus, pipeline, or feedback memory."
            ),
            (
                "Do not browse, call APIs, use outside knowledge, infer hidden topic, "
                "role, condition, source identity, or inspect unlisted workspace files."
            ),
        ],
        source_ids_to_hide=source_ids,
    )


def _exact_result_paths(
    *,
    result_root: Path,
    expected: set[Path],
    label: str,
) -> None:
    actual = (
        {path.resolve() for path in result_root.rglob("*") if path.is_file()}
        if result_root.is_dir()
        else set()
    )
    expected_resolved = {path.resolve() for path in expected}
    if actual != expected_resolved:
        missing = sorted(str(path) for path in expected_resolved - actual)
        extra = sorted(str(path) for path in actual - expected_resolved)
        raise AdaptiveOriginalAssignmentError(
            f"{label} result coverage differs: missing={missing}, extra={extra}"
        )


def validate_original_evaluation_results(
    *,
    batch_root: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    batch_root = batch_root.resolve()
    assignment_root = assignment_root.resolve()
    prepared = prepare_original_evaluation_assignment(
        batch_root=batch_root,
        assignment_root=assignment_root,
    )
    items = _validate_original_batch(batch_root)
    expected = {
        batch_root / "results" / "evaluation" / f"{item.packet_id}.json"
        for item in items
    }
    _exact_result_paths(
        result_root=batch_root / "results",
        expected=expected,
        label="Original-evaluation",
    )

    rows: list[dict[str, str]] = []
    for item in items:
        result_path = (
            batch_root / "results" / "evaluation" / f"{item.packet_id}.json"
        )
        try:
            result = AdaptiveOriginalEvaluationResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            validate_evaluation_result(result, item.packet)
        except Exception as exc:
            raise AdaptiveOriginalAssignmentError(
                "Original-evaluation result schema/packet binding differs: "
                f"{item.packet_id}"
            ) from exc
        rows.append(
            {
                "packet_id": item.packet_id,
                "packet_sha256": item.packet_sha256,
                "result_sha256": sha256_file(result_path),
            }
        )
    validation = {
        "schema_version": 1,
        "protocol_version": ORIGINAL_ASSIGNMENT_PROTOCOL,
        "status": "complete_results_validated",
        "batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "assignment_sha256": prepared["assignment_sha256"],
        "exact_coverage": True,
        "read_only": True,
        "candidate_modified": False,
        "feedback_memory_updated": False,
        "items": rows,
    }
    _write_idempotent(assignment_root / "validated_results_manifest.json", validation)
    return validation


def validate_semantic_audit_results(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    batch_root = batch_root.resolve()
    audit_root = audit_root.resolve()
    controller_manifest_path = controller_manifest_path.resolve()
    assignment_root = assignment_root.resolve()
    prepared = prepare_semantic_audit_assignment(
        batch_root=batch_root,
        audit_root=audit_root,
        controller_manifest_path=controller_manifest_path,
        assignment_root=assignment_root,
    )
    items = _validate_semantic_exchange(
        batch_root=batch_root,
        audit_root=audit_root,
        controller_manifest_path=controller_manifest_path,
    )
    expected = {item.result_path for item in items}
    _exact_result_paths(
        result_root=audit_root / "results",
        expected=expected,
        label="Semantic-audit",
    )

    rows: list[dict[str, str]] = []
    for item in items:
        try:
            result = AdaptiveOriginalSemanticAuditResult.model_validate_json(
                item.result_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise AdaptiveOriginalAssignmentError(
                f"Semantic-audit result schema differs: {item.audit_packet_id}"
            ) from exc
        if (
            result.audit_packet_id != item.audit_packet_id
            or result.auditor_role != item.packet.auditor_role
        ):
            raise AdaptiveOriginalAssignmentError(
                "Semantic-audit result/packet binding differs: "
                f"{item.audit_packet_id}"
            )
        rows.append(
            {
                "audit_packet_id": item.audit_packet_id,
                "packet_sha256": item.packet_sha256,
                "result_sha256": sha256_file(item.result_path),
            }
        )
    validation = {
        "schema_version": 1,
        "protocol_version": SEMANTIC_ASSIGNMENT_PROTOCOL,
        "status": "complete_results_validated",
        "source_batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "audit_protocol_sha256": sha256_file(audit_root / "protocol.json"),
        "audit_worklist_sha256": sha256_file(audit_root / "worklist.json"),
        "assignment_sha256": prepared["assignment_sha256"],
        "exact_coverage": True,
        "read_only": True,
        "candidate_or_evaluation_modified": False,
        "feedback_memory_updated": False,
        "items": rows,
    }
    _write_idempotent(assignment_root / "validated_results_manifest.json", validation)
    return validation


def _write_receipt(
    *,
    path: Path,
    identity: dict[str, Any],
    imported: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        receipt = _read_json(path)
        if not isinstance(receipt, dict) or any(
            receipt.get(key) != value for key, value in identity.items()
        ):
            raise AdaptiveOriginalAssignmentError(
                f"Existing import receipt conflicts with this exchange: {path}"
            )
        if receipt.get("status") != "imported":
            raise AdaptiveOriginalAssignmentError(
                f"Existing import receipt is malformed: {path}"
            )
        return {**receipt, "idempotent_replay": True}
    receipt = {
        **identity,
        "status": "imported",
        "imported": imported,
        "idempotent_replay": False,
    }
    _write_idempotent(path, receipt)
    return receipt


def validate_and_import_semantic_audit(
    *,
    batch_root: Path,
    audit_root: Path,
    controller_manifest_path: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    validate_semantic_audit_results(
        batch_root=batch_root,
        audit_root=audit_root,
        controller_manifest_path=controller_manifest_path,
        assignment_root=assignment_root,
    )
    validation_path = assignment_root.resolve() / "validated_results_manifest.json"
    semantic_path = batch_root.resolve() / "semantic_audit.json"
    archive_root = batch_root.resolve() / "semantic_audit_archive"
    if semantic_path.exists() or archive_root.exists():
        if not semantic_path.is_file() or not archive_root.is_dir():
            raise AdaptiveOriginalAssignmentError(
                "Partial pre-existing semantic-audit admission state"
            )
        validate_semantic_audit_archive(batch_root)
    imported = import_semantic_audit_exchange(
        batch_root=batch_root,
        audit_root=audit_root,
        controller_manifest_path=controller_manifest_path,
    )
    identity = {
        "schema_version": 1,
        "protocol_version": SEMANTIC_ASSIGNMENT_PROTOCOL,
        "validated_results_manifest_sha256": sha256_file(validation_path),
    }
    return _write_receipt(
        path=assignment_root.resolve() / "import_receipt.json",
        identity=identity,
        imported=imported,
    )


def validate_and_import_original_evaluation(
    *,
    output_root: Path,
    batch_root: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    validate_original_evaluation_results(
        batch_root=batch_root,
        assignment_root=assignment_root,
    )
    validate_semantic_audit_archive(batch_root)

    output_root = output_root.resolve()
    batch_root = batch_root.resolve()
    assignment_root = assignment_root.resolve()
    items = _validate_original_batch(batch_root)
    expected_sources: list[tuple[Path, Path]] = []
    for item in items:
        expected_sources.append(
            (
                batch_root / "results" / "evaluation" / f"{item.packet_id}.json",
                output_root / "results" / "evaluation" / f"{item.packet_id}.json",
            )
        )
    expected_sources.append(
        (
            batch_root / "semantic_audit.json",
            output_root / "audits" / "semantic_audit.json",
        )
    )
    archive_root = batch_root / "semantic_audit_archive"
    expected_sources.extend(
        (
            source,
            output_root
            / "audits"
            / "semantic_audit_archive"
            / source.relative_to(archive_root),
        )
        for source in sorted(path for path in archive_root.rglob("*") if path.is_file())
    )
    for source, target in expected_sources:
        if target.exists() and (
            not target.is_file() or target.read_bytes() != source.read_bytes()
        ):
            raise AdaptiveOriginalAssignmentError(
                f"Conflicting original-evaluation import target: {target}"
            )

    imported = import_blind_results(output_root, batch_root)
    if imported != {"expected": len(items), "imported": len(items)}:
        raise AdaptiveOriginalAssignmentError(
            f"Original-evaluation importer returned unexpected counts: {imported}"
        )
    validation_path = assignment_root / "validated_results_manifest.json"
    identity = {
        "schema_version": 1,
        "protocol_version": ORIGINAL_ASSIGNMENT_PROTOCOL,
        "validated_results_manifest_sha256": sha256_file(validation_path),
    }
    return _write_receipt(
        path=assignment_root / "import_receipt.json",
        identity=identity,
        imported=imported,
    )
