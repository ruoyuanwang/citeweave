from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    validate_evaluation_result,
)
from .io import atomic_write_bytes, sha256_file
from .judge_protocol import (
    FeedbackPacket,
    FeedbackResult,
    canonical_json,
    validate_feedback_result,
)

PacketKind = Literal["feedback", "evaluation"]
ImportResults = Callable[..., dict[str, int]]

_KINDS: tuple[PacketKind, ...] = ("feedback", "evaluation")
_MANIFEST_KEYS = {
    "schema_version",
    "blind_exchange",
    "condition_identity_visible",
    "packets",
    "result_directories",
}
_ROW_KEYS = {"packet_id", "kind", "packet_sha256"}
_RESULT_DIRECTORIES = {
    "feedback": "results/feedback",
    "evaluation": "results/evaluation",
}

_HUMAN_PROXY_INSTRUCTIONS = [
    (
        "Act only as the constrained LLM-based Human Proxy in the formal "
        "human-machine interaction experiment, and only for the packet supplied."
    ),
    (
        "Use only anonymous_candidate, risk_notice, canonical_evidence, and "
        "allowed_evidence_ids visible in that packet. Do not use hidden information, "
        "outside knowledge, another packet, or the complete source artifact."
    ),
    (
        "Address only risk_notice.flagged_text. Do not independently inspect or "
        "change any unflagged part of the artifact."
    ),
    (
        "Return accept, reject, or revise with exactly one local replace_span, "
        "delete_span, or append_caveat operation. Any target and replacement must "
        "each be at most 500 characters; a replace/delete target must lie inside "
        "the exact flagged_text."
    ),
    (
        "Do not browse, retrieve, call APIs or tools, rerun analysis, inspect the "
        "workspace, change code/query/corpus/evidence, or rewrite the full artifact."
    ),
    (
        "If the visible evidence is insufficient, reject or append a short caveat; "
        "never invent a correction."
    ),
]

_EVALUATION_INSTRUCTIONS = [
    (
        "Act only as a read-only quality Judge. Use only canonical_evidence, "
        "anonymous_candidate, and allowed_evidence_ids visible in the supplied packet."
    ),
    (
        "Score whether the candidate is supported by the visible evidence. Cite only "
        "visible evidence IDs."
    ),
    (
        "Do not propose, encode, or apply any revision. Do not modify the candidate, "
        "pipeline, source artifacts, feedback memory, query, corpus, or evidence."
    ),
    (
        "Do not browse, call APIs, inspect unlisted workspace files, use hidden "
        "condition identities, or use outside knowledge."
    ),
]


class AdaptiveAssignmentError(ValueError):
    """Raised when an adaptive subagent exchange must fail closed."""


@dataclass(frozen=True)
class BatchItem:
    packet_id: str
    kind: PacketKind
    packet_path: Path
    packet_sha256: str
    packet: BaseModel


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_idempotent(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise AdaptiveAssignmentError(
                f"Refusing to overwrite different conductor artifact: {path}"
            )
        return
    atomic_write_bytes(path, payload)


def _copy_idempotent(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise AdaptiveAssignmentError(
                f"Refusing to overwrite different isolated packet: {target}"
            )
        return
    atomic_write_bytes(target, payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveAssignmentError(f"Cannot read valid JSON: {path}") from exc


def _content_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_embedded_content_hash(packet: BaseModel, kind: PacketKind) -> None:
    if kind == "feedback":
        assert isinstance(packet, FeedbackPacket)
        visible = {
            "sample_id": packet.sample_id,
            "judge_id": packet.judge_id,
            "rubric_version": packet.rubric_version,
            "canonical_evidence": packet.canonical_evidence,
            "anonymous_candidate": packet.anonymous_candidate,
            "allowed_evidence_ids": packet.allowed_evidence_ids,
            "risk_notice": packet.risk_notice,
            "permitted_interventions": packet.permitted_interventions,
            "human_proxy_constraints": packet.human_proxy_constraints,
        }
    else:
        assert isinstance(packet, AdaptiveEvaluationPacket)
        visible = {
            "sample_id": packet.sample_id,
            "judge_id": packet.judge_id,
            "rubric_version": packet.rubric_version,
            "canonical_evidence": packet.canonical_evidence,
            "anonymous_candidate": packet.anonymous_candidate,
            "allowed_evidence_ids": packet.allowed_evidence_ids,
        }
    if packet.content_sha256 != _content_sha(visible):
        raise AdaptiveAssignmentError(
            f"Embedded packet content hash differs: {packet.packet_id}"
        )


def validate_frozen_batch(batch_root: Path) -> tuple[dict[str, Any], list[BatchItem]]:
    manifest_path = batch_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise AdaptiveAssignmentError("Adaptive batch manifest keys differ")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("blind_exchange") is not True
        or manifest.get("condition_identity_visible") is not False
        or manifest.get("result_directories") != _RESULT_DIRECTORIES
        or not isinstance(manifest.get("packets"), list)
        or not manifest["packets"]
    ):
        raise AdaptiveAssignmentError("Adaptive batch manifest contract differs")

    items: list[BatchItem] = []
    seen: set[str] = set()
    for row in manifest["packets"]:
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise AdaptiveAssignmentError("Adaptive batch packet row keys differ")
        packet_id = row.get("packet_id")
        kind = row.get("kind")
        expected_sha = row.get("packet_sha256")
        if (
            not isinstance(packet_id, str)
            or packet_id in seen
            or kind not in _KINDS
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise AdaptiveAssignmentError("Malformed or duplicate adaptive packet row")
        seen.add(packet_id)
        packet_path = batch_root / "packets" / kind / f"{packet_id}.json"
        if not packet_path.is_file() or sha256_file(packet_path) != expected_sha:
            raise AdaptiveAssignmentError(
                f"Adaptive packet file/hash differs: {packet_id}"
            )
        model = FeedbackPacket if kind == "feedback" else AdaptiveEvaluationPacket
        try:
            packet = model.model_validate_json(packet_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AdaptiveAssignmentError(
                f"Adaptive packet schema differs: {packet_id}"
            ) from exc
        if packet.packet_id != packet_id:
            raise AdaptiveAssignmentError(
                f"Adaptive packet identity differs: {packet_id}"
            )
        _validate_embedded_content_hash(packet, kind)
        items.append(
            BatchItem(
                packet_id=packet_id,
                kind=kind,
                packet_path=packet_path,
                packet_sha256=expected_sha,
                packet=packet,
            )
        )

    if [item.packet_id for item in items] != sorted(seen):
        raise AdaptiveAssignmentError(
            "Adaptive packet manifest must use deterministic packet-ID order"
        )
    actual_packets = {
        path.relative_to(batch_root).as_posix()
        for path in (batch_root / "packets").rglob("*")
        if path.is_file()
    }
    expected_packets = {
        item.packet_path.relative_to(batch_root).as_posix() for item in items
    }
    if actual_packets != expected_packets:
        raise AdaptiveAssignmentError(
            "Adaptive packet directory coverage differs from frozen manifest"
        )
    return manifest, items


def _result_schema(kind: PacketKind) -> dict[str, Any]:
    model = FeedbackResult if kind == "feedback" else AdaptiveEvaluationResult
    return model.model_json_schema()


def _assignment(
    *,
    kind: PacketKind,
    batch_root: Path,
    assignment_root: Path,
    items: list[BatchItem],
) -> dict[str, Any]:
    work = []
    for item in items:
        isolated_packet = (
            assignment_root / kind / "packets" / f"{item.packet_id}.json"
        )
        result_path = batch_root / "results" / kind / f"{item.packet_id}.json"
        work.append(
            {
                "packet_id": item.packet_id,
                "packet_file": str(isolated_packet.resolve()),
                "packet_sha256": item.packet_sha256,
                "result_file": str(result_path.resolve()),
            }
        )
    is_feedback = kind == "feedback"
    return {
        "schema_version": 1,
        "assignment_role": (
            "constrained_llm_human_proxy"
            if is_feedback
            else "independent_read_only_evaluation_judge"
        ),
        "packet_kind": kind,
        "condition_identity_visible": False,
        "only_listed_packets_are_visible": True,
        "may_read_unlisted_workspace_files": False,
        "may_modify_pipeline_or_source_artifacts": False,
        "may_directly_modify_candidate": False,
        "may_return_one_scoped_edit": is_feedback,
        "may_directly_update_feedback_memory": False,
        "validated_result_eligible_for_feedback_memory": is_feedback,
        "instructions": (
            _HUMAN_PROXY_INSTRUCTIONS
            if is_feedback
            else _EVALUATION_INSTRUCTIONS
        ),
        "result_schema": _result_schema(kind),
        "output_contract": (
            "Return exactly one JSON object per listed packet at its unique "
            "result_file. Do not create any other file."
        ),
        "write_scope": [row["result_file"] for row in work],
        "items": work,
    }


def prepare_assignments(
    *,
    batch_root: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    _manifest, items = validate_frozen_batch(batch_root)
    assignments: dict[str, Any] = {}
    for kind in _KINDS:
        role_items = [item for item in items if item.kind == kind]
        for item in role_items:
            _copy_idempotent(
                item.packet_path,
                assignment_root / kind / "packets" / f"{item.packet_id}.json",
            )
        expected_isolated = {
            f"packets/{item.packet_id}.json" for item in role_items
        }
        kind_root = assignment_root / kind
        actual_isolated = (
            {
                path.relative_to(kind_root).as_posix()
                for path in (kind_root / "packets").rglob("*")
                if path.is_file()
            }
            if (kind_root / "packets").exists()
            else set()
        )
        if actual_isolated != expected_isolated:
            raise AdaptiveAssignmentError(
                f"{kind} isolated packet coverage differs"
            )
        assignment = _assignment(
            kind=kind,
            batch_root=batch_root,
            assignment_root=assignment_root,
            items=role_items,
        )
        assignment_path = kind_root / "assignment.json"
        _write_idempotent(assignment_path, assignment)
        assignments[kind] = {
            "assignment_path": str(assignment_path.resolve()),
            "assignment_sha256": sha256_file(assignment_path),
            "packet_count": len(role_items),
            "status": "ready" if role_items else "no_packets_in_batch",
        }
    return {
        "status": "assignments_ready",
        "batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "assignments": assignments,
    }


def validate_complete_results(
    *,
    batch_root: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    _manifest, items = validate_frozen_batch(batch_root)
    prepared = prepare_assignments(
        batch_root=batch_root,
        assignment_root=assignment_root,
    )
    expected_result_paths = {
        (batch_root / "results" / item.kind / f"{item.packet_id}.json").resolve()
        for item in items
    }
    actual_result_paths = (
        {
            path.resolve()
            for path in (batch_root / "results").rglob("*")
            if path.is_file()
        }
        if (batch_root / "results").exists()
        else set()
    )
    if actual_result_paths != expected_result_paths:
        missing = sorted(str(path) for path in expected_result_paths - actual_result_paths)
        extra = sorted(str(path) for path in actual_result_paths - expected_result_paths)
        raise AdaptiveAssignmentError(
            f"Adaptive result coverage differs: missing={missing}, extra={extra}"
        )

    validated = []
    for item in items:
        result_path = (
            batch_root / "results" / item.kind / f"{item.packet_id}.json"
        )
        try:
            if item.kind == "feedback":
                assert isinstance(item.packet, FeedbackPacket)
                result = FeedbackResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                validate_feedback_result(result, item.packet)
            else:
                assert isinstance(item.packet, AdaptiveEvaluationPacket)
                result = AdaptiveEvaluationResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                validate_evaluation_result(result, item.packet)
        except Exception as exc:
            raise AdaptiveAssignmentError(
                f"Adaptive result schema/packet binding differs: {item.packet_id}"
            ) from exc
        validated.append(
            {
                "packet_id": item.packet_id,
                "kind": item.kind,
                "packet_sha256": item.packet_sha256,
                "result_path": result_path.relative_to(batch_root).as_posix(),
                "result_sha256": sha256_file(result_path),
            }
        )

    validation = {
        "schema_version": 1,
        "status": "complete_results_validated",
        "batch_manifest_sha256": sha256_file(batch_root / "manifest.json"),
        "assignment_sha256": {
            kind: prepared["assignments"][kind]["assignment_sha256"]
            for kind in _KINDS
        },
        "exact_coverage": True,
        "packet_and_result_schema_validated": True,
        "items": validated,
    }
    validation_path = batch_root / "validated_results_manifest.json"
    _write_idempotent(validation_path, validation)
    return validation


def validate_and_import(
    *,
    run_root: Path,
    batch_root: Path,
    assignment_root: Path,
    importer: ImportResults,
) -> dict[str, Any]:
    validation = validate_complete_results(
        batch_root=batch_root,
        assignment_root=assignment_root,
    )
    validation_path = batch_root / "validated_results_manifest.json"
    receipt_path = batch_root / "conductor_import_receipt.json"
    receipt_identity = {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "batch_manifest_sha256": validation["batch_manifest_sha256"],
        "validated_results_manifest_sha256": sha256_file(validation_path),
    }
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if not isinstance(receipt, dict) or any(
            receipt.get(key) != value for key, value in receipt_identity.items()
        ):
            raise AdaptiveAssignmentError(
                "Existing adaptive import receipt conflicts with this exchange"
            )
        if receipt.get("status") != "imported" or not isinstance(
            receipt.get("imported"), dict
        ):
            raise AdaptiveAssignmentError("Existing adaptive import receipt is malformed")
        return {**receipt, "idempotent_replay": True}

    imported = importer(run_root, batch_root, require_complete=True)
    expected_counts = {
        kind: sum(item["kind"] == kind for item in validation["items"])
        for kind in _KINDS
    }
    if imported != expected_counts:
        raise AdaptiveAssignmentError(
            f"Existing adaptive importer returned unexpected counts: {imported}"
        )
    receipt = {
        **receipt_identity,
        "status": "imported",
        "imported": imported,
        "idempotent_replay": False,
    }
    _write_idempotent(receipt_path, receipt)
    return receipt
