from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    FormalAdaptiveCase,
    load_cases,
    validate_evaluation_result,
    validate_formal_topic_sequence,
)
from .io import atomic_write_bytes, sha256_file, write_json
from .judge_protocol import canonical_json, collect_reference_ids, scan_condition_leaks

ExperimentMode = Literal["formal", "development_calibration"]

REVIEW_CONDITIONS = ("always_review", "static_review", "adaptive_review")
BASELINE_CONDITION = "baseline_original"
PROTOCOL_VERSION = "adaptive-original-evaluation-v1"
SEMANTIC_AUDIT_PROTOCOL_VERSION = "adaptive-original-semantic-audit-v1"


def _sha(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different {label}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _write_immutable_json(path: Path, value: Any, *, label: str) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    _write_immutable(path, payload, label=label)


def build_original_evaluation_packet(
    case: FormalAdaptiveCase,
    *,
    rubric_version: str,
    seed: int,
) -> AdaptiveEvaluationPacket:
    """Create one neutral packet for the untouched, pre-review candidate."""

    visible = {
        "sample_id": case.sample_id,
        "judge_id": "evaluation",
        "rubric_version": rubric_version,
        "canonical_evidence": case.canonical_evidence,
        "anonymous_candidate": case.anonymous_candidate,
        "allowed_evidence_ids": collect_reference_ids(case.canonical_evidence),
    }
    identity = {
        **visible,
        "evaluation_phase_hash": _sha(BASELINE_CONDITION),
        "seed": seed,
    }
    packet = AdaptiveEvaluationPacket(
        packet_id=f"EP{_sha(identity)[:20]}",
        content_sha256=_sha(visible),
        **visible,
    )
    leaks = scan_condition_leaks(
        packet.model_dump(mode="json"),
        [BASELINE_CONDITION, *REVIEW_CONDITIONS],
    )
    if leaks:
        raise RuntimeError(f"Condition-name leakage in original evaluation packet: {leaks}")
    return packet


def prepare_original_evaluation(
    *,
    cases_path: Path,
    reference_registry: Path,
    output_root: Path,
    experiment_mode: ExperimentMode = "formal",
    rubric_version: str = "adaptive-evaluation-v1",
    seed: int = 42,
) -> dict[str, Any]:
    """Materialize the immutable, one-packet-per-original-case evaluation set."""

    cases_path = cases_path.resolve()
    reference_registry = reference_registry.resolve()
    output_root = output_root.resolve()
    cases = load_cases(cases_path)
    topics = validate_formal_topic_sequence(
        cases,
        reference_registry,
        experiment_mode=experiment_mode,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        packet = build_original_evaluation_packet(
            case,
            rubric_version=rubric_version,
            seed=seed,
        )
        relative = Path("packets") / "evaluation" / f"{packet.packet_id}.json"
        packet_path = output_root / relative
        _write_immutable_json(
            packet_path,
            packet.model_dump(mode="json"),
            label="original evaluation packet",
        )
        rows.append(
            {
                "packet_id": packet.packet_id,
                "sample_id": case.sample_id,
                "dataset_id": case.dataset_id,
                "topic_role": case.topic_role,
                "artifact_type": case.artifact_type,
                "candidate_sha256": _sha(case.anonymous_candidate),
                "packet_path": relative.as_posix(),
                "packet_sha256": sha256_file(packet_path),
            }
        )

    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Original evaluation requires one unique packet per sample_id")
    if len({row["packet_id"] for row in rows}) != len(rows):
        raise ValueError("Original evaluation packet IDs must be unique")
    rows.sort(key=lambda row: row["packet_id"])

    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_target": "untouched_pre_intervention_original_candidate",
        "condition_identity_visible_to_judge": False,
        "judge_role": "evaluation_only",
        "judge_may_modify_artifacts": False,
        "evaluation_updates_feedback_memory": False,
        "formal_results_used": experiment_mode == "formal",
        "experiment_mode": experiment_mode,
        "rubric_version": rubric_version,
        "seed": seed,
        "cases_path": str(cases_path),
        "cases_sha256": sha256_file(cases_path),
        "reference_registry": str(reference_registry),
        "reference_registry_sha256": sha256_file(reference_registry),
        "topic_sequence": topics,
        "items": rows,
    }
    _write_immutable_json(
        output_root / "manifest.json",
        manifest,
        label="original evaluation manifest",
    )
    return manifest


def _load_verified_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("judge_may_modify_artifacts") is not False
        or manifest.get("evaluation_updates_feedback_memory") is not False
    ):
        raise RuntimeError("Original evaluation manifest violates the frozen protocol")
    seen: set[str] = set()
    for row in manifest.get("items") or []:
        packet_id = str(row["packet_id"])
        if packet_id in seen:
            raise RuntimeError(f"Duplicate original evaluation packet: {packet_id}")
        seen.add(packet_id)
        packet_path = output_root / row["packet_path"]
        if not packet_path.is_file() or sha256_file(packet_path) != row["packet_sha256"]:
            raise RuntimeError(f"Immutable original evaluation packet mismatch: {packet_path}")
        AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
    return manifest


def export_blind_batch(output_root: Path, batch_root: Path) -> dict[str, Any]:
    """Export only neutral packets and hashes; topic roles and phase are withheld."""

    output_root = output_root.resolve()
    batch_root = batch_root.resolve()
    if batch_root.exists():
        raise FileExistsError(f"Refusing to overwrite blind batch: {batch_root}")
    source_manifest = _load_verified_manifest(output_root)
    rows: list[dict[str, str]] = []
    for item in source_manifest["items"]:
        source = output_root / item["packet_path"]
        target = batch_root / "packets" / "evaluation" / source.name
        _write_immutable(target, source.read_bytes(), label="blind evaluation packet")
        rows.append(
            {
                "packet_id": item["packet_id"],
                "kind": "evaluation",
                "packet_sha256": sha256_file(target),
            }
        )
    manifest = {
        "schema_version": 1,
        "blind_exchange": True,
        "condition_identity_visible": False,
        "packets": rows,
        "result_directory": "results/evaluation",
    }
    write_json(batch_root / "manifest.json", manifest)
    return manifest


def import_blind_results(output_root: Path, batch_root: Path) -> dict[str, int]:
    """Validate and archive read-only judge decisions without touching policy memory."""

    output_root = output_root.resolve()
    batch_root = batch_root.resolve()
    source_manifest = _load_verified_manifest(output_root)
    source_by_id = {row["packet_id"]: row for row in source_manifest["items"]}
    batch_manifest = json.loads(
        (batch_root / "manifest.json").read_text(encoding="utf-8")
    )
    batch_rows = batch_manifest.get("packets") or []
    batch_by_id = {row["packet_id"]: row for row in batch_rows}
    if len(batch_by_id) != len(batch_rows):
        raise ValueError("Blind batch contains duplicate packet IDs")
    if set(batch_by_id) != set(source_by_id):
        raise ValueError("Blind batch packet set differs from original evaluation manifest")

    audit_path = batch_root / "semantic_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Independent semantic audit is required before result import: "
            f"{audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = audit.get("items")
    if (
        audit.get("all_consistent") is not True
        or audit.get("packets") != len(source_by_id)
        or not isinstance(audit_rows, list)
    ):
        raise ValueError("Semantic audit did not accept the complete blind result set")
    audit_by_id = {
        str(row.get("packet_id")): row
        for row in audit_rows
        if isinstance(row, dict)
    }
    if len(audit_by_id) != len(audit_rows) or set(audit_by_id) != set(source_by_id):
        raise ValueError("Semantic audit packet set differs from the blind result set")
    for packet_id, row in audit_by_id.items():
        if (
            row.get("semantic_consistent") is not True
            or row.get("decision_defensible") is not True
        ):
            raise ValueError(f"Semantic audit rejected result {packet_id}")
    if source_manifest.get("formal_results_used") is True:
        from .adaptive_semantic_audit import validate_semantic_audit_archive

        if (
            audit.get("protocol_version") != SEMANTIC_AUDIT_PROTOCOL_VERSION
            or audit.get("read_only") is not True
            or audit.get("judge_modified_artifacts") is not False
        ):
            raise ValueError(
                "Formal original evaluation requires the bound read-only "
                "semantic-audit protocol"
            )
        for packet_id, row in audit_by_id.items():
            source_row = source_by_id[packet_id]
            packet_path = output_root / source_row["packet_path"]
            result_path = (
                batch_root / "results" / "evaluation" / f"{packet_id}.json"
            )
            if (
                row.get("source_packet_sha256") != sha256_file(packet_path)
                or row.get("source_result_sha256") != sha256_file(result_path)
            ):
                raise ValueError(
                    f"Formal semantic audit hash binding differs: {packet_id}"
                )
        validate_semantic_audit_archive(batch_root)

    imported = 0
    for packet_id, source_row in source_by_id.items():
        batch_row = batch_by_id[packet_id]
        if batch_row.get("kind") != "evaluation":
            raise ValueError(f"Non-evaluation result is prohibited: {packet_id}")
        packet_path = output_root / source_row["packet_path"]
        if batch_row.get("packet_sha256") != sha256_file(packet_path):
            raise ValueError(f"Blind packet hash differs: {packet_id}")
        result_path = batch_root / "results" / "evaluation" / f"{packet_id}.json"
        if not result_path.is_file():
            continue
        packet = AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        result = AdaptiveEvaluationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        validate_evaluation_result(result, packet)
        target = output_root / "results" / "evaluation" / result_path.name
        _write_immutable(target, result_path.read_bytes(), label="original evaluation result")
        imported += 1
    _write_immutable(
        output_root / "audits" / "semantic_audit.json",
        audit_path.read_bytes(),
        label="original evaluation semantic audit",
    )
    if source_manifest.get("formal_results_used") is True:
        archive_root = batch_root / "semantic_audit_archive"
        for source in sorted(
            path for path in archive_root.rglob("*") if path.is_file()
        ):
            _write_immutable(
                output_root
                / "audits"
                / "semantic_audit_archive"
                / source.relative_to(archive_root),
                source.read_bytes(),
                label="original evaluation semantic audit archive",
            )
    return {"expected": len(source_by_id), "imported": imported}


def finalize_original_evaluation(output_root: Path) -> dict[str, Any]:
    """Score quality/error rates by topic after all independent results arrive."""

    output_root = output_root.resolve()
    manifest = _load_verified_manifest(output_root)
    expected_result_names = {
        f"{row['packet_id']}.json" for row in manifest["items"]
    }
    result_root = output_root / "results" / "evaluation"
    actual_result_names = (
        {path.name for path in result_root.iterdir()} if result_root.is_dir() else set()
    )
    if actual_result_names != expected_result_names:
        raise RuntimeError(
            "Original evaluation result set differs: "
            f"missing={sorted(expected_result_names - actual_result_names)}, "
            f"extra={sorted(actual_result_names - expected_result_names)}"
        )
    by_topic: dict[str, list[dict[str, Any]]] = {
        topic: [] for topic in manifest["topic_sequence"]
    }
    result_hashes: list[dict[str, str]] = []
    for row in manifest["items"]:
        result_path = (
            output_root / "results" / "evaluation" / f"{row['packet_id']}.json"
        )
        if not result_path.is_file():
            raise RuntimeError(f"Missing original evaluation result: {row['packet_id']}")
        packet = AdaptiveEvaluationPacket.model_validate_json(
            (output_root / row["packet_path"]).read_text(encoding="utf-8")
        )
        result = AdaptiveEvaluationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        validate_evaluation_result(result, packet)
        result_hashes.append(
            {
                "packet_id": row["packet_id"],
                "result_sha256": sha256_file(result_path),
            }
        )
        by_topic[row["dataset_id"]].append(
            {
                "sample_id": row["sample_id"],
                "artifact_type": row["artifact_type"],
                "decision": result.decision,
            }
        )

    topic_metrics: dict[str, Any] = {}
    for topic, rows in by_topic.items():
        if not rows:
            raise RuntimeError(f"No original evaluation rows for topic: {topic}")
        passed = sum(row["decision"] == "pass" for row in rows)
        failed = len(rows) - passed
        counts = {
            "items": len(rows),
            "review_requests": 0,
            "final_quality_passed": passed,
            "auto_accepts": len(rows),
            "unsafe_auto_accepts": failed,
            "quality_errors": failed,
            "final_quality_pass_rate": passed / len(rows),
            "quality_error_rate": failed / len(rows),
        }
        artifact_types: dict[str, Any] = {}
        for artifact_type in sorted({row["artifact_type"] for row in rows}):
            subset = [row for row in rows if row["artifact_type"] == artifact_type]
            subset_passed = sum(row["decision"] == "pass" for row in subset)
            artifact_types[artifact_type] = {
                "items": len(subset),
                "quality_passed": subset_passed,
                "quality_errors": len(subset) - subset_passed,
                "quality_error_rate": (len(subset) - subset_passed) / len(subset),
            }
        topic_metrics[topic] = {
            "topic_id": topic,
            "condition": BASELINE_CONDITION,
            "counts": counts,
            "by_artifact_type": artifact_types,
        }
        _write_immutable_json(
            output_root / "topic_counts" / f"{topic}.json",
            topic_metrics[topic],
            label="original evaluation topic metrics",
        )

    result_manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "results": result_hashes,
    }
    result_manifest_path = output_root / "result_manifest.json"
    _write_immutable_json(
        result_manifest_path,
        result_manifest,
        label="original evaluation result manifest",
    )

    all_rows = [row for rows in by_topic.values() for row in rows]
    total_passed = sum(row["decision"] == "pass" for row in all_rows)
    result = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "condition": BASELINE_CONDITION,
        "items": len(all_rows),
        "quality_passed": total_passed,
        "quality_errors": len(all_rows) - total_passed,
        "quality_pass_rate": total_passed / len(all_rows),
        "quality_error_rate": (len(all_rows) - total_passed) / len(all_rows),
        "evaluation_feedback_leakage": False,
        "judge_modified_artifacts": False,
        "topics": topic_metrics,
        "computed_at": datetime.now(UTC).isoformat(),
    }
    # computed_at is intentionally excluded from immutable result comparison.
    metrics_path = output_root / "metrics.json"
    if metrics_path.exists():
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        comparable_prior = {key: value for key, value in prior.items() if key != "computed_at"}
        comparable_new = {key: value for key, value in result.items() if key != "computed_at"}
        if comparable_prior != comparable_new:
            raise RuntimeError(f"Refusing to overwrite different metrics: {metrics_path}")
        return prior
    write_json(metrics_path, result)
    return result
