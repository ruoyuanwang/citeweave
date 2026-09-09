from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io import read_json, sha256_file


def _resolve_artifact_path(path_value: str, *, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.is_file():
        return path
    return manifest_path.parent / path


def validate_neural_embedding_indexes(
    *,
    neural_manifest: dict[str, Any],
    neural_manifest_path: Path,
    expected_document_schema: str = "semantic_entity_relation_json_v2",
) -> list[str]:
    reasons: list[str] = []
    indexes = neural_manifest.get("embedding_indexes") or []
    if not indexes:
        return ["neural_embedding_indexes_missing"]
    identities: set[tuple[str, str, str]] = set()
    for artifact in indexes:
        identity = (
            str(artifact.get("dataset_id")),
            str(artifact.get("network")),
            str(artifact.get("scale")),
        )
        label = "/".join(identity)
        if identity in identities:
            reasons.append(f"neural_embedding_index_duplicate:{label}")
            continue
        identities.add(identity)
        manifest_value = artifact.get("manifest_path")
        if not isinstance(manifest_value, str):
            reasons.append(f"neural_embedding_index_manifest_path_missing:{label}")
            continue
        index_manifest_path = _resolve_artifact_path(
            manifest_value, manifest_path=neural_manifest_path
        )
        if not index_manifest_path.is_file():
            reasons.append(f"neural_embedding_index_manifest_missing:{label}")
            continue
        if sha256_file(index_manifest_path) != artifact.get("manifest_sha256"):
            reasons.append(f"neural_embedding_index_manifest_hash_mismatch:{label}")
            continue
        index_manifest = read_json(index_manifest_path)
        if index_manifest.get("status") != "complete":
            reasons.append(f"neural_embedding_index_incomplete:{label}")
        if index_manifest.get("document_serialization") != expected_document_schema:
            reasons.append(f"neural_embedding_document_schema_mismatch:{label}")
        if (
            (index_manifest.get("cache_identity") or {}).get(
                "embedding_document_schema"
            )
            != expected_document_schema
        ):
            reasons.append(f"neural_embedding_cache_identity_schema_mismatch:{label}")
        if index_manifest.get("row_count") != artifact.get("row_count"):
            reasons.append(f"neural_embedding_index_row_count_mismatch:{label}")
        if index_manifest.get("embeddings_sha256") != artifact.get(
            "embeddings_sha256"
        ):
            reasons.append(f"neural_embedding_index_embedding_hash_mismatch:{label}")
        embedding_path = index_manifest_path.parent / "embeddings.npy"
        if not embedding_path.is_file() or sha256_file(embedding_path) != index_manifest.get(
            "embeddings_sha256"
        ):
            reasons.append(f"neural_embedding_file_missing_or_hash_mismatch:{label}")
        reuse = index_manifest.get("exact_embedding_reuse")
        if identity[2] in {"small", "medium"}:
            if not isinstance(reuse, dict):
                reasons.append(f"neural_embedding_scale_reuse_missing:{label}")
            elif reuse.get("reused_rows", 0) + reuse.get(
                "encoded_rows", 0
            ) != index_manifest.get("row_count"):
                reasons.append(f"neural_embedding_scale_reuse_count_mismatch:{label}")
        elif reuse is not None:
            reasons.append(f"neural_embedding_unexpected_large_reuse:{label}")
    return reasons


def validate_neural_runtime_amendments(
    *,
    representation_amendment_path: Path,
    representation_freeze_path: Path,
    execution_scope_amendment_path: Path,
    execution_scope_freeze_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    paths = (
        representation_amendment_path,
        representation_freeze_path,
        execution_scope_amendment_path,
        execution_scope_freeze_path,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {}, [f"neural_runtime_amendment_missing:{path}" for path in missing]
    representation = yaml.safe_load(
        representation_amendment_path.read_text(encoding="utf-8")
    )
    representation_freeze = read_json(representation_freeze_path)
    scope = yaml.safe_load(execution_scope_amendment_path.read_text(encoding="utf-8"))
    scope_freeze = read_json(execution_scope_freeze_path)
    representation_hash = sha256_file(representation_amendment_path)
    scope_hash = sha256_file(execution_scope_amendment_path)
    reasons = []
    if representation_freeze.get("sha256") != representation_hash:
        reasons.append("neural_runtime_representation_freeze_hash_mismatch")
    if representation.get("representation", {}).get("id") != (
        "semantic_entity_relation_json_v2"
    ):
        reasons.append("neural_runtime_representation_schema_mismatch")
    if representation.get("observed_before_freeze", {}).get(
        "formal_outcome_file_count"
    ) != 0:
        reasons.append("neural_runtime_representation_not_pre_outcome")
    builder_spec = representation.get("implementation") or {}
    builder_path = Path(str(builder_spec.get("builder_path") or ""))
    if not builder_path.is_file() or sha256_file(builder_path) != builder_spec.get(
        "builder_sha256"
    ):
        reasons.append("neural_runtime_builder_hash_mismatch")
    if scope_freeze.get("sha256") != scope_hash:
        reasons.append("neural_runtime_scope_freeze_hash_mismatch")
    if scope.get("prior_semantic_representation_amendment_sha256") != representation_hash:
        reasons.append("neural_runtime_scope_chain_mismatch")
    if scope.get("formal_outcome_file_count") != 0:
        reasons.append("neural_runtime_scope_not_pre_outcome")
    registered = scope.get("correction", {}).get("registered_execution_scope", {})
    audit_path = execution_scope_amendment_path.parent / str(
        registered.get("artifact") or ""
    )
    if not audit_path.is_file() or sha256_file(audit_path) != registered.get("sha256"):
        reasons.append("neural_runtime_scope_audit_hash_mismatch")
    if (
        registered.get("target_rows") != registered.get("reused_rows")
        or registered.get("newly_encoded_rows") != 0
        or registered.get("reuse_fraction") != 1.0
    ):
        reasons.append("neural_runtime_scope_reuse_claim_mismatch")
    return {
        "representation_amendment_path": str(
            representation_amendment_path.resolve()
        ),
        "representation_amendment_sha256": representation_hash,
        "execution_scope_amendment_path": str(execution_scope_amendment_path.resolve()),
        "execution_scope_amendment_sha256": scope_hash,
        "document_schema": representation.get("representation", {}).get("id"),
        "registered_execution_scope": registered,
    }, reasons
