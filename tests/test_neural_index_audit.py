from __future__ import annotations

import numpy as np

from citeweave.io import sha256_file, write_json
from citeweave.neural_index_audit import validate_neural_embedding_indexes


def test_validate_neural_embedding_indexes_accepts_semantic_v2(tmp_path) -> None:
    index_root = tmp_path / "dataset" / "network_large"
    index_root.mkdir(parents=True)
    embedding_path = index_root / "embeddings.npy"
    np.save(embedding_path, np.asarray([[1.0, 0.0]], dtype=np.float32))
    index_manifest_path = index_root / "manifest.json"
    index_manifest = {
        "status": "complete",
        "row_count": 1,
        "document_serialization": "semantic_entity_relation_json_v2",
        "cache_identity": {
            "embedding_document_schema": "semantic_entity_relation_json_v2"
        },
        "embeddings_sha256": sha256_file(embedding_path),
    }
    write_json(index_manifest_path, index_manifest)
    neural_manifest_path = tmp_path / "neural_dense_manifest.json"
    neural_manifest = {
        "embedding_indexes": [
            {
                "dataset_id": "dataset",
                "network": "network",
                "scale": "large",
                "manifest_path": str(index_manifest_path),
                "manifest_sha256": sha256_file(index_manifest_path),
                "embeddings_sha256": sha256_file(embedding_path),
                "row_count": 1,
            }
        ]
    }
    assert not validate_neural_embedding_indexes(
        neural_manifest=neural_manifest,
        neural_manifest_path=neural_manifest_path,
    )


def test_validate_neural_embedding_indexes_rejects_old_schema(tmp_path) -> None:
    index_root = tmp_path / "dataset" / "network_small"
    index_root.mkdir(parents=True)
    embedding_path = index_root / "embeddings.npy"
    np.save(embedding_path, np.asarray([[1.0, 0.0]], dtype=np.float32))
    index_manifest_path = index_root / "manifest.json"
    write_json(
        index_manifest_path,
        {
            "status": "complete",
            "row_count": 1,
            "document_serialization": "canonical_sorted_compact_json",
            "cache_identity": {},
            "embeddings_sha256": sha256_file(embedding_path),
        },
    )
    neural_manifest = {
        "embedding_indexes": [
            {
                "dataset_id": "dataset",
                "network": "network",
                "scale": "small",
                "manifest_path": str(index_manifest_path),
                "manifest_sha256": sha256_file(index_manifest_path),
                "embeddings_sha256": sha256_file(embedding_path),
                "row_count": 1,
            }
        ]
    }
    reasons = validate_neural_embedding_indexes(
        neural_manifest=neural_manifest,
        neural_manifest_path=tmp_path / "neural_dense_manifest.json",
    )
    assert "neural_embedding_document_schema_mismatch:dataset/network/small" in reasons
    assert "neural_embedding_scale_reuse_missing:dataset/network/small" in reasons
