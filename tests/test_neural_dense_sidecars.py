from __future__ import annotations

import hashlib
import json

import networkx as nx
import numpy as np

from citeweave.neural_dense_runtime import OnnxSentenceEncoder
from scripts.build_neural_dense_sidecars import (
    _build_group_contexts,
    _embedding_cache,
    _embedding_document,
    _merge_rankings,
    _prepare_exact_embedding_reuse,
)


def test_neural_dense_streaming_ranking_is_stable_and_bounded() -> None:
    rows = [
        {"evidence_id": "e1"},
        {"evidence_id": "e2"},
        {"evidence_id": "e3"},
    ]
    first = _merge_rankings([], scores=np.array([0.5, 0.9, 0.5]), rows=rows, limit=2)
    assert next(row[2]["evidence_id"] for row in first) == "e2"
    second = _merge_rankings(
        first,
        scores=np.array([0.95]),
        rows=[{"evidence_id": "e4"}],
        limit=2,
    )
    assert [row[2]["evidence_id"] for row in second] == ["e4", "e2"]


class _FakeEmbeddingModel:
    def __init__(self, *, reject_documents: bool = False) -> None:
        self.reject_documents = reject_documents
        self.encoded_documents: list[str] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        if self.reject_documents and any(text.startswith("{") for text in texts):
            raise AssertionError("Completed document batches should come from checkpoint")
        self.encoded_documents.extend(text for text in texts if text.startswith("{"))
        return np.asarray(
            [[float(len(text) % 7), 1.0] for text in texts], dtype=np.float32
        )


class _FakeTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.asarray([[len(text), 1] for text in texts]),
            "attention_mask": np.ones((len(texts), 2), dtype=np.int64),
        }


class _FakeOnnxSession:
    def run(
        self, _: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        values = feed["input_ids"][:, 0].astype(np.float32)
        return [np.column_stack([values, np.ones_like(values)])]


def test_onnx_encoder_batches_and_normalizes() -> None:
    encoder = object.__new__(OnnxSentenceEncoder)
    encoder._tokenizer = _FakeTokenizer()
    encoder._session = _FakeOnnxSession()
    encoder.input_names = {"input_ids", "attention_mask"}
    encoder.max_seq_length = 8
    embeddings = encoder.encode(
        ["a", "abcd", "xy"], batch_size=2, normalize_embeddings=True
    )
    assert embeddings.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_group_checkpoint_resumes_without_reencoding_documents(tmp_path) -> None:
    graph = nx.Graph()
    graph.add_node("a", label="Alpha", importance=2.0)
    graph.add_node("b", label="Beta", importance=1.0)
    graph.add_edge("a", "b", evidence_id="edge-ab", weight=3.0)
    tasks = [{"item_id": "task-1", "question": "Which edge?"}]
    checkpoint = tmp_path / "checkpoint.json"
    first = _build_group_contexts(
        model=_FakeEmbeddingModel(),
        graph=graph,
        network="keyword_cooccurrence",
        tasks=tasks,
        model_name="test-model",
        model_revision="revision",
        record_budget=2,
        batch_size=1,
        checkpoint_path=checkpoint,
        checkpoint_every_batches=1,
    )
    second = _build_group_contexts(
        model=_FakeEmbeddingModel(reject_documents=True),
        graph=graph,
        network="keyword_cooccurrence",
        tasks=tasks,
        model_name="test-model",
        model_revision="revision",
        record_budget=2,
        batch_size=1,
        checkpoint_path=checkpoint,
        checkpoint_every_batches=1,
    )
    assert first == second


def test_embedding_cache_reuses_documents_across_rankings(tmp_path) -> None:
    graph = nx.Graph()
    graph.add_node("a", label="Alpha", importance=2.0)
    graph.add_node("b", label="Beta", importance=1.0)
    graph.add_edge("a", "b", evidence_id="edge-ab", weight=3.0)
    tasks = [{"item_id": "task-1", "question": "Which edge?"}]
    cache_path = tmp_path / "embedding-cache"
    identity = {"model": "test-model", "runtime": "test"}
    first = _build_group_contexts(
        model=_FakeEmbeddingModel(),
        graph=graph,
        network="keyword_cooccurrence",
        tasks=tasks,
        model_name="test-model",
        model_revision="revision",
        record_budget=2,
        batch_size=1,
        embedding_cache_path=cache_path,
        embedding_cache_identity=identity,
        checkpoint_every_batches=1,
    )
    second = _build_group_contexts(
        model=_FakeEmbeddingModel(reject_documents=True),
        graph=graph,
        network="keyword_cooccurrence",
        tasks=tasks,
        model_name="test-model",
        model_revision="revision",
        record_budget=2,
        batch_size=1,
        embedding_cache_path=cache_path,
        embedding_cache_identity=identity,
        checkpoint_every_batches=1,
    )
    assert first == second


def test_exact_large_scale_reuse_only_encodes_unmatched_rows(tmp_path) -> None:
    identity = {"model": "test-model", "runtime": "test"}
    source_rows = [
        {
            "evidence_id": "network:node:shared-a",
            "node_id": "a",
            "label": "Alpha",
            "importance": 3,
        },
        {
            "evidence_id": "network:node:large-only",
            "node_id": "large",
            "label": "Large",
            "importance": 2,
        },
        {
            "evidence_id": "network:node:shared-b",
            "node_id": "b",
            "label": "Beta",
            "importance": 1,
        },
    ]
    source_ids_hash = hashlib.sha256(
        json.dumps(
            [row["evidence_id"] for row in source_rows], separators=(",", ":")
        ).encode()
    ).hexdigest()
    source_cache = tmp_path / "large"
    _embedding_cache(
        model=_FakeEmbeddingModel(),
        rows=source_rows,
        row_ids_sha256=source_ids_hash,
        embedding_dimension=2,
        batch_size=2,
        cache_path=source_cache,
        cache_identity=identity,
        checkpoint_every_batches=1,
    )
    target_rows = [
        {
            "evidence_id": "network:node:new",
            "node_id": "new",
            "label": "New",
            "importance": 1,
        },
        {**source_rows[2], "importance": 100, "community": 7},
        {**source_rows[0], "importance": 200, "community": 9},
    ]
    reordered, source_embeddings, source_indices, reuse_metadata = (
        _prepare_exact_embedding_reuse(
            rows=target_rows,
            source_rows=source_rows,
            source_cache_path=source_cache,
            cache_identity=identity,
        )
    )
    assert [row["evidence_id"] for row in reordered] == [
        "network:node:shared-b",
        "network:node:shared-a",
        "network:node:new",
    ]
    assert reuse_metadata["reused_rows"] == 2
    model = _FakeEmbeddingModel()
    target_ids_hash = hashlib.sha256(
        json.dumps(
            [row["evidence_id"] for row in reordered], separators=(",", ":")
        ).encode()
    ).hexdigest()
    embeddings, manifest = _embedding_cache(
        model=model,
        rows=reordered,
        row_ids_sha256=target_ids_hash,
        embedding_dimension=2,
        batch_size=2,
        cache_path=tmp_path / "target",
        cache_identity=identity,
        checkpoint_every_batches=1,
        reuse_embeddings=source_embeddings,
        reuse_source_indices=source_indices,
        reuse_metadata=reuse_metadata,
    )
    assert model.encoded_documents == [_embedding_document(target_rows[0])]
    expected = _FakeEmbeddingModel().encode(
        [_embedding_document(row) for row in reordered]
    )
    np.testing.assert_array_equal(np.asarray(embeddings), expected)
    assert manifest["exact_embedding_reuse"] == reuse_metadata
