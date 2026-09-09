from __future__ import annotations

from scripts.prepare_formal_v3_replication_workspaces import build_replication_config


def test_replication_config_preserves_frozen_scale_materialization() -> None:
    config = build_replication_config(
        {
            "id": "topic",
            "title": "Topic title",
            "keywords": ["term a", "term b"],
            "query_mode": "all",
            "year_from": 2010,
            "year_to": 2025,
            "source": "openalex",
            "document_types": ["article"],
        },
        protocol_hash="p" * 64,
        amendment_hash="a" * 64,
    )
    assert config.protocol.max_records is None
    assert config.acquisition.mode == "bulk"
    assert config.processing.candidate_pool_size == 2000
    assert config.processing.edge_row_limit == 1_000_000
    assert "protocol=" + "p" * 64 in config.protocol.notes
