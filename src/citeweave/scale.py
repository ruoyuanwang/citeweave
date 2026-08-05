from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb

from .io import write_json
from .models import ProjectConfig
from .transform import CanonicalTables


def scale_plan(tables: CanonicalTables, config: ProjectConfig) -> dict[str, Any]:
    documents = len(tables.works)
    relationships = len(tables.authorships) + len(tables.keywords) + len(tables.references)
    tier = (
        "small"
        if documents < 10_000
        else "medium"
        if documents < 100_000
        else "large"
        if documents < 1_000_000
        else "very_large"
    )
    candidate_pool = max(config.visualization_max_nodes * 8, 400)
    return {
        "documents": documents,
        "relationship_rows": relationships,
        "tier": tier,
        "policy": {
            "descriptive_statistics": "all canonical records; never sampled",
            "quality_and_field_coverage": "all canonical records; never sampled",
            "network_occurrence_counting": "all relationship rows",
            "network_candidate_pool": candidate_pool,
            "network_candidate_rule": (
                "retain the highest-occurrence eligible entities before pair expansion; "
                "minimum occurrence and group caps are recorded per network"
            ),
            "rendered_nodes_maximum": config.visualization_max_nodes,
            "rendered_label_budget": config.visualization_label_budget,
            "rendering_rule": (
                "rendered graphs are disclosed display subgraphs; complete candidate "
                "node/edge tables and VOSviewer exports remain available"
            ),
        },
        "capacity_guidance": {
            "under_100k": (
                "disk processing is the default for bulk projects; fixed Python batches "
                "and DuckDB materialization avoid relationship-sized Python lists"
            ),
            "100k_to_1m": (
                "use staged gzip JSONL, partitioned Parquet and DuckDB spill storage; "
                "size chunk_size from observed record width and keep candidate-first "
                "network construction enabled"
            ),
            "over_1m": (
                "benchmark on target hardware first; partition acquisition by "
                "non-overlapping year/source strata, preserve checkpoints, and prefer "
                "official snapshots for database-scale corpora"
            ),
        },
    }


def save_scale_plan(tables: CanonicalTables, config: ProjectConfig, output: Path) -> dict[str, Any]:
    plan = scale_plan(tables, config)
    write_json(output, plan)
    return plan


def run_duckdb_benchmark(
    *, documents: int = 100_000, terms_per_document: int = 5
) -> dict[str, Any]:
    """Exercise full-count aggregation and bounded candidate selection synthetically."""
    if documents < 1 or terms_per_document < 1:
        raise ValueError("documents and terms_per_document must be positive")
    started = time.perf_counter()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE works AS
            SELECT i AS work_id, 2000 + (i % 25) AS year, i % 500 AS source_id
            FROM range(?) AS t(i)
            """,
            [documents],
        )
        relationship_rows = documents * terms_per_document
        connection.execute(
            """
            CREATE TABLE memberships AS
            SELECT CAST(i / ? AS BIGINT) AS work_id,
                   'term-' || CAST(i % 2000 AS VARCHAR) AS item_id
            FROM range(?) AS t(i)
            """,
            [terms_per_document, relationship_rows],
        )
        annual_rows = connection.execute(
            "SELECT count(*) FROM (SELECT year, count(*) FROM works GROUP BY year)"
        ).fetchone()[0]
        candidates = connection.execute(
            """
            SELECT count(*) FROM (
              SELECT item_id, count(DISTINCT work_id) AS occurrence
              FROM memberships GROUP BY item_id
              ORDER BY occurrence DESC, item_id LIMIT 800
            )
            """
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "engine": "duckdb",
        "documents": documents,
        "relationship_rows": relationship_rows,
        "annual_groups": annual_rows,
        "selected_candidates": candidates,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "passed": annual_rows == 25 and candidates == min(800, 2000),
    }
