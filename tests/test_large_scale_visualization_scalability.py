from __future__ import annotations

import duckdb
import pandas as pd

from citeweave.large_scale_visualization import _linked_node_table


def test_linked_cocitation_nodes_exclude_unlinked_full_dimension(tmp_path) -> None:
    canonical = tmp_path / "canonical"
    visual = canonical / "visualization"
    visual.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "cited_work_id": "linked-a",
                "cited_doi": "10/a",
                "cited_title": "A",
                "cited_author": "Author A",
                "cited_year": 2020,
                "local_citations": 8,
            },
            {
                "cited_work_id": "linked-b",
                "cited_doi": "10/b",
                "cited_title": "B",
                "cited_author": "Author B",
                "cited_year": 2021,
                "local_citations": 5,
            },
            {
                "cited_work_id": "unlinked",
                "cited_doi": "10/c",
                "cited_title": "C",
                "cited_author": "Author C",
                "cited_year": 2022,
                "local_citations": 100,
            },
        ]
    ).to_parquet(visual / "reference_impact.parquet", index=False)
    edge_path = visual / "cocitation_edges.parquet"
    pd.DataFrame(
        [{"source_id": "linked-a", "target_id": "linked-b", "weight": 3}]
    ).to_parquet(edge_path, index=False)

    connection = duckdb.connect()
    try:
        result = _linked_node_table(
            connection,
            canonical,
            visual,
            network="cocitation",
            edge_path=edge_path,
        )
    finally:
        connection.close()

    assert set(result["id"]) == {"linked-a", "linked-b"}
    assert "unlinked" not in set(result["id"])


def test_linked_keyword_year_uses_all_occurrences_of_linked_terms(tmp_path) -> None:
    canonical = tmp_path / "canonical"
    visual = canonical / "visualization"
    visual.mkdir(parents=True)
    pd.DataFrame(
        [
            {"work_id": "w1", "year": 2000},
            {"work_id": "w2", "year": 2020},
            {"work_id": "w3", "year": 1990},
        ]
    ).to_parquet(canonical / "works.parquet", index=False)
    pd.DataFrame(
        [
            {"work_id": "w1", "keyword": "linked"},
            {"work_id": "w2", "keyword": "linked"},
            {"work_id": "w3", "keyword": "unlinked"},
        ]
    ).to_parquet(canonical / "keywords.parquet", index=False)
    pd.DataFrame(
        [
            {
                "keyword": "linked",
                "keyword_type": "source",
                "occurrences": 2,
            },
            {
                "keyword": "unlinked",
                "keyword_type": "source",
                "occurrences": 1,
            },
        ]
    ).to_parquet(visual / "keyword_occurrences.parquet", index=False)
    edge_path = visual / "keyword_cooccurrence_edges.parquet"
    pd.DataFrame(
        [{"source_id": "linked", "target_id": "missing", "weight": 1}]
    ).to_parquet(edge_path, index=False)

    connection = duckdb.connect()
    try:
        result = _linked_node_table(
            connection,
            canonical,
            visual,
            network="keyword_cooccurrence",
            edge_path=edge_path,
        )
    finally:
        connection.close()

    assert result["id"].tolist() == ["linked"]
    assert result.iloc[0]["average_year"] == 2010
