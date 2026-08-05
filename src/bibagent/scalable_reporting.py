from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analytics import AnalysisBundle, NetworkResult
from .io import read_json
from .models import ProjectPaths
from .transform import CanonicalTables


def _named_ranking(
    frame: pd.DataFrame,
    *,
    source_label: str,
    output_label: str = "name",
    limit: int = 30,
) -> pd.DataFrame:
    result = frame.dropna(subset=[source_label]).copy()
    result = result[result[source_label].astype(str).str.strip().ne("")]
    result = result.rename(columns={source_label: output_label, "documents": "publications"})
    return (
        result.sort_values(["publications", output_label], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )


def _bradford_sources(source_productivity: pd.DataFrame) -> pd.DataFrame:
    ranked = _named_ranking(
        source_productivity,
        source_label="source_name",
        limit=max(len(source_productivity), 1),
    )
    if ranked.empty:
        return pd.DataFrame(
            columns=["source_id", "name", "publications", "cumulative_publications", "zone"]
        )
    total = int(ranked["publications"].sum())
    ranked["cumulative_publications"] = ranked["publications"].cumsum()
    ranked["zone"] = np.minimum(
        3,
        np.floor((ranked["cumulative_publications"] - 1) / max(total / 3, 1)).astype(int) + 1,
    )
    return ranked[
        ["source_id", "name", "publications", "cumulative_publications", "zone"]
    ].reset_index(drop=True)


def _keyword_trends(tables: CanonicalTables, keyword_occurrences: pd.DataFrame) -> pd.DataFrame:
    top_keywords = (
        keyword_occurrences.dropna(subset=["keyword"])
        .sort_values(["occurrences", "keyword"], ascending=[False, True])
        .head(15)["keyword"]
        .astype(str)
        .tolist()
    )
    if not top_keywords:
        return pd.DataFrame(columns=["year", "keyword", "documents", "global_documents"])
    selected = tables.keywords[tables.keywords["keyword"].astype(str).isin(top_keywords)]
    selected = selected[["work_id", "keyword"]].drop_duplicates()
    merged = selected.merge(tables.works[["work_id", "year"]], on="work_id", how="inner")
    merged = merged.dropna(subset=["year"])
    grouped = (
        merged.groupby(["year", "keyword"])["work_id"]
        .nunique()
        .rename("documents")
        .reset_index()
    )
    globals_ = selected.groupby("keyword")["work_id"].nunique().rename("global_documents")
    grouped = grouped.join(globals_, on="keyword")
    grouped["year"] = grouped["year"].astype(int)
    return grouped.sort_values(
        ["global_documents", "keyword", "year"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _three_field(tables: CanonicalTables, visual: Path) -> pd.DataFrame:
    authors = pd.read_parquet(visual / "author_productivity.parquet").dropna(
        subset=["author_name"]
    )
    sources = pd.read_parquet(visual / "source_productivity.parquet").dropna(
        subset=["source_name"]
    )
    keywords = pd.read_parquet(visual / "keyword_occurrences.parquet").dropna(
        subset=["keyword"]
    )
    author_ids = set(authors.head(20)["author_id"].astype(str))
    source_ids = set(sources.head(20)["source_id"].astype(str))
    keyword_values = set(keywords.head(20)["keyword"].astype(str))
    memberships = tables.authorships[
        tables.authorships["author_id"].astype(str).isin(author_ids)
    ][["work_id", "author_id"]].drop_duplicates()
    work_sources = tables.works[
        tables.works["source_id"].astype(str).isin(source_ids)
    ][["work_id", "source_id"]]
    work_keywords = tables.keywords[
        tables.keywords["keyword"].astype(str).isin(keyword_values)
    ][["work_id", "keyword"]].drop_duplicates()
    merged = memberships.merge(work_sources, on="work_id").merge(work_keywords, on="work_id")
    if merged.empty:
        return pd.DataFrame(columns=["author", "source", "keyword", "documents"])
    author_names = tables.authors[["author_id", "name"]].rename(columns={"name": "author"})
    source_names = tables.sources[["source_id", "name"]].rename(columns={"name": "source"})
    merged = merged.merge(author_names, on="author_id").merge(source_names, on="source_id")
    return (
        merged.groupby(["author", "source", "keyword"])["work_id"]
        .nunique()
        .rename("documents")
        .reset_index()
        .sort_values(["documents", "author", "source", "keyword"], ascending=[False, True, True, True])
        .head(80)
        .reset_index(drop=True)
    )


def _selected_networks(paths: ProjectPaths) -> dict[str, NetworkResult]:
    directory = paths.analyses / "visualization"
    networks: dict[str, NetworkResult] = {}
    for node_path in sorted(directory.glob("*_nodes.parquet")):
        name = node_path.name.removesuffix("_nodes.parquet")
        edge_path = directory / f"{name}_edges.parquet"
        method_path = directory / f"{name}_method.json"
        nodes = pd.read_parquet(node_path)
        edges = pd.read_parquet(edge_path)
        if "weighted_degree" not in nodes:
            nodes["weighted_degree"] = pd.to_numeric(
                nodes.get("total_link_strength", 0), errors="coerce"
            ).fillna(0.0)
        if "occurrences" not in nodes:
            nodes["occurrences"] = 1
        if "cluster" not in nodes:
            nodes["cluster"] = 1
        metadata: dict[str, Any] = read_json(method_path) if method_path.exists() else {}
        metadata.setdefault("candidate_pool", metadata.get("candidate_nodes_with_links", len(nodes)))
        metadata.setdefault("full_candidate_edge_count", metadata.get("candidate_edges", len(edges)))
        metadata["displayed_nodes"] = len(nodes)
        metadata["displayed_edges"] = len(edges)
        networks[name] = NetworkResult(name, nodes, edges, metadata)
    return networks


def build_large_scale_analysis(paths: ProjectPaths, tables: CanonicalTables) -> AnalysisBundle:
    """Adapt disk-backed full-corpus aggregates to the report/evidence contract.

    No full adjacency matrix is created. Descriptive tables use all canonical
    records; network tables are the bounded, already selected visualization views.
    """
    visual = paths.canonical / "visualization"
    annual_source = pd.read_parquet(visual / "annual_output.parquet")
    annual = annual_source.rename(columns={"documents": "publications"})[
        ["year", "publications"]
    ].copy()
    annual["year"] = annual["year"].astype(int)
    source_productivity = pd.read_parquet(visual / "source_productivity.parquet")
    top_sources = _named_ranking(source_productivity, source_label="source_name")
    top_authors = _named_ranking(
        pd.read_parquet(visual / "author_productivity.parquet"),
        source_label="author_name",
    )
    top_institutions = _named_ranking(
        pd.read_parquet(visual / "institution_productivity.parquet"),
        source_label="institution_name",
    )
    document_types = pd.read_parquet(visual / "document_types.parquet").rename(
        columns={"documents": "publications"}
    )
    citations = pd.to_numeric(tables.works["cited_by_count"], errors="coerce").fillna(0)
    citation_distribution = pd.DataFrame(
        {
            "metric": ["mean", "median", "p75", "p90", "p95", "p98", "maximum", "zero_cited"],
            "value": [
                float(citations.mean()),
                float(citations.median()),
                float(citations.quantile(0.75)),
                float(citations.quantile(0.90)),
                float(citations.quantile(0.95)),
                float(citations.quantile(0.98)),
                float(citations.max()),
                int(citations.eq(0).sum()),
            ],
        }
    )
    top_cited_documents = (
        tables.works[["title", "year", "doi", "cited_by_count"]]
        .assign(cited_by_count=citations)
        .sort_values(["cited_by_count", "year", "title"], ascending=[False, False, True])
        .head(30)
        .reset_index(drop=True)
    )
    keyword_occurrences = pd.read_parquet(visual / "keyword_occurrences.parquet")
    first_year = int(annual["year"].min())
    last_year = int(annual["year"].max())
    first_count = int(annual.loc[annual["year"].eq(first_year), "publications"].iloc[0])
    last_count = int(annual.loc[annual["year"].eq(last_year), "publications"].iloc[0])
    growth = (
        ((last_count / first_count) ** (1 / (last_year - first_year)) - 1) * 100
        if first_count > 0 and last_year > first_year
        else None
    )
    summary = {
        "documents": len(tables.works),
        "authors": len(tables.authors),
        "institutions": len(tables.institutions),
        "sources": len(tables.sources),
        "references": len(tables.references),
        "timespan_start": first_year,
        "timespan_end": last_year,
        "annual_growth_rate": growth,
        "average_citations": float(citations.mean()),
        "median_citations": float(citations.median()),
        "analysis_scope": "all canonical records; network displays are bounded sparse views",
        "full_adjacency_matrix_materialized": False,
    }
    return AnalysisBundle(
        summary=summary,
        annual=annual,
        top_sources=top_sources,
        top_authors=top_authors,
        top_institutions=top_institutions,
        document_types=document_types[["document_type", "publications"]],
        citation_distribution=citation_distribution,
        top_cited_documents=top_cited_documents,
        bradford_sources=_bradford_sources(source_productivity),
        keyword_trends=_keyword_trends(tables, keyword_occurrences),
        three_field=_three_field(tables, visual),
        networks=_selected_networks(paths),
    )
