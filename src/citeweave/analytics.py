from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .transform import CanonicalTables


@dataclass
class NetworkResult:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class AnalysisBundle:
    summary: dict[str, Any]
    annual: pd.DataFrame
    top_sources: pd.DataFrame
    top_authors: pd.DataFrame
    top_institutions: pd.DataFrame
    document_types: pd.DataFrame
    citation_distribution: pd.DataFrame
    top_cited_documents: pd.DataFrame
    bradford_sources: pd.DataFrame
    keyword_trends: pd.DataFrame
    three_field: pd.DataFrame
    networks: dict[str, NetworkResult]


def _count_table(frame: pd.DataFrame, key: str, label: str, limit: int = 30) -> pd.DataFrame:
    if frame.empty or key not in frame:
        return pd.DataFrame(columns=[key, label])
    result = (
        frame.groupby(key, dropna=False)
        .size()
        .rename(label)
        .reset_index()
        .sort_values([label, key], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    return result


def _pairs(values: Iterable[str]) -> Iterable[tuple[str, str]]:
    unique = sorted({value for value in values if value})
    return itertools.combinations(unique, 2)


def _build_pair_network(
    memberships: pd.DataFrame,
    *,
    group_col: str,
    item_col: str,
    labels: pd.DataFrame,
    label_id: str,
    label_col: str,
    name: str,
    candidate_pool: int,
    max_group_items: int,
    min_occurrence: int = 2,
) -> NetworkResult:
    if memberships.empty:
        return NetworkResult(name, pd.DataFrame(), pd.DataFrame(), {"empty": True})
    clean = memberships[[group_col, item_col]].dropna().drop_duplicates()
    occurrences = clean[item_col].value_counts()
    candidates = set(
        occurrences[occurrences >= min_occurrence].head(candidate_pool).index.astype(str)
    )
    clean = clean[clean[item_col].astype(str).isin(candidates)]
    pair_counts: Counter[tuple[str, str]] = Counter()
    capped_groups = 0
    for _, group in clean.groupby(group_col, sort=False):
        values = group[item_col].astype(str).tolist()
        if len(values) > max_group_items:
            values = sorted(values, key=lambda value: (-occurrences.get(value, 0), value))[
                :max_group_items
            ]
            capped_groups += 1
        pair_counts.update(_pairs(values))
    edge_rows = []
    for (left, right), weight in pair_counts.items():
        left_occ = int(occurrences.get(left, 0))
        right_occ = int(occurrences.get(right, 0))
        association = weight / (left_occ * right_occ) if left_occ and right_occ else 0.0
        edge_rows.append(
            {
                "source": left,
                "target": right,
                "weight": int(weight),
                "association_strength": float(association),
            }
        )
    edges = pd.DataFrame(edge_rows)
    label_map = (
        labels.drop_duplicates(label_id).set_index(label_id)[label_col].to_dict()
        if not labels.empty and label_id in labels and label_col in labels
        else {}
    )
    node_rows = [
        {
            "id": str(item),
            "label": label_map.get(item, label_map.get(str(item), str(item))),
            "occurrences": int(count),
        }
        for item, count in occurrences.items()
        if str(item) in candidates
    ]
    nodes = pd.DataFrame(node_rows)
    nodes, edges = _network_metrics(nodes, edges)
    metadata = {
        "candidate_pool": candidate_pool,
        "candidate_nodes": len(candidates),
        "source_groups": int(memberships[group_col].nunique()),
        "max_group_items": max_group_items,
        "capped_groups": capped_groups,
        "min_occurrence": min_occurrence,
        "normalization": "association_strength = cooccurrence / (occ_i * occ_j)",
        "full_candidate_edge_count": len(edges),
    }
    return NetworkResult(name, nodes, edges, metadata)


def _network_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nodes.empty:
        return nodes, edges
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(str(row.id))
    if not edges.empty:
        for row in edges.itertuples(index=False):
            graph.add_edge(
                str(row.source),
                str(row.target),
                weight=float(row.weight),
                distance=1.0 / max(float(row.association_strength), 1e-12),
            )
    weighted_degree = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        community_map = {
            node: community_index + 1
            for community_index, community in enumerate(communities)
            for node in community
        }
        betweenness = nx.betweenness_centrality(
            graph, k=min(200, len(graph)) if len(graph) > 200 else None, weight="distance", seed=42
        )
    else:
        community_map = {node: 1 for node in graph}
        betweenness = {node: 0.0 for node in graph}
    result = nodes.copy()
    result["degree"] = result["id"].astype(str).map(degree).fillna(0).astype(int)
    result["weighted_degree"] = (
        result["id"].astype(str).map(weighted_degree).fillna(0).astype(float)
    )
    result["betweenness"] = result["id"].astype(str).map(betweenness).fillna(0.0)
    result["cluster"] = result["id"].astype(str).map(community_map).fillna(0).astype(int)
    return result, edges


def _citation_network(tables: CanonicalTables, candidate_pool: int) -> NetworkResult:
    works = tables.works
    references = tables.references
    if references.empty:
        return NetworkResult("citation", pd.DataFrame(), pd.DataFrame(), {"empty": True})
    corpus = set(works["work_id"])
    edges = references[
        references["citing_work_id"].isin(corpus) & references["cited_work_id"].isin(corpus)
    ][["citing_work_id", "cited_work_id"]].rename(
        columns={"citing_work_id": "source", "cited_work_id": "target"}
    )
    if edges.empty:
        return NetworkResult(
            "citation",
            pd.DataFrame(),
            pd.DataFrame(),
            {"empty": True, "reason": "no within-corpus citation edges"},
        )
    edges["weight"] = 1
    edges["association_strength"] = 1.0
    connected = set(edges["source"]) | set(edges["target"])
    selected = (
        works[works["work_id"].isin(connected)]
        .assign(score=lambda frame: frame["cited_by_count"].fillna(0))
        .nlargest(candidate_pool, "score")
    )
    selected_ids = set(selected["work_id"])
    edges = edges[
        edges["source"].isin(selected_ids) & edges["target"].isin(selected_ids)
    ].drop_duplicates()
    nodes = selected[["work_id", "title", "cited_by_count"]].rename(
        columns={"work_id": "id", "title": "label", "cited_by_count": "occurrences"}
    )
    nodes, edges = _network_metrics(nodes, edges)
    return NetworkResult(
        "citation",
        nodes,
        edges,
        {
            "candidate_pool": candidate_pool,
            "directed_semantics": True,
            "full_within_corpus_edges": len(
                references[
                    references["citing_work_id"].isin(corpus)
                    & references["cited_work_id"].isin(corpus)
                ]
            ),
        },
    )


def _cocitation_network(tables: CanonicalTables, candidate_pool: int) -> NetworkResult:
    refs = tables.references
    if refs.empty:
        return NetworkResult("cocitation", pd.DataFrame(), pd.DataFrame(), {"empty": True})
    labels = (
        refs[["cited_work_id", "cited_title", "cited_author", "cited_year"]]
        .drop_duplicates("cited_work_id")
        .copy()
    )

    def citation_label(row: pd.Series) -> str:
        title = row.get("cited_title")
        if pd.notna(title) and str(title).strip():
            return str(title).strip()
        author_year = " ".join(
            str(value)
            for value in [row.get("cited_author"), row.get("cited_year")]
            if pd.notna(value)
            and str(value).strip()
            and str(value).strip().casefold() not in {"author unknown", "unknown", "0"}
        )
        return author_year or "Unresolved reference"

    labels["label"] = labels.apply(citation_label, axis=1)
    return _build_pair_network(
        refs,
        group_col="citing_work_id",
        item_col="cited_work_id",
        labels=labels,
        label_id="cited_work_id",
        label_col="label",
        name="cocitation",
        candidate_pool=candidate_pool,
        max_group_items=100,
        min_occurrence=2,
    )


def _bibliographic_coupling(tables: CanonicalTables, candidate_pool: int) -> NetworkResult:
    refs = tables.references
    if refs.empty:
        return NetworkResult(
            "bibliographic_coupling", pd.DataFrame(), pd.DataFrame(), {"empty": True}
        )
    memberships = refs.rename(
        columns={"cited_work_id": "reference_id", "citing_work_id": "work_id"}
    )[["reference_id", "work_id"]]
    labels = tables.works[["work_id", "title"]].rename(columns={"title": "label"})
    return _build_pair_network(
        memberships,
        group_col="reference_id",
        item_col="work_id",
        labels=labels,
        label_id="work_id",
        label_col="label",
        name="bibliographic_coupling",
        candidate_pool=candidate_pool,
        max_group_items=100,
        min_occurrence=2,
    )


def analyze(
    tables: CanonicalTables,
    *,
    network_candidate_pool: int = 500,
) -> AnalysisBundle:
    works = tables.works
    annual = (
        works.dropna(subset=["year"])
        .assign(year=lambda frame: frame["year"].astype(int))
        .groupby("year")
        .size()
        .rename("publications")
        .reset_index()
        .sort_values("year")
    )
    work_sources = works[["work_id", "source_id"]].merge(
        tables.sources[["source_id", "name"]], on="source_id", how="left"
    )
    known_work_sources = work_sources.dropna(subset=["name"])
    top_sources = (
        known_work_sources.groupby(["source_id", "name"], dropna=False)
        .size()
        .rename("publications")
        .reset_index()
        .sort_values("publications", ascending=False)
        .head(30)
    )
    author_counts = (
        tables.authorships[["work_id", "author_id"]]
        .drop_duplicates()
        .groupby("author_id")
        .size()
        .rename("publications")
        .reset_index()
        if not tables.authorships.empty
        else pd.DataFrame(columns=["author_id", "publications"])
    )
    top_authors = (
        author_counts.merge(tables.authors[["author_id", "name"]], on="author_id", how="left")
        .sort_values("publications", ascending=False)
        .head(30)
    )
    institution_counts = (
        tables.authorships.dropna(subset=["institution_id"])[["work_id", "institution_id"]]
        .drop_duplicates()
        .groupby("institution_id")
        .size()
        .rename("publications")
        .reset_index()
        if not tables.authorships.empty
        else pd.DataFrame(columns=["institution_id", "publications"])
    )
    top_institutions = (
        institution_counts.merge(
            tables.institutions[["institution_id", "name"]],
            on="institution_id",
            how="left",
        )
        .sort_values("publications", ascending=False)
        .head(30)
    )
    document_types = _count_table(works, "document_type", "publications", 20)
    citations = pd.to_numeric(works["cited_by_count"], errors="coerce").fillna(0)
    citation_distribution = pd.DataFrame(
        {
            "metric": ["mean", "median", "p75", "p90", "maximum", "zero_cited"],
            "value": [
                float(citations.mean()),
                float(citations.median()),
                float(citations.quantile(0.75)),
                float(citations.quantile(0.90)),
                float(citations.max()),
                int((citations == 0).sum()),
            ],
        }
    )
    top_cited_documents = (
        works.assign(
            cited_by_count=pd.to_numeric(works["cited_by_count"], errors="coerce").fillna(0)
        )[["work_id", "title", "year", "doi", "cited_by_count"]]
        .sort_values(["cited_by_count", "year"], ascending=[False, False])
        .head(30)
        .reset_index(drop=True)
    )
    bradford_sources = top_sources.copy()
    if not known_work_sources.empty:
        bradford_sources = (
            known_work_sources.groupby(["source_id", "name"], dropna=False)
            .size()
            .rename("publications")
            .reset_index()
            .sort_values(["publications", "name"], ascending=[False, True])
            .reset_index(drop=True)
        )
        bradford_sources["cumulative_publications"] = bradford_sources["publications"].cumsum()
        zone_size = max(len(known_work_sources) / 3, 1)
        bradford_sources["zone"] = (
            np.ceil(bradford_sources["cumulative_publications"] / zone_size).clip(1, 3).astype(int)
        )
        bradford_sources["rank"] = np.arange(1, len(bradford_sources) + 1)

    keyword_trends = pd.DataFrame(columns=["year", "keyword", "documents", "global_documents"])
    if not tables.keywords.empty:
        keyword_year = (
            tables.keywords[["work_id", "keyword"]]
            .drop_duplicates()
            .merge(works[["work_id", "year"]], on="work_id", how="inner")
            .dropna(subset=["year", "keyword"])
        )
        global_counts = keyword_year.groupby("keyword")["work_id"].nunique()
        top_keyword_values = set(global_counts.nlargest(15).index)
        keyword_trends = (
            keyword_year[keyword_year["keyword"].isin(top_keyword_values)]
            .assign(year=lambda frame: frame["year"].astype(int))
            .groupby(["year", "keyword"])["work_id"]
            .nunique()
            .rename("documents")
            .reset_index()
        )
        keyword_trends["global_documents"] = keyword_trends["keyword"].map(global_counts)

    three_field = pd.DataFrame(
        columns=["author_id", "author", "source_id", "source", "keyword", "documents"]
    )
    if not tables.authorships.empty and not tables.keywords.empty:
        top_author_ids = set(top_authors.head(8)["author_id"])
        top_source_ids = set(top_sources.head(8)["source_id"])
        keyword_global = (
            tables.keywords[["work_id", "keyword"]]
            .drop_duplicates()
            .groupby("keyword")["work_id"]
            .nunique()
            .nlargest(10)
        )
        top_keywords = set(keyword_global.index)
        triples = (
            tables.authorships[tables.authorships["author_id"].isin(top_author_ids)][
                ["work_id", "author_id"]
            ]
            .drop_duplicates()
            .merge(works[["work_id", "source_id"]], on="work_id")
        )
        triples = triples[triples["source_id"].isin(top_source_ids)].merge(
            tables.keywords[tables.keywords["keyword"].isin(top_keywords)][
                ["work_id", "keyword"]
            ].drop_duplicates(),
            on="work_id",
        )
        if not triples.empty:
            three_field = (
                triples.groupby(["author_id", "source_id", "keyword"])["work_id"]
                .nunique()
                .rename("documents")
                .reset_index()
                .merge(
                    tables.authors[["author_id", "name"]].rename(columns={"name": "author"}),
                    on="author_id",
                    how="left",
                )
                .merge(
                    tables.sources[["source_id", "name"]].rename(columns={"name": "source"}),
                    on="source_id",
                    how="left",
                )
            )[["author_id", "author", "source_id", "source", "keyword", "documents"]]
    coauthor_memberships = tables.authorships[["work_id", "author_id"]].drop_duplicates()
    coauthorship = _build_pair_network(
        coauthor_memberships,
        group_col="work_id",
        item_col="author_id",
        labels=tables.authors,
        label_id="author_id",
        label_col="name",
        name="coauthorship",
        candidate_pool=network_candidate_pool,
        max_group_items=50,
        min_occurrence=2,
    )
    keyword_memberships = tables.keywords[["work_id", "keyword"]].drop_duplicates()
    keyword_labels = pd.DataFrame({"keyword": keyword_memberships["keyword"].dropna().unique()})
    keyword_labels["label"] = keyword_labels["keyword"]
    keyword_network = _build_pair_network(
        keyword_memberships,
        group_col="work_id",
        item_col="keyword",
        labels=keyword_labels,
        label_id="keyword",
        label_col="label",
        name="keyword_cooccurrence",
        candidate_pool=network_candidate_pool,
        max_group_items=30,
        min_occurrence=2,
    )
    institution_memberships = tables.authorships.dropna(subset=["institution_id"])[
        ["work_id", "institution_id"]
    ].drop_duplicates()
    institution_network = _build_pair_network(
        institution_memberships,
        group_col="work_id",
        item_col="institution_id",
        labels=tables.institutions,
        label_id="institution_id",
        label_col="name",
        name="institution_collaboration",
        candidate_pool=network_candidate_pool,
        max_group_items=50,
        min_occurrence=2,
    )
    networks = {
        "coauthorship": coauthorship,
        "institution_collaboration": institution_network,
        "keyword_cooccurrence": keyword_network,
        "citation": _citation_network(tables, network_candidate_pool),
        "cocitation": _cocitation_network(tables, network_candidate_pool),
        "bibliographic_coupling": _bibliographic_coupling(tables, network_candidate_pool),
    }
    years = annual["year"] if not annual.empty else pd.Series(dtype=float)
    summary = {
        "documents": len(works),
        "sources": int(works["source_id"].nunique()),
        "authors": int(tables.authors["author_id"].nunique()) if not tables.authors.empty else 0,
        "institutions": int(tables.institutions["institution_id"].nunique())
        if not tables.institutions.empty
        else 0,
        "timespan_start": int(years.min()) if not years.empty else None,
        "timespan_end": int(years.max()) if not years.empty else None,
        "references": len(tables.references),
        "average_citations": float(citations.mean()) if len(citations) else 0.0,
        "median_citations": float(citations.median()) if len(citations) else 0.0,
        "annual_growth_rate": _compound_growth(annual),
    }
    return AnalysisBundle(
        summary=summary,
        annual=annual,
        top_sources=top_sources,
        top_authors=top_authors,
        top_institutions=top_institutions,
        document_types=document_types,
        citation_distribution=citation_distribution,
        top_cited_documents=top_cited_documents,
        bradford_sources=bradford_sources,
        keyword_trends=keyword_trends,
        three_field=three_field,
        networks=networks,
    )


def _compound_growth(annual: pd.DataFrame) -> float | None:
    nonzero = annual[annual["publications"] > 0]
    if len(nonzero) < 2:
        return None
    first = float(nonzero.iloc[0]["publications"])
    last = float(nonzero.iloc[-1]["publications"])
    periods = int(nonzero.iloc[-1]["year"] - nonzero.iloc[0]["year"])
    if periods <= 0 or first <= 0:
        return None
    return (math.pow(last / first, 1 / periods) - 1) * 100
