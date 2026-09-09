from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .io import write_json

Scale = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class GraphScale:
    name: Scale
    requested_nodes: int | None
    nodes: int
    edges: int
    connected_components: int


@dataclass(frozen=True)
class DiscoveryTask:
    item_id: str
    dataset_id: str
    network: str
    scale: Scale
    task_type: str
    complexity: int
    question: str
    answer: dict[str, Any]
    evidence_ids: list[str]
    operator_trace: list[dict[str, Any]]
    interpretation_contract: dict[str, Any]
    answer_alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DenseRetrievalIndex:
    """Query-independent latent semantic index for one canonical graph."""

    node_records: list[dict[str, Any]]
    edge_records: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    documents: list[str]
    vectorizer: TfidfVectorizer
    document_tfidf: Any
    projector: TruncatedSVD | None
    dense_documents: Any
    dimensions: int


NETWORK_SPECS = {
    "keyword_cooccurrence": {
        "nodes": "keyword_occurrences.parquet",
        "edges": "keyword_cooccurrence_edges.parquet",
        "id": "keyword",
        "label": "keyword",
        "importance": "occurrences",
    },
    "institution_collaboration": {
        "nodes": "institution_productivity.parquet",
        "edges": "institution_collaboration_edges.parquet",
        "id": "institution_id",
        "label": "institution_name",
        "importance": "documents",
    },
    "coauthorship": {
        "nodes": "author_productivity.parquet",
        "edges": "coauthor_edges.parquet",
        "id": "author_id",
        "label": "author_name",
        "importance": "documents",
    },
}

SCALE_LIMITS: dict[Scale, int | None] = {
    "small": 100,
    # The observed full bibliometric graphs are often dense: a 300-node induced
    # subgraph already contains tens of thousands of relations. This tier keeps
    # a meaningful gap from both the 100-node display graph and the full graph.
    "medium": 300,
    "large": None,
}

BM25_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def _bm25_top_rows(
    question: str,
    documents: list[str],
    rows: list[dict[str, Any]],
    *,
    limit: int,
    tie_key: str,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    query_terms = set(BM25_TOKEN.findall(question.casefold()))
    tokenized = [BM25_TOKEN.findall(document.casefold()) for document in documents]
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    document_frequency = {
        term: sum(term in tokens for tokens in tokenized) for term in query_terms
    }
    total = len(tokenized)
    scores = []
    for index, tokens in enumerate(tokenized):
        counts = {term: tokens.count(term) for term in query_terms if term in tokens}
        score = 0.0
        for term, frequency in counts.items():
            frequency_in_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (total - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(tokens) / max(1.0, average_length)
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        scores.append((score, rows[index]))
    return [
        row
        for _, row in sorted(
            scores,
            key=lambda pair: (-pair[0], _hash([tie_key, pair[1]["evidence_id"]])),
        )[:limit]
    ]


def _load_graph(workspace: Path, network: str, scale: Scale) -> tuple[nx.Graph, GraphScale]:
    spec = NETWORK_SPECS[network]
    root = workspace / "canonical" / "visualization"
    nodes = pd.read_parquet(root / spec["nodes"]).copy()
    edges = pd.read_parquet(root / spec["edges"]).copy()
    id_col = spec["id"]
    nodes[id_col] = nodes[id_col].astype(str)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    nodes = nodes.drop_duplicates(id_col, keep="first")
    limit = SCALE_LIMITS[scale]
    if limit is not None and len(nodes) > limit:
        nodes = nodes.sort_values(
            [spec["importance"], id_col], ascending=[False, True]
        ).head(limit)
    allowed = set(nodes[id_col])
    edges = edges[
        edges["source_id"].isin(allowed) & edges["target_id"].isin(allowed)
    ].copy()
    graph = nx.Graph()
    for row in nodes.to_dict("records"):
        node_id = str(row[id_col])
        graph.add_node(
            node_id,
            label=str(row.get(spec["label"]) or node_id),
            importance=float(row.get(spec["importance"]) or 0.0),
        )
    ordered_edges = edges.sort_values(
        ["source_id", "target_id", "weight"], ascending=[True, True, False]
    )
    for index, row in enumerate(ordered_edges.to_dict("records"), 1):
        source = str(row["source_id"])
        target = str(row["target_id"])
        weight = float(row.get("weight") or 1.0)
        graph.add_edge(
            source,
            target,
            weight=weight,
            distance=1.0 / max(weight, 1e-12),
            evidence_id=f"{network}:edge:{index:06d}",
        )
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    summary = GraphScale(
        name=scale,
        requested_nodes=limit,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        connected_components=nx.number_connected_components(graph) if graph else 0,
    )
    return graph, summary


def _communities(graph: nx.Graph) -> dict[str, int]:
    if not graph:
        return {}
    groups = nx.community.louvain_communities(graph, weight="weight", seed=42)
    ordered = sorted(
        (sorted(group) for group in groups), key=lambda group: (-len(group), group[0])
    )
    return {node: index for index, group in enumerate(ordered) for node in group}


def _node_evidence_id(network: str, node: str) -> str:
    return f"{network}:node:{hashlib.sha256(node.encode()).hexdigest()[:16]}"


def _node_record(graph: nx.Graph, network: str, node: str, community: int) -> dict[str, Any]:
    attrs = graph.nodes[node]
    return {
        "evidence_id": _node_evidence_id(network, node),
        "node_id": node,
        "label": attrs["label"],
        "importance": _round(attrs["importance"]),
        "community": int(community),
        "weighted_degree": _round(graph.degree(node, weight="weight")),
    }


def _edge_record(graph: nx.Graph, source: str, target: str) -> dict[str, Any]:
    attrs = graph.edges[source, target]
    return {
        "evidence_id": attrs["evidence_id"],
        "source": source,
        "target": target,
        "source_label": graph.nodes[source]["label"],
        "target_label": graph.nodes[target]["label"],
        "weight": _round(attrs["weight"]),
    }


def _hierarchical_summary_records(
    graph: nx.Graph,
    network: str,
    communities: dict[str, int],
    *,
    metric_schema: Literal["v1_count_share", "v2_weight_share"] = "v1_count_share",
) -> list[dict[str, Any]]:
    """Build query-independent deterministic summaries for a GraphRAG baseline.

    These records are an index over the canonical graph, not task support: no
    benchmark answer, evidence list, or operator trace is consulted here.
    """
    schema_suffix = "" if metric_schema == "v1_count_share" else ":v2"
    records: list[dict[str, Any]] = [
        {
            "summary_id": f"{network}:summary{schema_suffix}:global",
            "summary_type": "global_graph",
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "connected_components": nx.number_connected_components(graph),
            "density": _round(nx.density(graph)),
            "total_node_importance": _round(
                sum(float(graph.nodes[node]["importance"]) for node in graph)
            ),
        }
    ]
    community_ids = sorted(set(communities.values()))
    for community in community_ids:
        members = [node for node in graph if communities[node] == community]
        internal = 0
        external = 0
        internal_weight = 0.0
        external_weight = 0.0
        for source, target, attrs in graph.edges(data=True):
            source_inside = communities[source] == community
            target_inside = communities[target] == community
            if source_inside and target_inside:
                internal += 1
                internal_weight += float(attrs["weight"])
            elif source_inside != target_inside:
                external += 1
                external_weight += float(attrs["weight"])
        top_members = sorted(
            members,
            key=lambda node: (
                -graph.degree(node, weight="weight"),
                -graph.nodes[node]["importance"],
                graph.nodes[node]["label"],
            ),
        )[:8]
        importance_representative = min(
            members,
            key=lambda node: (
                -graph.nodes[node]["importance"], graph.nodes[node]["label"]
            ),
        )
        summary = {
            "summary_id": f"{network}:summary{schema_suffix}:community:{community}",
            "summary_type": "community",
            "community": community,
            "member_count": len(members),
            "total_node_importance": _round(
                sum(float(graph.nodes[node]["importance"]) for node in members)
            ),
            "internal_edge_count": internal,
            "external_edge_count": external,
            "internal_weight": _round(internal_weight),
            "external_weight": _round(external_weight),
            "top_members": [
                {
                    "node_id": node,
                    "label": graph.nodes[node]["label"],
                    "importance": _round(graph.nodes[node]["importance"]),
                    "weighted_degree": _round(graph.degree(node, weight="weight")),
                    "evidence_id": _node_evidence_id(network, node),
                }
                for node in top_members
            ],
        }
        if metric_schema == "v1_count_share":
            summary["external_edge_share"] = _round(
                external / max(1, internal + external)
            )
        else:
            summary.update(
                {
                    "metric_schema": "explicit_count_and_weight_shares_v2",
                    "external_edge_count_share": _round(
                        external / max(1, internal + external)
                    ),
                    "external_edge_share": _round(
                        external_weight / max(1.0, internal_weight + external_weight)
                    ),
                    "external_edge_share_definition": (
                        "external_weight / (internal_weight + external_weight)"
                    ),
                    "importance_representative": {
                        "node_id": importance_representative,
                        "label": graph.nodes[importance_representative]["label"],
                        "evidence_id": _node_evidence_id(
                            network, importance_representative
                        ),
                    },
                }
            )
        records.append(
            summary
        )

    boundary_groups: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for source, target in graph.edges:
        left, right = sorted((communities[source], communities[target]))
        if left == right:
            continue
        boundary_groups.setdefault((left, right), []).append((source, target))
    for (left, right), edges in sorted(boundary_groups.items()):
        ranked = sorted(
            edges,
            key=lambda pair: (
                -graph.edges[pair]["weight"],
                graph.edges[pair]["evidence_id"],
            ),
        )
        records.append(
            {
                "summary_id": (
                    f"{network}:summary{schema_suffix}:boundary:{left}:{right}"
                ),
                "summary_type": "community_boundary",
                "communities": [left, right],
                "edge_count": len(edges),
                "total_weight": _round(
                    sum(float(graph.edges[pair]["weight"]) for pair in edges)
                ),
                "representative_edges": [
                    _edge_record(graph, source, target) for source, target in ranked[:5]
                ],
            }
        )
    return records


def _build_dense_retrieval_index(
    graph: nx.Graph,
    network: str,
    communities: dict[str, int],
    *,
    max_features: int = 4096,
    max_dimensions: int = 48,
) -> DenseRetrievalIndex:
    """Fit a bounded, query-independent LSA index over graph rows."""
    node_records = [
        _node_record(graph, network, node, communities[node]) for node in graph.nodes
    ]
    edge_records = [_edge_record(graph, source, target) for source, target in graph.edges]
    rows: list[dict[str, Any]] = [
        *({"row_type": "node", **row} for row in node_records),
        *({"row_type": "edge", **row} for row in edge_records),
    ]
    documents = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )
    document_tfidf = vectorizer.fit_transform(documents)
    dimensions = min(
        max_dimensions,
        max(0, document_tfidf.shape[0] - 1),
        max(0, document_tfidf.shape[1] - 1),
    )
    if dimensions >= 2:
        projector = TruncatedSVD(
            n_components=dimensions,
            algorithm="randomized",
            n_iter=5,
            random_state=42,
        )
        dense_documents = normalize(projector.fit_transform(document_tfidf))
    else:
        projector = None
        dense_documents = None
    return DenseRetrievalIndex(
        node_records=node_records,
        edge_records=edge_records,
        rows=rows,
        documents=documents,
        vectorizer=vectorizer,
        document_tfidf=document_tfidf,
        projector=projector,
        dense_documents=dense_documents,
        dimensions=dimensions,
    )


def _path_task(
    graph: nx.Graph,
    *,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    ranked = sorted(
        graph.nodes,
        key=lambda node: (-graph.nodes[node]["importance"], graph.nodes[node]["label"]),
    )[:40]
    candidates: list[tuple[int, float, str, str, list[str]]] = []
    for i, source in enumerate(ranked):
        for target in ranked[i + 1 :]:
            if communities[source] == communities[target]:
                continue
            try:
                path = nx.shortest_path(graph, source, target, weight="distance")
            except nx.NetworkXNoPath:
                continue
            hops = len(path) - 1
            if 2 <= hops <= 4:
                importance = graph.nodes[source]["importance"] + graph.nodes[target]["importance"]
                candidates.append((hops, importance, source, target, path))
    if not candidates:
        return None
    _, _, source, target, path = min(
        candidates, key=lambda item: (-item[0], -item[1], item[2], item[3])
    )
    edge_ids = [
        graph.edges[left, right]["evidence_id"]
        for left, right in itertools.pairwise(path)
    ]
    node_ids = [_node_evidence_id(network, node) for node in path]
    labels = [graph.nodes[node]["label"] for node in path]
    answer = {
        "source_label": labels[0],
        "target_label": labels[-1],
        "intermediate_labels": labels[1:-1],
        "hops": len(path) - 1,
        "total_inverse_weight_distance": _round(
            sum(graph.edges[left, right]["distance"] for left, right in itertools.pairwise(path))
        ),
    }
    alternatives: list[dict[str, Any]] = []
    for candidate_path in itertools.islice(
        nx.all_shortest_paths(graph, source, target, weight="distance"), 500
    ):
        candidate_labels = [graph.nodes[node]["label"] for node in candidate_path]
        candidate_answer = {
            "source_label": candidate_labels[0],
            "target_label": candidate_labels[-1],
            "intermediate_labels": candidate_labels[1:-1],
            "hops": len(candidate_path) - 1,
            "total_inverse_weight_distance": _round(
                sum(
                    graph.edges[left, right]["distance"]
                    for left, right in itertools.pairwise(candidate_path)
                )
            ),
        }
        if candidate_answer != answer:
            alternatives.append(candidate_answer)
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:multi_hop_connector",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="multi_hop_connector",
        complexity=len(path) - 1,
        question=(
            f"Find the inverse-edge-weight shortest cross-community connection from "
            f"'{labels[0]}' to "
            f"'{labels[-1]}'. Report every intermediate node and explain what the path "
            "does and does not establish."
        ),
        answer=answer,
        evidence_ids=node_ids + edge_ids,
        operator_trace=[
            {"operator": "community_filter", "different_communities": True},
            {
                "operator": "shortest_path",
                "weight": "inverse_edge_weight",
                "path": path,
                "total_distance": answer["total_inverse_weight_distance"],
                "edge_evidence_ids": edge_ids,
            },
            {"operator": "path_projection", "labels": labels, "hops": len(path) - 1},
        ],
        interpretation_contract={
            "allowed": "structural connection in the observed bibliometric network",
            "forbidden": ["causality", "semantic equivalence", "knowledge transfer"],
            "required_limitation": "The path depends on the corpus and graph construction.",
        },
        answer_alternatives=alternatives,
    )


def _bridge_task(
    graph: nx.Graph,
    *,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    cross_edges = [
        (u, v) for u, v in graph.edges if communities.get(u) != communities.get(v)
    ]
    if not cross_edges:
        return None
    # Exact edge betweenness is intentionally used at small scale. At larger scales,
    # deterministic endpoint sampling keeps benchmark construction tractable.
    if graph.number_of_nodes() <= 300:
        centrality = nx.edge_betweenness_centrality(graph, weight="distance")
    else:
        sample = sorted(graph.nodes)[: min(96, graph.number_of_nodes())]
        centrality = nx.edge_betweenness_centrality_subset(
            graph, sources=sample, targets=list(graph.nodes), weight="distance"
        )
    source, target = min(
        cross_edges,
        key=lambda edge: (
            -centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)),
            -graph.edges[edge]["weight"],
            edge[0],
            edge[1],
        ),
    )
    before_components = nx.number_connected_components(graph)
    counterfactual = graph.copy()
    counterfactual.remove_edge(source, target)
    after_components = nx.number_connected_components(counterfactual)
    try:
        alternate_hops: int | None = nx.shortest_path_length(counterfactual, source, target)
    except nx.NetworkXNoPath:
        alternate_hops = None
    score = centrality.get((source, target), centrality.get((target, source), 0.0))
    labels = [graph.nodes[source]["label"], graph.nodes[target]["label"]]
    edge_id = graph.edges[source, target]["evidence_id"]
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:bridge_counterfactual",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="bridge_counterfactual",
        complexity=4,
        question=(
            "Identify the highest-betweenness cross-community edge selected by the "
            "registered graph operator. Then use edge deletion to state whether it is "
            "an irreplaceable bridge or a redundant connector."
        ),
        answer={
            "source_label": labels[0],
            "target_label": labels[1],
            "edge_weight": _round(graph.edges[source, target]["weight"]),
            "alternate_hops_after_deletion": alternate_hops,
            "component_increase": after_components - before_components,
        },
        evidence_ids=[
            _node_evidence_id(network, source),
            _node_evidence_id(network, target),
            edge_id,
        ],
        operator_trace=[
            {"operator": "community_boundary_edges", "candidate_count": len(cross_edges)},
            {"operator": "edge_betweenness", "selected_edge": [source, target], "score": _round(score)},
            {
                "operator": "delete_edge_counterfactual",
                "components_before": before_components,
                "components_after": after_components,
                "alternate_hops": alternate_hops,
            },
        ],
        interpretation_contract={
            "allowed": "structural brokerage and redundancy under a stated deletion",
            "forbidden": ["causal influence", "author intent", "information flow"],
            "required_limitation": "Betweenness is parameter- and corpus-dependent.",
        },
    )


def _community_task(
    graph: nx.Graph,
    *,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    rows: list[dict[str, Any]] = []
    for community in sorted(set(communities.values())):
        members = {node for node, value in communities.items() if value == community}
        if len(members) < 2:
            continue
        internal = 0.0
        external = 0.0
        for source, target, attrs in graph.edges(data=True):
            weight = float(attrs["weight"])
            source_in = source in members
            target_in = target in members
            if source_in and target_in:
                internal += weight
            elif source_in or target_in:
                external += weight
        total = internal + external
        representative = min(
            members,
            key=lambda node: (-graph.nodes[node]["importance"], graph.nodes[node]["label"]),
        )
        rows.append(
            {
                "community": community,
                "nodes": len(members),
                "importance": sum(graph.nodes[node]["importance"] for node in members),
                "external_share": external / total if total else 0.0,
                "representative": representative,
            }
        )
    if len(rows) < 2:
        return None
    dominant = min(rows, key=lambda row: (-row["importance"], row["community"]))
    outward_ranking = sorted(
        rows, key=lambda row: (-row["external_share"], row["community"])
    )
    outward = outward_ranking[0]
    if dominant["community"] == outward["community"]:
        outward = outward_ranking[1]
    dominant_label = graph.nodes[dominant["representative"]]["label"]
    outward_label = graph.nodes[outward["representative"]]["label"]
    evidence = [
        _node_evidence_id(network, dominant["representative"]),
        _node_evidence_id(network, outward["representative"]),
    ]
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:community_role_contrast",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="community_role_contrast",
        complexity=4,
        question=(
            "Contrast the community with the greatest total node importance against a "
            "different community with the largest external-edge share. Explain the "
            "global phenomenon without treating a community label as a scientific topic."
        ),
        answer={
            "dominant_community": int(dominant["community"]),
            "dominant_representative": dominant_label,
            "dominant_nodes": int(dominant["nodes"]),
            "outward_community": int(outward["community"]),
            "outward_representative": outward_label,
            "outward_external_share": _round(outward["external_share"]),
        },
        evidence_ids=evidence,
        operator_trace=[
            {"operator": "louvain", "seed": 42, "communities": len(rows)},
            {"operator": "community_aggregate", "metrics": rows},
            {
                "operator": "role_contrast",
                "dominant": dominant["community"],
                "outward": outward["community"],
            },
        ],
        interpretation_contract={
            "allowed": "contrast in aggregate structural roles",
            "forbidden": ["community equals research theme", "quality ranking", "causality"],
            "required_limitation": "Community identifiers are algorithmic, not semantic labels.",
        },
    )


def _resilience_task(
    graph: nx.Graph,
    *,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    if graph.number_of_nodes() < 3:
        return None
    ranked = sorted(
        graph.nodes,
        key=lambda node: (-graph.degree(node, weight="weight"), graph.nodes[node]["label"]),
    )
    target = ranked[0]
    base_largest = max(map(len, nx.connected_components(graph)))
    counterfactual = graph.copy()
    counterfactual.remove_node(target)
    after_components = nx.number_connected_components(counterfactual) if counterfactual else 0
    after_largest = max(map(len, nx.connected_components(counterfactual))) if counterfactual else 0
    second = min(
        counterfactual.nodes,
        key=lambda node: (
            -counterfactual.degree(node, weight="weight"),
            counterfactual.nodes[node]["label"],
        ),
    )
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:hub_removal_resilience",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="hub_removal_resilience",
        complexity=4,
        question=(
            "Remove the weighted-degree hub and quantify the structural response. State "
            "whether the observed graph is hub-dependent, including the replacement hub "
            "and the largest component after deletion as a fraction of the original node count."
        ),
        answer={
            "removed_hub": graph.nodes[target]["label"],
            "replacement_hub": graph.nodes[second]["label"],
            "components_after": after_components,
            "largest_component_fraction_of_original_nodes": _round(
                after_largest / max(1, graph.number_of_nodes())
            ),
        },
        evidence_ids=[
            _node_evidence_id(network, target),
            _node_evidence_id(network, second),
        ],
        operator_trace=[
            {
                "operator": "weighted_degree_argmax",
                "node": target,
                "value": _round(graph.degree(target, weight="weight")),
            },
            {"operator": "delete_node", "node": target},
            {
                "operator": "component_recompute",
                "components_after": after_components,
                "largest_before": base_largest,
                "largest_after": after_largest,
            },
            {"operator": "weighted_degree_argmax", "node": second, "after_deletion": True},
        ],
        interpretation_contract={
            "allowed": "structural sensitivity to a node-deletion counterfactual",
            "forbidden": ["real-world collapse", "causal indispensability", "quality"],
            "required_limitation": "Node deletion is a model perturbation, not an intervention.",
        },
    )


def _simple_control_from_complex(
    graph: nx.Graph,
    *,
    parent: DiscoveryTask,
) -> DiscoveryTask:
    if parent.task_type == "bridge_counterfactual":
        source, target = parent.operator_trace[1]["selected_edge"]
        source_label = graph.nodes[source]["label"]
        target_label = graph.nodes[target]["label"]
        edge_id = graph.edges[source, target]["evidence_id"]
        return DiscoveryTask(
            item_id=(
                f"{parent.dataset_id}:{parent.network}:{parent.scale}:direct_edge_lookup"
            ),
            dataset_id=parent.dataset_id,
            network=parent.network,
            scale=parent.scale,
            task_type="direct_edge_lookup",
            complexity=1,
            question=(
                f"In the supplied graph records, report the direct edge weight between "
                f"'{source_label}' and '{target_label}'. Do not infer brokerage, causality, "
                "or what would happen if the edge were removed."
            ),
            answer={
                "source_label": source_label,
                "target_label": target_label,
                "edge_weight": _round(graph.edges[source, target]["weight"]),
            },
            evidence_ids=[
                _node_evidence_id(parent.network, source),
                _node_evidence_id(parent.network, target),
                edge_id,
            ],
            operator_trace=[
                {
                    "operator": "edge_attribute_lookup",
                    "edge": [source, target],
                    "attribute": "weight",
                    "value": _round(graph.edges[source, target]["weight"]),
                }
            ],
            interpretation_contract={
                "allowed": "a direct edge attribute in the supplied graph",
                "forbidden": ["brokerage", "causality", "counterfactual effect"],
                "required_limitation": "The edge weight depends on corpus indexing and graph construction.",
            },
        )
    if parent.task_type == "hub_removal_resilience":
        node = parent.operator_trace[0]["node"]
        label = graph.nodes[node]["label"]
        return DiscoveryTask(
            item_id=(
                f"{parent.dataset_id}:{parent.network}:{parent.scale}:node_attribute_lookup"
            ),
            dataset_id=parent.dataset_id,
            network=parent.network,
            scale=parent.scale,
            task_type="node_attribute_lookup",
            complexity=1,
            question=(
                f"In the supplied graph records, report the occurrence/productivity value "
                f"and weighted degree of '{label}'. Do not remove the node or infer network "
                "resilience."
            ),
            answer={
                "node_label": label,
                "importance": _round(graph.nodes[node]["importance"]),
                "weighted_degree": _round(graph.degree(node, weight="weight")),
            },
            evidence_ids=[_node_evidence_id(parent.network, node)],
            operator_trace=[
                {
                    "operator": "node_attribute_lookup",
                    "node": node,
                    "attributes": {
                        "importance": _round(graph.nodes[node]["importance"]),
                        "weighted_degree": _round(graph.degree(node, weight="weight")),
                    },
                }
            ],
            interpretation_contract={
                "allowed": "two node attributes in the supplied graph",
                "forbidden": ["resilience", "causal importance", "scientific quality"],
                "required_limitation": "Node attributes depend on corpus indexing and graph construction.",
            },
        )
    raise ValueError(f"No registered simple control for {parent.task_type}")


def _temporal_structure_task(
    graph: nx.Graph,
    *,
    workspace: Path,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    if network != "keyword_cooccurrence":
        return None
    trends_path = workspace / "analyses" / "keyword_trends.parquet"
    if not trends_path.is_file():
        return None
    trends = pd.read_parquet(trends_path)
    required = {"year", "keyword", "documents"}
    if not required.issubset(trends.columns):
        return None
    trends = trends[trends["keyword"].astype(str).isin(graph.nodes)].copy()
    if trends.empty or trends["year"].nunique() < 6:
        return None
    years = sorted(int(year) for year in trends["year"].dropna().unique())
    early_years = set(years[:3])
    recent_years = set(years[-3:])
    rows = []
    for keyword, group in trends.groupby("keyword"):
        node = str(keyword)
        if node not in graph:
            continue
        early = float(group[group["year"].isin(early_years)]["documents"].sum())
        recent = float(group[group["year"].isin(recent_years)]["documents"].sum())
        rows.append(
            {
                "node": node,
                "early_documents": early,
                "recent_documents": recent,
                "growth_ratio": (recent + 1.0) / (early + 1.0),
                "weighted_degree": float(graph.degree(node, weight="weight")),
            }
        )
    if len(rows) < 2:
        return None
    recent_floor = sorted(row["recent_documents"] for row in rows)[len(rows) // 2]
    emerging_pool = [row for row in rows if row["recent_documents"] >= recent_floor]
    emerging = min(
        emerging_pool,
        key=lambda row: (-row["growth_ratio"], -row["recent_documents"], row["node"]),
    )
    established = min(
        (row for row in rows if row["node"] != emerging["node"]),
        key=lambda row: (-row["weighted_degree"], -row["early_documents"], row["node"]),
    )
    emerging_label = graph.nodes[emerging["node"]]["label"]
    established_label = graph.nodes[established["node"]]["label"]
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:temporal_structural_shift",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="temporal_structural_shift",
        complexity=5,
        question=(
            "Contrast the registered high-growth recent keyword with the established "
            "weighted-degree hub. Use both time aggregation and topology, and state whether "
            "the two occupy the same detected community."
        ),
        answer={
            "emerging_label": emerging_label,
            "emerging_recent_documents": int(emerging["recent_documents"]),
            "emerging_growth_ratio": _round(emerging["growth_ratio"]),
            "established_label": established_label,
            "established_weighted_degree": _round(established["weighted_degree"]),
            "same_community": communities[emerging["node"]]
            == communities[established["node"]],
        },
        evidence_ids=[
            _node_evidence_id(network, emerging["node"]),
            _node_evidence_id(network, established["node"]),
        ],
        operator_trace=[
            {
                "operator": "temporal_window_aggregate",
                "early_years": sorted(early_years),
                "recent_years": sorted(recent_years),
                "selected": emerging,
            },
            {
                "operator": "weighted_degree_argmax",
                "selected": established,
            },
            {
                "operator": "cross_layer_join",
                "same_community": communities[emerging["node"]]
                == communities[established["node"]],
            },
        ],
        interpretation_contract={
            "allowed": "a temporal-structural contrast in the observed corpus",
            "forbidden": ["future prediction", "scientific breakthrough", "causal emergence"],
            "required_limitation": "Growth depends on the chosen windows and indexing coverage.",
        },
    )


def _rank_divergence_task(
    graph: nx.Graph,
    *,
    dataset_id: str,
    network: str,
    scale: Scale,
    communities: dict[str, int],
) -> DiscoveryTask | None:
    if network not in {"institution_collaboration", "coauthorship"}:
        return None
    productivity = sorted(
        graph.nodes,
        key=lambda node: (-graph.nodes[node]["importance"], graph.nodes[node]["label"]),
    )
    centrality = sorted(
        graph.nodes,
        key=lambda node: (-graph.degree(node, weight="weight"), graph.nodes[node]["label"]),
    )
    productive = productivity[0]
    central = centrality[0]
    productivity_rank = {node: index + 1 for index, node in enumerate(productivity)}
    centrality_rank = {node: index + 1 for index, node in enumerate(centrality)}
    candidates = set(productivity[:50]) | set(centrality[:50])
    divergent = min(
        candidates,
        key=lambda node: (
            -abs(productivity_rank[node] - centrality_rank[node]),
            min(productivity_rank[node], centrality_rank[node]),
            graph.nodes[node]["label"],
        ),
    )
    direction = (
        "more_central_than_productive"
        if centrality_rank[divergent] < productivity_rank[divergent]
        else "more_productive_than_central"
        if productivity_rank[divergent] < centrality_rank[divergent]
        else "aligned"
    )
    return DiscoveryTask(
        item_id=f"{dataset_id}:{network}:{scale}:productivity_centrality_divergence",
        dataset_id=dataset_id,
        network=network,
        scale=scale,
        task_type="productivity_centrality_divergence",
        complexity=5,
        question=(
            "Compare the most productive actor with the weighted-degree collaboration hub. "
            "Then identify the registered actor with the largest cross-rank divergence among "
            "the union of the top 50 actors on either layer."
        ),
        answer={
            "most_productive_label": graph.nodes[productive]["label"],
            "collaboration_hub_label": graph.nodes[central]["label"],
            "same_top_actor": productive == central,
            "largest_divergence_label": graph.nodes[divergent]["label"],
            "divergence_productivity_rank": productivity_rank[divergent],
            "divergence_centrality_rank": centrality_rank[divergent],
            "divergence_direction": direction,
        },
        evidence_ids=[
            _node_evidence_id(network, productive),
            _node_evidence_id(network, central),
            _node_evidence_id(network, divergent),
        ],
        operator_trace=[
            {
                "operator": "rank",
                "metric": "productivity",
                "selected_node": productive,
                "cross_rank": centrality_rank[productive],
            },
            {
                "operator": "rank",
                "metric": "weighted_degree",
                "selected_node": central,
                "cross_rank": productivity_rank[central],
            },
            {
                "operator": "rank_divergence",
                "candidate_rule": "union_top_50",
                "selected_node": divergent,
                "productivity_rank": productivity_rank[divergent],
                "centrality_rank": centrality_rank[divergent],
                "direction": direction,
            },
        ],
        interpretation_contract={
            "allowed": "rank alignment or divergence between two observed layers",
            "forbidden": ["research quality", "causal productivity", "individual influence"],
            "required_limitation": "Ranks inherit author/institution disambiguation and corpus scope.",
        },
    )


def _contexts(
    graph: nx.Graph,
    task: DiscoveryTask,
    communities: dict[str, int],
    *,
    record_budget: int = 160,
    dense_index: DenseRetrievalIndex | None = None,
) -> dict[str, Any]:
    if dense_index is None:
        dense_index = _build_dense_retrieval_index(
            graph, task.network, communities
        )
    node_records = dense_index.node_records
    edge_records = dense_index.edge_records
    rows = dense_index.rows
    documents = dense_index.documents
    flat_rows = sorted(
        rows,
        key=lambda row: _hash([task.item_id, row["evidence_id"]]),
    )[:record_budget]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([task.question, *documents])
    similarities = (matrix[1:] @ matrix[0].T).toarray().ravel()
    candidate_limit = min(len(rows), max(record_budget, record_budget * 4))
    tfidf_ranked = [
        row
        for _, row in sorted(
            zip(similarities, rows),
            key=lambda pair: (-pair[0], _hash([task.item_id, pair[1]["evidence_id"]])),
        )[:candidate_limit]
    ]
    tfidf_rows = tfidf_ranked[:record_budget]
    bm25_ranked = _bm25_top_rows(
        task.question,
        documents,
        rows,
        limit=candidate_limit,
        tie_key=task.item_id,
    )
    bm25_rows = bm25_ranked[:record_budget]

    dense_query_tfidf = dense_index.vectorizer.transform([task.question])
    if dense_index.projector is not None:
        dense_query = normalize(dense_index.projector.transform(dense_query_tfidf))[0]
        dense_similarities = np.asarray(
            dense_index.dense_documents @ dense_query
        ).ravel()
    else:
        dense_similarities = (
            dense_index.document_tfidf @ dense_query_tfidf.T
        ).toarray().ravel()
    lsa_ranked = [
        row
        for _, row in sorted(
            zip(dense_similarities, rows),
            key=lambda pair: (-pair[0], _hash([task.item_id, pair[1]["evidence_id"]])),
        )[:candidate_limit]
    ]
    lsa_rows = lsa_ranked[:record_budget]
    reciprocal_rank_scores: dict[str, float] = {}
    row_by_evidence = {row["evidence_id"]: row for row in rows}
    for ranking in (bm25_ranked, tfidf_ranked, lsa_ranked):
        for rank, row in enumerate(ranking, 1):
            evidence_id = row["evidence_id"]
            reciprocal_rank_scores[evidence_id] = (
                reciprocal_rank_scores.get(evidence_id, 0.0) + 1.0 / (60 + rank)
            )
    hybrid_rows = [
        row_by_evidence[evidence_id]
        for evidence_id, _ in sorted(
            reciprocal_rank_scores.items(),
            key=lambda pair: (-pair[1], _hash([task.item_id, pair[0]])),
        )[:record_budget]
    ]

    node_by_id = {row["node_id"]: row for row in node_records}
    edge_by_id = {row["evidence_id"]: row for row in edge_records}

    def assemble_graph_records(
        seed_nodes: set[str], seed_edge_ids: set[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selected_node_ids = set(seed_nodes)
        selected_edges = [edge_by_id[value] for value in sorted(seed_edge_ids) if value in edge_by_id]
        for row in selected_edges:
            selected_node_ids.update((row["source"], row["target"]))
        selected_nodes = [node_by_id[node] for node in sorted(selected_node_ids)]
        candidates = sorted(
            (
                row
                for row in edge_records
                if row["evidence_id"] not in seed_edge_ids
                and (
                    row["source"] in selected_node_ids
                    or row["target"] in selected_node_ids
                )
            ),
            key=lambda row: (-row["weight"], row["evidence_id"]),
        )
        for row in candidates:
            new_nodes = {
                node
                for node in (row["source"], row["target"])
                if node not in selected_node_ids
            }
            added = 1 + len(new_nodes)
            if len(selected_nodes) + len(selected_edges) + added > record_budget:
                continue
            selected_edges.append(row)
            for node in sorted(new_nodes):
                selected_node_ids.add(node)
                selected_nodes.append(node_by_id[node])
        for row in sorted(
            node_records,
            key=lambda value: (-value["importance"], value["node_id"]),
        ):
            if len(selected_nodes) + len(selected_edges) >= record_budget:
                break
            if row["node_id"] not in selected_node_ids:
                selected_node_ids.add(row["node_id"])
                selected_nodes.append(row)
        return selected_nodes, selected_edges

    # Oracle support context is retained as an upper-bound diagnostic. It must
    # never be described as an end-to-end query retriever in paper claims.
    evidence = set(task.evidence_ids)
    oracle_seed_nodes = {
        row["node_id"] for row in node_records if row["evidence_id"] in evidence
    }
    oracle_seed_edges = {
        row["evidence_id"] for row in edge_records if row["evidence_id"] in evidence
    }
    graph_nodes, graph_edges = assemble_graph_records(
        oracle_seed_nodes, oracle_seed_edges
    )

    # Non-oracle query retrieval uses only labels explicitly present in the
    # question. For entity-pair questions it executes a shortest-path traversal;
    # global questions fall back to salient seeds and therefore require a graph
    # program rather than silently consulting benchmark gold.
    anchored = sorted(
        (
            (task.question.casefold().find(f"'{graph.nodes[node]['label']}'".casefold()), node)
            for node in graph.nodes
            if f"'{graph.nodes[node]['label']}'".casefold() in task.question.casefold()
        ),
        key=lambda value: (value[0], value[1]),
    )
    query_seed_nodes = {node for _, node in anchored}
    query_seed_edges: set[str] = set()
    if len(anchored) >= 2:
        try:
            query_path = nx.shortest_path(graph, anchored[0][1], anchored[-1][1])
        except nx.NetworkXNoPath:
            query_path = []
        query_seed_nodes.update(query_path)
        query_seed_edges.update(
            graph.edges[left, right]["evidence_id"]
            for left, right in itertools.pairwise(query_path)
        )
    if not query_seed_nodes:
        query_seed_nodes.update(
            sorted(
                graph.nodes,
                key=lambda node: (
                    -graph.nodes[node]["importance"],
                    graph.nodes[node]["label"],
                ),
            )[:8]
        )
    query_nodes, query_edges = assemble_graph_records(
        query_seed_nodes, query_seed_edges
    )
    # Entity-pair queries retain traversed edges first; global queries retain
    # salient nodes first. This routing uses only observable query anchors.
    query_raw_records: list[tuple[str, dict[str, Any]]]
    if anchored:
        query_raw_records = [
            *(("edge", row) for row in query_edges),
            *(("node", row) for row in query_nodes),
        ]
    else:
        query_raw_records = [
            *(("node", row) for row in query_nodes),
            *(("edge", row) for row in query_edges),
        ]

    def retrieve_hierarchy(
        hierarchy_records: list[dict[str, Any]], representation: str
    ) -> dict[str, Any]:
        hierarchy_documents = [
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in hierarchy_records
        ]
        hierarchy_vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
        hierarchy_matrix = hierarchy_vectorizer.fit_transform(
            [task.question, *hierarchy_documents]
        )
        hierarchy_similarities = (
            hierarchy_matrix[1:] @ hierarchy_matrix[0].T
        ).toarray().ravel()
        summary_budget = min(len(hierarchy_records), max(1, record_budget // 2))
        selected_summaries = [hierarchy_records[0]]
        selected_summaries.extend(
            row
            for _, row in sorted(
                zip(hierarchy_similarities[1:], hierarchy_records[1:]),
                key=lambda pair: (-pair[0], pair[1]["summary_id"]),
            )[: max(0, summary_budget - 1)]
        )
        remaining = max(0, record_budget - len(selected_summaries))
        selected_raw = query_raw_records[:remaining]
        return {
            "representation": representation,
            "record_budget": record_budget,
            "retrieval_anchors": [graph.nodes[node]["label"] for _, node in anchored],
            "summary_index_scope": "query_independent_full_graph",
            "summaries": selected_summaries,
            "nodes": [row for row_type, row in selected_raw if row_type == "node"],
            "edges": [row for row_type, row in selected_raw if row_type == "edge"],
        }

    hierarchical_context = retrieve_hierarchy(
        _hierarchical_summary_records(graph, task.network, communities),
        "query_retrieved_hierarchical_graph_summaries_v1_count_share",
    )
    hierarchical_context_v2 = retrieve_hierarchy(
        _hierarchical_summary_records(
            graph,
            task.network,
            communities,
            metric_schema="v2_weight_share",
        ),
        "query_retrieved_hierarchical_graph_summaries_v2_explicit_metrics",
    )
    graph_context = {
        "representation": "connected_graph_oracle_support_upper_bound",
        "record_budget": record_budget,
        "nodes": graph_nodes,
        "edges": graph_edges,
    }
    query_graph_context = {
        "representation": "query_anchored_graph_retrieval",
        "record_budget": record_budget,
        "retrieval_anchors": [graph.nodes[node]["label"] for _, node in anchored],
        "nodes": query_nodes,
        "edges": query_edges,
    }
    program_context = {
        **graph_context,
        "representation": "graph_program_retrieval",
        "operator_trace": task.operator_trace,
        "interpretation_contract": task.interpretation_contract,
    }
    # Representation-only ablation: preserve exactly the graph-program evidence
    # and derived facts, but serialize every record into one unordered table.
    # This prevents retrieval quality from contaminating graph-vs-flat claims.
    flat_program_rows = [
        *({"row_type": "node", **row} for row in graph_nodes),
        *({"row_type": "edge", **row} for row in graph_edges),
    ]
    return {
        "flat_retrieval": {
            "representation": "flat_random_budget_control",
            "record_budget": record_budget,
            "rows": flat_rows,
        },
        "flat_tfidf": {
            "representation": "flat_tfidf_budget_matched_retrieval",
            "record_budget": record_budget,
            "rows": tfidf_rows,
        },
        "flat_bm25": {
            "representation": "flat_bm25_budget_matched_retrieval",
            "record_budget": record_budget,
            "parameters": {"k1": 1.2, "b": 0.75},
            "rows": bm25_rows,
        },
        "flat_lsa": {
            "representation": "flat_lsa_latent_dense_budget_matched_retrieval",
            "record_budget": record_budget,
            "parameters": {
                "dimensions": dense_index.dimensions,
                "max_features": 4096,
                "random_state": 42,
            },
            "rows": lsa_rows,
        },
        "flat_hybrid": {
            "representation": "flat_bm25_tfidf_lsa_rrf_hybrid_retrieval",
            "record_budget": record_budget,
            "parameters": {
                "fusion": "reciprocal_rank_fusion",
                "rrf_k": 60,
                "candidate_depth_per_retriever": candidate_limit,
            },
            "rows": hybrid_rows,
        },
        "flat_program": {
            "representation": "flat_program_ablation",
            "record_budget": record_budget,
            "rows": flat_program_rows,
            "derived_rows": task.operator_trace,
            "interpretation_contract": task.interpretation_contract,
        },
        "graph_query_retrieval": query_graph_context,
        "graph_hierarchical_retrieval": hierarchical_context,
        "graph_hierarchical_retrieval_v2": hierarchical_context_v2,
        "graph_retrieval": graph_context,
        "operator_only": {
            "representation": "operator_trace_without_raw_graph",
            "operator_trace": task.operator_trace,
            "interpretation_contract": task.interpretation_contract,
        },
        "graph_program": program_context,
    }


def build_discovery_benchmark(
    workspace: Path,
    output_dir: Path,
    *,
    scales: tuple[Scale, ...] = ("small", "medium", "large"),
    networks: tuple[str, ...] = tuple(NETWORK_SPECS),
    task_types: tuple[str, ...] | None = None,
    record_budget: int = 160,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    dataset_id = workspace.name
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    scale_records: list[dict[str, Any]] = []
    for network in networks:
        if network not in NETWORK_SPECS:
            raise ValueError(f"Unsupported scalable network: {network}")
        for scale in scales:
            graph, summary = _load_graph(workspace, network, scale)
            scale_records.append({"network": network, **asdict(summary)})
            if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:
                continue
            communities = _communities(graph)
            dense_index = _build_dense_retrieval_index(graph, network, communities)
            builders = (_path_task, _bridge_task, _community_task, _resilience_task)
            built_tasks: dict[str, DiscoveryTask] = {}
            for builder in builders:
                builder_task_type = {
                    _path_task: "multi_hop_connector",
                    _bridge_task: "bridge_counterfactual",
                    _community_task: "community_role_contrast",
                    _resilience_task: "hub_removal_resilience",
                }[builder]
                if task_types is not None and builder_task_type not in task_types:
                    continue
                task = builder(
                    graph,
                    dataset_id=dataset_id,
                    network=network,
                    scale=scale,
                    communities=communities,
                )
                if task is None:
                    continue
                built_tasks[task.task_type] = task
                record = asdict(task)
                record["contexts"] = _contexts(
                    graph,
                    task,
                    communities,
                    record_budget=record_budget,
                    dense_index=dense_index,
                )
                record["context_hashes"] = {
                    name: _hash(context) for name, context in record["contexts"].items()
                }
                tasks.append(record)
            simple_controls = {
                "direct_edge_lookup": ("bridge_counterfactual", _bridge_task),
                "node_attribute_lookup": ("hub_removal_resilience", _resilience_task),
            }
            for simple_type, (parent_type, parent_builder) in simple_controls.items():
                if task_types is not None and simple_type not in task_types:
                    continue
                parent = built_tasks.get(parent_type)
                if parent is None:
                    parent = parent_builder(
                        graph,
                        dataset_id=dataset_id,
                        network=network,
                        scale=scale,
                        communities=communities,
                    )
                if parent is None:
                    continue
                task = _simple_control_from_complex(graph, parent=parent)
                record = asdict(task)
                record["matched_complex_item_id"] = parent.item_id
                record["contexts"] = _contexts(
                    graph,
                    task,
                    communities,
                    record_budget=record_budget,
                    dense_index=dense_index,
                )
                record["context_hashes"] = {
                    name: _hash(context) for name, context in record["contexts"].items()
                }
                tasks.append(record)
            temporal = _temporal_structure_task(
                graph,
                workspace=workspace,
                dataset_id=dataset_id,
                network=network,
                scale=scale,
                communities=communities,
            )
            rank_divergence = _rank_divergence_task(
                graph,
                dataset_id=dataset_id,
                network=network,
                scale=scale,
                communities=communities,
            )
            for task in (temporal, rank_divergence):
                if task is None:
                    continue
                if task_types is not None and task.task_type not in task_types:
                    continue
                record = asdict(task)
                record["contexts"] = _contexts(
                    graph,
                    task,
                    communities,
                    record_budget=record_budget,
                    dense_index=dense_index,
                )
                record["context_hashes"] = {
                    name: _hash(context) for name, context in record["contexts"].items()
                }
                tasks.append(record)
    benchmark = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "design": {
            "independent_variables": ["graph_scale", "task_complexity", "retrieval_condition"],
            "conditions": [
                "no_reference",
                "flat_retrieval",
                "flat_tfidf",
                "flat_bm25",
                "flat_lsa",
                "flat_hybrid",
                "graph_retrieval",
                "flat_program",
                "graph_query_retrieval",
                "graph_hierarchical_retrieval",
                "graph_hierarchical_retrieval_v2",
                "graph_program",
                "operator_only",
            ],
            "record_budget": record_budget,
            "primary_contrast": "graph_program_vs_flat_tfidf",
            "sparse_retrieval_contrast": "graph_program_vs_flat_bm25",
            "mechanism_contrast": "graph_program_vs_graph_retrieval",
            "representation_contrast": "graph_program_vs_flat_program",
            "raw_evidence_contrast": "graph_program_vs_operator_only",
        },
        "scales": scale_records,
        "tasks": tasks,
    }
    write_json(output_dir / "benchmark.json", benchmark)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "benchmark_sha256": _hash(benchmark),
            "tasks": len(tasks),
            "task_types": sorted({task["task_type"] for task in tasks}),
            "networks": sorted({task["network"] for task in tasks}),
            "scale_records": scale_records,
        },
    )
    return benchmark


def build_simple_control_benchmark(
    workspace: Path,
    output_dir: Path,
    *,
    parent_tasks: list[dict[str, Any]],
    record_budget: int = 160,
) -> dict[str, Any]:
    """Build lookup controls from already-frozen complex-task anchors.

    Reusing the registered parent operator outputs avoids rerunning the task-selection
    operator after a complexity-extension protocol has been frozen.
    """
    workspace = workspace.resolve()
    dataset_id = workspace.name
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_parents = {
        (network, scale, task_type)
        for network in {str(task["network"]) for task in parent_tasks}
        for scale in {str(task["scale"]) for task in parent_tasks}
        for task_type in ("bridge_counterfactual", "hub_removal_resilience")
    }
    indexed = {
        (str(task["network"]), str(task["scale"]), str(task["task_type"])): task
        for task in parent_tasks
        if task["task_type"] in {"bridge_counterfactual", "hub_removal_resilience"}
    }
    if set(indexed) != expected_parents:
        raise ValueError("Frozen parent tasks must cover both matched types at every scale")

    tasks: list[dict[str, Any]] = []
    scale_records: list[dict[str, Any]] = []
    for network, scale, parent_type in sorted(indexed):
        if parent_type != "bridge_counterfactual":
            continue
        typed_scale: Scale = scale  # type: ignore[assignment]
        graph, summary = _load_graph(workspace, network, typed_scale)
        scale_records.append({"network": network, **asdict(summary)})
        communities = _communities(graph)
        dense_index = _build_dense_retrieval_index(graph, network, communities)
        for matched_parent_type in (
            "bridge_counterfactual",
            "hub_removal_resilience",
        ):
            raw = indexed[(network, scale, matched_parent_type)]
            parent = DiscoveryTask(
                **{
                    field.name: raw[field.name]
                    for field in DiscoveryTask.__dataclass_fields__.values()
                }
            )
            if parent.dataset_id != dataset_id:
                raise ValueError("Frozen parent task belongs to a different dataset")
            task = _simple_control_from_complex(graph, parent=parent)
            record = asdict(task)
            record["matched_complex_item_id"] = parent.item_id
            record["contexts"] = _contexts(
                graph,
                task,
                communities,
                record_budget=record_budget,
                dense_index=dense_index,
            )
            record["context_hashes"] = {
                name: _hash(context) for name, context in record["contexts"].items()
            }
            tasks.append(record)
    benchmark = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "design": {
            "independent_variables": ["graph_scale", "retrieval_condition"],
            "record_budget": record_budget,
            "control_role": "anchor_matched_simple_negative_control",
        },
        "scales": scale_records,
        "tasks": tasks,
    }
    write_json(output_dir / "benchmark.json", benchmark)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "benchmark_sha256": _hash(benchmark),
            "tasks": len(tasks),
            "parent_item_ids": sorted(task["matched_complex_item_id"] for task in tasks),
            "scale_records": scale_records,
        },
    )
    return benchmark


def _equal(gold: Any, observed: Any, *, tolerance: float = 1e-4) -> bool:
    if isinstance(gold, bool) or gold is None or isinstance(gold, str):
        return gold == observed
    if isinstance(gold, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(gold), float(observed), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(gold, list) and isinstance(observed, list):
        return len(gold) == len(observed) and all(
            _equal(left, right, tolerance=tolerance) for left, right in zip(gold, observed)
        )
    if isinstance(gold, dict) and isinstance(observed, dict):
        return set(gold) == set(observed) and all(
            _equal(gold[key], observed[key], tolerance=tolerance) for key in gold
        )
    return False


def score_discovery_response(task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    observed_evidence = set(response.get("evidence_ids") or [])
    gold_evidence = set(task["evidence_ids"])
    overlap = len(observed_evidence & gold_evidence)
    precision = overlap / len(observed_evidence) if observed_evidence else 0.0
    recall = overlap / len(gold_evidence) if gold_evidence else 1.0
    evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    limitation = str(response.get("limitation") or "").strip()
    accepted_answers = [task["answer"], *(task.get("answer_alternatives") or [])]
    return {
        "item_id": task["item_id"],
        "answer_exact": any(_equal(answer, response.get("answer")) for answer in accepted_answers),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": evidence_f1,
        "has_required_limitation": bool(limitation),
        "abstain": bool(response.get("abstain")),
    }
