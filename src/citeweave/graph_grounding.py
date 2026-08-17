from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .analytics import AnalysisBundle, NetworkResult
from .io import write_json
from .transform import CanonicalTables


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    network: str
    fact_type: str
    statement: str
    value: dict[str, Any]
    evidence_nodes: list[str]
    evidence_edges: list[str]
    caveat: str


@dataclass
class GraphGroundingBundle:
    graph: nx.MultiDiGraph
    facts: list[GraphFact]

    def prompt_packet(
        self,
        *,
        network: str | None = None,
        query: str | None = None,
        limit: int = 16,
    ) -> str:
        facts = retrieve_graph_facts(self.facts, network=network, query=query, limit=limit)
        return json.dumps([asdict(fact) for fact in facts], ensure_ascii=False, indent=2)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(child) for child in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _keyword_id(value: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(value).strip().casefold())
    if normalized.startswith("keyword:"):
        return normalized
    return f"keyword:{normalized}"


def _network_node_id(network: str, raw_id: Any) -> str:
    value = str(raw_id)
    if network == "coauthorship":
        return value if value.startswith("author:") else f"author:{value}"
    if network == "institution_collaboration":
        return value if value.startswith("institution:") else f"institution:{value}"
    if network == "keyword_cooccurrence":
        return _keyword_id(value)
    if network in {"citation", "bibliographic_coupling"}:
        return value if value.startswith("work:") else f"work:{value}"
    if network == "cocitation":
        return value if value.startswith("referenced_work:") else f"referenced_work:{value}"
    return f"{network}:{value}"


def _edge_id(network: str, source: str, target: str) -> str:
    left, right = sorted((source, target))
    return f"edge:{network}:{left}|{right}"


def _entity_id(prefix: str, raw_id: Any) -> str:
    value = str(raw_id)
    return value if value.startswith(f"{prefix}:") else f"{prefix}:{value}"


def _add_table_nodes(graph: nx.MultiDiGraph, tables: CanonicalTables) -> None:
    for row in tables.works.itertuples(index=False):
        graph.add_node(
            f"work:{row.work_id}",
            kind="work",
            label=_json_value(row.title) or str(row.work_id),
            work_id=str(row.work_id),
            doi=_json_value(row.doi),
            year=_json_value(row.year),
            cited_by_count=_json_value(row.cited_by_count),
        )
    for row in tables.authors.itertuples(index=False):
        graph.add_node(
            _entity_id("author", row.author_id),
            kind="author",
            label=_json_value(row.name) or str(row.author_id),
            author_id=str(row.author_id),
            orcid=_json_value(row.orcid),
        )
    for row in tables.institutions.itertuples(index=False):
        graph.add_node(
            _entity_id("institution", row.institution_id),
            kind="institution",
            label=_json_value(row.name) or str(row.institution_id),
            institution_id=str(row.institution_id),
            country_code=_json_value(row.country_code),
        )
    for row in tables.sources.itertuples(index=False):
        graph.add_node(
            f"source:{row.source_id}",
            kind="source",
            label=_json_value(row.name) or str(row.source_id),
            source_id=str(row.source_id),
            issn=_json_value(row.issn),
        )
    for keyword in tables.keywords.get("keyword", pd.Series(dtype=str)).dropna().unique():
        graph.add_node(
            _keyword_id(keyword),
            kind="keyword",
            label=str(keyword),
        )


def _add_canonical_relations(graph: nx.MultiDiGraph, tables: CanonicalTables) -> None:
    work_source = tables.works[["work_id", "source_id"]].dropna().drop_duplicates()
    for row in work_source.itertuples(index=False):
        graph.add_edge(
            f"work:{row.work_id}",
            f"source:{row.source_id}",
            key=f"published_in:{row.work_id}:{row.source_id}",
            relation="published_in",
        )

    for row in tables.authorships.dropna(subset=["work_id", "author_id"]).itertuples(index=False):
        author = _entity_id("author", row.author_id)
        graph.add_edge(
            author,
            f"work:{row.work_id}",
            key=f"authored:{row.author_id}:{row.work_id}",
            relation="authored",
            position=_json_value(row.position),
        )
        if pd.notna(row.institution_id):
            institution = _entity_id("institution", row.institution_id)
            graph.add_edge(
                author,
                institution,
                key=f"affiliated_with:{row.author_id}:{row.institution_id}",
                relation="affiliated_with",
            )

    for row in tables.keywords.dropna(subset=["work_id", "keyword"]).itertuples(index=False):
        graph.add_edge(
            f"work:{row.work_id}",
            _keyword_id(row.keyword),
            key=f"has_keyword:{row.work_id}:{_keyword_id(row.keyword)}",
            relation="has_keyword",
            keyword_type=_json_value(row.keyword_type),
        )

    for row in tables.references.dropna(
        subset=["citing_work_id", "cited_work_id"]
    ).itertuples(index=False):
        cited = f"work:{row.cited_work_id}"
        if cited not in graph:
            cited = f"referenced_work:{row.cited_work_id}"
            graph.add_node(
                cited,
                kind="referenced_work",
                label=_json_value(row.cited_title) or str(row.cited_work_id),
                cited_doi=_json_value(row.cited_doi),
                cited_year=_json_value(row.cited_year),
            )
        graph.add_edge(
            f"work:{row.citing_work_id}",
            cited,
            key=f"cites:{row.citing_work_id}:{row.cited_work_id}",
            relation="cites",
        )


def _add_network_relations(
    graph: nx.MultiDiGraph,
    network_name: str,
    network: NetworkResult,
) -> None:
    if network.nodes.empty:
        return
    network_id = f"network:{network_name}"
    graph.add_node(
        network_id,
        kind="network",
        label=network_name.replace("_", " "),
        network=network_name,
        node_count=len(network.nodes),
        edge_count=len(network.edges),
    )
    for row in network.nodes.to_dict(orient="records"):
        raw_id = row["id"]
        node_id = _network_node_id(network_name, raw_id)
        if node_id not in graph:
            graph.add_node(
                node_id,
                kind="network_entity",
                label=_json_value(row.get("label")) or str(raw_id),
                raw_id=str(raw_id),
            )
        for attribute in ("occurrences", "degree", "weighted_degree", "betweenness"):
            if attribute in row:
                graph.nodes[node_id][f"{network_name}_{attribute}"] = _json_value(
                    row.get(attribute)
                )
        graph.add_edge(
            node_id,
            network_id,
            key=f"participates_in:{network_name}:{raw_id}",
            relation="participates_in_network",
            network=network_name,
        )
        cluster = _json_value(row.get("cluster"))
        if cluster not in (None, 0):
            cluster_id = f"cluster:{network_name}:{int(cluster)}"
            graph.add_node(
                cluster_id,
                kind="cluster",
                label=f"{network_name} cluster {int(cluster)}",
                network=network_name,
                cluster=int(cluster),
            )
            graph.add_edge(
                node_id,
                cluster_id,
                key=f"member_of:{network_name}:{raw_id}:{int(cluster)}",
                relation="member_of_cluster",
                network=network_name,
            )

    for row in network.edges.to_dict(orient="records"):
        source = _network_node_id(network_name, row["source"])
        target = _network_node_id(network_name, row["target"])
        edge_id = _edge_id(network_name, source, target)
        graph.add_edge(
            source,
            target,
            key=edge_id,
            edge_id=edge_id,
            relation=network_name,
            network=network_name,
            weight=_json_value(row.get("weight")),
            association_strength=_json_value(row.get("association_strength")),
        )


def _network_facts(analyses: AnalysisBundle) -> list[GraphFact]:
    facts: list[GraphFact] = []

    def add(
        network: str,
        fact_type: str,
        statement: str,
        value: dict[str, Any],
        nodes: list[str],
        edges: list[str] | None = None,
        caveat: str = (
            "This fact describes the frozen corpus and network parameterization; "
            "it does not imply causality or research quality."
        ),
    ) -> None:
        fact_id = f"G{len(facts) + 1:03d}"
        facts.append(
            GraphFact(
                fact_id=fact_id,
                network=network,
                fact_type=fact_type,
                statement=statement,
                value=_json_value(value),
                evidence_nodes=[
                    f"fact:{fact_id}",
                    f"network:{network}",
                    *nodes,
                ],
                evidence_edges=edges or [],
                caveat=caveat,
            )
        )

    for name, network in analyses.networks.items():
        if network.nodes.empty:
            continue
        add(
            name,
            "network_size",
            f"The {name.replace('_', ' ')} network contains {len(network.nodes)} nodes "
            f"and {len(network.edges)} edges.",
            {"nodes": len(network.nodes), "edges": len(network.edges)},
            [],
        )

        ranked = network.nodes.sort_values(
            ["weighted_degree", "occurrences", "label"],
            ascending=[False, False, True],
        )
        top = ranked.iloc[0]
        top_node = _network_node_id(name, top["id"])
        add(
            name,
            "highest_weighted_degree",
            f"{top['label']} has the highest weighted degree in the "
            f"{name.replace('_', ' ')} network.",
            {
                "id": str(top["id"]),
                "label": str(top["label"]),
                "weighted_degree": float(top["weighted_degree"]),
                "cluster": int(top["cluster"]),
            },
            [top_node],
        )

        clusters = sorted(
            {
                int(value)
                for value in network.nodes.get("cluster", pd.Series(dtype=int)).dropna()
                if int(value) > 0
            }
        )
        add(
            name,
            "cluster_count",
            f"The {name.replace('_', ' ')} network contains {len(clusters)} detected clusters.",
            {"clusters": len(clusters), "cluster_ids": clusters},
            [f"cluster:{name}:{cluster}" for cluster in clusters],
        )

        if not network.edges.empty:
            edge_frame = network.edges.copy()
            edge_frame["_source"] = edge_frame["source"].astype(str)
            edge_frame["_target"] = edge_frame["target"].astype(str)
            strongest = edge_frame.sort_values(
                ["weight", "_source", "_target"],
                ascending=[False, True, True],
            ).iloc[0]
            source = _network_node_id(name, strongest["source"])
            target = _network_node_id(name, strongest["target"])
            labels = network.nodes.set_index(network.nodes["id"].astype(str))["label"].to_dict()
            source_label = str(labels.get(str(strongest["source"]), strongest["source"]))
            target_label = str(labels.get(str(strongest["target"]), strongest["target"]))
            source_display = source_label
            target_display = target_label
            if source_label == target_label:
                source_display = f"{source_label} [{strongest['source']}]"
                target_display = f"{target_label} [{strongest['target']}]"
            edge_id = _edge_id(name, source, target)
            add(
                name,
                "strongest_edge",
                f"The strongest observed {name.replace('_', ' ')} connection is between "
                f"{source_display} and {target_display}, "
                f"with weight {float(strongest['weight']):g}.",
                {
                    "source": str(strongest["source"]),
                    "source_label": source_label,
                    "target": str(strongest["target"]),
                    "target_label": target_label,
                    "weight": float(strongest["weight"]),
                    "association_strength": float(strongest["association_strength"]),
                },
                [source, target],
                [edge_id],
            )
    return facts


def _add_fact_nodes(graph: nx.MultiDiGraph, facts: list[GraphFact]) -> None:
    for fact in facts:
        fact_node = f"fact:{fact.fact_id}"
        network_node = f"network:{fact.network}"
        graph.add_node(
            fact_node,
            kind="graph_fact",
            label=fact.statement,
            fact_id=fact.fact_id,
            fact_type=fact.fact_type,
            network=fact.network,
            value=fact.value,
            caveat=fact.caveat,
            evidence_edges=fact.evidence_edges,
        )
        graph.add_edge(
            fact_node,
            network_node,
            key=f"describes_network:{fact.fact_id}:{fact.network}",
            relation="describes_network",
        )
        for evidence_node in fact.evidence_nodes:
            if evidence_node in {fact_node, network_node} or evidence_node not in graph:
                continue
            graph.add_edge(
                evidence_node,
                fact_node,
                key=f"supports_fact:{fact.fact_id}:{evidence_node}",
                relation="supports_fact",
            )


def build_bibliometric_knowledge_graph(
    tables: CanonicalTables,
    analyses: AnalysisBundle,
) -> GraphGroundingBundle:
    graph = nx.MultiDiGraph()
    graph.graph.update(
        {
            "name": "CiteWeave bibliometric knowledge graph",
            "schema_version": 2,
        }
    )
    _add_table_nodes(graph, tables)
    _add_canonical_relations(graph, tables)
    for name, network in analyses.networks.items():
        _add_network_relations(graph, name, network)
    facts = _network_facts(analyses)
    _add_fact_nodes(graph, facts)
    return GraphGroundingBundle(graph=graph, facts=facts)


def build_bounded_network_knowledge_graph(
    analyses: AnalysisBundle,
) -> GraphGroundingBundle:
    """Build graph-grounding artifacts from already bounded network views.

    This deliberately excludes corpus-wide canonical entity and relation tables.
    It is the scalable counterpart of :func:`build_bibliometric_knowledge_graph`
    for formal datasets whose full relations must remain on disk.
    """
    graph = nx.MultiDiGraph()
    graph.graph.update(
        {
            "name": "CiteWeave bounded bibliometric network knowledge graph",
            "schema_version": 2,
            "scope": "selected visualization networks; canonical relations remain disk-backed",
        }
    )
    for name, network in analyses.networks.items():
        _add_network_relations(graph, name, network)
    facts = _network_facts(analyses)
    _add_fact_nodes(graph, facts)
    return GraphGroundingBundle(graph=graph, facts=facts)


def retrieve_graph_facts(
    facts: list[GraphFact],
    *,
    network: str | None = None,
    query: str | None = None,
    limit: int = 16,
) -> list[GraphFact]:
    candidates = [fact for fact in facts if network is None or fact.network == network]
    if not query:
        return candidates[:limit]
    terms = set(re.findall(r"[a-z0-9]+", query.casefold()))

    def score(fact: GraphFact) -> tuple[int, str]:
        haystack = " ".join(
            [
                fact.network,
                fact.fact_type,
                fact.statement,
                json.dumps(fact.value, ensure_ascii=False),
            ]
        ).casefold()
        overlap = sum(term in haystack for term in terms)
        return (-overlap, fact.fact_id)

    return sorted(candidates, key=score)[:limit]


def build_graph_qa_items(bundle: GraphGroundingBundle, dataset_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fact in bundle.facts:
        question_by_type = {
            "network_size": f"How many nodes and edges are in the {fact.network} network?",
            "highest_weighted_degree": (
                f"Which node has the highest weighted degree in the {fact.network} network?"
            ),
            "cluster_count": f"How many clusters are detected in the {fact.network} network?",
            "strongest_edge": (
                f"Which pair has the strongest connection in the {fact.network} network?"
            ),
        }
        items.append(
            {
                "item_id": f"{dataset_id}:{fact.fact_id}",
                "dataset_id": dataset_id,
                "network": fact.network,
                "task_type": fact.fact_type,
                "question": question_by_type[fact.fact_type],
                "answerable": True,
                "gold_answer": fact.value,
                "gold_statement": fact.statement,
                "gold_evidence_nodes": fact.evidence_nodes,
                "gold_evidence_edges": fact.evidence_edges,
                "caveat": fact.caveat,
            }
        )
    for network in sorted({fact.network for fact in bundle.facts}):
        items.append(
            {
                "item_id": f"{dataset_id}:U:{network}",
                "dataset_id": dataset_id,
                "network": network,
                "task_type": "unanswerable_false_premise",
                "question": (
                    f"Which cluster contains the node 'CiteWeave absent benchmark node' "
                    f"in the {network} network?"
                ),
                "answerable": False,
                "gold_answer": None,
                "gold_statement": "The named node is absent, so the question is not answerable.",
                "gold_evidence_nodes": [],
                "gold_evidence_edges": [],
                "caveat": "The system should abstain rather than invent a cluster.",
                "evidence_operation": {
                    "type": "node_absence_check",
                    "network_node": f"network:{network}",
                    "target_label": "CiteWeave absent benchmark node",
                },
            }
        )
        items[-1]["gold_evidence_nodes"] = [f"network:{network}"]
    return items


def _graphml_safe_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    safe = nx.MultiDiGraph()
    safe.graph.update({key: str(value) for key, value in graph.graph.items()})
    for node, data in graph.nodes(data=True):
        safe.add_node(
            node,
            **{
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple, set))
                    else ("" if value is None else value)
                )
                for key, value in data.items()
            },
        )
    for source, target, key, data in graph.edges(keys=True, data=True):
        safe.add_edge(
            source,
            target,
            key=key,
            **{
                field: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple, set))
                    else ("" if value is None else value)
                )
                for field, value in data.items()
            },
        )
    return safe


def save_graph_grounding(
    bundle: GraphGroundingBundle,
    out_dir: Path,
    *,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "bibliometric_kg.json",
        {
            "schema_version": 2,
            "nodes": [
                {"id": node, **{key: _json_value(value) for key, value in data.items()}}
                for node, data in bundle.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "key": key,
                    **{field: _json_value(value) for field, value in data.items()},
                }
                for source, target, key, data in bundle.graph.edges(keys=True, data=True)
            ],
        },
    )
    write_json(out_dir / "graph_facts.json", [asdict(fact) for fact in bundle.facts])
    if dataset_id:
        write_json(
            out_dir / "graph_qa_benchmark.json",
            build_graph_qa_items(bundle, dataset_id),
        )
    nx.write_graphml(_graphml_safe_graph(bundle.graph), out_dir / "bibliometric_kg.graphml")
    summary = {
        "schema_version": 2,
        "nodes": bundle.graph.number_of_nodes(),
        "edges": bundle.graph.number_of_edges(),
        "facts": len(bundle.facts),
        "qa_items": len(build_graph_qa_items(bundle, dataset_id)) if dataset_id else 0,
    }
    write_json(out_dir / "graph_grounding_manifest.json", summary)
    return summary
