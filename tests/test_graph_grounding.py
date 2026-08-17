import json

import networkx as nx

from citeweave.analytics import analyze
from citeweave.graph_grounding import (
    _network_node_id,
    build_bibliometric_knowledge_graph,
    build_graph_qa_items,
    retrieve_graph_facts,
    save_graph_grounding,
)
from citeweave.transform import Canonicalizer


def _bundle(crossref_records):
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    analyses = analyze(tables, network_candidate_pool=100)
    return tables, build_bibliometric_knowledge_graph(tables, analyses)


def test_bibliometric_kg_contains_entities_relations_and_network_facts(crossref_records):
    tables, bundle = _bundle(crossref_records)

    first_work = f"work:{tables.works.iloc[0]['work_id']}"
    first_author = str(tables.authors.iloc[0]["author_id"])
    assert first_work in bundle.graph
    assert first_author in bundle.graph
    assert f"author:{first_author}" not in bundle.graph
    assert any(data["relation"] == "authored" for *_, data in bundle.graph.edges(data=True))
    assert any(data["relation"] == "keyword_cooccurrence" for *_, data in bundle.graph.edges(data=True))
    assert {fact.fact_type for fact in bundle.facts} >= {
        "network_size",
        "highest_weighted_degree",
        "cluster_count",
        "strongest_edge",
    }
    assert all(f"fact:{fact.fact_id}" in bundle.graph for fact in bundle.facts)
    assert all(f"network:{fact.network}" in fact.evidence_nodes for fact in bundle.facts)


def test_network_node_mapping_is_idempotent():
    assert _network_node_id("coauthorship", "author:a1") == "author:a1"
    assert _network_node_id("citation", "work:w1") == "work:w1"
    assert _network_node_id("keyword_cooccurrence", "keyword:graph") == "keyword:graph"


def test_graph_qa_has_answerable_and_false_premise_items(crossref_records):
    _, bundle = _bundle(crossref_records)
    items = build_graph_qa_items(bundle, "fixture")

    assert any(item["answerable"] for item in items)
    assert any(not item["answerable"] for item in items)
    assert all(item["question"].isascii() for item in items)
    assert all(item["dataset_id"] == "fixture" for item in items)
    assert all(item["gold_evidence_nodes"] for item in items)


def test_graph_fact_retrieval_prefers_matching_network(crossref_records):
    _, bundle = _bundle(crossref_records)
    facts = retrieve_graph_facts(
        bundle.facts,
        network="keyword_cooccurrence",
        query="strongest keyword connection",
        limit=2,
    )

    assert facts
    assert all(fact.network == "keyword_cooccurrence" for fact in facts)
    assert facts[0].fact_type == "strongest_edge"


def test_graph_grounding_artifacts_are_serializable(crossref_records, tmp_path):
    _, bundle = _bundle(crossref_records)
    summary = save_graph_grounding(bundle, tmp_path, dataset_id="fixture")

    assert summary["nodes"] == bundle.graph.number_of_nodes()
    assert summary["edges"] == bundle.graph.number_of_edges()
    assert summary["facts"] == len(bundle.facts)
    assert json.loads((tmp_path / "graph_qa_benchmark.json").read_text(encoding="utf-8"))
    loaded = nx.read_graphml(tmp_path / "bibliometric_kg.graphml")
    assert loaded.number_of_nodes() == bundle.graph.number_of_nodes()
