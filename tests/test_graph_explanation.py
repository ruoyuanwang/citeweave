import json
from pathlib import Path

import pandas as pd

from citeweave.analytics import NetworkResult
from citeweave.graph_explanation import (
    alias_graph,
    build_prompt,
    displayed_graph,
    explain_network,
    graph_rag_context,
    score_response,
    verify_response,
)
from citeweave.models import GraphExplanationPolicy
from citeweave.visualization import FigureArtifact


def sample_network() -> NetworkResult:
    nodes = pd.DataFrame(
        [
            {
                "id": "a",
                "label": "Alpha",
                "occurrences": 10,
                "degree": 2,
                "weighted_degree": 8.0,
                "betweenness": 0.0,
                "cluster": 1,
            },
            {
                "id": "b",
                "label": "Beta",
                "occurrences": 8,
                "degree": 3,
                "weighted_degree": 9.0,
                "betweenness": 0.7,
                "cluster": 1,
            },
            {
                "id": "c",
                "label": "Gamma",
                "occurrences": 7,
                "degree": 2,
                "weighted_degree": 7.0,
                "betweenness": 0.4,
                "cluster": 2,
            },
            {
                "id": "d",
                "label": "Delta",
                "occurrences": 5,
                "degree": 1,
                "weighted_degree": 3.0,
                "betweenness": 0.0,
                "cluster": 2,
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "a", "target": "b", "weight": 5, "association_strength": 0.2},
            {"source": "b", "target": "c", "weight": 3, "association_strength": 0.15},
            {"source": "c", "target": "d", "weight": 2, "association_strength": 0.1},
        ]
    )
    return NetworkResult("keyword_cooccurrence", nodes, edges, {})


class FakeClient:
    def complete(self, image_path: Path, prompt: str):
        assert "最终论文" in prompt
        return (
            json.dumps(
                {
                    "overview": "网络包含两个由桥接关系连接的社区。",
                    "claims": [
                        {
                            "type": "multi_hop",
                            "statement": "Alpha经由Beta和Gamma连接Delta。",
                            "nodes": ["N1", "N2", "N4", "N3"],
                            "communities": {"N1": 1, "N2": 1, "N4": 2, "N3": 2},
                            "path": ["N1", "N2", "N4", "N3"],
                            "evidence_edges": [["N1", "N2"], ["N2", "N4"], ["N4", "N3"]],
                        },
                        {
                            "type": "cross_community",
                            "statement": "Alpha与Delta直接相连。",
                            "nodes": ["N1", "N3"],
                            "communities": {"N1": 1, "N3": 2},
                            "path": [],
                            "evidence_edges": [["N1", "N3"]],
                        },
                    ],
                    "caveats": ["共现不表示因果"],
                },
                ensure_ascii=False,
            ),
            {"total_tokens": 123},
        )


def test_explanation_keeps_only_verified_relations(tmp_path):
    image = tmp_path / "network_keyword_cooccurrence.png"
    image.write_bytes(b"not-read-by-fake-client")
    figure = FigureArtifact(
        "network_keyword_cooccurrence",
        image,
        tmp_path / "network_keyword_cooccurrence.svg",
        {},
        {},
    )
    policy = GraphExplanationPolicy(mode="graph_rag", max_paths=4, max_hops=4)

    result = explain_network(
        sample_network(),
        figure,
        policy,
        max_nodes=10,
        client=FakeClient(),
    )

    assert result["status"] == "complete"
    assert result["verification"] == {
        "reported_claims": 2,
        "verified_claims": 1,
        "rejected_claims": 1,
        "claim_support_rate": 0.5,
    }
    assert result["verified_claims"][0]["type"] == "multi_hop"
    assert result["rejected_claims"][0]["reason"] == "unsupported evidence edge"
    assert result["metrics"]["edge_hallucination_rate"] == 0.25
    assert result["metrics"]["path_validity_rate"] == 1.0
    assert result["metrics"]["verified_complex_claims"] == 1


def test_vlm_prompt_does_not_leak_graph_structure():
    graph, lookup = displayed_graph(sample_network(), 10)
    _, _, nodes, edges = alias_graph(graph, lookup)

    prompt = build_prompt("vlm", nodes, edges, None)

    assert '"weighted_degree"' not in prompt
    assert '"occurrences"' not in prompt
    assert '"community":' not in prompt
    assert '"association_strength"' not in prompt
    assert "本组只提供PNG" in prompt
    assert "禁止输出“hub_or_bridge”" in prompt
    assert "focus_node中填写唯一" in prompt
    assert [node["label"] for node in nodes] == ["Alpha", "Beta", "Delta", "Gamma"]

    flat_prompt = build_prompt("flat_kg", nodes, edges, None)
    assert '"weighted_degree"' in flat_prompt
    assert '"association_strength"' in flat_prompt


def test_graph_rag_context_exposes_verifier_defined_hub_evidence():
    graph, lookup = displayed_graph(sample_network(), 10)
    id_to_alias, _, nodes, edges = alias_graph(graph, lookup)

    retrieval = graph_rag_context(
        graph,
        lookup,
        id_to_alias,
        max_paths=4,
        max_hops=4,
    )

    assert retrieval["hub_candidates"] == [
        {
            "node": "N2",
            "degree": 3,
            "weighted_degree": 9.0,
            "incident_edges": [
                {
                    "source": "N2",
                    "target": "N1",
                    "weight": 5,
                    "source_community": 1,
                    "target_community": 1,
                },
                {
                    "source": "N2",
                    "target": "N4",
                    "weight": 3,
                    "source_community": 1,
                    "target_community": 2,
                },
            ],
        }
    ]
    prompt = build_prompt("graph_rag", nodes, edges, retrieval)
    assert "hub槽必须从hub_candidates选择focus_node" in prompt
    assert '"hub_candidates"' in prompt


def test_graph_rag_context_exposes_copy_ready_multi_hop_candidates():
    graph, lookup = displayed_graph(sample_network(), 10)
    # Make Delta the representative of community 2 so the representative path
    # crosses Beta -> Gamma -> Delta and has the required two hops.
    lookup["d"]["betweenness"] = 2.0
    id_to_alias, _, nodes, edges = alias_graph(graph, lookup)

    retrieval = graph_rag_context(
        graph,
        lookup,
        id_to_alias,
        max_paths=4,
        max_hops=4,
    )

    assert retrieval["multi_hop_paths"] == [
        {
            "path_id": "P1",
            "nodes": ["N2", "N4", "N3"],
            "communities": {"N2": 1, "N4": 2, "N3": 2},
            "path": ["N2", "N4", "N3"],
            "evidence_edges": [["N2", "N4"], ["N4", "N3"]],
        }
    ]
    prompt = build_prompt("graph_rag", nodes, edges, retrieval)
    assert "选择两个不同的path_id" in prompt
    assert "四个字段逐字复制" in prompt


def test_verifier_rejects_duplicate_and_non_simple_paths():
    aliases = {"N1": "a", "N2": "b", "N3": "c"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1, "weighted_degree": 3},
        {"alias": "N2", "label": "Beta", "community": 1, "weighted_degree": 2},
        {"alias": "N3", "label": "Gamma", "community": 2, "weighted_degree": 1},
    ]
    edges = [
        {"source": "N1", "target": "N2", "weight": 1},
        {"source": "N2", "target": "N3", "weight": 1},
    ]
    valid = {
        "type": "multi_hop",
        "statement": "有效路径。",
        "nodes": ["N1", "N2", "N3"],
        "communities": {"N1": 1, "N2": 1, "N3": 2},
        "path": ["N1", "N2", "N3"],
        "evidence_edges": [["N1", "N2"], ["N2", "N3"]],
    }
    cyclic = {
        **valid,
        "statement": "循环路径。",
        "path": ["N1", "N2", "N1"],
        "evidence_edges": [["N1", "N2"]],
    }

    verified, rejected = verify_response(
        {"claims": [valid, valid, cyclic]}, aliases, nodes, edges
    )

    assert len(verified) == 1
    assert rejected[0]["reason"] == "duplicate structural claim"
    assert rejected[1]["reason"] == "invalid multi-hop path"


def test_verifier_requires_all_cross_edges_to_cross_communities():
    aliases = {"N1": "a", "N2": "b", "N3": "c"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1, "weighted_degree": 3},
        {"alias": "N2", "label": "Beta", "community": 1, "weighted_degree": 2},
        {"alias": "N3", "label": "Gamma", "community": 2, "weighted_degree": 1},
    ]
    edges = [
        {"source": "N1", "target": "N2", "weight": 1},
        {"source": "N2", "target": "N3", "weight": 1},
    ]
    parsed = {
        "claims": [
            {
                "type": "cross_community",
                "statement": "混合了内部边。",
                "nodes": ["N1", "N2", "N3"],
                "communities": {"N1": 1, "N2": 1, "N3": 2},
                "evidence_edges": [["N1", "N2"], ["N2", "N3"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert rejected[0]["reason"] == "claim has no cross-community edge"


def test_score_response_counts_edges_and_paths():
    graph, lookup = displayed_graph(sample_network(), 10)
    _, aliases, nodes, edges = alias_graph(graph, lookup)
    parsed = json.loads(FakeClient().complete(Path("unused.png"), "最终论文")[0])
    verified, _ = verify_response(parsed, aliases, nodes, edges)

    metrics = score_response(parsed, nodes, edges, verified)

    assert metrics["reported_claims"] == 2
    assert metrics["verified_claims"] == 1
    assert metrics["reported_edges"] == 4
    assert metrics["supported_edges"] == 3
    assert metrics["unsupported_edges"] == 1
    assert metrics["edge_hallucination_rate"] == 0.25
    assert metrics["reported_multi_hop_claims"] == 1
    assert metrics["valid_multi_hop_claims"] == 1


def test_verifier_rejects_wrong_community_assignment():
    aliases = {"N1": "a", "N2": "b"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1},
        {"alias": "N2", "label": "Beta", "community": 2},
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 1}]
    parsed = {
        "claims": [
            {
                "type": "cross_community",
                "statement": "关系存在。",
                "nodes": ["N1", "N2"],
                "communities": {"N1": 9, "N2": 2},
                "evidence_edges": [["N1", "N2"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert rejected[0]["reason"] == "incorrect community assignment"


def test_verifier_rejects_non_numeric_community_without_crashing():
    aliases = {"N1": "a", "N2": "b"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1},
        {"alias": "N2", "label": "Beta", "community": 2},
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 1}]
    parsed = {
        "claims": [
            {
                "type": "cross_community",
                "statement": "关系存在。",
                "nodes": ["N1", "N2"],
                "communities": {"N1": "unknown", "N2": 2},
                "evidence_edges": [["N1", "N2"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert rejected[0]["reason"] == "incorrect community assignment"


def test_verifier_rejects_non_object_communities_without_crashing():
    aliases = {"N1": "a", "N2": "b"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1},
        {"alias": "N2", "label": "Beta", "community": 2},
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 1}]
    parsed = {
        "claims": [
            {
                "type": "cross_community",
                "statement": "关系存在。",
                "nodes": ["N1", "N2"],
                "communities": [1, 2],
                "evidence_edges": [["N1", "N2"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert rejected[0]["reason"] == "communities is not an object"


def test_verifier_accepts_one_hub_with_ordinary_evidence_neighbor():
    aliases = {f"N{index}": f"id-{index}" for index in range(1, 11)}
    nodes = [
        {
            "alias": f"N{index}",
            "label": f"Node {index}",
            "community": 1,
            "degree": 11 - index,
            "weighted_degree": 11 - index,
        }
        for index in range(1, 11)
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 5}]
    parsed = {
        "claims": [
            {
                "type": "hub",
                "statement": "N1是hub。",
                "focus_node": "N1",
                "nodes": ["N2", "N1"],
                "communities": {"N1": 1, "N2": 1},
                "path": [],
                "evidence_edges": [["N1", "N2"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not rejected
    assert len(verified) == 1
    assert verified[0]["focus_node"] == "N1"
    assert verified[0]["nodes"][0]["alias"] == "N2"
    assert "Node 1" in verified[0]["statement"]
    assert "Node 2" not in verified[0]["statement"]


def test_verifier_normalizes_valid_focus_node_omitted_from_nodes():
    aliases = {f"N{index}": f"id-{index}" for index in range(1, 11)}
    nodes = [
        {
            "alias": f"N{index}",
            "label": f"Node {index}",
            "community": 1,
            "degree": 11 - index,
            "weighted_degree": 11 - index,
        }
        for index in range(1, 11)
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 5}]
    parsed = {
        "claims": [
            {
                "type": "hub",
                "statement": "N1是hub。",
                "focus_node": "N1",
                "nodes": ["N2"],
                "communities": {"N2": 1},
                "path": [],
                "evidence_edges": [["N1", "N2"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not rejected
    assert [node["alias"] for node in verified[0]["nodes"]] == ["N1", "N2"]
    assert verified[0]["normalizations"] == [
        "focus_node_added_to_nodes",
        "focus_node_community_added",
    ]


def test_focus_normalization_does_not_accept_false_hub_edge():
    aliases = {f"N{index}": f"id-{index}" for index in range(1, 11)}
    nodes = [
        {
            "alias": f"N{index}",
            "label": f"Node {index}",
            "community": 1,
            "degree": 11 - index,
            "weighted_degree": 11 - index,
        }
        for index in range(1, 11)
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 5}]
    parsed = {
        "claims": [
            {
                "type": "hub",
                "statement": "N1是hub。",
                "focus_node": "N1",
                "nodes": ["N3"],
                "communities": {"N3": 1},
                "path": [],
                "evidence_edges": [["N1", "N3"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert rejected[0]["reason"] == "unsupported evidence edge"


def test_verifier_rejects_non_hub_or_ambiguous_hub_focus():
    aliases = {f"N{index}": f"id-{index}" for index in range(1, 11)}
    nodes = [
        {
            "alias": f"N{index}",
            "label": f"Node {index}",
            "community": 1,
            "degree": 11 - index,
            "weighted_degree": 11 - index,
        }
        for index in range(1, 11)
    ]
    edges = [{"source": "N1", "target": "N2", "weight": 5}]
    base = {
        "type": "hub",
        "statement": "hub声明。",
        "nodes": ["N1", "N2"],
        "communities": {"N1": 1, "N2": 1},
        "path": [],
        "evidence_edges": [["N1", "N2"]],
    }
    parsed = {
        "claims": [
            {**base, "focus_node": "N2"},
            {**base, "focus_node": ["N1", "N2"]},
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert "claimed hub is outside the top weighted-degree decile" in rejected[0]["reason"]
    assert "missing or invalid focus node" in rejected[1]["reason"]


def test_verifier_accepts_bridge_focus_independent_of_node_order():
    aliases = {"N1": "a", "N2": "b", "N3": "c"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1, "weighted_degree": 1},
        {"alias": "N2", "label": "Beta", "community": 1, "weighted_degree": 2},
        {"alias": "N3", "label": "Gamma", "community": 2, "weighted_degree": 1},
    ]
    edges = [
        {"source": "N1", "target": "N2", "weight": 2},
        {"source": "N2", "target": "N3", "weight": 3},
    ]
    parsed = {
        "claims": [
            {
                "type": "bridge",
                "statement": "Beta是割点。",
                "focus_node": "N2",
                "nodes": ["N3", "N1", "N2"],
                "communities": {"N1": 1, "N2": 1, "N3": 2},
                "path": [],
                "evidence_edges": [["N1", "N2"], ["N2", "N3"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not rejected
    assert len(verified) == 1
    assert verified[0]["focus_node"] == "N2"
    assert "Beta" in verified[0]["statement"]
    assert "割点" in verified[0]["statement"]


def test_verifier_rejects_wrong_bridge_focus():
    aliases = {"N1": "a", "N2": "b", "N3": "c"}
    nodes = [
        {"alias": "N1", "label": "Alpha", "community": 1, "weighted_degree": 1},
        {"alias": "N2", "label": "Beta", "community": 1, "weighted_degree": 2},
        {"alias": "N3", "label": "Gamma", "community": 2, "weighted_degree": 1},
    ]
    edges = [
        {"source": "N1", "target": "N2", "weight": 2},
        {"source": "N2", "target": "N3", "weight": 3},
    ]
    parsed = {
        "claims": [
            {
                "type": "bridge",
                "statement": "Alpha是割点。",
                "focus_node": "N1",
                "nodes": ["N1", "N2", "N3"],
                "communities": {"N1": 1, "N2": 1, "N3": 2},
                "path": [],
                "evidence_edges": [["N1", "N2"], ["N2", "N3"]],
            }
        ]
    }

    verified, rejected = verify_response(parsed, aliases, nodes, edges)

    assert not verified
    assert "claimed bridge is not an articulation point" in rejected[0]["reason"]
    assert "bridge evidence edges are not incident" in rejected[0]["reason"]
