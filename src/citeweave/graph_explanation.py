from __future__ import annotations

import base64
import json
import os
import re
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import httpx
import networkx as nx

from .analytics import AnalysisBundle, NetworkResult
from .exceptions import ConfigurationError
from .models import GraphExplanationPolicy
from .visualization import FigureArtifact, _select_network

SYSTEM_PROMPT = """你是CiteWeave中的图解释节点。你的任务不是回答独立图问答，而是为论文结果章节
生成可验证的图解释证据。只能使用当前图片和结构化图证据，不得依据关键词常识补边，不得把共现
解释为因果。每条关系声明必须列出支撑它的节点、社区和真实证据边。只输出JSON对象。"""

OUTPUT_SCHEMA = {
    "overview": "一句不含新关系的总体观察",
    "claims": [
        {
            "type": "hub|community_structure|cross_community|multi_hop|bridge",
            "statement": "可直接用于论文写作的克制表述",
            "focus_node": "N1（仅hub或bridge使用；其他类型留空）",
            "nodes": ["N1", "N2"],
            "communities": {"N1": 1, "N2": 2},
            "path": ["N1", "N3", "N2"],
            "evidence_edges": [["N1", "N3"], ["N3", "N2"]],
        }
    ],
    "abstentions": [
        {
            "slot": "community_structure|cross_community|multi_hop_1|multi_hop_2|hub_or_bridge",
            "reason": "当前输入不足以支持该槽位",
        }
    ],
    "caveats": ["解释边界"],
}

ALLOWED_CLAIM_TYPES = {
    "hub",
    "community_structure",
    "cross_community",
    "multi_hop",
    "bridge",
}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left).upper(), str(right).upper())))


def _hub_candidate_aliases(nodes: list[dict[str, Any]]) -> list[str]:
    """Return the verifier-defined top weighted-degree decile deterministically."""
    hub_budget = max(1, round(len(nodes) * 0.1))
    return [
        str(node["alias"])
        for node in sorted(
            nodes,
            key=lambda item: (
                -float(item.get("weighted_degree") or 0),
                -float(item.get("degree") or 0),
                str(item.get("alias") or ""),
            ),
        )[:hub_budget]
    ]


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ConfigurationError("Graph explanation model did not return a JSON object.")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ConfigurationError("Graph explanation model returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError("Graph explanation response must be a JSON object.")
    return parsed


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenVisionClient:
    def __init__(self, policy: GraphExplanationPolicy):
        self.policy = policy
        self.api_key = os.getenv(policy.api_key_env)
        if not self.api_key:
            raise ConfigurationError(
                f"{policy.api_key_env} is required when graph_explanation.mode is enabled."
            )
        self.client = httpx.Client(timeout=180, follow_redirects=True)

    def complete(self, image_path: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        response = self.client.post(
            f"{self.policy.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.policy.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_url(image_path)},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "temperature": self.policy.temperature,
                "stream": False,
                "enable_thinking": False,
            },
        )
        if response.status_code >= 400:
            sanitized = response.text.replace(self.api_key, "***")
            raise ConfigurationError(
                f"Graph explanation API returned {response.status_code}: {sanitized[:500]}"
            )
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"].strip()
        return answer, _json_safe(payload.get("usage", {}))


def displayed_graph(
    network: NetworkResult, max_nodes: int
) -> tuple[nx.Graph, dict[str, dict[str, Any]]]:
    nodes, edges, _ = _select_network(network, max_nodes)
    graph = nx.Graph()
    lookup: dict[str, dict[str, Any]] = {}
    for row in nodes.to_dict("records"):
        node_id = str(row["id"])
        lookup[node_id] = _json_safe(row)
        graph.add_node(node_id, **lookup[node_id])
    for row in edges.to_dict("records"):
        left, right = str(row["source"]), str(row["target"])
        if left in graph and right in graph:
            graph.add_edge(left, right, **_json_safe(row))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    lookup = {node: lookup[node] for node in graph}
    return graph, lookup


def alias_graph(
    graph: nx.Graph, lookup: dict[str, dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        graph.nodes,
        key=lambda node: (
            str(lookup[node].get("label") or node).casefold(),
            str(node).casefold(),
        ),
    )
    id_to_alias = {node: f"N{index}" for index, node in enumerate(ordered, start=1)}
    alias_to_id = {alias: node for node, alias in id_to_alias.items()}
    nodes = [
        {
            "alias": id_to_alias[node],
            "id": node,
            "label": str(lookup[node].get("label") or node),
            "community": int(lookup[node].get("cluster") or 0),
            "occurrences": int(lookup[node].get("occurrences") or 0),
            "degree": int(lookup[node].get("degree") or graph.degree(node)),
            "weighted_degree": float(lookup[node].get("weighted_degree") or 0),
            "betweenness": float(lookup[node].get("betweenness") or 0),
        }
        for node in ordered
    ]
    edges = [
        {
            "source": id_to_alias[left],
            "target": id_to_alias[right],
            "weight": int(data.get("weight") or 0),
            "association_strength": float(
                data.get("association_strength") or data.get("normalized") or 0
            ),
        }
        for left, right, data in sorted(
            graph.edges(data=True),
            key=lambda item: (-float(item[2].get("weight") or 0), item[0], item[1]),
        )
    ]
    return id_to_alias, alias_to_id, nodes, edges


def graph_rag_context(
    graph: nx.Graph,
    lookup: dict[str, dict[str, Any]],
    id_to_alias: dict[str, str],
    *,
    max_paths: int,
    max_hops: int,
) -> dict[str, Any]:
    ranked = sorted(
        graph.nodes,
        key=lambda node: (
            -float(lookup[node].get("betweenness") or 0),
            -float(lookup[node].get("weighted_degree") or 0),
            -float(lookup[node].get("occurrences") or 0),
        ),
    )
    community_representatives: dict[int, list[str]] = {}
    for node in ranked:
        community = int(lookup[node].get("cluster") or 0)
        community_representatives.setdefault(community, [])
        if len(community_representatives[community]) < 3:
            community_representatives[community].append(id_to_alias[node])

    cross_edges = []
    community_edges: dict[int, list[dict[str, Any]]] = {}
    for left, right, data in graph.edges(data=True):
        source_community = int(lookup[left].get("cluster") or 0)
        target_community = int(lookup[right].get("cluster") or 0)
        edge_record = {
            "source": id_to_alias[left],
            "target": id_to_alias[right],
            "weight": int(data.get("weight") or 0),
            "source_community": source_community,
            "target_community": target_community,
        }
        if source_community == target_community:
            community_edges.setdefault(source_community, []).append(edge_record)
            continue
        cross_edges.append(edge_record)
    cross_edges.sort(key=lambda item: (-item["weight"], item["source"], item["target"]))
    for values in community_edges.values():
        values.sort(key=lambda item: (-item["weight"], item["source"], item["target"]))

    representative_ids = [
        alias_list[0] for _, alias_list in sorted(community_representatives.items()) if alias_list
    ]
    alias_to_id = {alias: node for node, alias in id_to_alias.items()}
    paths: list[list[str]] = []
    for source_alias, target_alias in combinations(representative_ids, 2):
        source, target = alias_to_id[source_alias], alias_to_id[target_alias]
        try:
            path = nx.shortest_path(graph, source, target)
        except nx.NetworkXNoPath:
            continue
        if 2 <= len(path) - 1 <= max_hops:
            paths.append([id_to_alias[node] for node in path])
    paths.sort(key=lambda path: (len(path), path))
    multi_hop_candidates = []
    for index, path in enumerate(paths[:max_paths], start=1):
        evidence_edges = [
            [left, right] for left, right in pairwise(path)
        ]
        multi_hop_candidates.append(
            {
                "path_id": f"P{index}",
                "nodes": path,
                "communities": {
                    alias: int(lookup[alias_to_id[alias]].get("cluster") or 0)
                    for alias in path
                },
                "path": path,
                "evidence_edges": evidence_edges,
            }
        )

    articulation = sorted(
        nx.articulation_points(graph),
        key=lambda node: (-float(lookup[node].get("betweenness") or 0), id_to_alias[node]),
    )
    articulation_evidence = []
    for node in articulation[:8]:
        incident_edges = []
        for neighbor, data in graph[node].items():
            incident_edges.append(
                {
                    "source": id_to_alias[node],
                    "target": id_to_alias[neighbor],
                    "weight": int(data.get("weight") or 0),
                    "source_community": int(lookup[node].get("cluster") or 0),
                    "target_community": int(lookup[neighbor].get("cluster") or 0),
                }
            )
        incident_edges.sort(
            key=lambda item: (-item["weight"], item["source"], item["target"])
        )
        articulation_evidence.append(
            {"node": id_to_alias[node], "incident_edges": incident_edges[:6]}
        )

    hub_nodes = [
        {
            "alias": id_to_alias[node],
            "degree": int(lookup[node].get("degree") or graph.degree(node)),
            "weighted_degree": float(lookup[node].get("weighted_degree") or 0),
        }
        for node in graph.nodes
    ]
    hub_candidates = []
    for alias in _hub_candidate_aliases(hub_nodes):
        node = alias_to_id[alias]
        incident_edges = []
        for neighbor, data in graph[node].items():
            incident_edges.append(
                {
                    "source": alias,
                    "target": id_to_alias[neighbor],
                    "weight": int(data.get("weight") or 0),
                    "source_community": int(lookup[node].get("cluster") or 0),
                    "target_community": int(lookup[neighbor].get("cluster") or 0),
                }
            )
        incident_edges.sort(
            key=lambda item: (-item["weight"], item["source"], item["target"])
        )
        hub_candidates.append(
            {
                "node": alias,
                "degree": int(lookup[node].get("degree") or graph.degree(node)),
                "weighted_degree": float(lookup[node].get("weighted_degree") or 0),
                "incident_edges": incident_edges[:6],
            }
        )
    return {
        "retrieval_method": "salience + community anchors + verified multi-hop paths",
        "salient_nodes": [id_to_alias[node] for node in ranked[:12]],
        "community_representatives": {
            f"C{community}": aliases
            for community, aliases in sorted(community_representatives.items())
        },
        "community_internal_edges": {
            f"C{community}": values[:4]
            for community, values in sorted(community_edges.items())
        },
        "cross_community_edges": cross_edges[:12],
        "multi_hop_paths": multi_hop_candidates,
        "hub_candidates": hub_candidates,
        "articulation_candidates": [id_to_alias[node] for node in articulation[:8]],
        "articulation_evidence": articulation_evidence,
    }


def build_prompt(
    mode: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    retrieval: dict[str, Any] | None,
) -> str:
    alias_legend = [
        {"alias": node["alias"], "label": node["label"]}
        for node in nodes
    ]
    if mode == "vlm":
        evidence = (
            "本组只提供PNG；别名图例仅用于统一输出标识，不提供社区、边或中心性信息。"
            "不得把视觉上接近但没有连线的节点写成关系。"
        )
    elif mode == "flat_kg":
        evidence = "完整显示子图：\n" + json.dumps(
            {"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2
        )
    elif mode == "graph_rag":
        evidence = (
            "按图结构检索出的解释证据。所有evidence_edges只能逐字选自"
            "community_internal_edges、cross_community_edges、multi_hop_paths或"
            "hub_candidates中的incident_edges、articulation_evidence中明确列出的边，"
            "不得根据PNG猜边。两个multi_hop槽必须选择两个不同的path_id，并把候选中的"
            "nodes、communities、path和evidence_edges四个字段逐字复制，禁止跨候选拼接或"
            "自行改写路径。hub槽必须从hub_candidates选择focus_node及其incident_edges：\n"
            + json.dumps(retrieval or {}, ensure_ascii=False, indent=2)
        )
    else:
        raise ValueError(mode)
    return f"""请为CiteWeave最终论文的“概念结构”章节生成图解释证据计划。
这不是问答题。为保证不同输入条件可比较，请尝试完成固定的5个任务槽：1条社区内部关系、
1条跨社区关系、2条不同的多跳关系、1条hub或bridge关系。多跳关系至少包含3个节点。
最后一个槽位的type必须在“hub”和“bridge”中二选一，禁止输出“hub_or_bridge”等组合类型。
hub或bridge声明必须在focus_node中填写唯一被主张为hub或割点的节点，且该节点必须出现在nodes中；
nodes中的其他节点只是证据邻居，不会被视为hub或割点。hub或bridge的每条evidence_edges都必须连接focus_node。
其他类型的focus_node必须留空字符串。
某个槽位证据不足时不要猜测，将其写入abstentions并说明原因；不得用重复声明填充槽位。
statement不得使用“导致”“促进”“演化”“迁移趋势”“建立关联”等超出共现结构的语义；
共现边只允许表述为结构连接或共同出现。最终系统会忽略你的自由表述并根据证据重新生成克制句子。

节点别名图例：
{json.dumps(alias_legend, ensure_ascii=False, indent=2)}

当前证据条件：
{evidence}

严格按照下面的JSON结构输出：
{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}
"""


def _grounded_statement(
    claim_type: str,
    aliases: list[str],
    focus_node: str,
    path: list[str],
    reported_edges: list[tuple[str, str]],
    node_lookup: dict[str, dict[str, Any]],
    graph_edges: dict[tuple[str, str], dict[str, Any]],
) -> str:
    """Render a conservative sentence from verified structure, not model prose."""

    def node_text(alias: str) -> str:
        node = node_lookup[alias]
        return f"{node['label']}（{alias}，社区{node['community']}）"

    def edge_text(edge: tuple[str, str]) -> str:
        data = graph_edges[edge]
        source, target = str(data["source"]), str(data["target"])
        weight = int(data.get("weight") or 0)
        return f"{node_text(source)}—{node_text(target)}（共现权重{weight}）"

    if claim_type == "multi_hop":
        labels = " → ".join(node_text(alias) for alias in path)
        weights = [int(graph_edges[edge].get("weight") or 0) for edge in map(_canonical_edge, path, path[1:])]
        return (
            f"显示子图中存在一条{len(path) - 1}跳共现路径：{labels}；"
            f"路径边权依次为{weights}。该路径只表示网络结构连通性，不表示时序迁移或因果关系。"
        )
    if claim_type == "cross_community":
        relations = "；".join(edge_text(edge) for edge in reported_edges)
        return f"显示子图中检测到以下跨社区共现边：{relations}。"
    if claim_type == "community_structure":
        community = node_lookup[aliases[0]]["community"]
        labels = "、".join(node_text(alias) for alias in aliases)
        relations = "；".join(edge_text(edge) for edge in reported_edges)
        return f"{labels}同属社区{community}；已核验的社区内部共现边包括：{relations}。"
    if claim_type == "bridge":
        relations = "；".join(edge_text(edge) for edge in reported_edges)
        return f"{node_text(focus_node)}是显示子图中的割点；其已核验连接包括：{relations}。"
    if claim_type == "hub":
        node = node_lookup[focus_node]
        return (
            f"{node_text(focus_node)}是显示子图中的高连接节点，"
            f"其度为{node['degree']}、加权度为{node['weighted_degree']}。"
        )
    relations = "；".join(edge_text(edge) for edge in reported_edges)
    return f"显示子图中已核验以下共现关系：{relations}。"


def verify_response(
    parsed: dict[str, Any],
    alias_to_id: dict[str, str],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_lookup = {node["alias"]: node for node in nodes}
    graph_edges = {_canonical_edge(edge["source"], edge["target"]): edge for edge in edges}
    verification_graph = nx.Graph()
    verification_graph.add_nodes_from(node_lookup)
    verification_graph.add_edges_from(graph_edges)
    articulation_points = set(nx.articulation_points(verification_graph))
    hub_candidates = set(_hub_candidate_aliases(nodes))
    verified, rejected = [], []
    verified_signatures: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(parsed.get("claims") or [], start=1):
        if not isinstance(raw, dict):
            rejected.append({"claim_index": index, "reason": "claim is not an object", "raw": raw})
            continue
        claim = _json_safe(raw)
        claim_type = str(claim.get("type") or "").lower()
        raw_focus_node = claim.get("focus_node")
        focus_node = (
            raw_focus_node.upper() if isinstance(raw_focus_node, str) else ""
        )
        aliases = [str(value).upper() for value in claim.get("nodes") or []]
        raw_communities = claim.get("communities") or {}
        communities_are_valid = isinstance(raw_communities, dict)
        communities = raw_communities if communities_are_valid else {}
        normalizations = []
        if (
            claim_type in {"hub", "bridge"}
            and focus_node
            and focus_node in alias_to_id
            and focus_node not in aliases
        ):
            # focus_node already names the unique structural subject. Requiring the
            # model to duplicate it in nodes caused valid, fully grounded hub claims
            # to be rejected. Add only that known alias; all structural checks remain.
            aliases = [focus_node, *aliases]
            claim["nodes"] = aliases
            normalizations.append("focus_node_added_to_nodes")
            if isinstance(communities, dict) and focus_node not in {
                str(alias).upper() for alias in communities
            }:
                communities = {
                    **communities,
                    focus_node: int(node_lookup[focus_node]["community"]),
                }
                claim["communities"] = communities
                normalizations.append("focus_node_community_added")
        path = [str(value).upper() for value in claim.get("path") or []]
        reported_edges = []
        malformed_edge = False
        for value in claim.get("evidence_edges") or []:
            if not isinstance(value, list) or len(value) != 2:
                malformed_edge = True
                continue
            reported_edges.append(_canonical_edge(value[0], value[1]))
        path_edge_keys = [_canonical_edge(left, right) for left, right in pairwise(path)]

        reasons = []
        if not communities_are_valid:
            reasons.append("communities is not an object")
        referenced_aliases = (
            aliases
            + path
            + [alias for edge in reported_edges for alias in edge]
            + ([focus_node] if focus_node else [])
        )
        if claim_type not in ALLOWED_CLAIM_TYPES:
            reasons.append("unknown claim type")
        if not aliases or any(alias not in alias_to_id for alias in referenced_aliases):
            reasons.append("unknown or missing node alias")
        if any(alias not in aliases for edge in reported_edges for alias in edge):
            reasons.append("evidence edge endpoint missing from claim nodes")
        if malformed_edge or not reported_edges:
            reasons.append("missing or malformed evidence edges")
        if any(edge not in graph_edges for edge in reported_edges):
            reasons.append("unsupported evidence edge")
        if claim_type == "multi_hop":
            if (
                len(path) < 3
                or len(set(path)) != len(path)
                or any(edge not in graph_edges for edge in path_edge_keys)
                or set(path_edge_keys) != set(reported_edges)
            ):
                reasons.append("invalid multi-hop path")
        elif path:
            reasons.append("path supplied for non-multi-hop claim")
        if (
            claim_type == "cross_community"
            and reported_edges
            and not all(
                node_lookup[left]["community"] != node_lookup[right]["community"]
                for left, right in reported_edges
                if left in node_lookup and right in node_lookup
            )
        ):
            reasons.append("claim has no cross-community edge")
        if (
            claim_type == "community_structure"
            and aliases
            and len(
                {
                    node_lookup[alias]["community"]
                    for alias in aliases
                    if alias in node_lookup
                }
            )
            != 1
        ):
            reasons.append("community-structure nodes span multiple communities")
        if claim_type in {"hub", "bridge"} and (
            not focus_node or focus_node not in alias_to_id
        ):
            reasons.append("missing or invalid focus node")
        if claim_type == "bridge" and focus_node:
            if focus_node not in articulation_points:
                reasons.append("claimed bridge is not an articulation point")
            if len(reported_edges) < 2 or any(
                focus_node not in edge for edge in reported_edges
            ):
                reasons.append("bridge evidence edges are not incident to the claimed bridge")
        if claim_type == "hub" and focus_node:
            if focus_node not in hub_candidates:
                reasons.append("claimed hub is outside the top weighted-degree decile")
            if any(focus_node not in edge for edge in reported_edges):
                reasons.append("hub evidence edges are not incident to the claimed hub")
        for alias, community in communities.items():
            alias = str(alias).upper()
            try:
                community_matches = (
                    alias in node_lookup
                    and int(community) == int(node_lookup[alias]["community"])
                )
            except (TypeError, ValueError):
                community_matches = False
            if not community_matches:
                reasons.append("incorrect community assignment")
                break
        signature = (
            claim_type,
            tuple(sorted(set(reported_edges))),
            tuple(path),
        )
        if signature in verified_signatures:
            reasons.append("duplicate structural claim")
        if reasons:
            rejected.append(
                {"claim_index": index, "reason": "; ".join(sorted(set(reasons))), "raw": claim}
            )
            continue

        verified_signatures.add(signature)
        claim["claim_id"] = f"GC{len(verified) + 1:03d}"
        claim["focus_node"] = focus_node
        if normalizations:
            claim["normalizations"] = normalizations
        claim["model_statement"] = claim.get("statement")
        claim["statement"] = _grounded_statement(
            claim_type,
            aliases,
            focus_node,
            path,
            reported_edges,
            node_lookup,
            graph_edges,
        )
        claim["nodes"] = [
            {
                "alias": alias,
                "id": alias_to_id[alias],
                "label": node_lookup[alias]["label"],
                "community": node_lookup[alias]["community"],
            }
            for alias in aliases
        ]
        claim["evidence_edges"] = [
            {
                **graph_edges[edge],
                "source_id": alias_to_id[str(graph_edges[edge]["source"])],
                "target_id": alias_to_id[str(graph_edges[edge]["target"])],
            }
            for edge in reported_edges
        ]
        claim["path"] = [
            {
                "alias": alias,
                "id": alias_to_id[alias],
                "label": node_lookup[alias]["label"],
            }
            for alias in path
        ]
        claim["verified"] = True
        verified.append(claim)
    return verified, rejected


def score_response(
    parsed: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    verified: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic ablation metrics against the displayed graph."""
    node_aliases = {str(node["alias"]).upper() for node in nodes}
    graph_edges = {_canonical_edge(edge["source"], edge["target"]) for edge in edges}
    claims = [claim for claim in (parsed.get("claims") or []) if isinstance(claim, dict)]
    reported_edges = 0
    supported_edges = 0
    unsupported_edge_claims = 0
    reported_multi_hop_claims = 0
    valid_multi_hop_claims = 0
    reported_by_type = {claim_type: 0 for claim_type in sorted(ALLOWED_CLAIM_TYPES)}

    for claim in claims:
        claim_type = str(claim.get("type") or "").lower()
        if claim_type in reported_by_type:
            reported_by_type[claim_type] += 1
        claim_has_unsupported_edge = False
        claim_edge_keys: list[tuple[str, str]] = []
        for value in claim.get("evidence_edges") or []:
            reported_edges += 1
            if not isinstance(value, list) or len(value) != 2:
                claim_has_unsupported_edge = True
                continue
            edge = _canonical_edge(value[0], value[1])
            claim_edge_keys.append(edge)
            if all(alias in node_aliases for alias in edge) and edge in graph_edges:
                supported_edges += 1
            else:
                claim_has_unsupported_edge = True
        if claim_has_unsupported_edge:
            unsupported_edge_claims += 1

        if claim_type == "multi_hop":
            reported_multi_hop_claims += 1
            path = [str(value).upper() for value in claim.get("path") or []]
            path_edges = [_canonical_edge(left, right) for left, right in pairwise(path)]
            if (
                len(path) >= 3
                and len(set(path)) == len(path)
                and all(alias in node_aliases for alias in path)
                and all(edge in graph_edges for edge in path_edges)
                and set(path_edges) == set(claim_edge_keys)
            ):
                valid_multi_hop_claims += 1

    verified_by_type = {claim_type: 0 for claim_type in sorted(ALLOWED_CLAIM_TYPES)}
    for claim in verified:
        claim_type = str(claim.get("type") or "").lower()
        if claim_type in verified_by_type:
            verified_by_type[claim_type] += 1
    complex_types = {"cross_community", "multi_hop", "bridge"}
    verified_complex_claims = sum(verified_by_type[value] for value in complex_types)
    unsupported_edges = reported_edges - supported_edges
    abstentions = [
        item for item in (parsed.get("abstentions") or []) if isinstance(item, dict)
    ]

    def fulfilled_slots(counts: dict[str, int]) -> int:
        return (
            min(counts["community_structure"], 1)
            + min(counts["cross_community"], 1)
            + min(counts["multi_hop"], 2)
            + min(counts["hub"] + counts["bridge"], 1)
        )

    return {
        "requested_slots": 5,
        "reported_abstentions": len(abstentions),
        "abstention_rate": round(len(abstentions) / 5, 4),
        "reported_claims": len(parsed.get("claims") or []),
        "verified_claims": len(verified),
        "claim_support_rate": (
            round(len(verified) / len(parsed.get("claims") or []), 4)
            if parsed.get("claims")
            else None
        ),
        "reported_edges": reported_edges,
        "supported_edges": supported_edges,
        "unsupported_edges": unsupported_edges,
        "edge_hallucination_rate": (
            round(unsupported_edges / reported_edges, 4) if reported_edges else None
        ),
        "unsupported_edge_claims": unsupported_edge_claims,
        "reported_multi_hop_claims": reported_multi_hop_claims,
        "valid_multi_hop_claims": valid_multi_hop_claims,
        "path_validity_rate": (
            round(valid_multi_hop_claims / reported_multi_hop_claims, 4)
            if reported_multi_hop_claims
            else None
        ),
        "verified_complex_claims": verified_complex_claims,
        "reported_slot_coverage": round(fulfilled_slots(reported_by_type) / 5, 4),
        "verified_slot_coverage": round(fulfilled_slots(verified_by_type) / 5, 4),
        "reported_claims_by_type": reported_by_type,
        "verified_claims_by_type": verified_by_type,
    }


def explain_network(
    network: NetworkResult,
    figure: FigureArtifact,
    policy: GraphExplanationPolicy,
    *,
    max_nodes: int,
    client: QwenVisionClient | None = None,
) -> dict[str, Any]:
    graph, lookup = displayed_graph(network, max_nodes)
    if graph.number_of_edges() == 0:
        return {
            "figure_name": figure.name,
            "network_name": network.name,
            "mode": policy.mode,
            "status": "skipped_empty_graph",
            "verified_claims": [],
            "rejected_claims": [],
        }
    id_to_alias, alias_to_id, nodes, edges = alias_graph(graph, lookup)
    retrieval = (
        graph_rag_context(
            graph,
            lookup,
            id_to_alias,
            max_paths=policy.max_paths,
            max_hops=policy.max_hops,
        )
        if policy.mode == "graph_rag"
        else None
    )
    prompt = build_prompt(policy.mode, nodes, edges, retrieval)
    active_client = client or QwenVisionClient(policy)
    raw_answer, usage = active_client.complete(figure.png, prompt)
    parsed = _parse_json(raw_answer)
    verified, rejected = verify_response(parsed, alias_to_id, nodes, edges)
    metrics = score_response(parsed, nodes, edges, verified)
    return {
        "figure_name": figure.name,
        "network_name": network.name,
        "mode": policy.mode,
        "model": policy.model,
        "status": "complete",
        "overview": parsed.get("overview"),
        "verified_claims": verified,
        "rejected_claims": rejected,
        "caveats": parsed.get("caveats") or [],
        "verification": {
            "reported_claims": len(parsed.get("claims") or []),
            "verified_claims": len(verified),
            "rejected_claims": len(rejected),
            "claim_support_rate": (
                round(len(verified) / len(parsed.get("claims") or []), 4)
                if parsed.get("claims")
                else None
            ),
        },
        "metrics": metrics,
        "retrieval": retrieval,
        "usage": usage,
        "prompt": prompt,
        "raw_answer": raw_answer,
    }


def generate_graph_explanations(
    analyses: AnalysisBundle,
    figures: list[FigureArtifact],
    policy: GraphExplanationPolicy,
    *,
    max_nodes: int,
    client: QwenVisionClient | None = None,
) -> list[dict[str, Any]]:
    if policy.mode == "disabled":
        return []
    figure_lookup = {figure.name: figure for figure in figures}
    results = []
    for network_name in policy.networks:
        network = analyses.networks.get(network_name)
        figure = figure_lookup.get(f"network_{network_name}")
        if network is None or figure is None:
            continue
        results.append(explain_network(network, figure, policy, max_nodes=max_nodes, client=client))
    return results
