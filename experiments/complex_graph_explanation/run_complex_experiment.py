from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "generated_benchmark" / "benchmark.jsonl"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-plus-2025-12-19"
CONDITIONS = (
    "vlm",
    "vlm_flat_kg",
    "vlm_graph_rag",
    "graphrag",
    "graph_operator_rag",
    "no_graph",
)

SYSTEM_PROMPT = """你是一名严谨的图结构推理评测助手。你只能根据当前输入中的图回答，
不得根据关键词含义猜测不存在的边。图中的边均为无向边。社区编号只表示算法社区，
不代表因果或语义等价。必须输出一个JSON对象，不要使用Markdown代码块，也不要添加JSON外文字。

统一输出格式：
{
  "shortest_distance": 整数或null,
  "path": ["N1", "N2"]或[],
  "bridge_node": "N1"或null,
  "cross_community_edges": [["N1", "N2"]]或[],
  "direct_edge_supported": true、false或null,
  "indirect_path": ["N1", "N2", "N3"]或[],
  "evidence_edges": [["N1", "N2"]]或[],
  "explanation": "不超过120字的解释"
}

所有路径和证据边都必须真实存在。无法判断时使用null或空数组。"""


def load_benchmark(path: Path, sample_ids: list[str] | None, limit: int | None):
    if not path.exists():
        raise SystemExit(f"缺少benchmark：{path}")
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
    if sample_ids:
        wanted = set(sample_ids)
        samples = [item for item in samples if item["sample_id"] in wanted]
        missing = wanted - {item["sample_id"] for item in samples}
        if missing:
            raise SystemExit(f"benchmark中不存在：{sorted(missing)}")
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise SystemExit("没有可运行的样本。")
    return samples


def resolve_api_key(value: str | None):
    key = value or os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key.strip()
    key = getpass.getpass("请输入阿里云百炼API Key（输入不会显示）：").strip()
    if not key:
        raise SystemExit("未输入API Key。")
    return key


def graph_context(sample: dict[str, Any]) -> str:
    node_lines = [
        f"[{node['alias']}] label={node['label']}; community=C{node['community']}; occurrences={node['occurrences']}"
        for node in sample["graph"]["nodes"]
    ]
    edge_lines = [
        f"[{index}] {edge['source']} -- {edge['target']}; weight={edge['weight']}; "
        f"association_strength={edge['association_strength']}"
        for index, edge in enumerate(sample["graph"]["edges"], start=1)
    ]
    return "\n".join(
        [
            "--- QUERY-FOCUSED KG SUBGRAPH ---",
            "Nodes:",
            *node_lines,
            "Edges:",
            *edge_lines,
            "--- END KG SUBGRAPH ---",
        ]
    )


def task_output_instruction(task_type: str) -> str:
    instructions = {
        "path_trace": (
            "本题只填写 shortest_distance、path、evidence_edges 和 explanation；"
            "bridge_node与direct_edge_supported填null，cross_community_edges与indirect_path填空数组。"
        ),
        "cross_community_path": (
            "本题填写 shortest_distance、path、cross_community_edges、evidence_edges 和 explanation；"
            "bridge_node与direct_edge_supported填null，indirect_path填空数组。"
        ),
        "bridge_node": (
            "本题只填写 bridge_node、evidence_edges 和 explanation；shortest_distance与"
            "direct_edge_supported填null，其余数组填空数组。不要把度数最高节点直接等同于桥接节点。"
        ),
        "unsupported_edge": (
            "本题填写 shortest_distance、direct_edge_supported、indirect_path、evidence_edges和explanation；"
            "path与cross_community_edges填空数组，bridge_node填null。"
        ),
    }
    return instructions[task_type]


def sample_graph(sample: dict[str, Any]):
    aliases = {node["alias"] for node in sample["graph"]["nodes"]}
    communities = {node["alias"]: str(node["community"]) for node in sample["graph"]["nodes"]}
    adjacency = {alias: set() for alias in aliases}
    edge_lookup = {}
    for edge in sample["graph"]["edges"]:
        left, right = edge["source"], edge["target"]
        adjacency[left].add(right)
        adjacency[right].add(left)
        edge_lookup[tuple(sorted((left, right)))] = edge
    return aliases, communities, adjacency, edge_lookup


def graph_shortest_path(adjacency, source: str, target: str, blocked: str | None = None):
    if source == blocked or target == blocked:
        return None
    queue = [(source, [source])]
    seen = {source}
    cursor = 0
    while cursor < len(queue):
        current, path = queue[cursor]
        cursor += 1
        if current == target:
            return path
        for neighbor in sorted(adjacency[current]):
            if neighbor != blocked and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def disconnected_pairs_after_removal(aliases, adjacency, removed: str):
    remaining = sorted(aliases - {removed})
    unseen = set(remaining)
    component_sizes = []
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor != removed and neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(len(component))
    total_pairs = len(remaining) * (len(remaining) - 1) // 2
    connected_pairs = sum(size * (size - 1) // 2 for size in component_sizes)
    return total_pairs - connected_pairs, sorted(component_sizes, reverse=True)


def components_after_node_removal(aliases, adjacency, removed: str):
    unseen = set(aliases - {removed})
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in sorted(adjacency[current]):
                if neighbor != removed and neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def simple_paths_within(adjacency, source: str, target: str, max_edges=4, cap=300):
    paths = []

    def visit(current, path):
        if len(paths) >= cap:
            return
        if current == target:
            paths.append(path)
            return
        if len(path) - 1 >= max_edges:
            return
        for neighbor in sorted(adjacency[current]):
            if neighbor not in path:
                visit(neighbor, path + [neighbor])

    visit(source, [source])
    return paths


def multihop_retrieval_context(sample: dict[str, Any]) -> str:
    """Retrieve graph evidence without returning a final answer or reading gold fields."""
    aliases, communities, adjacency, edge_lookup = sample_graph(sample)
    task_type = sample["task_type"]
    mentioned = []
    for alias in re.findall(r"\bN\d+\b", sample["question"], flags=re.IGNORECASE):
        alias = alias.upper()
        if alias in aliases and alias not in mentioned:
            mentioned.append(alias)

    evidence: dict[str, Any] = {
        "retrieval_method": None,
        "task_type": task_type,
        "note": "Candidates are evidence, not a final answer; the model must compare and reason.",
    }
    if task_type in {"path_trace", "cross_community_path", "unsupported_edge"}:
        if len(mentioned) != 2:
            raise ValueError(f"{sample['sample_id']}无法识别检索端点：{mentioned}")
        source, target = mentioned
        paths = simple_paths_within(adjacency, source, target, max_edges=4)
        bfs_path = graph_shortest_path(adjacency, source, target)
        unique = {tuple(path) for path in paths}
        if bfs_path:
            unique.add(tuple(bfs_path))
        ordered = sorted(
            unique,
            key=lambda path: hashlib.sha256(
                f"{sample['sample_id']}|{'-'.join(path)}".encode("utf-8")
            ).hexdigest(),
        )
        if bfs_path and tuple(bfs_path) not in ordered[:12]:
            selected_paths = ordered[:11] + [tuple(bfs_path)]
            selected_paths.sort(
                key=lambda path: hashlib.sha256(
                    f"selected|{sample['sample_id']}|{'-'.join(path)}".encode("utf-8")
                ).hexdigest()
            )
        else:
            selected_paths = ordered[:12]

        retrieved_nodes = sorted({alias for path in selected_paths for alias in path})
        retrieved_edge_keys = {
            tuple(sorted((left, right)))
            for path in selected_paths
            for left, right in zip(path, path[1:])
        }
        evidence.update(
            {
                "retrieval_method": "anchor-aware multi-hop path retrieval",
                "anchors": [source, target],
                "candidate_relation_chains_unranked": [list(path) for path in selected_paths],
                "node_communities": {
                    alias: f"C{communities[alias]}" for alias in retrieved_nodes
                },
                "retrieved_edges": [
                    {
                        "source": key[0],
                        "target": key[1],
                        "weight": edge_lookup[key]["weight"],
                    }
                    for key in sorted(retrieved_edge_keys)
                ],
                "source_neighbors": sorted(adjacency[source]),
                "target_neighbors": sorted(adjacency[target]),
            }
        )
    elif task_type == "bridge_node":
        probes = []
        for alias in sorted(aliases):
            probes.append(
                {
                    "candidate": alias,
                    "community": f"C{communities[alias]}",
                    "neighbors": sorted(adjacency[alias]),
                    "components_observed_after_removal": components_after_node_removal(
                        aliases, adjacency, alias
                    ),
                }
            )
        evidence.update(
            {
                "retrieval_method": "counterfactual node-removal evidence retrieval",
                "candidate_probes_unranked": probes,
                "selection_rule": (
                    "Choose by comparing the returned components; no candidate is labelled as correct."
                ),
            }
        )
    else:
        raise ValueError(task_type)

    return "\n".join(
        [
            "--- MULTI-HOP GRAPH RETRIEVAL EVIDENCE ---",
            json.dumps(evidence, ensure_ascii=False, indent=2),
            "--- END MULTI-HOP GRAPH RETRIEVAL EVIDENCE ---",
        ]
    )


def graph_operator_context(sample: dict[str, Any]) -> str:
    """Run deterministic graph operators without reading any gold-answer field."""
    aliases, communities, adjacency, edge_lookup = sample_graph(sample)
    mentioned = []
    for alias in re.findall(r"\bN\d+\b", sample["question"], flags=re.IGNORECASE):
        alias = alias.upper()
        if alias in aliases and alias not in mentioned:
            mentioned.append(alias)
    task_type = sample["task_type"]
    result: dict[str, Any] = {"operator": None, "task_type": task_type}

    if task_type in {"path_trace", "cross_community_path", "unsupported_edge"}:
        if len(mentioned) != 2:
            raise ValueError(f"{sample['sample_id']}无法从问题中识别两个端点：{mentioned}")
        source, target = mentioned
        direct_edge = tuple(sorted((source, target))) in edge_lookup
        path = graph_shortest_path(adjacency, source, target)
        path_edges_value = [[left, right] for left, right in zip(path or [], (path or [])[1:])]
        cross_edges = [
            [left, right]
            for left, right in zip(path or [], (path or [])[1:])
            if communities[left] != communities[right]
        ]
        result.update(
            {
                "operator": "BFS shortest path + adjacency verification",
                "source": source,
                "target": target,
                "direct_edge_exists": direct_edge,
                "shortest_distance": None if path is None else len(path) - 1,
                "shortest_path": path,
                "path_edges": path_edges_value,
                "path_communities": [f"{alias}:C{communities[alias]}" for alias in (path or [])],
            }
        )
        if task_type == "cross_community_path":
            result["cross_community_edges"] = cross_edges
        if task_type == "unsupported_edge":
            result["indirect_path_if_no_direct_edge"] = path if not direct_edge else None

    elif task_type == "bridge_node":
        ranking = []
        for alias in sorted(aliases):
            disconnected, sizes = disconnected_pairs_after_removal(aliases, adjacency, alias)
            ranking.append(
                {
                    "node": alias,
                    "disconnected_pairs": disconnected,
                    "remaining_component_sizes": sizes,
                }
            )
        ranking.sort(key=lambda item: (-item["disconnected_pairs"], item["node"]))
        candidate = ranking[0]["node"]
        incident_edges = [
            [candidate, neighbor] for neighbor in sorted(adjacency[candidate])
        ]
        result.update(
            {
                "operator": "node-deletion connectivity audit",
                "ranking": ranking,
                "top_bridge_candidate": candidate,
                "top_candidate_incident_edges": incident_edges,
            }
        )
    else:
        raise ValueError(task_type)

    return "\n".join(
        [
            "--- DETERMINISTIC GRAPH OPERATOR RESULTS ---",
            json.dumps(result, ensure_ascii=False, indent=2),
            "--- END GRAPH OPERATOR RESULTS ---",
        ]
    )


def image_data_url(path: Path):
    if not path.exists():
        raise SystemExit(f"缺少实验图片：{path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_input(condition: str, sample: dict[str, Any], benchmark_dir: Path):
    shared = (
        f"样本：{sample['sample_id']}\n"
        f"任务类型：{sample['task_type']}\n"
        f"问题：{sample['question']}\n\n"
    )
    output_instruction = task_output_instruction(sample["task_type"])
    if condition == "vlm":
        image_path = benchmark_dir / sample["image"]
        text = shared + (
            "你只能使用随消息提供的PNG图。节点圆圈内是别名，右侧图例给出关键词和社区；"
            f"连线表示无向边，线宽按共现权重对数缩放。{output_instruction}请直接输出统一JSON。"
        )
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            {"type": "text", "text": text},
        ]
        return content, text, str(image_path)
    if condition == "vlm_flat_kg":
        image_path = benchmark_dir / sample["image"]
        evidence = graph_context(sample)
        text = shared + (
            "你同时获得PNG图和该子图的完整结构化节点边表。这是简单融合基线；"
            "需要自行从全部边中完成推理，不得编造未列出的关系。\n\n"
            f"{evidence}\n\n{output_instruction}请直接输出统一JSON。"
        )
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            {"type": "text", "text": text},
        ]
        return content, text, f"{image_path} + complete structured KG subgraph"
    if condition == "vlm_graph_rag":
        image_path = benchmark_dir / sample["image"]
        retrieved = multihop_retrieval_context(sample)
        text = shared + (
            "你同时获得PNG图和按当前问题检索出的多跳KG证据。候选关系链或节点删除探针"
            "没有标注最终答案，需要你比较证据后完成推理，并用PNG理解整体布局。\n\n"
            f"{retrieved}\n\n{output_instruction}请直接输出统一JSON。"
        )
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            {"type": "text", "text": text},
        ]
        return content, text, f"{image_path} + multi-hop GraphRAG evidence"
    if condition == "graphrag":
        evidence = graph_context(sample)
        text = shared + (
            "下面是从底层关键词KG中检索出的、与当前问题对应的结构化子图。"
            "只依据节点和边记录推理，不能补充未列出的边。\n\n"
            f"{evidence}\n\n{output_instruction}请直接输出统一JSON。"
        )
        return text, text, "structured query-focused KG subgraph"
    if condition == "graph_operator_rag":
        evidence = graph_context(sample)
        operator_evidence = graph_operator_context(sample)
        text = shared + (
            "下面先提供从底层KG检索出的结构化子图，再提供由确定性图算法计算的任务证据。"
            "图算子结果不是语言模型猜测；回答必须与其保持一致。\n\n"
            f"{evidence}\n\n{operator_evidence}\n\n"
            f"{output_instruction}请在输出前检查数值与路径长度是否一致，然后直接输出统一JSON。"
        )
        return text, text, "structured KG subgraph + deterministic graph operators"
    if condition == "no_graph":
        text = shared + (
            "本组不提供图片或KG。不得把关键词常识冒充为当前图中的边；"
            f"证据不足时使用null或空数组。{output_instruction}请直接输出统一JSON。"
        )
        return text, text, None
    raise ValueError(condition)


def call_qwen(base_url, api_key, model, user_content, temperature, seed, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API返回HTTP {exc.code}：{body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"无法连接API：{exc}") from exc
    result["client_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def parse_json_answer(answer: str):
    text = answer.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def canonical_edge(edge):
    if not isinstance(edge, list) or len(edge) != 2:
        return None
    left, right = str(edge[0]).upper(), str(edge[1]).upper()
    if not re.fullmatch(r"N\d+", left) or not re.fullmatch(r"N\d+", right):
        return None
    return tuple(sorted((left, right)))


def aliases_from(value):
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        alias = str(item).upper()
        if re.fullmatch(r"N\d+", alias):
            result.append(alias)
    return result


def path_edges(path):
    return [tuple(sorted((left, right))) for left, right in zip(path, path[1:])]


def score_record(sample, parsed):
    graph_edges = {
        tuple(sorted((edge["source"], edge["target"]))) for edge in sample["graph"]["edges"]
    }
    communities = {node["alias"]: str(node["community"]) for node in sample["graph"]["nodes"]}
    gold = sample["gold"]
    score = {
        "json_parse_ok": int(parsed is not None),
        "task_correct": 0,
        "path_valid": None,
        "path_shortest": None,
        "distance_consistent": None,
        "bridge_correct": None,
        "bridge_evidence_valid": None,
        "cross_edges_correct": None,
        "direct_edge_correct": None,
        "claimed_edge_count": 0,
        "unsupported_claimed_edge_count": 0,
        "hallucination_rate": None,
    }
    if parsed is None:
        return score

    path = aliases_from(parsed.get("path"))
    indirect_path = aliases_from(parsed.get("indirect_path"))
    task_type = sample["task_type"]
    chosen_path = indirect_path if task_type == "unsupported_edge" else path
    chosen_edges = path_edges(chosen_path)
    if task_type in {"path_trace", "cross_community_path", "unsupported_edge"} and chosen_path:
        expected_endpoints = (
            gold.get("queried_pair", [gold["source"], gold["target"]])
            if sample["task_type"] == "unsupported_edge"
            else [gold["source"], gold["target"]]
        )
        endpoints_ok = chosen_path[0] == expected_endpoints[0] and chosen_path[-1] == expected_endpoints[1]
        score["path_valid"] = int(endpoints_ok and all(edge in graph_edges for edge in chosen_edges))
        score["path_shortest"] = int(
            score["path_valid"] == 1 and len(chosen_path) - 1 == gold["shortest_distance"]
        )
        stated_distance = parsed.get("shortest_distance")
        score["distance_consistent"] = int(
            isinstance(stated_distance, int) and stated_distance == len(chosen_path) - 1
        )
    elif task_type in {"path_trace", "cross_community_path", "unsupported_edge"}:
        score["path_valid"] = 0
        score["path_shortest"] = 0
        score["distance_consistent"] = 0

    claimed_edges = list(chosen_edges)
    for field in ("evidence_edges", "cross_community_edges"):
        value = parsed.get(field)
        if isinstance(value, list):
            for item in value:
                edge = canonical_edge(item)
                if edge:
                    claimed_edges.append(edge)
    if parsed.get("direct_edge_supported") is True and sample["task_type"] == "unsupported_edge":
        queried = canonical_edge(gold["queried_pair"])
        if queried:
            claimed_edges.append(queried)
    deduplicated_claims = set(claimed_edges)
    unsupported = {edge for edge in deduplicated_claims if edge not in graph_edges}
    score["claimed_edge_count"] = len(deduplicated_claims)
    score["unsupported_claimed_edge_count"] = len(unsupported)
    if deduplicated_claims:
        score["hallucination_rate"] = round(len(unsupported) / len(deduplicated_claims), 4)

    if task_type == "path_trace":
        distance_ok = parsed.get("shortest_distance") == gold["shortest_distance"]
        score["task_correct"] = int(score["path_shortest"] == 1 and distance_ok)
    elif task_type == "cross_community_path":
        expected_cross = {
            edge
            for edge in chosen_edges
            if edge[0] in communities
            and edge[1] in communities
            and communities[edge[0]] != communities[edge[1]]
        }
        reported_cross = {
            edge
            for item in (parsed.get("cross_community_edges") or [])
            if (edge := canonical_edge(item)) is not None
        }
        score["cross_edges_correct"] = int(reported_cross == expected_cross and bool(chosen_path))
        score["task_correct"] = int(
            score["path_shortest"] == 1 and score["cross_edges_correct"] == 1
        )
    elif task_type == "bridge_node":
        bridge = parsed.get("bridge_node")
        bridge = str(bridge).upper() if bridge is not None else None
        score["bridge_correct"] = int(bridge == gold["bridge_node"])
        reported_evidence = {
            edge
            for item in (parsed.get("evidence_edges") or [])
            if (edge := canonical_edge(item)) is not None
        }
        valid_incident_evidence = {
            edge for edge in reported_evidence if edge in graph_edges and bridge in edge
        }
        score["bridge_evidence_valid"] = int(len(valid_incident_evidence) >= 2)
        score["task_correct"] = int(
            score["bridge_correct"] == 1 and score["bridge_evidence_valid"] == 1
        )
    elif task_type == "unsupported_edge":
        score["direct_edge_correct"] = int(parsed.get("direct_edge_supported") is False)
        score["task_correct"] = int(
            score["direct_edge_correct"] == 1 and score["path_valid"] == 1
        )
    return score


def write_summary(records, output_dir: Path):
    completed = [item for item in records if item["status"] == "complete"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in completed:
        groups.setdefault((record["condition"], record["task_type"]), []).append(record)
        groups.setdefault((record["condition"], "ALL"), []).append(record)

    def metric_mean(items, key):
        values = [item["score"].get(key) for item in items if item["score"].get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    rows = []
    for (condition, task_type), items in sorted(groups.items()):
        count = len(items)
        correct = sum(item["score"]["task_correct"] for item in items)
        parsed = sum(item["score"]["json_parse_ok"] for item in items)
        claims = sum(item["score"]["claimed_edge_count"] for item in items)
        unsupported = sum(item["score"]["unsupported_claimed_edge_count"] for item in items)
        rows.append(
            {
                "condition": condition,
                "task_type": task_type,
                "n": count,
                "accuracy": round(correct / count, 4) if count else None,
                "json_parse_rate": round(parsed / count, 4) if count else None,
                "shortest_path_rate": metric_mean(items, "path_shortest"),
                "distance_consistency_rate": metric_mean(items, "distance_consistent"),
                "bridge_accuracy": metric_mean(items, "bridge_correct"),
                "cross_edge_exact_accuracy": metric_mean(items, "cross_edges_correct"),
                "direct_edge_accuracy": metric_mean(items, "direct_edge_correct"),
                "claimed_edges": claims,
                "unsupported_claimed_edges": unsupported,
                "edge_hallucination_rate": round(unsupported / claims, 4) if claims else None,
            }
        )

    fields = list(rows[0]) if rows else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Complex Graph Experiment Summary",
        "",
        "| Condition | Task | N | Strict accuracy | Shortest path | Distance consistency | Bridge | Cross-edge exact | Direct-edge | Edge hallucination |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def display(value):
        return "NA" if value is None else f"{value:.3f}"

    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['task_type']} | {row['n']} | "
            f"{display(row['accuracy'])} | {display(row['shortest_path_rate'])} | "
            f"{display(row['distance_consistency_rate'])} | {display(row['bridge_accuracy'])} | "
            f"{display(row['cross_edge_exact_accuracy'])} | {display(row['direct_edge_accuracy'])} | "
            f"{display(row['edge_hallucination_rate'])} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(args):
    benchmark_path = Path(args.benchmark)
    benchmark_dir = benchmark_path.parent
    samples = load_benchmark(benchmark_path, args.sample_ids, args.limit)

    if args.dry_run:
        checked = 0
        for sample in samples:
            for condition in args.conditions:
                user_content, prompt, artifact = build_input(condition, sample, benchmark_dir)
                if condition in {"vlm", "vlm_flat_kg", "vlm_graph_rag"} and not isinstance(user_content, list):
                    raise SystemExit(f"VLM输入未包含图片：{sample['sample_id']}")
                if condition == "graphrag" and "QUERY-FOCUSED KG SUBGRAPH" not in prompt:
                    raise SystemExit(f"GraphRAG输入未包含KG：{sample['sample_id']}")
                if condition == "graph_operator_rag" and "DETERMINISTIC GRAPH OPERATOR RESULTS" not in prompt:
                    raise SystemExit(f"图算子GraphRAG输入未包含计算结果：{sample['sample_id']}")
                if condition == "vlm_flat_kg" and "QUERY-FOCUSED KG SUBGRAPH" not in prompt:
                    raise SystemExit(f"VLM+完整KG输入缺少结构化图：{sample['sample_id']}")
                if condition == "vlm_graph_rag" and "MULTI-HOP GRAPH RETRIEVAL EVIDENCE" not in prompt:
                    raise SystemExit(f"VLM+GraphRAG输入缺少检索证据：{sample['sample_id']}")
                checked += 1
                print(
                    f"OK {sample['sample_id']} / {condition}; prompt_chars={len(prompt)}; "
                    f"artifact={artifact or 'none'}"
                )
        print(f"\nDRY RUN OK：已验证 {checked} 组输入，未调用API、未产生费用。")
        return

    api_key = resolve_api_key(args.api_key)
    run_id = datetime.now().strftime("complex-graph-%Y%m%d-%H%M%S")
    output_dir = Path(args.results_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    total = len(samples) * len(args.conditions) * args.repeats
    completed_count = 0

    for sample in samples:
        for repeat in range(1, args.repeats + 1):
            for condition in args.conditions:
                completed_count += 1
                seed = args.seed + repeat - 1
                user_content, saved_prompt, artifact = build_input(condition, sample, benchmark_dir)
                print(f"[{completed_count}/{total}] {sample['sample_id']} / {condition} / r{repeat}")
                try:
                    raw = call_qwen(
                        args.base_url,
                        api_key,
                        args.model,
                        user_content,
                        args.temperature,
                        seed,
                        args.max_tokens,
                        args.timeout,
                    )
                    choice = raw.get("choices", [{}])[0]
                    answer = (choice.get("message", {}).get("content") or "").strip()
                    parsed = parse_json_answer(answer)
                    record = {
                        "sample_id": sample["sample_id"],
                        "task_type": sample["task_type"],
                        "condition": condition,
                        "repeat": repeat,
                        "status": "complete",
                        "question": sample["question"],
                        "artifact": artifact,
                        "model": raw.get("model") or args.model,
                        "seed": seed,
                        "temperature": args.temperature,
                        "answer": answer,
                        "parsed_answer": parsed,
                        "gold": sample["gold"],
                        "score": score_record(sample, parsed),
                        "elapsed_seconds": raw.get("client_elapsed_seconds"),
                        "usage": raw.get("usage", {}),
                        "saved_prompt": saved_prompt,
                    }
                except RuntimeError as exc:
                    record = {
                        "sample_id": sample["sample_id"],
                        "task_type": sample["task_type"],
                        "condition": condition,
                        "repeat": repeat,
                        "status": "error",
                        "question": sample["question"],
                        "error": str(exc),
                    }
                records.append(record)
                with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
                    for item in records:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                if record["status"] == "error":
                    raise SystemExit(f"请求失败，已有结果已保存：{record['error']}")
                print(
                    f"  correct={record['score']['task_correct']}; "
                    f"hallucination={record['score']['hallucination_rate']}; "
                    f"elapsed={record['elapsed_seconds']}s"
                )
                if args.delay and completed_count < total:
                    time.sleep(args.delay)

    write_summary(records, output_dir)
    manifest = {
        "benchmark": str(benchmark_path.resolve()),
        "model": args.model,
        "conditions": args.conditions,
        "sample_count": len(samples),
        "repeats": args.repeats,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n实验完成：{output_dir.resolve()}")
    print(f"汇总：{(output_dir / 'summary.md').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM vs GraphRAG on complex CiteWeave subgraphs.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=["vlm", "vlm_flat_kg", "vlm_graph_rag"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", help="不建议写进命令；优先使用DASHSCOPE_API_KEY环境变量。")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
