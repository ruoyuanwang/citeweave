from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

from run_complex_experiment import graph_operator_context, multihop_retrieval_context, score_record


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "generated_benchmark" / "benchmark.jsonl"


def shortest_distance(nodes, edges, source, target):
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def correct_response(sample):
    gold = sample["gold"]
    task_type = sample["task_type"]
    response = {
        "shortest_distance": gold["shortest_distance"],
        "path": gold["one_shortest_path"] if task_type != "unsupported_edge" else [],
        "bridge_node": gold.get("bridge_node"),
        "cross_community_edges": gold["cross_community_edges_on_path"]
        if task_type == "cross_community_path"
        else [],
        "direct_edge_supported": False if task_type == "unsupported_edge" else None,
        "indirect_path": gold.get("indirect_path", []),
        "evidence_edges": gold["evidence_edges"],
        "explanation": "gold validation response",
    }
    return response


def operator_response(sample):
    text = graph_operator_context(sample)
    payload = text.split("--- DETERMINISTIC GRAPH OPERATOR RESULTS ---", 1)[1]
    payload = payload.split("--- END GRAPH OPERATOR RESULTS ---", 1)[0].strip()
    result = json.loads(payload)
    task_type = sample["task_type"]
    path = result.get("shortest_path") or []
    response = {
        "shortest_distance": result.get("shortest_distance"),
        "path": path if task_type in {"path_trace", "cross_community_path"} else [],
        "bridge_node": result.get("top_bridge_candidate") if task_type == "bridge_node" else None,
        "cross_community_edges": result.get("cross_community_edges", []),
        "direct_edge_supported": result.get("direct_edge_exists") if task_type == "unsupported_edge" else None,
        "indirect_path": result.get("indirect_path_if_no_direct_edge") or [],
        "evidence_edges": (
            result.get("top_candidate_incident_edges", [])[:2]
            if task_type == "bridge_node"
            else result.get("path_edges", [])
        ),
        "explanation": "operator validation response",
    }
    return response


def main(args):
    path = Path(args.benchmark)
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise SystemExit("benchmark为空。")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise SystemExit("sample_id存在重复。")

    source_aliases = []
    target_aliases = []
    for sample in samples:
        aliases = {node["alias"] for node in sample["graph"]["nodes"]}
        edges = {
            tuple(sorted((edge["source"], edge["target"]))) for edge in sample["graph"]["edges"]
        }
        if not 6 <= len(aliases) <= 10:
            raise SystemExit(f"{sample['sample_id']}节点数不在6—10范围内。")
        if any(left not in aliases or right not in aliases for left, right in edges):
            raise SystemExit(f"{sample['sample_id']}存在未知边端点。")
        image_path = path.parent / sample["image"]
        if not image_path.exists():
            raise SystemExit(f"{sample['sample_id']}缺少PNG：{image_path}")

        gold = sample["gold"]
        source_aliases.append(gold["source"])
        target_aliases.append(gold["target"])
        distance = shortest_distance(aliases, edges, gold["source"], gold["target"])
        if distance != gold["shortest_distance"]:
            raise SystemExit(f"{sample['sample_id']}最短距离gold错误。")
        if any(tuple(sorted(edge)) not in edges for edge in gold["evidence_edges"]):
            raise SystemExit(f"{sample['sample_id']}gold包含不存在的证据边。")
        if sample["task_type"] == "unsupported_edge":
            queried = tuple(sorted(gold["queried_pair"]))
            if queried in edges or gold["direct_edge_supported"] is not False:
                raise SystemExit(f"{sample['sample_id']}负例不是有效非边。")

        score = score_record(sample, correct_response(sample))
        if score["task_correct"] != 1 or score["unsupported_claimed_edge_count"] != 0:
            raise SystemExit(f"{sample['sample_id']}自动评分器未能识别标准答案。")
        operator_score = score_record(sample, operator_response(sample))
        if operator_score["task_correct"] != 1 or operator_score["unsupported_claimed_edge_count"] != 0:
            raise SystemExit(f"{sample['sample_id']}图算子结果与底层图或评分器不一致。")

        retrieval_text = multihop_retrieval_context(sample)
        retrieval_payload = retrieval_text.split("--- MULTI-HOP GRAPH RETRIEVAL EVIDENCE ---", 1)[1]
        retrieval_payload = retrieval_payload.split(
            "--- END MULTI-HOP GRAPH RETRIEVAL EVIDENCE ---", 1
        )[0].strip()
        retrieval = json.loads(retrieval_payload)
        forbidden = {"gold", "top_bridge_candidate", "shortest_distance", "direct_edge_exists"}
        if forbidden & set(retrieval):
            raise SystemExit(f"{sample['sample_id']}检索上下文泄露最终答案字段。")
        if sample["task_type"] in {"path_trace", "cross_community_path", "unsupported_edge"}:
            candidate_paths = retrieval.get("candidate_relation_chains_unranked") or []
            if not candidate_paths:
                raise SystemExit(f"{sample['sample_id']}没有检索到候选关系链。")
            retrieved_min = min(len(path_value) - 1 for path_value in candidate_paths)
            if retrieved_min != gold["shortest_distance"]:
                raise SystemExit(f"{sample['sample_id']}候选关系链没有覆盖真实最短路径。")
        elif len(retrieval.get("candidate_probes_unranked") or []) != len(aliases):
            raise SystemExit(f"{sample['sample_id']}节点删除证据不完整。")

    print(f"VALIDATION OK：{len(samples)}个样本全部通过。")
    print("任务分布：", dict(Counter(sample["task_type"] for sample in samples)))
    print("source别名分布：", dict(Counter(source_aliases)))
    print("target别名分布：", dict(Counter(target_aliases)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    main(parser.parse_args())
