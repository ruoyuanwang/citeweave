from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_DIR = Path(__file__).resolve().parent
# When installed under <repo>/experiments/complex_graph_explanation, infer the
# CiteWeave root. In a standalone copy, pass --run-dir explicitly.
DEFAULT_PROJECT = (
    EXPERIMENT_DIR.parents[1]
    if EXPERIMENT_DIR.parent.name == "experiments"
    else EXPERIMENT_DIR
)
DEFAULT_RUN = DEFAULT_PROJECT / "runs" / "pilot-llm-bibliometrics"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "generated_benchmark"

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
]


def clean_number(value: str | float | int | None) -> int | float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def load_graph(nodes_path: Path, edges_path: Path):
    with nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
        node_rows = list(csv.DictReader(handle))
    with edges_path.open("r", encoding="utf-8-sig", newline="") as handle:
        edge_rows = list(csv.DictReader(handle))

    nodes: dict[str, dict[str, Any]] = {}
    for row in node_rows:
        label = row.get("id") or row.get("label")
        if not label:
            continue
        nodes[label] = {
            "label": row.get("label") or label,
            "occurrences": clean_number(row.get("occurrences")),
            "degree": clean_number(row.get("degree")),
            "weighted_degree": clean_number(row.get("weighted_degree")),
            "betweenness": clean_number(row.get("betweenness")),
            "community": str(row.get("cluster") or "unknown"),
        }

    adjacency: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    edges: dict[frozenset[str], dict[str, Any]] = {}
    for row in edge_rows:
        source = row["source"]
        target = row["target"]
        if source not in nodes or target not in nodes or source == target:
            continue
        attrs = {
            "weight": clean_number(row.get("weight")) or 0,
            "association_strength": clean_number(row.get("association_strength")),
        }
        adjacency[source][target] = attrs
        adjacency[target][source] = attrs
        edges[frozenset((source, target))] = attrs

    for node in nodes:
        adjacency.setdefault(node, {})
    return nodes, adjacency, edges


def shortest_path(adjacency, source: str, target: str) -> list[str] | None:
    queue = deque([source])
    parent = {source: None}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for neighbor in sorted(adjacency[current]):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if target not in parent:
        return None
    path = []
    current: str | None = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def path_candidates(nodes, adjacency) -> list[list[str]]:
    labels = sorted(nodes)
    candidates = []
    for index, source in enumerate(labels):
        for target in labels[index + 1 :]:
            path = shortest_path(adjacency, source, target)
            if not path or len(path) not in {4, 5}:
                continue
            communities = {nodes[item]["community"] for item in path}
            if len(communities) < 2:
                continue
            candidates.append(path)
    return candidates


def induced_edges(selected: list[str], edges):
    rows = []
    for index, source in enumerate(selected):
        for target in selected[index + 1 :]:
            attrs = edges.get(frozenset((source, target)))
            if attrs is not None:
                rows.append((source, target, attrs))
    return rows


def add_distractors(path, nodes, adjacency, edges, rng, target_size=8):
    selected = list(path)
    pool = set()
    for path_node in path[1:-1] or path:
        pool.update(adjacency[path_node])
    pool.difference_update(selected)

    candidates = list(pool)
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda item: (
            sum(1 for node in selected if item in adjacency[node]),
            nodes[item]["occurrences"] or 0,
        ),
        reverse=True,
    )
    for candidate in candidates:
        if len(selected) >= target_size:
            break
        links = sum(1 for node in selected if candidate in adjacency[node])
        if not 1 <= links <= 3:
            continue
        proposal = selected + [candidate]
        if len(induced_edges(proposal, edges)) <= 16:
            selected.append(candidate)

    if len(selected) < target_size:
        fallback = [item for item in candidates if item not in selected]
        for candidate in fallback:
            if len(selected) >= target_size:
                break
            selected.append(candidate)
    return selected


def components_after_removal(selected, sub_edges, removed: str):
    remaining = [item for item in selected if item != removed]
    local = {item: set() for item in remaining}
    for source, target, _ in sub_edges:
        if source == removed or target == removed:
            continue
        local[source].add(target)
        local[target].add(source)
    components = []
    unseen = set(remaining)
    while unseen:
        root = unseen.pop()
        component = {root}
        queue = [root]
        while queue:
            current = queue.pop()
            for neighbor in local[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def unique_bridge(selected, sub_edges):
    scored = []
    for node in selected:
        components = components_after_removal(selected, sub_edges, node)
        sizes = [len(item) for item in components]
        total_pairs = (len(selected) - 1) * (len(selected) - 2) // 2
        connected_pairs = sum(size * (size - 1) // 2 for size in sizes)
        scored.append((total_pairs - connected_pairs, node, sizes))
    scored.sort(reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0]


def force_layout(aliases, edge_rows, rng, width=1160, height=920):
    count = len(aliases)
    positions = {}
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.34
    for index, alias in enumerate(aliases):
        angle = 2 * math.pi * index / count + rng.uniform(-0.12, 0.12)
        positions[alias] = [
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        ]

    area = width * height
    ideal = math.sqrt(area / max(count, 1)) * 0.55
    for iteration in range(180):
        displacement = {alias: [0.0, 0.0] for alias in aliases}
        for index, left in enumerate(aliases):
            for right in aliases[index + 1 :]:
                dx = positions[left][0] - positions[right][0]
                dy = positions[left][1] - positions[right][1]
                distance = max(math.hypot(dx, dy), 1.0)
                force = ideal * ideal / distance
                ux, uy = dx / distance, dy / distance
                displacement[left][0] += ux * force
                displacement[left][1] += uy * force
                displacement[right][0] -= ux * force
                displacement[right][1] -= uy * force
        for source, target, _ in edge_rows:
            dx = positions[source][0] - positions[target][0]
            dy = positions[source][1] - positions[target][1]
            distance = max(math.hypot(dx, dy), 1.0)
            force = distance * distance / ideal
            ux, uy = dx / distance, dy / distance
            displacement[source][0] -= ux * force
            displacement[source][1] -= uy * force
            displacement[target][0] += ux * force
            displacement[target][1] += uy * force
        temperature = 85 * (1 - iteration / 180) + 2
        for alias in aliases:
            dx, dy = displacement[alias]
            magnitude = max(math.hypot(dx, dy), 1.0)
            positions[alias][0] += dx / magnitude * min(magnitude, temperature)
            positions[alias][1] += dy / magnitude * min(magnitude, temperature)
            positions[alias][0] = min(max(positions[alias][0], 95), width - 95)
            positions[alias][1] = min(max(positions[alias][1], 100), height - 90)
    return positions


def get_font(size: int, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def wrap_label(text: str, width=34):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if len(proposal) <= width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def render_sample(sample, output_path: Path, seed: int):
    image = Image.new("RGB", (1800, 1100), "#FAFAFA")
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    body_font = get_font(24)
    small_font = get_font(19)
    node_font = get_font(28, bold=True)

    aliases = [item["alias"] for item in sample["graph"]["nodes"]]
    node_lookup = {item["alias"]: item for item in sample["graph"]["nodes"]}
    edge_rows = [
        (item["source"], item["target"], item) for item in sample["graph"]["edges"]
    ]
    positions = force_layout(aliases, edge_rows, random.Random(seed), 1160, 920)
    draw.text((45, 28), f"Complex keyword graph - {sample['sample_id']}", font=title_font, fill="#222222")
    draw.text((45, 73), "Node color = community; edge width = co-occurrence weight (log-scaled)", font=body_font, fill="#444444")

    for source, target, attrs in edge_rows:
        left = positions[source]
        right = positions[target]
        weight = float(attrs["weight"])
        line_width = max(2, min(9, int(round(2 + math.log1p(weight) * 1.8))))
        draw.line((left[0], left[1], right[0], right[1]), fill="#6B7280", width=line_width)

    for alias in aliases:
        node = node_lookup[alias]
        community_index = int(node["community"]) - 1 if str(node["community"]).isdigit() else 0
        color = PALETTE[community_index % len(PALETTE)]
        x, y = positions[alias]
        radius = 39
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#FFFFFF", width=5)
        draw_centered(draw, (x, y), alias, node_font, "#FFFFFF")

    panel_x = 1210
    draw.rounded_rectangle((1190, 105, 1760, 1045), radius=18, fill="#FFFFFF", outline="#D1D5DB", width=2)
    draw.text((panel_x, 130), "Node legend", font=title_font, fill="#111827")
    y = 190
    for node in sample["graph"]["nodes"]:
        community_index = int(node["community"]) - 1 if str(node["community"]).isdigit() else 0
        color = PALETTE[community_index % len(PALETTE)]
        draw.ellipse((panel_x, y + 5, panel_x + 28, y + 33), fill=color)
        draw.text((panel_x + 40, y), f"{node['alias']}  [C{node['community']}]", font=body_font, fill="#111827")
        y += 35
        for line in wrap_label(node["label"]):
            draw.text((panel_x + 40, y), line, font=small_font, fill="#374151")
            y += 25
        y += 13

    image.save(output_path, format="PNG", optimize=True)


def make_sample(sample_id, task_type, path, selected, nodes, edges, rng):
    # Randomize aliases so the answer cannot be inferred from fixed N1-N2-N3-N4 ordering.
    alias_order = list(selected)
    rng.shuffle(alias_order)
    alias_by_label = {label: f"N{index + 1}" for index, label in enumerate(alias_order)}
    graph_nodes = [
        {
            "alias": alias_by_label[label],
            "label": nodes[label]["label"],
            "community": nodes[label]["community"],
            "occurrences": nodes[label]["occurrences"],
        }
        for label in alias_order
    ]
    sub_edges = induced_edges(selected, edges)
    graph_edges = [
        {
            "source": alias_by_label[source],
            "target": alias_by_label[target],
            "weight": attrs["weight"],
            "association_strength": attrs["association_strength"],
        }
        for source, target, attrs in sub_edges
    ]
    alias_path = [alias_by_label[item] for item in path]
    source_alias, target_alias = alias_path[0], alias_path[-1]
    source_label, target_label = nodes[path[0]]["label"], nodes[path[-1]]["label"]
    cross_edges = []
    for left, right in zip(path, path[1:]):
        if nodes[left]["community"] != nodes[right]["community"]:
            cross_edges.append([alias_by_label[left], alias_by_label[right]])

    gold: dict[str, Any] = {
        "source": source_alias,
        "target": target_alias,
        "shortest_distance": len(path) - 1,
        "one_shortest_path": alias_path,
        "evidence_edges": [[left, right] for left, right in zip(alias_path, alias_path[1:])],
        "cross_community_edges_on_path": cross_edges,
    }

    if task_type == "path_trace":
        question = (
            f"从 {source_alias}（{source_label}）到 {target_alias}（{target_label}）的最少跳数是多少？"
            "请给出一条完整最短路径，并列出路径上的证据边。"
        )
    elif task_type == "cross_community_path":
        question = (
            f"寻找从 {source_alias}（{source_label}）到 {target_alias}（{target_label}）的一条最短路径。"
            "除完整路径外，指出该路径上哪些边跨越了不同社区。"
        )
    elif task_type == "bridge_node":
        bridge = unique_bridge(selected, sub_edges)
        if bridge is None:
            return None
        disconnected_pairs, bridge_label, component_sizes = bridge
        bridge_alias = alias_by_label[bridge_label]
        gold.update(
            {
                "bridge_node": bridge_alias,
                "disconnected_pairs_after_removal": disconnected_pairs,
                "component_sizes_after_removal": component_sizes,
            }
        )
        question = (
            "如果删除一个节点，哪个节点会使剩余图中互相不可达的节点对数量最多？"
            "请返回该桥接节点，并列出至少两条与判断直接相关的证据边。"
        )
    elif task_type == "unsupported_edge":
        non_edges = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                if frozenset((left, right)) not in edges:
                    local_path = shortest_path(
                        {
                            node: {
                                neighbor: attrs
                                for neighbor, attrs in {
                                    other: edges[frozenset((node, other))]
                                    for other in selected
                                    if other != node and frozenset((node, other)) in edges
                                }.items()
                            }
                            for node in selected
                        },
                        left,
                        right,
                    )
                    if local_path and len(local_path) >= 3:
                        non_edges.append((left, right, local_path))
        if not non_edges:
            return None
        left, right, local_path = sorted(non_edges, key=lambda item: (-len(item[2]), item[0], item[1]))[0]
        left_alias, right_alias = alias_by_label[left], alias_by_label[right]
        alias_local_path = [alias_by_label[item] for item in local_path]
        gold.update(
            {
                "queried_pair": [left_alias, right_alias],
                "direct_edge_supported": False,
                "indirect_path": alias_local_path,
            }
        )
        question = (
            f"图中是否存在 {left_alias}（{nodes[left]['label']}）与 {right_alias}（{nodes[right]['label']}）之间的直接边？"
            "如果不存在，请明确回答否，并给出一条图中真实存在的间接路径；不得把间接路径说成直接关系。"
        )
    else:
        raise ValueError(task_type)

    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "question": question,
        "image": f"images/{sample_id}.png",
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "gold": gold,
    }


def write_outputs(samples, output_dir: Path, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    csv_fields = [
        "sample_id",
        "task_type",
        "question",
        "image",
        "node_count",
        "edge_count",
        "gold_json",
    ]
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "task_type": sample["task_type"],
                    "question": sample["question"],
                    "image": sample["image"],
                    "node_count": len(sample["graph"]["nodes"]),
                    "edge_count": len(sample["graph"]["edges"]),
                    "gold_json": json.dumps(sample["gold"], ensure_ascii=False),
                }
            )
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(args):
    run_dir = Path(args.run_dir)
    nodes_path = run_dir / "analyses" / "network_keyword_cooccurrence_nodes.csv"
    edges_path = run_dir / "analyses" / "network_keyword_cooccurrence_edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise SystemExit(f"缺少图数据：\n{nodes_path}\n{edges_path}")

    output_dir = Path(args.output)
    nodes, adjacency, edges = load_graph(nodes_path, edges_path)
    candidates = path_candidates(nodes, adjacency)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    task_types = ["path_trace", "cross_community_path", "bridge_node", "unsupported_edge"]
    counts = {item: 0 for item in task_types}
    samples = []
    used_signatures = set()
    candidate_cursor = 0
    attempts = 0
    while min(counts.values()) < args.per_type and attempts < len(candidates) * 3:
        task_type = task_types[len(samples) % len(task_types)]
        if counts[task_type] >= args.per_type:
            task_type = next(item for item in task_types if counts[item] < args.per_type)
        path = candidates[candidate_cursor % len(candidates)]
        candidate_cursor += 1
        attempts += 1
        signature = (task_type, path[0], path[-1])
        if signature in used_signatures:
            continue
        selected = add_distractors(path, nodes, adjacency, edges, rng, args.nodes_per_sample)
        sample_id = f"CG{len(samples) + 1:03d}"
        sample = make_sample(sample_id, task_type, path, selected, nodes, edges, rng)
        if sample is None:
            continue
        used_signatures.add(signature)
        samples.append(sample)
        counts[task_type] += 1

    if min(counts.values()) < args.per_type:
        raise SystemExit(f"未能生成足够样本，实际数量：{counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        render_sample(sample, output_dir / sample["image"], args.seed + index)

    metadata = {
        "source_run": str(run_dir.resolve()),
        "source_nodes": len(nodes),
        "source_edges": len(edges),
        "seed": args.seed,
        "nodes_per_sample": args.nodes_per_sample,
        "per_type": args.per_type,
        "sample_count": len(samples),
        "task_counts": counts,
        "construction": {
            "path_length_edges": [3, 4],
            "minimum_path_communities": 2,
            "distractors": "neighbors of internal path nodes",
            "subgraph_edges": "induced edges among selected nodes",
        },
    }
    write_outputs(samples, output_dir, metadata)
    print(f"已生成 {len(samples)} 个复杂图样本：{output_dir.resolve()}")
    print("任务分布：" + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Benchmark：{(output_dir / 'benchmark.jsonl').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a complex graph-reasoning benchmark from CiteWeave outputs.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--nodes-per-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
