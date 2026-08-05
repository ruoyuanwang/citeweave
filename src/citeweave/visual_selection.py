from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import VisualizationPolicy

MIN_OCCURRENCE = {
    "coauthorship": 2,
    "institution_collaboration": 2,
    "keyword_cooccurrence": 5,
    "cocitation": 3,
    "citation": 1,
    "bibliographic_coupling": 2,
}

LOW_INFORMATION_KEYWORDS = {
    "dan",
    "las",
    "los",
    "nda",
    "two decades",
    "study",
    "research",
    "article",
    "analysis",
}


@dataclass
class SelectedNetwork:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    positions: dict[str, np.ndarray]
    disclosure: dict[str, Any]
    layout_qa: dict[str, float]


def adaptive_occurrence_threshold(
    occurrences: pd.Series,
    *,
    network_name: str,
    target_candidates: int,
) -> int:
    """Choose a data-driven occurrence threshold without materializing a matrix."""
    values = pd.to_numeric(occurrences, errors="coerce").fillna(0).astype(int)
    values = values[values >= MIN_OCCURRENCE.get(network_name, 1)]
    if values.empty:
        return MIN_OCCURRENCE.get(network_name, 1)
    if len(values) <= target_candidates:
        return int(values.min())
    kth = int(values.nlargest(target_candidates).iloc[-1])
    return max(MIN_OCCURRENCE.get(network_name, 1), kth)


def _association_strength(edges: pd.DataFrame, occurrences: dict[str, float]) -> pd.Series:
    left = edges["source"].astype(str).map(occurrences).fillna(1).clip(lower=1)
    right = edges["target"].astype(str).map(occurrences).fillna(1).clip(lower=1)
    return pd.to_numeric(edges["weight"], errors="coerce").fillna(0) / (left * right)


def _merge_small_clusters(
    graph: nx.Graph,
    communities: list[set[str]],
    min_size: int,
) -> dict[str, int]:
    large = [set(group) for group in communities if len(group) >= min_size]
    small_groups = [set(group) for group in communities if len(group) < min_size]
    if not large:
        large = [set(group) for group in communities]
        small_groups = []
    for small_group in small_groups:
        scores = []
        for index, group in enumerate(large):
            score = sum(
                float(graph[node][neighbor].get("strength", 0))
                for node in small_group
                for neighbor in graph.neighbors(node)
                if neighbor in group
            )
            scores.append((score, -len(group), -index))
        if scores and max(scores)[0] > 0:
            best = max(range(len(scores)), key=lambda index: scores[index])
            large[best].update(small_group)
        else:
            # Disconnected author teams or topic islands must not be assigned a
            # misleading color merely to satisfy a minimum cluster size.
            large.append(small_group)
    ordered = sorted(
        large,
        key=lambda group: (
            -sum(float(graph.degree(node, weight="strength")) for node in group),
            min(group),
        ),
    )
    return {node: index + 1 for index, group in enumerate(ordered) for node in group}


def _community_quotas(
    nodes: pd.DataFrame,
    max_nodes: int,
    max_clusters: int,
) -> dict[int, int]:
    weights = nodes.groupby("cluster")["importance"].sum().clip(lower=0)
    if weights.empty:
        return {}
    if len(weights) > max_clusters:
        weights = weights.nlargest(max_clusters)
    if 2 * len(weights) > max_nodes:
        weights = weights.nlargest(max(max_nodes // 2, 1))
    shares = weights / max(float(weights.sum()), 1)
    quotas = {int(cluster): max(2, round(max_nodes * share)) for cluster, share in shares.items()}
    while sum(quotas.values()) > max_nodes:
        candidates = [cluster for cluster, value in quotas.items() if value > 2]
        if not candidates:
            break
        cluster = max(candidates, key=lambda item: quotas[item] - max_nodes * shares.loc[item])
        quotas[cluster] -= 1
    while sum(quotas.values()) < max_nodes:
        cluster = max(
            quotas,
            key=lambda item: max_nodes * shares.loc[item] - quotas[item],
        )
        quotas[cluster] += 1
    return quotas


def _sparsify_edges(
    graph: nx.Graph,
    edge_frame: pd.DataFrame,
    max_edges_per_node: int,
) -> pd.DataFrame:
    if edge_frame.empty:
        return edge_frame
    edge_frame = edge_frame.copy()
    edge_frame["_score"] = edge_frame["association_strength"] * np.log1p(edge_frame["weight"])
    backbone: set[tuple[str, str]] = set()
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        tree = nx.maximum_spanning_tree(subgraph, weight="strength")
        backbone.update(tuple(sorted((str(left), str(right)))) for left, right in tree.edges())
    per_node: set[tuple[str, str]] = set(backbone)
    for node in sorted(graph):
        incident = edge_frame[
            (edge_frame["source"] == node) | (edge_frame["target"] == node)
        ].nlargest(max_edges_per_node, "_score")
        per_node.update(
            tuple(sorted((str(row.source), str(row.target))))
            for row in incident.itertuples(index=False)
        )
    edge_frame["_key"] = [
        tuple(sorted((str(left), str(right))))
        for left, right in zip(edge_frame["source"], edge_frame["target"])
    ]
    result = edge_frame[edge_frame["_key"].isin(per_node)].copy()
    global_cap = max(len(graph) - 1, max_edges_per_node * len(graph))
    if len(result) > global_cap:
        mandatory = result[result["_key"].isin(backbone)]
        remainder = result[~result["_key"].isin(backbone)].nlargest(
            max(global_cap - len(mandatory), 0), "_score"
        )
        result = pd.concat([mandatory, remainder], ignore_index=True)
    return result.drop(columns=["_score", "_key"], errors="ignore").drop_duplicates(
        ["source", "target"]
    )


def _normalize_positions(positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = list(positions)
    points = np.asarray([positions[node] for node in ids], dtype=float)
    points -= points.mean(axis=0)
    if len(points) > 2:
        _, _, vh = np.linalg.svd(points, full_matrices=False)
        points = points @ vh.T
    span = np.ptp(points, axis=0)
    if span[1] > span[0]:
        points = points[:, ::-1]
        span = span[::-1]
    points[:, 0] /= max(span[0], 1e-9)
    points[:, 1] /= max(span[1], 1e-9)
    points[:, 0] *= 2.2
    points[:, 1] *= 1.45
    return {node: points[index] for index, node in enumerate(ids)}


def _resolve_node_overlaps(
    positions: dict[str, np.ndarray],
    radii: dict[str, float],
    *,
    padding: float = 1.18,
    iterations: int = 160,
) -> dict[str, np.ndarray]:
    """Apply a deterministic collision pass without altering graph topology."""
    result = {node: np.asarray(point, dtype=float).copy() for node, point in positions.items()}
    nodes = sorted(result)
    for _ in range(iterations):
        maximum_shift = 0.0
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                delta = result[right] - result[left]
                distance = float(np.linalg.norm(delta))
                target = padding * (float(radii.get(left, 0.02)) + float(radii.get(right, 0.02)))
                if distance >= target:
                    continue
                if distance < 1e-9:
                    angle = (
                        (sum(ord(character) for character in left + right) % 360) * math.pi / 180
                    )
                    direction = np.asarray([math.cos(angle), math.sin(angle)])
                else:
                    direction = delta / distance
                shift = 0.52 * (target - distance)
                result[left] -= direction * shift
                result[right] += direction * shift
                maximum_shift = max(maximum_shift, shift)
        if maximum_shift < 1e-4:
            break
    center = np.mean(list(result.values()), axis=0)
    return {node: point - center for node, point in result.items()}


def _layout_score(
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    node_radii: dict[str, float],
    clusters: dict[str, int] | None = None,
) -> tuple[float, dict[str, float]]:
    if len(graph) < 2:
        return 0.0, {"overlap_ratio": 0.0, "mean_edge_length": 0.0, "layout_score": 0.0}
    distances = []
    weighted_lengths = []
    overlaps = 0
    pairs = 0
    nodes = list(graph)
    for left, right, data in graph.edges(data=True):
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        distances.append(distance)
        weighted_lengths.append(distance * math.sqrt(max(float(data.get("strength", 0)), 1e-12)))
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            pairs += 1
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if distance < node_radii[left] + node_radii[right]:
                overlaps += 1
    overlap_ratio = overlaps / max(pairs, 1)
    mean_edge_length = float(np.mean(distances)) if distances else 0.0
    weighted_edge_length = float(np.mean(weighted_lengths)) if weighted_lengths else 0.0
    cluster_separation = 0.0
    if clusters and len(set(clusters.values())) > 1:
        within = []
        between = []
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                distance = float(np.linalg.norm(positions[left] - positions[right]))
                (within if clusters[left] == clusters[right] else between).append(distance)
        if within and between:
            cluster_separation = float(np.mean(within) / max(np.mean(between), 1e-9))
    score = (
        30 * overlap_ratio
        + weighted_edge_length
        + 0.05 * mean_edge_length
        + 0.30 * cluster_separation
    )
    return score, {
        "overlap_ratio": overlap_ratio,
        "mean_edge_length": mean_edge_length,
        "weighted_edge_length": weighted_edge_length,
        "within_between_distance_ratio": cluster_separation,
        "layout_score": score,
    }


def multi_start_forceatlas2(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    *,
    seed: int,
    restarts: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Use several deterministic starts and retain the best readable layout."""
    occurrence = nodes.set_index("id")["occurrences"].astype(float).clip(lower=1)
    lo, hi = occurrence.quantile([0.05, 0.95]).tolist()
    clipped = occurrence.clip(lower=lo, upper=max(hi, lo + 1))
    radii = 0.018 + 0.026 * np.sqrt((clipped - lo) / max(hi - lo, 1))
    node_radii = radii.to_dict()
    best_score = math.inf
    best_positions: dict[str, np.ndarray] = {}
    best_qa: dict[str, float] = {}
    for restart in range(restarts):
        run_seed = seed + restart * 1_009
        initial = nx.random_layout(graph, seed=run_seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            positions = nx.forceatlas2_layout(
                graph,
                pos=initial,
                max_iter=iterations,
                scaling_ratio=2.8,
                gravity=0.45,
                node_mass={node: 1 + math.log1p(float(occurrence.get(node, 1))) for node in graph},
                node_size={node: float(node_radii.get(node, 0.02)) for node in graph},
                weight="strength",
                linlog=True,
                seed=run_seed,
            )
        normalized = _normalize_positions(positions)
        normalized = _resolve_node_overlaps(normalized, node_radii)
        score, qa = _layout_score(graph, normalized, node_radii)
        if score < best_score:
            best_score = score
            best_positions = normalized
            best_qa = {**qa, "selected_seed": float(run_seed)}
    return best_positions, best_qa


def _community_initial_positions(
    graph: nx.Graph,
    clusters: dict[str, int],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(set(clusters.values()))
    quotient = nx.Graph()
    quotient.add_nodes_from(cluster_ids)
    for left, right, data in graph.edges(data=True):
        left_cluster = clusters[left]
        right_cluster = clusters[right]
        if left_cluster == right_cluster:
            continue
        current = quotient.get_edge_data(left_cluster, right_cluster, {}).get("weight", 0.0)
        quotient.add_edge(
            left_cluster,
            right_cluster,
            weight=current + float(data.get("strength", 0)),
        )
    if len(quotient) == 1:
        centers = {cluster_ids[0]: np.zeros(2)}
    elif quotient.number_of_edges():
        centers = nx.spring_layout(quotient, seed=seed, weight="weight", scale=1.8)
    else:
        centers = nx.circular_layout(quotient, scale=1.8)
    positions = {}
    for cluster in cluster_ids:
        members = sorted(node for node in graph if clusters[node] == cluster)
        radius = 0.08 + 0.025 * math.sqrt(len(members))
        angles = rng.uniform(0, 2 * math.pi, len(members))
        jitter = rng.uniform(0.35, 1.0, len(members)) * radius
        for index, node in enumerate(members):
            positions[node] = np.asarray(centers[cluster]) + jitter[index] * np.asarray(
                [math.cos(angles[index]), math.sin(angles[index])]
            )
    return positions


def multi_start_vos_layout(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    *,
    seed: int,
    restarts: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Optimize a VOS-style similarity map with all-pairs repulsion."""
    ids = sorted(graph)
    index = {node: position for position, node in enumerate(ids)}
    size = len(ids)
    similarity = np.zeros((size, size), dtype=float)
    positive = []
    for left, right, data in graph.edges(data=True):
        value = max(float(data.get("strength", 0)), 1e-12)
        positive.append(value)
        similarity[index[left], index[right]] = value
        similarity[index[right], index[left]] = value
    scale = float(np.median(positive)) if positive else 1.0
    similarity /= max(scale, 1e-12)
    positive_similarity = similarity[similarity > 0]
    cap = float(np.quantile(positive_similarity, 0.95)) if len(positive_similarity) else 1.0
    similarity = np.sqrt(np.clip(similarity, 0, cap))
    clusters = nodes.set_index("id")["cluster"].astype(int).to_dict()
    occurrence = nodes.set_index("id")["occurrences"].astype(float).clip(lower=1)
    lo, hi = occurrence.quantile([0.05, 0.95]).tolist()
    clipped = occurrence.clip(lower=lo, upper=max(hi, lo + 1))
    radii = 0.018 + 0.026 * np.sqrt((clipped - lo) / max(hi - lo, 1))
    node_radii = radii.to_dict()
    repulsion = 0.035
    diagonal_mask = ~np.eye(size, dtype=bool)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        points = flat.reshape(size, 2)
        delta = points[:, None, :] - points[None, :, :]
        distance_squared = np.sum(delta * delta, axis=2) + 1e-5
        attraction = 0.5 * float(np.sum(similarity * distance_squared))
        repulsive = -0.5 * repulsion * float(np.log(distance_squared[diagonal_mask]).sum())
        center = points.mean(axis=0)
        center_penalty = 25.0 * float(center @ center)
        coefficients = similarity - repulsion / distance_squared
        np.fill_diagonal(coefficients, 0.0)
        gradient = 2 * np.sum(coefficients[:, :, None] * delta, axis=1)
        gradient += 50.0 * center / size
        return attraction + repulsive + center_penalty, gradient.ravel()

    best_score = math.inf
    best_positions: dict[str, np.ndarray] = {}
    best_qa: dict[str, float] = {}
    for restart in range(restarts):
        run_seed = seed + restart * 1_009
        initial = _community_initial_positions(graph, clusters, run_seed)
        flat = np.asarray([initial[node] for node in ids], dtype=float).ravel()
        result = minimize(
            objective,
            flat,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": min(iterations, 1_000), "ftol": 1e-10, "maxls": 30},
        )
        coordinates = result.x.reshape(size, 2)
        normalized = _normalize_positions({node: coordinates[index[node]] for node in ids})
        normalized = _resolve_node_overlaps(normalized, node_radii)
        score, qa = _layout_score(graph, normalized, node_radii, clusters)
        if score < best_score:
            best_score = score
            best_positions = normalized
            best_qa = {
                **qa,
                "selected_seed": float(run_seed),
                "optimizer_iterations": float(result.nit),
                "optimizer_converged": float(bool(result.success)),
            }
    return best_positions, best_qa


def multi_start_cluster_layout(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    *,
    seed: int,
    restarts: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Pack communities first, then optimize association-strength structure within each."""
    clusters = nodes.set_index("id")["cluster"].astype(int).to_dict()
    occurrence = nodes.set_index("id")["occurrences"].astype(float).clip(lower=1)
    lo, hi = occurrence.quantile([0.05, 0.95]).tolist()
    clipped = occurrence.clip(lower=lo, upper=max(hi, lo + 1))
    radii = 0.018 + 0.026 * np.sqrt((clipped - lo) / max(hi - lo, 1))
    node_radii = radii.to_dict()
    cluster_ids = sorted(set(clusters.values()))
    quotient = nx.Graph()
    quotient.add_nodes_from(cluster_ids)
    for left, right, data in graph.edges(data=True):
        left_cluster = clusters[left]
        right_cluster = clusters[right]
        if left_cluster == right_cluster:
            continue
        current = quotient.get_edge_data(left_cluster, right_cluster, {}).get("weight", 0.0)
        quotient.add_edge(
            left_cluster,
            right_cluster,
            weight=current + math.sqrt(max(float(data.get("strength", 0)), 1e-12)),
        )
    best_score = math.inf
    best_positions: dict[str, np.ndarray] = {}
    best_qa: dict[str, float] = {}
    for restart in range(restarts):
        run_seed = seed + restart * 1_009
        if len(quotient) == 1:
            centers = {cluster_ids[0]: np.zeros(2)}
        elif quotient.number_of_edges():
            centers = nx.spring_layout(
                quotient,
                seed=run_seed,
                weight="weight",
                k=1.45 / math.sqrt(len(quotient)),
                iterations=min(iterations, 500),
                scale=1.15,
            )
        else:
            centers = nx.circular_layout(quotient, scale=1.15)
        positions: dict[str, np.ndarray] = {}
        for cluster in cluster_ids:
            members = sorted(node for node in graph if clusters[node] == cluster)
            subgraph = graph.subgraph(members)
            if len(members) == 1:
                local = {members[0]: np.zeros(2)}
            else:
                local_scale = min(0.58, 0.18 + 0.065 * math.sqrt(len(members)))
                local = nx.spring_layout(
                    subgraph,
                    seed=run_seed + cluster * 37,
                    weight="strength",
                    k=0.9 / math.sqrt(len(members)),
                    iterations=min(iterations, 500),
                    scale=local_scale,
                )
            center = 1.10 * np.asarray(centers[cluster])
            for node, point in local.items():
                positions[node] = center + np.asarray(point)
        normalized = _normalize_positions(positions)
        normalized = _resolve_node_overlaps(normalized, node_radii)
        score, qa = _layout_score(graph, normalized, node_radii, clusters)
        if score < best_score:
            best_score = score
            best_positions = normalized
            best_qa = {
                **qa,
                "selected_seed": float(run_seed),
                "optimizer_iterations": float(min(iterations, 500)),
                "optimizer_converged": 1.0,
            }
    return best_positions, best_qa


def select_network(
    name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    policy: VisualizationPolicy,
    *,
    seed: int,
) -> SelectedNetwork | None:
    """Candidate-first network selection with no all-pairs adjacency matrix."""
    if nodes.empty or edges.empty:
        return None
    node_frame = nodes.copy()
    edge_frame = edges.rename(columns={"source_id": "source", "target_id": "target"}).copy()
    node_frame["id"] = node_frame["id"].astype(str)
    if name == "keyword_cooccurrence":
        normalized_labels = node_frame["label"].astype(str).str.strip().str.casefold()
        node_frame = node_frame[
            normalized_labels.str.len().ge(3) & ~normalized_labels.isin(LOW_INFORMATION_KEYWORDS)
        ].copy()
    edge_frame["source"] = edge_frame["source"].astype(str)
    edge_frame["target"] = edge_frame["target"].astype(str)
    node_frame["occurrences"] = pd.to_numeric(node_frame["occurrences"], errors="coerce").fillna(0)
    linked_ids = set(edge_frame["source"]) | set(edge_frame["target"])
    linked_nodes = node_frame[node_frame["id"].isin(linked_ids)].copy()
    if linked_nodes.empty:
        return None
    target_candidates = min(
        policy.max_candidate_nodes,
        max(policy.max_display_nodes * policy.candidate_multiplier, policy.max_display_nodes),
    )
    threshold = adaptive_occurrence_threshold(
        linked_nodes["occurrences"],
        network_name=name,
        target_candidates=target_candidates,
    )
    eligible = linked_nodes[linked_nodes["occurrences"] >= threshold].nlargest(
        target_candidates, "occurrences"
    )
    eligible_ids = set(eligible["id"])
    edge_frame = edge_frame[
        edge_frame["source"].isin(eligible_ids) & edge_frame["target"].isin(eligible_ids)
    ].copy()
    if edge_frame.empty:
        return None
    occurrence_lookup = eligible.set_index("id")["occurrences"].to_dict()
    edge_frame["association_strength"] = _association_strength(edge_frame, occurrence_lookup)
    candidate_graph = nx.Graph()
    for row in edge_frame.itertuples(index=False):
        candidate_graph.add_edge(
            row.source,
            row.target,
            weight=float(row.weight),
            strength=float(row.association_strength),
        )
    candidate_graph.remove_nodes_from(list(nx.isolates(candidate_graph)))
    if not candidate_graph:
        return None
    communities = list(
        nx.community.louvain_communities(
            candidate_graph, weight="strength", resolution=1.0, seed=seed
        )
    )
    cluster_map = _merge_small_clusters(candidate_graph, communities, policy.min_cluster_size)
    total_links = dict(candidate_graph.degree(weight="weight"))
    candidate_nodes = eligible[eligible["id"].isin(candidate_graph)].copy()
    candidate_nodes["cluster"] = candidate_nodes["id"].map(cluster_map).astype(int)
    candidate_nodes["total_link_strength"] = candidate_nodes["id"].map(total_links).fillna(0)
    occurrence_rank = candidate_nodes["occurrences"].rank(pct=True)
    strength_rank = candidate_nodes["total_link_strength"].rank(pct=True)
    candidate_nodes["importance"] = 0.42 * occurrence_rank + 0.58 * strength_rank
    semantic_network = name in {
        "keyword_cooccurrence",
        "cocitation",
        "bibliographic_coupling",
    }
    max_clusters = (
        min(policy.max_display_clusters, 8) if semantic_network else policy.max_display_clusters
    )
    display_node_limit = min(
        policy.max_display_nodes,
        60 if name in {"coauthorship", "citation"} else policy.max_display_nodes,
    )
    quotas = _community_quotas(
        candidate_nodes,
        min(display_node_limit, len(candidate_nodes)),
        max_clusters,
    )
    selected_parts = []
    for cluster, quota in quotas.items():
        group = candidate_nodes[candidate_nodes["cluster"] == cluster]
        selected_parts.append(group.nlargest(quota, "importance"))
    selected_nodes = pd.concat(selected_parts, ignore_index=True).drop_duplicates("id")
    if len(selected_nodes) < min(display_node_limit, len(candidate_nodes)):
        remainder = candidate_nodes[~candidate_nodes["id"].isin(selected_nodes["id"])]
        selected_nodes = pd.concat(
            [
                selected_nodes,
                remainder.nlargest(display_node_limit - len(selected_nodes), "importance"),
            ],
            ignore_index=True,
        )
    selected_ids = set(selected_nodes["id"])
    selected_edges = edge_frame[
        edge_frame["source"].isin(selected_ids) & edge_frame["target"].isin(selected_ids)
    ].copy()
    display_graph = nx.Graph()
    for row in selected_edges.itertuples(index=False):
        display_graph.add_edge(
            row.source,
            row.target,
            weight=float(row.weight),
            strength=float(row.association_strength),
        )
    if not display_graph:
        return None
    # Do not show isolated quota members: maps communicate relations, not ranked lists.
    selected_nodes = selected_nodes[selected_nodes["id"].isin(display_graph)].copy()
    selected_edges = _sparsify_edges(display_graph, selected_edges, policy.max_edges_per_node)
    final_graph = nx.Graph()
    for row in selected_edges.itertuples(index=False):
        final_graph.add_edge(
            row.source,
            row.target,
            weight=float(row.weight),
            strength=float(row.association_strength),
        )
    selected_nodes = selected_nodes[selected_nodes["id"].isin(final_graph)].copy()
    forceatlas_layout = policy.layout_algorithm == "forceatlas2" or name == "coauthorship"
    layout_function = multi_start_forceatlas2 if forceatlas_layout else multi_start_cluster_layout
    positions, layout_qa = layout_function(
        final_graph,
        selected_nodes,
        seed=seed,
        restarts=policy.layout_restarts,
        iterations=policy.layout_iterations,
    )
    disclosure = {
        "method": "candidate-first association-strength map",
        "matrix_materialized": False,
        "counting_method": policy.counting_method,
        "minimum_occurrence": int(threshold),
        "eligible_items": len(eligible),
        "candidate_nodes_with_links": len(candidate_nodes),
        "candidate_edges": len(edge_frame),
        "displayed_nodes": len(selected_nodes),
        "displayed_edges": len(selected_edges),
        "displayed_clusters": int(selected_nodes["cluster"].nunique()),
        "maximum_display_clusters": max_clusters,
        "maximum_display_nodes": display_node_limit,
        "maximum_edges_per_node": policy.max_edges_per_node,
        "normalization": "association strength: c_ij / (w_i * w_j)",
        "edge_reduction": "maximum-spanning forest plus strongest edges per node",
        "community_detection": "Louvain on association strength; small clusters merged",
        "layout": (
            "ForceAtlas2 LinLog; best of deterministic multiple starts"
            if forceatlas_layout
            else "two-level VOS-style association-strength community layout; best of multiple starts"
        ),
        "layout_restarts": policy.layout_restarts,
        "layout_iterations": policy.layout_iterations,
    }
    return SelectedNetwork(
        name=name,
        nodes=selected_nodes,
        edges=selected_edges,
        positions=positions,
        disclosure=disclosure,
        layout_qa=layout_qa,
    )
