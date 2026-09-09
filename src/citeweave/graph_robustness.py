"""Outcome-independent sensitivity of fixed bibliometric graph phenomena.

Exploratory diagnostics, not a new gold generator or a model evaluation. Keep all
baseline vertices when deleting edges so isolation cannot disappear from the
denominator. Community comparisons use memberships, never integer label equality.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def load_verified_trends(workspace: Path, original_trends: Path) -> pd.DataFrame:
    """Recompute from corrected non-author canonicals; never silently trust a fallback."""
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    try:
        actual = connection.execute(
            """
            WITH selected AS (
              SELECT keyword, occurrences AS global_documents FROM read_parquet(?)
              WHERE keyword IS NOT NULL ORDER BY occurrences DESC, keyword LIMIT 15
            )
            SELECT w.year, k.keyword, count(DISTINCT k.work_id) AS documents,
                   max(s.global_documents) AS global_documents
            FROM read_parquet(?) k JOIN selected s USING(keyword)
            JOIN read_parquet(?) w USING(work_id)
            WHERE w.year IS NOT NULL GROUP BY w.year, k.keyword
            ORDER BY global_documents DESC, keyword, year
            """,
            [
                str(workspace / "canonical/visualization/keyword_occurrences.parquet"),
                str(workspace / "canonical/keywords.parquet"),
                str(workspace / "canonical/works.parquet"),
            ],
        ).df()
    finally:
        connection.close()
    expected = pd.read_parquet(original_trends)
    columns = ["year", "keyword", "documents", "global_documents"]
    left = actual[columns].sort_values(["keyword", "year"]).reset_index(drop=True)
    right = expected[columns].sort_values(["keyword", "year"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    return actual


def registered_variants() -> list[dict]:
    base = {
        "minimum_weight": 1,
        "drop_fraction": 0.0,
        "drop_seed": 11,
        "community_seed": 42,
        "resolution": 1.0,
        "window_years": 3,
    }
    changes = [("baseline", "baseline", {})]
    changes += [(f"weight_{n}", "minimum_weight", {"minimum_weight": n}) for n in (2, 3, 5)]
    changes += [(f"seed_{n}", "community_seed", {"community_seed": n}) for n in (0, 1, 17)]
    changes += [(f"resolution_{n}", "resolution", {"resolution": n}) for n in (0.5, 1.5, 2.0)]
    changes += [
        (f"drop_{fraction}_{seed}", "edge_dropout", {"drop_fraction": fraction, "drop_seed": seed})
        for fraction in (0.01, 0.05)
        for seed in (11, 29, 47)
    ]
    changes += [(f"window_{n}", "temporal_window", {"window_years": n}) for n in (2, 4)]
    return [
        {**base, **change, "variant_id": name, "family": family} for name, family, change in changes
    ]


def perturb_graph(graph: nx.Graph, spec: dict) -> nx.Graph:
    changed = graph.copy()
    minimum = float(spec["minimum_weight"])
    fraction = float(spec["drop_fraction"])
    if minimum < 0 or not 0 <= fraction <= 1:
        raise ValueError("Invalid perturbation")
    edges = []
    for u, v, attrs in graph.edges(data=True):
        weight = float(attrs["weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Positive finite edge weights required")
        if weight < minimum:
            changed.remove_edge(u, v)
        else:
            edges.append(tuple(sorted((u, v))))
    # Exact fixed count, SHA ordering independent of input iteration and Python RNG.
    seed = spec["drop_seed"]
    edges.sort(
        key=lambda edge: hashlib.sha256(
            f"edge-drop-v1|{seed}|{len(edge[0])}:{edge[0]}|{edge[1]}".encode()
        ).digest()
    )
    changed.remove_edges_from(edges[: math.floor(fraction * len(edges))])
    return changed


def partition(graph: nx.Graph, spec: dict) -> dict[str, int]:
    if graph.number_of_edges() == 0:
        groups = [{node} for node in graph]
    else:
        groups = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=int(spec["community_seed"]),
            resolution=float(spec["resolution"]),
        )
    ordered = sorted((sorted(group) for group in groups), key=lambda group: (-len(group), group[0]))
    return {node: index for index, group in enumerate(ordered) for node in group}


def community_roles(graph: nx.Graph, communities: dict[str, int]) -> dict:
    members: dict[int, list[str]] = defaultdict(list)
    for node, group in communities.items():
        members[group].append(node)
    internal, external = defaultdict(float), defaultdict(float)
    for u, v, attrs in graph.edges(data=True):
        a, b, weight = communities[u], communities[v], float(attrs["weight"])
        if a == b:
            internal[a] += weight
        else:
            external[a] += weight
            external[b] += weight
    rows = []
    for group, nodes in members.items():
        if len(nodes) < 2:
            continue
        total = internal[group] + external[group]
        representative = min(
            nodes, key=lambda n: (-graph.nodes[n]["importance"], graph.nodes[n]["label"])
        )
        rows.append(
            {
                "community": group,
                "members": sorted(nodes),
                "nodes": len(nodes),
                "importance": sum(graph.nodes[n]["importance"] for n in nodes),
                "external_share": external[group] / total if total else 0.0,
                "representative": graph.nodes[representative]["label"],
            }
        )
    disconnected = sum(not nx.is_connected(graph.subgraph(nodes)) for nodes in members.values())
    if len(rows) < 2:
        return {"status": "insufficient_communities", "disconnected_communities": disconnected}
    dominant = min(rows, key=lambda r: (-r["importance"], r["community"]))
    outward = min(
        (r for r in rows if r["community"] != dominant["community"]),
        key=lambda r: (-r["external_share"], r["community"]),
    )
    return {
        "status": "measured",
        "dominant": dominant,
        "outward": outward,
        "disconnected_communities": disconnected,
    }


def _trace(task: dict, operator: str) -> dict:
    return next(step for step in task["operator_trace"] if step["operator"] == operator)


def measure_fixed_path(graph: nx.Graph, communities: dict, task: dict) -> dict:
    original = _trace(task, "shortest_path")["path"]
    source, target = original[0], original[-1]
    if source not in graph or target not in graph:
        return {"status": "anchor_missing"}
    cross = communities[source] != communities[target]
    try:
        path = nx.shortest_path(graph, source, target, weight="distance")
    except nx.NetworkXNoPath:
        return {"status": "disconnected", "cross_community": cross}
    distance = sum(graph.edges[a, b]["distance"] for a, b in pairwise(path))
    original_present = all(graph.has_edge(a, b) for a, b in pairwise(original))
    original_distance = (
        sum(graph.edges[a, b]["distance"] for a, b in pairwise(original))
        if original_present
        else None
    )
    return {
        "status": "measured",
        "total_inverse_weight_distance": distance,
        "hops": len(path) - 1,
        "path": path,
        "cross_community": cross,
        "original_path_still_optimal": original_present
        and math.isclose(original_distance, distance, rel_tol=1e-9, abs_tol=1e-9),
    }


def measure_fixed_edge(graph: nx.Graph, communities: dict, task: dict) -> dict:
    source, target = _trace(task, "edge_betweenness")["selected_edge"]
    if not graph.has_edge(source, target):
        return {"status": "anchor_edge_absent"}
    before = nx.number_connected_components(graph)
    changed = graph.copy()
    changed.remove_edge(source, target)
    after = nx.number_connected_components(changed)
    try:
        hops = nx.shortest_path_length(changed, source, target)
    except nx.NetworkXNoPath:
        hops = None
    return {
        "status": "measured",
        "component_increase": after - before,
        "alternate_hops_after_deletion": hops,
        "redundant_connector": hops is not None,
        "cross_community": communities[source] != communities[target],
        "highest_betweenness_selection_retested": False,
    }


def measure_fixed_hub(graph: nx.Graph, task: dict) -> dict:
    node = _trace(task, "delete_node")["node"]
    if node not in graph:
        return {"status": "anchor_missing"}
    ranking = sorted(
        graph, key=lambda n: (-graph.degree(n, weight="weight"), graph.nodes[n]["label"])
    )
    changed = graph.copy()
    changed.remove_node(node)
    components = list(nx.connected_components(changed))
    largest = max(map(len, components), default=0)
    replacement = min(
        changed,
        key=lambda n: (-changed.degree(n, weight="weight"), changed.nodes[n]["label"]),
        default=None,
    )
    return {
        "status": "measured",
        "frozen_hub_still_argmax": ranking[0] == node,
        "frozen_hub_rank": ranking.index(node) + 1,
        "reselected_hub": graph.nodes[ranking[0]]["label"],
        "replacement_hub": graph.nodes[replacement]["label"] if replacement else None,
        "components_after": len(components),
        "largest_component_fraction_of_original_nodes": largest / len(graph),
    }


def measure_temporal(graph: nx.Graph, communities: dict, trends: pd.DataFrame, window: int) -> dict:
    trends = trends[trends.keyword.astype(str).isin(graph)].copy()
    years = sorted(int(year) for year in trends.year.dropna().unique())
    if len(years) < 2 * window:
        return {"status": "insufficient_nonoverlapping_years"}
    early_years, recent_years = years[:window], years[-window:]
    rows = []
    for node, group in trends.groupby("keyword"):
        early = float(group[group.year.isin(early_years)].documents.sum())
        recent = float(group[group.year.isin(recent_years)].documents.sum())
        rows.append(
            {
                "node": str(node),
                "early": early,
                "recent": recent,
                "growth": (recent + 1) / (early + 1),
                "weighted_degree": float(graph.degree(str(node), weight="weight")),
            }
        )
    if len(rows) < 2:
        return {"status": "insufficient_nodes"}
    floor = sorted(r["recent"] for r in rows)[len(rows) // 2]
    emerging = min(
        (r for r in rows if r["recent"] >= floor),
        key=lambda r: (-r["growth"], -r["recent"], r["node"]),
    )
    established = min(
        (r for r in rows if r["node"] != emerging["node"]),
        key=lambda r: (-r["weighted_degree"], -r["early"], r["node"]),
    )
    return {
        "status": "measured",
        "early_years": early_years,
        "recent_years": recent_years,
        "emerging_label": graph.nodes[emerging["node"]]["label"],
        "emerging_recent_documents": int(emerging["recent"]),
        "emerging_growth_ratio": emerging["growth"],
        "established_label": graph.nodes[established["node"]]["label"],
        "established_weighted_degree": established["weighted_degree"],
        "same_community": communities[emerging["node"]] == communities[established["node"]],
        "graph_is_full_period_not_window_specific": True,
        "candidate_keywords_are_fixed_original_trend_table": True,
    }


def measure_variant(
    graph: nx.Graph,
    tasks: list[dict],
    trends: pd.DataFrame,
    spec: dict,
    *,
    precomputed_partition: dict | None = None,
) -> tuple[dict, dict]:
    communities = (
        precomputed_partition if precomputed_partition is not None else partition(graph, spec)
    )
    components = list(nx.connected_components(graph))
    measurements = {}
    for task in tasks:
        kind = task["task_type"]
        if kind == "multi_hop_connector":
            value = measure_fixed_path(graph, communities, task)
        elif kind == "bridge_counterfactual":
            value = measure_fixed_edge(graph, communities, task)
        elif kind == "hub_removal_resilience":
            value = measure_fixed_hub(graph, task)
        elif kind == "community_role_contrast":
            value = community_roles(graph, communities)
        elif kind == "temporal_structural_shift":
            value = measure_temporal(graph, communities, trends, int(spec["window_years"]))
        else:
            raise ValueError(f"Unsupported sensitivity task: {kind}")
        measurements[kind] = {"item_id": task["item_id"], **value}
    return {
        "variant": spec,
        "graph": {
            "nodes": len(graph),
            "edges": graph.number_of_edges(),
            "isolates": nx.number_of_isolates(graph),
            "components": len(components),
            "largest_component_fraction": max(map(len, components), default=0) / max(1, len(graph)),
            "communities": len(set(communities.values())),
        },
        "measurements": measurements,
    }, communities


def baseline_checks(measurements: dict, tasks: list[dict]) -> list[dict]:
    records = []
    fields = {
        "multi_hop_connector": ("total_inverse_weight_distance",),
        "bridge_counterfactual": ("component_increase", "alternate_hops_after_deletion"),
        "hub_removal_resilience": (
            "replacement_hub",
            "components_after",
            "largest_component_fraction_of_original_nodes",
        ),
        "temporal_structural_shift": (
            "emerging_label",
            "emerging_recent_documents",
            "emerging_growth_ratio",
            "established_label",
            "established_weighted_degree",
            "same_community",
        ),
    }
    for task in tasks:
        measured = measurements[task["task_type"]]
        if measured["status"] != "measured":
            records.append(
                {"item_id": task["item_id"], "passed": False, "reason": measured["status"]}
            )
            continue
        if task["task_type"] == "community_role_contrast":
            measured = {
                "dominant_community": measured["dominant"]["community"],
                "dominant_representative": measured["dominant"]["representative"],
                "dominant_nodes": measured["dominant"]["nodes"],
                "outward_community": measured["outward"]["community"],
                "outward_representative": measured["outward"]["representative"],
                "outward_external_share": measured["outward"]["external_share"],
            }
            selected_fields = tuple(measured)
        else:
            selected_fields = fields[task["task_type"]]
        mismatches = []
        for field in selected_fields:
            a, b = measured[field], task["answer"][field]
            equal = (
                math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6)
                if isinstance(a, (float, int)) and isinstance(b, (float, int))
                else a == b
            )
            if not equal:
                mismatches.append(field)
        records.append(
            {
                "item_id": task["item_id"],
                "passed": not mismatches,
                "checked_fields": list(selected_fields),
                "mismatches": mismatches,
            }
        )
    return records


def compare_variants(
    baseline: dict, changed: dict, base_partition: dict, new_partition: dict
) -> dict:
    nodes = sorted(base_partition)
    if set(nodes) != set(new_partition):
        raise ValueError("Sensitivity must retain the baseline vertex universe")
    result: dict[str, Any] = {
        "partition_adjusted_rand_index": float(
            adjusted_rand_score(
                [base_partition[n] for n in nodes], [new_partition[n] for n in nodes]
            )
        )
    }
    b, c = baseline["measurements"], changed["measurements"]
    for kind in b:
        old, new = b[kind], c[kind]
        if old["status"] != "measured" or new["status"] != "measured":
            result[kind] = {
                "comparable": False,
                "baseline_status": old["status"],
                "variant_status": new["status"],
            }
            continue
        if kind == "multi_hop_connector":
            metrics = {
                "distance_change": new["total_inverse_weight_distance"]
                - old["total_inverse_weight_distance"],
                "distance_equal": math.isclose(
                    new["total_inverse_weight_distance"],
                    old["total_inverse_weight_distance"],
                    rel_tol=1e-8,
                    abs_tol=1e-6,
                ),
                "cross_community_equal": new["cross_community"] == old["cross_community"],
                "original_path_still_optimal": new["original_path_still_optimal"],
            }
        elif kind == "bridge_counterfactual":
            metrics = {
                "redundancy_equal": new["redundant_connector"] == old["redundant_connector"],
                "alternate_hops_equal": new["alternate_hops_after_deletion"]
                == old["alternate_hops_after_deletion"],
                "cross_community_equal": new["cross_community"] == old["cross_community"],
            }
        elif kind == "hub_removal_resilience":
            metrics = {
                "frozen_hub_still_argmax": new["frozen_hub_still_argmax"],
                "replacement_hub_equal": new["replacement_hub"] == old["replacement_hub"],
                "largest_component_fraction_change": new[
                    "largest_component_fraction_of_original_nodes"
                ]
                - old["largest_component_fraction_of_original_nodes"],
                "components_after_change": new["components_after"] - old["components_after"],
            }
        elif kind == "community_role_contrast":
            metrics = {}
            for role in ("dominant", "outward"):
                left, right = set(old[role]["members"]), set(new[role]["members"])
                metrics[f"{role}_membership_jaccard"] = len(left & right) / len(left | right)
                metrics[f"{role}_representative_equal"] = (
                    old[role]["representative"] == new[role]["representative"]
                )
            metrics["outward_external_share_change"] = (
                new["outward"]["external_share"] - old["outward"]["external_share"]
            )
        else:
            metrics = {
                f"{key}_equal": new[key] == old[key]
                for key in ("emerging_label", "established_label", "same_community")
            }
            metrics["growth_ratio_change"] = (
                new["emerging_growth_ratio"] - old["emerging_growth_ratio"]
            )
        result[kind] = {"comparable": True, **metrics}
    return result
