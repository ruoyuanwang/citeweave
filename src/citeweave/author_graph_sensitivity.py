"""Independent sparse arithmetic for author repair/scope sensitivity.

This is neither a replacement gold generator nor an independent validation of
community detection or betweenness-based task selection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .io import read_json, sha256_file


class SparseAuthorGraph:
    def __init__(self, edges: pd.DataFrame):
        pairs = edges[["source_id", "target_id"]].astype(str)
        self.nodes = sorted(set(pairs.source_id) | set(pairs.target_id))
        self.index = {node: i for i, node in enumerate(self.nodes)}
        a = pairs.source_id.map(self.index).to_numpy()
        b = pairs.target_id.map(self.index).to_numpy()
        weights = edges.weight.to_numpy(dtype=float)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("Positive finite edge weights required")
        self.weights = csr_matrix(
            (np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
            shape=(len(self.nodes), len(self.nodes)),
        )
        if self.weights.nnz != 2 * len(edges):
            raise ValueError("Duplicate/self-loop edge rows not accepted")

    def summary(self) -> dict[str, Any]:
        count, labels = connected_components(self.weights, directed=False)
        return {
            "nodes": len(self.nodes),
            "edges": self.weights.nnz // 2,
            "components": count,
            "largest_component": int(np.bincount(labels).max()) if len(labels) else 0,
        }

    def path_distance(self, source: str, target: str) -> float | None:
        if source not in self.index or target not in self.index:
            return None
        distances = self.weights.copy()
        distances.data = 1 / distances.data
        distance = float(
            dijkstra(distances, directed=False, indices=self.index[source])[self.index[target]]
        )
        return distance if math.isfinite(distance) else None

    def edge_removal(self, source: str, target: str) -> dict[str, Any] | None:
        if source not in self.index or target not in self.index:
            return None
        a, b = self.index[source], self.index[target]
        if self.weights[a, b] == 0:
            return None
        before = connected_components(self.weights, directed=False, return_labels=False)
        changed = self.weights.copy()
        changed[a, b] = changed[b, a] = 0
        changed.eliminate_zeros()
        after = connected_components(changed, directed=False, return_labels=False)
        hops = float(dijkstra(changed, directed=False, indices=a, unweighted=True)[b])
        return {
            "component_increase": int(after - before),
            "alternate_hops_after_deletion": int(hops) if math.isfinite(hops) else None,
        }

    def node_removal(self, node: str) -> dict[str, Any] | None:
        if node not in self.index:
            return None
        keep = np.arange(len(self.nodes)) != self.index[node]
        changed = self.weights[keep][:, keep]
        count, labels = connected_components(changed, directed=False)
        largest = int(np.bincount(labels).max()) if len(labels) else 0
        return {
            "components_after": int(count),
            "largest_component_fraction_of_original_nodes": largest / max(1, len(self.nodes)),
        }


def task_arithmetic(graph: SparseAuthorGraph, task: dict[str, Any]) -> dict[str, Any] | None:
    trace = task["operator_trace"]
    if task["task_type"] == "multi_hop_connector":
        path = next(step["path"] for step in trace if step["operator"] == "shortest_path")
        if path[0] not in graph.index or path[-1] not in graph.index:
            return None
        return {"total_inverse_weight_distance": graph.path_distance(path[0], path[-1])}
    if task["task_type"] == "bridge_counterfactual":
        edge = next(
            step["selected_edge"] for step in trace if step["operator"] == "edge_betweenness"
        )
        return graph.edge_removal(*edge)
    if task["task_type"] == "hub_removal_resilience":
        node = next(step["node"] for step in trace if step["operator"] == "delete_node")
        return graph.node_removal(node)
    raise ValueError("Unsupported arithmetic")


def _same_fields(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is right
    return all(
        (a == b if a is None or b is None else math.isclose(float(a), float(b), abs_tol=1e-6))
        for k, a in left.items()
        for b in [right.get(k)]
    )


def audit_author_graph_sensitivity(
    workspace: Path, correction: Path, benchmark: Path | None
) -> dict[str, Any]:
    tasks = (
        []
        if benchmark is None
        else [
            t
            for t in read_json(benchmark)["tasks"]
            if t["network"] == "coauthorship"
            and t["scale"] == "large"
            and t["task_type"]
            in {"multi_hop_connector", "bridge_counterfactual", "hub_removal_resilience"}
        ]
    )
    paths = {
        "original_bounded": workspace / "canonical" / "visualization" / "coauthor_edges.parquet",
        "corrected_bounded": correction / "canonical" / "visualization" / "coauthor_edges.parquet",
        "corrected_uncapped": correction / "uncapped" / "coauthor_edges.parquet",
    }
    results = {}
    for name, path in paths.items():
        graph = SparseAuthorGraph(pd.read_parquet(path))
        results[name] = {
            "source_sha256": sha256_file(path),
            "graph": graph.summary(),
            "tasks": {t["item_id"]: task_arithmetic(graph, t) for t in tasks},
        }
        del graph
    comparisons = []
    for task in tasks:
        key = task["item_id"]
        original = results["original_bounded"]["tasks"][key]
        bounded = results["corrected_bounded"]["tasks"][key]
        uncapped = results["corrected_uncapped"]["tasks"][key]
        comparisons.append(
            {
                "item_id": key,
                "original_arithmetic_matches_registered_fields": _same_fields(
                    original, task["answer"]
                ),
                "anchor_survives_correction": bounded is not None,
                "registered_arithmetic_changes_after_repair": not _same_fields(original, bounded),
                "registered_anchor_arithmetic_changes_after_uncapping": (
                    not _same_fields(bounded, uncapped)
                    if bounded is not None and uncapped is not None
                    else None
                ),
            }
        )
    return {
        "dataset_id": workspace.name,
        "graphs": results,
        "task_comparisons": comparisons,
        "benchmark_sha256": sha256_file(benchmark) if benchmark else None,
        "limits": [
            "Fixed old anchors only; no new task selection, no LLM outcomes.",
            "Checks path distance and deletion arithmetic, not full answers, community assignments or highest-betweenness selection.",
            "Uncapped includes unresolved per-work author occurrences, not an assertion that all nodes identify distinct real people.",
        ],
    }
