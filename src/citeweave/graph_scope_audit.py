"""Read-only scope accounting against canonical memberships, not graph-task gold.

Full pair-incidence mass is computed as sum choose(k_work, 2), without constructing
the uncapped graph. It is not the number of unique full-graph edges.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

from .io import read_json, sha256_file

SPECS = {
    "keyword_cooccurrence": (
        "keywords.parquet",
        "keyword",
        "keyword_occurrences.parquet",
        "keyword_cooccurrence_edges.parquet",
        "occurrences",
    ),
    "institution_collaboration": (
        "authorships.parquet",
        "institution_id",
        "institution_productivity.parquet",
        "institution_collaboration_edges.parquet",
        "documents",
    ),
    "coauthorship": (
        "authorships.parquet",
        "author_id",
        "author_productivity.parquet",
        "coauthor_edges.parquet",
        "documents",
    ),
}
LIMITS = {"small": 100, "medium": 300, "large": None}


def _parquet(path: Path) -> str:
    return "read_parquet('" + str(path.resolve()).replace("'", "''") + "')"


def audit_workspace_graph_scope(
    workspace: Path,
    requests: set[tuple[str, str]],
) -> dict[str, Any]:
    canonical = workspace / "canonical"
    process = read_json(workspace / "audit" / "processing_manifest.json")
    connection = duckdb.connect(":memory:")
    connection.execute("SET threads=1")
    connection.execute("SET memory_limit='512MB'")
    records = []
    source_hashes = {}
    try:
        for network in sorted({network for network, _ in requests}):
            membership_name, entity_column, nodes_name, edges_name, importance_column = SPECS[
                network
            ]
            membership_path = canonical / membership_name
            nodes_path = canonical / "visualization" / nodes_name
            edges_path = canonical / "visualization" / edges_name
            for path in (membership_path, nodes_path, edges_path):
                source_hashes[str(path.resolve())] = sha256_file(path)
            connection.execute(
                "CREATE OR REPLACE TEMP TABLE memberships AS "
                f"SELECT DISTINCT cast(work_id AS VARCHAR) AS work_id, "
                f"cast({entity_column} AS VARCHAR) AS entity_id FROM {_parquet(membership_path)} "
                f"WHERE work_id IS NOT NULL AND {entity_column} IS NOT NULL"
            )
            raw_entities, raw_works, raw_memberships = connection.execute(
                "SELECT count(DISTINCT entity_id), count(DISTINCT work_id), count(*) FROM memberships"
            ).fetchone()
            full_pair_mass = connection.execute(
                "SELECT coalesce(sum(n * (n - 1) / 2), 0) FROM "
                "(SELECT work_id, count(*) AS n FROM memberships GROUP BY work_id)"
            ).fetchone()[0]
            raw_nodes = connection.execute(
                f"SELECT cast({entity_column} AS VARCHAR), {importance_column} FROM {_parquet(nodes_path)}"
            ).fetchall()
            node_importance = {}
            for node_id, importance in raw_nodes:
                node_importance.setdefault(node_id, float(importance or 0))
            raw_edges = connection.execute(
                f"SELECT cast(source_id AS VARCHAR), cast(target_id AS VARCHAR), weight "
                f"FROM {_parquet(edges_path)} ORDER BY source_id, target_id, weight DESC"
            ).fetchall()
            requested_scales = sorted(scale for net, scale in requests if net == network)
            for scale in requested_scales:
                limit = LIMITS[scale]
                ordered = sorted(node_importance, key=lambda node: (-node_importance[node], node))
                selected = set(ordered[:limit] if limit is not None else ordered)
                edges = [
                    (a, b, float(w or 1))
                    for a, b, w in raw_edges
                    if a in selected and b in selected
                ]
                retained_nodes = {node for a, b, _ in edges for node in (a, b)}
                pair_weights = {}
                for a, b, weight in edges:
                    pair_weights[tuple(sorted((a, b)))] = weight
                retained_mass = sum(pair_weights.values())
                retained_memberships, covered_works = connection.execute(
                    "WITH selected AS (SELECT unnest(?) AS entity_id) "
                    "SELECT count(*), count(DISTINCT work_id) FROM memberships JOIN selected USING(entity_id)",
                    [sorted(retained_nodes)],
                ).fetchone()
                issues = []
                if len(pair_weights) != len(edges):
                    issues.append("duplicate_undirected_edge_rows")
                if any(a == b for a, b in pair_weights):
                    issues.append("self_loop_pair_mass_incompatible")
                if retained_mass > float(full_pair_mass) + 1e-6:
                    issues.append("retained_pair_mass_exceeds_canonical_memberships")
                records.append(
                    {
                        "dataset_id": workspace.name,
                        "network": network,
                        "scale": scale,
                        "processing_candidate_pool": process.get("candidate_pool_size"),
                        "processing_edge_row_limit": process.get("edge_row_limit"),
                        "raw_distinct_entities": raw_entities,
                        "raw_eligible_works": raw_works,
                        "raw_unique_work_entity_memberships": raw_memberships,
                        "raw_pair_incidence_mass": float(full_pair_mass),
                        "projected_nodes_before_isolate_removal": len(selected),
                        "retained_graph_nodes": len(retained_nodes),
                        "retained_graph_edges": len(pair_weights),
                        "retained_edge_weight_mass": retained_mass,
                        "retained_entity_fraction": len(retained_nodes) / raw_entities
                        if raw_entities
                        else None,
                        "retained_membership_fraction": retained_memberships / raw_memberships
                        if raw_memberships
                        else None,
                        "work_coverage_any_retained_entity": covered_works / raw_works
                        if raw_works
                        else None,
                        "retained_pair_incidence_fraction": retained_mass / full_pair_mass
                        if full_pair_mass
                        else None,
                        "is_uncapped_corpus_graph": False
                        if process.get("candidate_pool_size")
                        else None,
                        "accounting_issues": issues,
                    }
                )
    finally:
        connection.close()
    return {"dataset_id": workspace.name, "records": records, "source_sha256": source_hashes}


def audit_registered_graph_scope(
    benchmark_roots: list[Path],
    workspace_roots: list[Path],
) -> dict[str, Any]:
    requests: dict[str, set[tuple[str, str]]] = defaultdict(set)
    benchmark_hashes = {}
    for root in benchmark_roots:
        for path in sorted(root.glob("*/benchmark.json")):
            benchmark = read_json(path)
            benchmark_hashes[str(path.resolve())] = sha256_file(path)
            requests[benchmark["dataset_id"]].update(
                (task["network"], task["scale"]) for task in benchmark["tasks"]
            )
    results = []
    for topic, graph_requests in sorted(requests.items()):
        candidates = [
            root / topic for root in workspace_roots if (root / topic / "canonical").is_dir()
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one canonical workspace for {topic}")
        results.append(audit_workspace_graph_scope(candidates[0], graph_requests))
    rows = [row for result in results for row in result["records"]]
    return {
        "schema_version": 1,
        "status": "scope_accounted"
        if rows and not any(row["accounting_issues"] for row in rows)
        else "requires_investigation",
        "topics": len(results),
        "registered_graphs": len(rows),
        "benchmark_sha256": benchmark_hashes,
        "datasets": results,
        "interpretation": [
            "Large means the largest registered candidate-capped graph, not all corpus entities.",
            "Pair-incidence fraction is retained summed edge weight divided by canonical sum choose(k_work,2), not unique-edge recall.",
            "Entity, membership and work coverage are scope descriptors, not GraphRAG answer accuracy.",
            "No benchmark, source, prompt, index or result was changed by this audit.",
        ],
    }
