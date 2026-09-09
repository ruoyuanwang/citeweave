"""Detect sentinel identities independently of graph-task construction."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import duckdb

from .io import read_json, sha256_file

PLACEHOLDER_TOKENS = {"", "none", "null", "nan"}
IDENTITY_COLUMNS = {
    "works": "work_id",
    "authors": "author_id",
    "institutions": "institution_id",
    "sources": "source_id",
}


def is_placeholder_identity(value: Any) -> bool:
    if value is None:
        return True
    return (
        str(value).strip().rstrip("/").rsplit(":", 1)[-1].rsplit("/", 1)[-1].casefold()
        in PLACEHOLDER_TOKENS
    )


def _contains_identity(value: Any, identities: set[str]) -> bool:
    if isinstance(value, str):
        return value in identities
    if isinstance(value, dict):
        return any(_contains_identity(v, identities) for v in value.values())
    if isinstance(value, list):
        return any(_contains_identity(v, identities) for v in value)
    return False


def audit_canonical_identities(workspace: Path, benchmark_roots: list[Path]) -> dict[str, Any]:
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute("SET memory_limit='512MB'")
    findings = []
    hashes = {}
    try:
        for table, column in IDENTITY_COLUMNS.items():
            path = workspace / "canonical" / f"{table}.parquet"
            hashes[str(path.resolve())] = sha256_file(path)
            identities = c.execute(f"SELECT {column} FROM read_parquet(?)", [str(path)]).fetchall()
            bad = sorted({str(row[0]) for row in identities if is_placeholder_identity(row[0])})
            finding: dict[str, Any] = {"table": table, "column": column, "placeholder_ids": bad}
            if bad and table in {"authors", "institutions"}:
                membership = workspace / "canonical" / "authorships.parquet"
                hashes[str(membership.resolve())] = sha256_file(membership)
                finding["membership_counts"] = [
                    {"identity": r[0], "rows": r[1], "distinct_works": r[2]}
                    for r in c.execute(
                        f"SELECT {column}, count(*), count(DISTINCT work_id) FROM read_parquet(?) "
                        f"WHERE {column} IN (SELECT unnest(?)) GROUP BY {column}",
                        [str(membership), bad],
                    ).fetchall()
                ]
            findings.append(finding)
        placeholders = {i for f in findings for i in f["placeholder_ids"]}
        bad_authors = next(set(f["placeholder_ids"]) for f in findings if f["table"] == "authors")
        graph_path = workspace / "canonical" / "visualization" / "coauthor_edges.parquet"
        hashes[str(graph_path.resolve())] = sha256_file(graph_path)
        incident_edges, incident_mass = c.execute(
            "SELECT count(*), coalesce(sum(weight),0) FROM read_parquet(?) "
            "WHERE source_id IN (SELECT unnest(?)) OR target_id IN (SELECT unnest(?))",
            [str(graph_path), sorted(bad_authors), sorted(bad_authors)],
        ).fetchone()
        task_records = []
        for root in benchmark_roots:
            path = root / workspace.name / "benchmark.json"
            if not path.is_file():
                continue
            hashes[str(path.resolve())] = sha256_file(path)
            for task in read_json(path)["tasks"]:
                core = {k: v for k, v in task.items() if k not in {"contexts", "context_hashes"}}
                task_records.append(
                    {
                        "benchmark": str(path.resolve()),
                        "item_id": task["item_id"],
                        "network": task["network"],
                        "placeholder_in_core": _contains_identity(core, placeholders),
                        "contexts_containing_placeholder": [
                            name
                            for name, context in task.get("contexts", {}).items()
                            if _contains_identity(context, placeholders)
                        ],
                    }
                )
        # Raw names prove the collision without trusting canonical first-name deduplication.
        raw_names: set[str] = set()
        raw_missing = 0
        raw_files = []
        for path in sorted((workspace / "raw").rglob("openalex-page-*.json.gz")):
            raw_files.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict) or "results" not in payload:
                raise ValueError(f"Unexpected raw page schema: {path}")
            for work in payload["results"]:
                for authorship in work.get("authorships") or []:
                    author = authorship.get("author") or {}
                    if is_placeholder_identity(author.get("id")) and not author.get("orcid"):
                        raw_missing += 1
                        if author.get("display_name"):
                            raw_names.add(str(author["display_name"]))
    finally:
        c.close()
    return {
        "dataset_id": workspace.name,
        "status": "identity_collision_detected" if placeholders else "no_sentinel_identity",
        "table_findings": findings,
        "coauthor_edges_incident_to_placeholder": incident_edges,
        "coauthor_edge_weight_incident_to_placeholder": incident_mass,
        "raw_missing_author_occurrences": raw_missing,
        "raw_distinct_names_merged": len(raw_names),
        "raw_names_examples": sorted(raw_names)[:5],
        "tasks": task_records,
        "source_sha256": hashes,
        "raw_pages": raw_files,
        "limits": [
            "Detects missing/sentinel IDs, not all author disambiguation errors.",
            "Core mention counts understate impact: shared community/rank computations can affect every task on a contaminated network.",
            "Raw occurrence counts are before canonical deduplication and may differ from membership counts.",
        ],
    }
