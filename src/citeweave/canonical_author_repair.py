"""Isolated author-identity correction, without mutating frozen corpora.

Missing authors remain explicitly unresolved work-position occurrences. No names
or affiliations are used as evidence that two records denote the same person.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .canonical_identity_audit import is_placeholder_identity
from .io import read_json, sha256_file, write_json, write_parquet
from .transform import stable_id


def _sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def repair_author_layer(
    workspace: Path, output: Path, *, candidate_pool: int = 2000
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Refusing to overwrite author correction output: {output}")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")
    canonical = workspace / "canonical"
    inputs = {name: canonical / f"{name}.parquet" for name in ("authors", "authorships", "works")}
    input_hashes = {str(p.resolve()): sha256_file(p) for p in inputs.values()}
    authors = pd.read_parquet(inputs["authors"])
    memberships = pd.read_parquet(inputs["authorships"])
    original_memberships = memberships.copy()
    missing = memberships.author_id.map(is_placeholder_identity)
    if memberships.loc[missing, ["work_id", "position"]].isna().any().any():
        raise ValueError("Missing work/position cannot be safely assigned an occurrence identity")
    replacements = []
    for (work_id, position), group in memberships.loc[missing].groupby(["work_id", "position"]):
        author_id = stable_id("openalex-author-occurrence", work_id, int(position))
        memberships.loc[group.index, "author_id"] = author_id
        replacements.append(
            {
                "author_id": author_id,
                "name": f"Unresolved author occurrence {author_id.rsplit(':', 1)[1]}",
                "given_name": None,
                "family_name": None,
                "orcid": None,
            }
        )
    authors = authors.loc[~authors.author_id.map(is_placeholder_identity)].copy()
    authors = pd.concat(
        [authors, pd.DataFrame(replacements, columns=authors.columns)], ignore_index=True
    )
    if authors.author_id.duplicated().any():
        raise ValueError("Correction would create duplicate author identities")
    if not memberships.loc[~missing].equals(original_memberships.loc[~missing]):
        raise ValueError("Known author memberships changed unexpectedly")
    if not memberships.drop(columns="author_id").equals(
        original_memberships.drop(columns="author_id")
    ):
        raise ValueError("Non-author membership fields changed unexpectedly")
    authors = authors.sort_values("author_id").reset_index(drop=True)
    output.mkdir(parents=True)
    write_parquet(output / "canonical" / "authors.parquet", authors)
    write_parquet(output / "canonical" / "authorships.parquet", memberships)
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute("SET memory_limit='512MB'")
    try:
        c.register("authors", authors)
        c.register("authorships", memberships)
        c.execute(
            "CREATE TEMP TABLE membership AS SELECT DISTINCT work_id, author_id FROM authorships"
        )
        c.execute(f"CREATE VIEW works AS SELECT * FROM read_parquet({_sql_path(inputs['works'])})")
        productivity = c.execute("""
            SELECT rel.author_id, authors.name AS author_name, authors.orcid,
                   count(DISTINCT rel.work_id) AS documents,
                   sum(coalesce(works.cited_by_count,0)) AS citations
            FROM membership rel JOIN works USING(work_id) LEFT JOIN authors USING(author_id)
            GROUP BY rel.author_id, authors.name, authors.orcid
            ORDER BY documents DESC, citations DESC, rel.author_id
        """).fetchdf()
        # Construct the actual uncapped pair graph once; no dense adjacency matrix.
        c.execute("""
            CREATE TEMP TABLE all_edges AS
            SELECT l.author_id AS source_id, r.author_id AS target_id, count(*) AS weight
            FROM membership l JOIN membership r ON l.work_id=r.work_id AND l.author_id<r.author_id
            GROUP BY source_id,target_id
        """)
        c.execute(f"""
            CREATE TEMP TABLE candidates AS
            SELECT author_id FROM membership GROUP BY author_id
            ORDER BY count(DISTINCT work_id) DESC, author_id LIMIT {candidate_pool}
        """)
        process = read_json(workspace / "audit" / "processing_manifest.json")
        edge_limit = int(process["edge_row_limit"])
        if edge_limit < 1:
            raise ValueError("Invalid frozen edge row limit")
        c.execute(f"""
            CREATE TEMP TABLE bounded_edges AS
            SELECT * FROM all_edges WHERE source_id IN (SELECT author_id FROM candidates)
            AND target_id IN (SELECT author_id FROM candidates)
            ORDER BY weight DESC,source_id,target_id LIMIT {edge_limit}
        """)
        out_graph = output / "canonical" / "visualization"
        write_parquet(out_graph / "author_productivity.parquet", productivity)
        for table, target in (
            ("bounded_edges", out_graph / "coauthor_edges.parquet"),
            ("all_edges", output / "uncapped" / "coauthor_edges.parquet"),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            c.execute(
                f"COPY (SELECT * FROM {table} ORDER BY source_id,target_id) TO {_sql_path(target)} (FORMAT PARQUET)"
            )
        stats = {}
        for table in ("bounded_edges", "all_edges"):
            nodes = c.execute(
                f"SELECT count(*) FROM (SELECT source_id AS id FROM {table} UNION SELECT target_id FROM {table})"
            ).fetchone()[0]
            edges, mass = c.execute(
                f"SELECT count(*),coalesce(sum(weight),0) FROM {table}"
            ).fetchone()
            stats[table] = {"nonisolated_nodes": nodes, "edges": edges, "pair_incidence_mass": mass}
        expected_mass = c.execute(
            "SELECT coalesce(sum(n*(n-1)/2),0) FROM (SELECT work_id,count(*) AS n FROM membership GROUP BY work_id)"
        ).fetchone()[0]
        if stats["all_edges"]["pair_incidence_mass"] != expected_mass:
            raise ValueError("Uncapped pair mass is inconsistent with canonical memberships")
    finally:
        c.close()
    if any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError("Frozen source changed during correction")
    receipt = {
        "schema_version": 1,
        "status": "isolated_correction_not_promoted",
        "dataset_id": workspace.name,
        "source_workspace": str(workspace.resolve()),
        "source_sha256": input_hashes,
        "changed_membership_rows": int(missing.sum()),
        "unresolved_occurrence_nodes": len(replacements),
        "candidate_pool_size": candidate_pool,
        "edge_row_limit": edge_limit,
        "graph_statistics": stats,
        "uncapped_pair_mass_verified": True,
        "output_sha256": {
            str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*.parquet"))
        },
        "limits": [
            "Unknown identities are work-position occurrences, not disambiguated people.",
            "Their labels intentionally do not reuse the old placeholder's arbitrarily retained person name.",
            "Known identities and every non-author membership field are unchanged.",
            "Original frozen data, task gold, indexes and results were not modified.",
            "Promotion requires prospective amendment, benchmark regeneration and new readiness audits.",
        ],
    }
    write_json(output / "repair_receipt.json", receipt)
    return receipt
