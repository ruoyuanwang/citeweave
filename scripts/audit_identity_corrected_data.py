"""Verify scoped author corrections and unchanged registered graph layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from citeweave.author_benchmark_rebuild import UNCHANGED_GRAPH_FILES
from citeweave.io import read_json, sha256_file, write_json


def compare_memberships(old: Path, new: Path) -> dict:
    c = duckdb.connect()
    c.execute("SET threads=1")
    try:
        old_known = "SELECT * FROM read_parquet(?) WHERE author_id IS NOT NULL AND NOT regexp_matches(lower(trim(author_id)), '(^|:)(none|null|nan|)$')"
        new_known = (
            "SELECT * FROM read_parquet(?) WHERE author_id NOT LIKE 'openalex-author-occurrence:%'"
        )
        differences = c.execute(
            f"SELECT count(*) FROM (({old_known} EXCEPT ALL {new_known}) UNION ALL ({new_known} EXCEPT ALL {old_known}))",
            [str(old), str(new), str(new), str(old)],
        ).fetchone()[0]
        unresolved, maximum_works = c.execute(
            """
            SELECT count(*),coalesce(max(works),0) FROM (
                SELECT author_id,count(DISTINCT work_id) AS works FROM read_parquet(?)
                WHERE author_id LIKE 'openalex-author-occurrence:%' GROUP BY author_id
            )
        """,
            [str(new)],
        ).fetchone()
        placeholders = c.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE author_id IS NULL OR regexp_matches(lower(trim(author_id)), '(^|:)(none|null|nan|)$')",
            [str(new)],
        ).fetchone()[0]
        other_fields = "SELECT * EXCLUDE(author_id) FROM read_parquet(?)"
        other_differences = c.execute(
            f"SELECT count(*) FROM (({other_fields} EXCEPT ALL {other_fields}) UNION ALL ({other_fields} EXCEPT ALL {other_fields}))",
            [str(old), str(new), str(new), str(old)],
        ).fetchone()[0]
    finally:
        c.close()
    return {
        "known_author_membership_multiset_differences": differences,
        "unresolved_author_occurrences": unresolved,
        "max_canonical_works_per_unresolved_id": maximum_works,
        "remaining_placeholder_membership_rows": placeholders,
        "non_author_membership_field_multiset_differences": other_differences,
        "passed": differences == 0
        and placeholders == 0
        and maximum_works <= 1
        and other_differences == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for group, old_root in {
        "primary": Path("experiments/formal_v3_workspaces"),
        "replication": Path("experiments/formal_v3_replication_workspaces"),
    }.items():
        for old in sorted(old_root.iterdir()):
            if not (old / "project.yml").is_file():
                continue
            new = args.corrected_root / group / old.name
            path = new / "identity_correction_verification.json"
            if not path.is_file():
                records.append(
                    {"dataset_id": old.name, "passed": False, "reason": "reprocessing_incomplete"}
                )
                continue
            verification = read_json(path)
            issues = []
            if (
                verification["status"] != "reprocessed_not_promoted"
                or not verification["processing_acceptance"]["passed"]
            ):
                issues.append("processing_verification_failed")
            for name, row in verification["unaffected_tables"].items():
                if (
                    row["row_multiset_differences"]
                    or sha256_file(old / "canonical" / name) != row["old_sha256"]
                    or sha256_file(new / "canonical" / name) != row["new_sha256"]
                ):
                    issues.append(f"non_author_table_drift:{name}")
            graphs = {}
            for name in UNCHANGED_GRAPH_FILES:
                original_path = old / "canonical" / "visualization" / name
                new_path = new / "canonical" / "visualization" / name
                same = pd.read_parquet(original_path).equals(pd.read_parquet(new_path))
                graphs[name] = {
                    "ordered_rows_equal": same,
                    "old_sha256": sha256_file(original_path),
                    "new_sha256": sha256_file(new_path),
                }
                if not same:
                    issues.append(f"unchanged_graph_drift:{name}")
            membership = compare_memberships(
                old / "canonical" / "authorships.parquet", new / "canonical" / "authorships.parquet"
            )
            if not membership["passed"]:
                issues.append("author_membership_correction_invariant_failed")
            record = {
                "dataset_id": old.name,
                "group": group,
                "passed": not issues,
                "issues": issues,
                "membership": membership,
                "unchanged_graphs": graphs,
                "reprocessing_verification_sha256": sha256_file(path),
                "old_membership_sha256": sha256_file(old / "canonical" / "authorships.parquet"),
                "new_membership_sha256": sha256_file(new / "canonical" / "authorships.parquet"),
            }
            records.append(record)
            print(
                json.dumps(
                    {"topic": old.name, "passed": record["passed"], "membership": membership}
                ),
                flush=True,
            )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "scoped_data_invariants_passed"
            if len(records) == 8 and all(r["passed"] for r in records)
            else "incomplete_or_failed",
            "datasets": records,
            "limitations": [
                "Does not prove complete real-person disambiguation or scientific validity of all source metadata.",
                "Formal model and real human outcomes remain unmeasured.",
            ],
        },
    )


if __name__ == "__main__":
    main()
