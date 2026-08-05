from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .bulk_processing import PART_TABLE_COLUMNS
from .io import read_json, sha256_file
from .models import ProjectPaths


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _sql_string(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def verify_large_processing(root: Path) -> dict[str, Any]:
    """Independently verify disk-normalized tables and visualization contracts."""
    paths = ProjectPaths(root)
    manifest_path = paths.audit / "processing_manifest.json"
    quality_path = paths.quality / "processing_report.json"
    schema_path = paths.canonical / "schema.json"
    checks: list[dict[str, Any]] = []
    if not manifest_path.exists():
        checks.append(_check("processing_manifest", False, "<missing>"))
        return _result(paths, checks)

    manifest = read_json(manifest_path)
    checks.append(
        _check(
            "processing_complete",
            manifest.get("status") == "complete"
            and int(manifest.get("records_processed", 0)) > 0
            and int(manifest.get("batches_completed", 0)) > 0,
            {
                "status": manifest.get("status"),
                "records": manifest.get("records_processed"),
                "batches": manifest.get("batches_completed"),
            },
        )
    )

    harvest_path = paths.audit / "harvest_manifest.json"
    harvest_records = None
    if harvest_path.exists():
        harvest_records = int(read_json(harvest_path).get("unique_records", 0))
    checks.append(
        _check(
            "input_count_contract",
            harvest_records is None
            or int(manifest.get("records_processed", -1)) == harvest_records,
            {
                "processed": manifest.get("records_processed"),
                "harvest_unique": harvest_records,
            },
        )
    )

    expected_hashes = (manifest.get("outputs") or {}).get("canonical_sha256") or {}
    missing: list[str] = []
    hash_errors: list[str] = []
    for table in PART_TABLE_COLUMNS:
        path = paths.canonical / f"{table}.parquet"
        if not path.exists():
            missing.append(table)
        elif expected_hashes.get(table) != sha256_file(path):
            hash_errors.append(table)
    checks.append(
        _check(
            "canonical_file_integrity",
            not missing and not hash_errors and len(expected_hashes) == len(PART_TABLE_COLUMNS),
            {"missing": missing, "hash_errors": hash_errors},
        )
    )

    independent: dict[str, Any] = {}
    relation_errors: dict[str, int] = {}
    unique_errors: dict[str, dict[str, int]] = {}
    if not missing:
        connection = duckdb.connect(":memory:")
        try:
            for table in PART_TABLE_COLUMNS:
                connection.execute(
                    f"""
                    CREATE VIEW "{table}" AS
                    SELECT * FROM read_parquet(
                        {_sql_string(paths.canonical / f"{table}.parquet")}
                    )
                    """
                )
            independent["table_rows"] = {
                table: int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
                for table in PART_TABLE_COLUMNS
            }
            for table, identifier in {
                "works": "work_id",
                "authors": "author_id",
                "institutions": "institution_id",
                "sources": "source_id",
            }.items():
                rows, distinct = connection.execute(
                    f'SELECT count(*), count(DISTINCT {identifier}) FROM "{table}"'
                ).fetchone()
                if rows != distinct:
                    unique_errors[table] = {"rows": int(rows), "distinct": int(distinct)}
            relation_queries = {
                "authorship_work": """
                    SELECT count(*) FROM authorships rel
                    LEFT JOIN works parent USING (work_id)
                    WHERE parent.work_id IS NULL
                """,
                "authorship_author": """
                    SELECT count(*) FROM authorships rel
                    LEFT JOIN authors parent USING (author_id)
                    WHERE parent.author_id IS NULL
                """,
                "keyword_work": """
                    SELECT count(*) FROM keywords rel
                    LEFT JOIN works parent USING (work_id)
                    WHERE parent.work_id IS NULL
                """,
                "reference_citing_work": """
                    SELECT count(*) FROM "references" rel
                    LEFT JOIN works parent ON rel.citing_work_id = parent.work_id
                    WHERE parent.work_id IS NULL
                """,
                "provenance_work": """
                    SELECT count(*) FROM provenance rel
                    LEFT JOIN works parent USING (work_id)
                    WHERE parent.work_id IS NULL
                """,
            }
            relation_errors = {
                name: int(connection.execute(query).fetchone()[0])
                for name, query in relation_queries.items()
            }
        finally:
            connection.close()
    checks.append(
        _check(
            "primary_and_foreign_key_integrity",
            not missing
            and not unique_errors
            and relation_errors
            and all(value == 0 for value in relation_errors.values()),
            {"uniqueness": unique_errors, "foreign_key_orphans": relation_errors},
        )
    )

    if quality_path.exists():
        quality = read_json(quality_path)
        scope = quality.get("scope_filter") or {}
        exclusion_path = paths.quality / "excluded_records.parquet"
        exclusion_hash = (manifest.get("outputs") or {}).get("exclusion_sha256")
        exclusion_ok = (
            exclusion_path.exists()
            and exclusion_hash == sha256_file(exclusion_path)
            and int(scope.get("input_records_accounted", -1))
            == int(manifest.get("records_processed", -2))
            and int(scope.get("out_of_scope_in_canonical", -1)) == 0
        )
        quality_ok = (
            quality.get("passed") is True
            and all(
                int(value) == 0 for value in (quality.get("foreign_key_orphans") or {}).values()
            )
            and int(quality.get("canonical_records", 0)) > 0
            and exclusion_ok
        )
    else:
        quality = "<missing>"
        quality_ok = False
    checks.append(_check("processing_quality_gate", quality_ok, quality))

    visualization_errors: list[str] = []
    visualization_rows: dict[str, int] = {}
    contract_matches = False
    if schema_path.exists():
        schema = read_json(schema_path)
        contract_matches = schema.get("processing_contract") == manifest.get("run_contract")
        visualization = schema.get("visualization_tables") or {}
        for name, item in visualization.items():
            path = paths.root / item["path"]
            if not path.exists() or sha256_file(path) != item.get("sha256"):
                visualization_errors.append(name)
            else:
                connection = duckdb.connect(":memory:")
                try:
                    count = int(
                        connection.execute(
                            f"SELECT count(*) FROM read_parquet({_sql_string(path)})"
                        ).fetchone()[0]
                    )
                finally:
                    connection.close()
                visualization_rows[name] = count
                if count != int(item.get("rows", -1)):
                    visualization_errors.append(name)
        required = {
            "annual_output",
            "source_productivity",
            "author_productivity",
            "keyword_occurrences",
            "reference_impact",
            "coauthor_edges",
            "keyword_cooccurrence_edges",
            "cocitation_edges",
        }
        missing_visualizations = sorted(required - set(visualization))
    else:
        schema = "<missing>"
        missing_visualizations = ["schema.json"]
    checks.append(
        _check(
            "visualization_ready_contract",
            not visualization_errors
            and not missing_visualizations
            and bool(visualization_rows)
            and contract_matches,
            # The schema must describe the exact protocol and parameters
            # recorded by the processing checkpoint.
            {
                "missing": missing_visualizations,
                "integrity_errors": visualization_errors,
                "processing_contract_matches": contract_matches,
                "rows": visualization_rows,
            },
        )
    )

    return _result(paths, checks, independent)


def _result(
    paths: ProjectPaths,
    checks: list[dict[str, Any]],
    independent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    passed = bool(checks) and all(item["passed"] for item in checks)
    return {
        "project_root": str(paths.root),
        "passed": passed,
        "passed_checks": sum(bool(item["passed"]) for item in checks),
        "total_checks": len(checks),
        "checks": checks,
        "independent_profile": independent or {},
    }
