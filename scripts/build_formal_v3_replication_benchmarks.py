from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import yaml

from citeweave.formal_protocol_amendment import effective_query_datasets
from citeweave.graph_discovery import build_discovery_benchmark
from citeweave.harvest_acceptance import verify_bulk_harvest
from citeweave.io import read_json, sha256_file, write_json, write_parquet
from citeweave.processing_acceptance import verify_large_processing

TASKS = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _materialize_keyword_trends(workspace: Path) -> dict[str, Any]:
    output = workspace / "analyses" / "keyword_trends.parquet"
    canonical = workspace / "canonical"
    visual = canonical / "visualization"
    connection = duckdb.connect()
    try:
        trends = connection.execute(
            f"""
            WITH top_keywords AS (
              SELECT keyword, occurrences AS global_documents
              FROM read_parquet('{_sql_path(visual / 'keyword_occurrences.parquet')}')
              WHERE keyword IS NOT NULL
              ORDER BY occurrences DESC, keyword
              LIMIT 15
            )
            SELECT works.year, keywords.keyword,
                   count(DISTINCT keywords.work_id) AS documents,
                   max(top_keywords.global_documents) AS global_documents
            FROM read_parquet('{_sql_path(canonical / 'keywords.parquet')}') keywords
            JOIN top_keywords USING (keyword)
            JOIN read_parquet('{_sql_path(canonical / 'works.parquet')}') works USING (work_id)
            WHERE works.year IS NOT NULL
            GROUP BY works.year, keywords.keyword
            ORDER BY global_documents DESC, keyword, year
            """
        ).df()
    finally:
        connection.close()
    write_parquet(output, trends)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "rows": len(trends),
        "keywords": int(trends["keyword"].nunique()),
        "years": int(trends["year"].nunique()),
    }


def _data_gate(dataset: dict[str, Any], workspace: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    harvest = verify_bulk_harvest(workspace)
    processing = verify_large_processing(workspace)
    quality = read_json(workspace / "quality" / "processing_report.json")
    gates = protocol["data_gates"]
    bounds = quality["year_bounds"]
    checks = {
        "harvest_acceptance": bool(harvest["passed"]),
        "processing_acceptance": bool(processing["passed"]),
        "minimum_canonical_works": quality["canonical_records"]
        >= gates["minimum_canonical_works"],
        "complete_year_bounds": bounds["minimum"] == dataset["year_from"]
        and bounds["maximum"] == dataset["year_to"],
        "minimum_temporal_years": bounds["maximum"] - bounds["minimum"] + 1
        >= gates["minimum_temporal_years"],
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "canonical_works": quality["canonical_records"],
        "duplicates_removed": quality["duplicates_removed"],
        "year_bounds": bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    amendment_hash = sha256_file(args.amendment)
    if read_json(args.protocol_freeze).get("protocol_sha256") != protocol_hash:
        raise SystemExit("Replication protocol freeze hash mismatch")
    if read_json(args.amendment_freeze).get("sha256") != amendment_hash:
        raise SystemExit("Replication amendment freeze hash mismatch")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    datasets = effective_query_datasets(protocol, amendment=amendment)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in datasets:
        dataset_id = dataset["id"]
        workspace = args.workspace_root / dataset_id
        project = yaml.safe_load((workspace / "project.yml").read_text(encoding="utf-8"))
        expected_query = {
            "keywords": dataset["keywords"],
            "query_mode": dataset["query_mode"],
            "year_from": dataset["year_from"],
            "year_to": dataset["year_to"],
            "source": dataset["source"],
            "document_types": dataset["document_types"],
            "max_records": None,
        }
        observed_query = {
            key: project["protocol"].get(key) for key in expected_query
        }
        if observed_query != expected_query:
            raise SystemExit(f"Replication project query mismatch: {dataset_id}")
        gate = _data_gate(dataset, workspace, protocol)
        output = args.output_root / dataset_id
        write_json(output / "data_gate_audit.json", gate)
        if not gate["eligible"]:
            records.append({"dataset_id": dataset_id, "status": "ineligible"})
            continue
        temporal = _materialize_keyword_trends(workspace)
        component = build_discovery_benchmark(
            workspace,
            output / "components" / "keyword_multiscale",
            networks=("keyword_cooccurrence",),
            scales=("small", "medium", "large"),
            task_types=TASKS,
            record_budget=protocol["benchmark"]["record_budget"],
        )
        if len(component["tasks"]) != protocol["benchmark"][
            "expected_tasks_per_eligible_dataset"
        ]:
            raise SystemExit(f"Replication task count mismatch: {dataset_id}")
        large = next(row for row in component["scales"] if row["name"] == "large")
        if large["nodes"] < protocol["data_gates"]["minimum_large_graph_nodes"]:
            raise SystemExit(f"Replication large graph gate failed: {dataset_id}")
        benchmark = {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "formal_v3_replication": {
                "confirmatory": True,
                "status": "constructed_not_executed",
                "protocol_sha256": protocol_hash,
                "query_amendment_sha256": amendment_hash,
                "temporal_prerequisite": temporal,
            },
            "design": {
                **component["design"],
                "conditions": protocol["conditions"]["confirmatory"],
                "primary_hypotheses": ["R1"],
            },
            "scales": component["scales"],
            "tasks": component["tasks"],
        }
        benchmark_path = output / "benchmark.json"
        write_json(benchmark_path, benchmark)
        record = {
            "dataset_id": dataset_id,
            "status": "constructed_not_executed",
            "tasks": len(benchmark["tasks"]),
            "benchmark_sha256": sha256_file(benchmark_path),
            "canonical_works": gate["canonical_works"],
            "graph_scales": benchmark["scales"],
        }
        records.append(record)
        write_json(
            args.output_root / "construction_manifest.json",
            {
                "schema_version": 1,
                "status": "constructing",
                "protocol_sha256": protocol_hash,
                "query_amendment_sha256": amendment_hash,
                "records": records,
            },
        )
        print(dataset_id, len(benchmark["tasks"]), large["nodes"], large["edges"])
    status = (
        "constructed_not_executed"
        if len(records) == len(datasets)
        and all(row["status"] == "constructed_not_executed" for row in records)
        else "incomplete"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "protocol_sha256": protocol_hash,
        "query_amendment_sha256": amendment_hash,
        "records": records,
    }
    write_json(args.output_root / "construction_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
