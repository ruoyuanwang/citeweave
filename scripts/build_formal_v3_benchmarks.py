from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import yaml

from citeweave.graph_discovery import build_discovery_benchmark
from citeweave.harvest_acceptance import verify_bulk_harvest
from citeweave.io import read_json, sha256_file, write_json, write_parquet
from citeweave.processing_acceptance import verify_large_processing

PRIMARY_TASKS = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)
EXTERNAL_TASKS = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "productivity_centrality_divergence",
)


def _protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_project_matches(
    dataset: dict[str, Any], workspace: Path, amendment: dict[str, Any]
) -> None:
    project = yaml.safe_load((workspace / "project.yml").read_text(encoding="utf-8"))
    source = project["protocol"]
    expected = {
        "keywords": dataset["keywords"],
        "year_from": dataset["year_from"],
        "year_to": dataset["year_to"],
        "source": dataset["source"],
        "query_mode": dataset["query_mode"],
        "document_types": dataset["document_types"],
        "max_records": dataset["max_records"],
    }
    observed = {key: source.get(key) for key in expected}
    materialization = amendment["changes"]["graph_materialization"]
    expected_processing = {
        key: materialization[key] for key in ("candidate_pool_size", "edge_row_limit")
    }
    observed_processing = {
        key: project["processing"].get(key) for key in expected_processing
    }
    if (
        observed != expected
        or project.get("project_id") != dataset["id"]
        or observed_processing != expected_processing
    ):
        raise ValueError(
            f"Workspace protocol differs from freeze for {dataset['id']}: "
            f"expected={expected!r}, observed={observed!r}, "
            f"expected_processing={expected_processing!r}, "
            f"observed_processing={observed_processing!r}"
        )


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _materialize_keyword_trends(workspace: Path) -> dict[str, Any]:
    """Materialize the deterministic full-corpus prerequisite for temporal tasks."""
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


def _data_gate_audit(
    dataset: dict[str, Any], workspace: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    harvest = verify_bulk_harvest(workspace)
    processing = verify_large_processing(workspace)
    quality = read_json(workspace / "quality" / "processing_report.json")
    gates = protocol["data_gates"]
    year_bounds = quality["year_bounds"]
    checks = {
        "harvest_acceptance": bool(harvest["passed"]),
        "processing_acceptance": bool(processing["passed"]),
        "minimum_canonical_works": (
            quality["canonical_records"] >= gates["minimum_canonical_works"]
        ),
        "complete_year_bounds": (
            year_bounds["minimum"] == dataset["year_from"]
            and year_bounds["maximum"] == dataset["year_to"]
        ),
        "minimum_temporal_years": (
            year_bounds["maximum"] - year_bounds["minimum"] + 1
            >= gates["minimum_temporal_years"]
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "canonical_works": quality["canonical_records"],
        "duplicates_removed": quality["duplicates_removed"],
        "year_bounds": year_bounds,
        "harvest": harvest,
        "processing": processing,
    }


def _combine(
    primary: dict[str, Any], external: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    tasks = [*primary["tasks"], *external["tasks"]]
    item_ids = [task["item_id"] for task in tasks]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Formal benchmark components contain duplicate item IDs")
    expected = protocol["benchmark"]["expected_tasks_per_eligible_dataset"]
    if len(tasks) != expected:
        counts: dict[str, int] = {}
        for task in tasks:
            key = f"{task['network']}:{task['scale']}:{task['task_type']}"
            counts[key] = counts.get(key, 0) + 1
        raise ValueError(
            f"Expected exactly {expected} registered tasks, got {len(tasks)}: {counts}"
        )
    return {
        "schema_version": 2,
        "dataset_id": primary["dataset_id"],
        "formal_v3": {
            "confirmatory": True,
            "status": "constructed_not_executed",
            "primary_task_count": len(primary["tasks"]),
            "external_validity_task_count": len(external["tasks"]),
        },
        "design": {
            **primary["design"],
            "conditions": protocol["conditions"]["primary"],
            "diagnostic_conditions": protocol["conditions"]["diagnostic"],
            "primary_hypotheses": [
                row["id"] for row in protocol["hypotheses"] if row.get("primary")
            ],
        },
        "scales": [*primary["scales"], *external["scales"]],
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_hash = _protocol_hash(args.protocol)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    freeze = read_json(args.freeze)
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    amendment_hash = sha256_file(args.amendment)
    if freeze["protocol_sha256"] != protocol_hash:
        raise SystemExit("Protocol hash differs from freeze attestation")
    if (
        amendment["base_protocol_sha256"] != protocol_hash
        or amendment["status"] != "frozen_before_model_execution"
        or amendment["model_outcomes_inspected"] is not False
    ):
        raise SystemExit("Amendment does not attest a pre-model change to this protocol")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in protocol["datasets"]:
        dataset_id = dataset["id"]
        workspace = args.workspace_root / dataset_id
        output = args.output_root / dataset_id
        _assert_project_matches(dataset, workspace, amendment)
        gate_audit = _data_gate_audit(dataset, workspace, protocol)
        write_json(output / "data_gate_audit.json", gate_audit)
        if not gate_audit["eligible"]:
            records.append({"dataset_id": dataset_id, "status": "ineligible"})
            continue
        temporal_prerequisite = _materialize_keyword_trends(workspace)
        primary = build_discovery_benchmark(
            workspace,
            output / "components" / "primary_keyword_multiscale",
            networks=("keyword_cooccurrence",),
            scales=("small", "medium", "large"),
            task_types=PRIMARY_TASKS,
            record_budget=protocol["benchmark"]["record_budget"],
        )
        external = build_discovery_benchmark(
            workspace,
            output / "components" / "external_networks_large",
            networks=("institution_collaboration", "coauthorship"),
            scales=("large",),
            task_types=EXTERNAL_TASKS,
            record_budget=protocol["benchmark"]["record_budget"],
        )
        benchmark = _combine(primary, external, protocol)
        benchmark["formal_v3"]["protocol_sha256"] = protocol_hash
        benchmark["formal_v3"]["amendment_sha256"] = amendment_hash
        benchmark["formal_v3"]["temporal_prerequisite"] = temporal_prerequisite
        benchmark_path = output / "benchmark.json"
        write_json(benchmark_path, benchmark)
        record = {
            "dataset_id": dataset_id,
            "status": "constructed_not_executed",
            "tasks": len(benchmark["tasks"]),
            "benchmark_sha256": sha256_file(benchmark_path),
            "canonical_works": gate_audit["canonical_works"],
            "graph_scales": benchmark["scales"],
        }
        records.append(record)
        write_json(
            args.output_root / "construction_manifest.json",
            {
                "schema_version": 1,
                "protocol_sha256": protocol_hash,
                "amendment_sha256": amendment_hash,
                "amendment_path": str(args.amendment.resolve()),
                "protocol_path": str(args.protocol.resolve()),
                "status": "constructing",
                "records": records,
            },
        )
        print(dataset_id, record["status"], record["tasks"])
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "amendment_sha256": amendment_hash,
        "amendment_path": str(args.amendment.resolve()),
        "protocol_path": str(args.protocol.resolve()),
        "status": (
            "constructed_not_executed"
            if records and all(row["status"] == "constructed_not_executed" for row in records)
            else "incomplete"
        ),
        "records": records,
    }
    write_json(args.output_root / "construction_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
