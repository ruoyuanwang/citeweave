from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.formal_protocol_amendment import effective_query_datasets
from citeweave.io import read_json, sha256_file, write_json
from citeweave.models import (
    AcquisitionPolicy,
    ProcessingPolicy,
    ProjectConfig,
    SearchProtocol,
    SourceName,
)
from citeweave.workflow import create_project


def build_replication_config(
    dataset: dict[str, Any], *, protocol_hash: str, amendment_hash: str
) -> ProjectConfig:
    return ProjectConfig(
        project_id=dataset["id"],
        protocol=SearchProtocol(
            title=dataset["title"],
            keywords=dataset["keywords"],
            query_mode=dataset["query_mode"],
            year_from=dataset["year_from"],
            year_to=dataset["year_to"],
            source=SourceName(dataset["source"]),
            document_types=dataset["document_types"],
            max_records=None,
            include_abstracts=True,
            include_references=True,
            notes=(
                "CiteWeave v3 independent scale-interaction replication; "
                f"protocol={protocol_hash}; query_amendment={amendment_hash}."
            ),
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            partition_strategy="adaptive_date",
            target_slice_records=25_000,
            page_size=None,
            max_retries=8,
            max_slice_restarts=3,
            compress_raw=True,
        ),
        processing=ProcessingPolicy(
            mode="disk",
            chunk_size=1_000,
            duckdb_memory_limit="4GB",
            candidate_pool_size=2_000,
            edge_row_limit=1_000_000,
            keep_partitions=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--query-judgment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    amendment_hash = sha256_file(args.amendment)
    protocol_freeze = read_json(args.protocol_freeze)
    amendment_freeze = read_json(args.amendment_freeze)
    judgment = read_json(args.query_judgment)
    if protocol_freeze.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Replication protocol freeze hash mismatch")
    if amendment_freeze.get("sha256") != amendment_hash:
        raise SystemExit("Replication query amendment freeze hash mismatch")
    if judgment.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Query judgment protocol hash mismatch")
    if judgment.get("query_amendment_sha256") != amendment_hash:
        raise SystemExit("Query judgment amendment hash mismatch")
    if not judgment.get("judgment", {}).get("overall_approved"):
        raise SystemExit("Replication queries lack independent pre-acquisition approval")

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    datasets = effective_query_datasets(protocol, amendment=amendment)
    expected_ids = {row["id"] for row in datasets}
    approved_ids = {
        row["id"]
        for row in judgment["judgment"]["judgments"]
        if row.get("approved")
    }
    if approved_ids != expected_ids:
        raise SystemExit("Not every effective replication query is approved")

    records = []
    for dataset in datasets:
        workspace = args.output_root / dataset["id"]
        if workspace.exists():
            raise SystemExit(f"Refusing to overwrite replication workspace: {workspace}")
        config = build_replication_config(
            dataset,
            protocol_hash=protocol_hash,
            amendment_hash=amendment_hash,
        )
        create_project(workspace, config)
        records.append(
            {
                "dataset_id": dataset["id"],
                "workspace": str(workspace.resolve()),
                "project_sha256": sha256_file(workspace / "project.yml"),
                "keywords": dataset["keywords"],
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "initialized_pre_acquisition",
        "protocol_sha256": protocol_hash,
        "query_amendment_sha256": amendment_hash,
        "query_judgment_sha256": sha256_file(args.query_judgment),
        "records": records,
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
