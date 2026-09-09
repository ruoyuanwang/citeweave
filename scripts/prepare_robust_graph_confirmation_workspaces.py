from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.models import (
    AcquisitionPolicy,
    ProcessingPolicy,
    ProjectConfig,
    SearchProtocol,
    SourceName,
)
from citeweave.workflow import create_project


def _config(dataset: dict[str, Any], *, protocol_sha256: str) -> ProjectConfig:
    return ProjectConfig(
        project_id=dataset["id"],
        protocol=SearchProtocol(
            title=dataset["title"],
            keywords=dataset["keywords"],
            query_mode="all",
            year_from=int(dataset["year_from"]),
            year_to=int(dataset["year_to"]),
            source=SourceName.openalex,
            document_types=["article"],
            max_records=None,
            include_abstracts=True,
            include_references=True,
            notes=(
                "Held-out robust graph synthesis confirmation candidate; "
                f"priority={dataset['priority']}; protocol={protocol_sha256}."
            ),
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            partition_strategy="adaptive_date",
            target_slice_records=25_000,
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
        random_seed=20_260_908,
    )


def prepare(
    protocol_path: Path,
    freeze_path: Path,
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    protocol_sha = sha256_file(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    if freeze.get("sha256") != protocol_sha:
        raise ValueError("Confirmation protocol differs from its freeze artifact")
    if protocol.get("status") != "frozen_before_any_candidate_acquisition":
        raise ValueError("Confirmation protocol is not pre-acquisition frozen")
    candidates = list(protocol.get("candidate_topics") or [])
    if len(candidates) != 12:
        raise ValueError("Exactly 12 ordered candidates are required")
    if [row.get("priority") for row in candidates] != list(range(1, 13)):
        raise ValueError("Candidate priority must be exactly 1 through 12")
    if len({row.get("id") for row in candidates}) != 12:
        raise ValueError("Candidate ids must be unique")
    if manifest_path.exists() or output_root.exists():
        raise ValueError("Refusing to overwrite initialized confirmation artifacts")

    records = []
    for dataset in candidates:
        workspace = output_root / dataset["id"]
        create_project(workspace, _config(dataset, protocol_sha256=protocol_sha))
        records.append(
            {
                "dataset_id": dataset["id"],
                "priority": dataset["priority"],
                "domain": dataset["domain"],
                "workspace": str(workspace.resolve()),
                "project_sha256": sha256_file(workspace / "project.yml"),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "initialized_before_candidate_acquisition",
        "confirmatory": True,
        "protocol_sha256": protocol_sha,
        "candidate_count": len(records),
        "selection_rule": "first_eight_eligible_candidates_by_frozen_priority",
        "records": records,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.protocol, args.freeze, args.output_root, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
