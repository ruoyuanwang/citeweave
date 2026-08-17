from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from citeweave.bulk_acquisition import _fingerprint
from citeweave.formal_graph_experiment import (
    build_formal_graph_grounding,
    verify_formal_graph_grounding,
)
from citeweave.formal_protocol import verify_frozen_query_registry
from citeweave.harvest_acceptance import verify_bulk_harvest
from citeweave.io import read_json
from citeweave.large_scale_evidence import (
    prepare_large_scale_evidence,
    verify_large_scale_evidence,
)
from citeweave.large_scale_visualization import render_large_project
from citeweave.models import (
    AcquisitionPolicy,
    ProcessingPolicy,
    ProjectConfig,
    SearchProtocol,
    SourceName,
)
from citeweave.processing_acceptance import verify_large_processing
from citeweave.visual_acceptance import verify_visualization
from citeweave.workflow import harvest_project, process_project

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets.yml"
DEFAULT_WORKSPACES = ROOT / "experiments" / "formal_workspaces"


def _config(entry: dict[str, Any]) -> ProjectConfig:
    if entry.get("max_records") is not None:
        raise ValueError(f"{entry['id']}: formal datasets must have max_records: null")
    if entry.get("query_status") != "frozen":
        raise ValueError(f"{entry['id']}: query must pass Query Judge and be frozen")
    return ProjectConfig(
        project_id=entry["id"],
        protocol=SearchProtocol(
            title=entry["title"],
            keywords=entry["keywords"],
            query_mode=entry["query_mode"],
            year_from=entry["year_from"],
            year_to=entry["year_to"],
            source=SourceName(entry["source"]),
            search_expression=entry.get("search_expression"),
            search_scope=entry.get("search_scope", "fulltext"),
            document_types=entry.get("document_types", []),
            language=entry.get("language"),
            max_records=None,
            include_references=entry.get("include_references", True),
            notes="Formal full-year experiment protocol; independently generated query.",
        ),
        crossref_mailto=os.getenv("CROSSREF_MAILTO"),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            partition_strategy="adaptive_date",
            target_slice_records=25_000,
            page_size=1_000,
            max_retries=8,
        ),
        processing=ProcessingPolicy(
            mode="disk",
            chunk_size=1_000,
            duckdb_memory_limit="2GB",
            keep_partitions=True,
        ),
        llm_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        llm_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _require_pass(stage: str, result: dict[str, Any]) -> None:
    if not result.get("passed"):
        raise RuntimeError(f"{stage} acceptance failed: {json.dumps(result, ensure_ascii=False)}")


def _require_harvest_for_config(workspace: Path, config: ProjectConfig) -> dict[str, Any]:
    manifest_path = workspace / "audit" / "harvest_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Harvest manifest is missing: {manifest_path}")
    harvest = read_json(manifest_path)
    if harvest.get("query_fingerprint") != _fingerprint(config):
        raise RuntimeError(
            "Harvest query fingerprint does not match the frozen formal registry contract."
        )
    acceptance = verify_bulk_harvest(workspace)
    _require_pass("harvest", acceptance)
    return acceptance


def run_one(entry: dict[str, Any], *, stage: str) -> dict[str, Any]:
    workspace = DEFAULT_WORKSPACES / entry["id"]
    config = _config(entry)
    output: dict[str, Any] = {"dataset_id": entry["id"], "workspace": str(workspace)}
    if stage in {"harvest", "all"}:
        output["harvest"] = harvest_project(workspace, config, resume=True)
        output["harvest_acceptance"] = verify_bulk_harvest(workspace)
        _require_pass("harvest", output["harvest_acceptance"])
    if stage in {"process", "all"}:
        output["harvest_acceptance"] = _require_harvest_for_config(workspace, config)
        output["processing"] = process_project(workspace, config, resume=True)
        output["processing_acceptance"] = verify_large_processing(workspace)
        _require_pass("processing", output["processing_acceptance"])
    if stage in {"visualize", "all"}:
        output["harvest_acceptance"] = _require_harvest_for_config(workspace, config)
        output["processing_acceptance"] = verify_large_processing(workspace)
        _require_pass("processing", output["processing_acceptance"])
        output["visualization"] = render_large_project(workspace)
        output["visualization_acceptance"] = verify_visualization(workspace)
        _require_pass("visualization", output["visualization_acceptance"])
    if stage in {"evidence", "all"}:
        output["harvest_acceptance"] = _require_harvest_for_config(workspace, config)
        output["processing_acceptance"] = verify_large_processing(workspace)
        _require_pass("processing", output["processing_acceptance"])
        output["visualization_acceptance"] = verify_visualization(workspace)
        _require_pass("visualization", output["visualization_acceptance"])
        output["evidence"] = prepare_large_scale_evidence(workspace)
        output["evidence_acceptance"] = verify_large_scale_evidence(workspace)
        _require_pass("evidence", output["evidence_acceptance"])
        output["formal_graph_experiment"] = build_formal_graph_grounding(workspace)
        output["formal_graph_experiment_acceptance"] = verify_formal_graph_grounding(
            workspace
        )
        _require_pass(
            "formal_graph_experiment",
            output["formal_graph_experiment_acceptance"],
        )
    if stage == "graph":
        output["harvest_acceptance"] = _require_harvest_for_config(workspace, config)
        output["processing_acceptance"] = verify_large_processing(workspace)
        _require_pass("processing", output["processing_acceptance"])
        output["visualization_acceptance"] = verify_visualization(workspace)
        _require_pass("visualization", output["visualization_acceptance"])
        output["evidence_acceptance"] = verify_large_scale_evidence(workspace)
        _require_pass("evidence", output["evidence_acceptance"])
        output["formal_graph_experiment"] = build_formal_graph_grounding(workspace)
        output["formal_graph_experiment_acceptance"] = verify_formal_graph_grounding(
            workspace
        )
        _require_pass(
            "formal_graph_experiment",
            output["formal_graph_experiment_acceptance"],
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["harvest", "process", "visualize", "evidence", "graph", "all"],
        default="all",
    )
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    verify_frozen_query_registry(registry)
    entries = registry["datasets"]
    selected = {item["id"] for item in entries} if args.all_datasets else set(args.dataset or [])
    if not selected:
        raise SystemExit("Pass --dataset ID or --all-datasets.")
    unknown = selected - {item["id"] for item in entries}
    if unknown:
        raise SystemExit(f"Unknown dataset ids: {sorted(unknown)}")
    results = [run_one(item, stage=args.stage) for item in entries if item["id"] in selected]
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
