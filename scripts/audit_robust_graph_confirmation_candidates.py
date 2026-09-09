from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from citeweave.io import read_json, sha256_file, write_json


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sample_packet(
    *,
    protocol_sha256: str,
    dataset: dict[str, Any],
    works: pd.DataFrame,
) -> dict[str, Any]:
    ranked = sorted(
        works.to_dict(orient="records"),
        key=lambda row: hashlib.sha256(
            f"{protocol_sha256}:{dataset['id']}:{row['work_id']}".encode()
        ).hexdigest(),
    )[:100]
    candidate_code = "CAND-" + hashlib.sha256(
        f"{protocol_sha256}:{dataset['id']}".encode()
    ).hexdigest()[:12]
    return {
        "schema_version": 1,
        "status": "blind_query_relevance_packet",
        "candidate_code": candidate_code,
        "concepts": dataset["keywords"],
        "instructions": (
            "Mark relevant only when both registered concepts are scientifically central "
            "to the title/abstract. Do not infer relevance from candidate priority."
        ),
        "items": [
            {
                "item_code": "QR-"
                + hashlib.sha256(
                    f"{candidate_code}:{row['work_id']}".encode()
                ).hexdigest()[:16],
                "title": _json_safe(row.get("title")),
                "abstract": _json_safe(row.get("abstract")),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            }
            for row in ranked
        ],
    }


def audit_candidates(
    protocol_path: Path,
    freeze_path: Path,
    initialization_path: Path,
    packet_root: Path,
) -> dict[str, Any]:
    protocol_sha = sha256_file(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    initialization = read_json(initialization_path)
    if freeze.get("sha256") != protocol_sha:
        raise ValueError("Confirmation protocol freeze mismatch")
    if initialization.get("protocol_sha256") != protocol_sha:
        raise ValueError("Initialization protocol identity mismatch")
    datasets = {row["id"]: row for row in protocol["candidate_topics"]}
    if set(datasets) != {row["dataset_id"] for row in initialization["records"]}:
        raise ValueError("Initialized candidate set differs from protocol")
    gates = protocol["selection"]["data_gates"]
    packet_root.mkdir(parents=True, exist_ok=True)
    records = []
    for initialized in sorted(initialization["records"], key=lambda row: row["priority"]):
        dataset_id = initialized["dataset_id"]
        dataset = datasets[dataset_id]
        workspace = Path(initialized["workspace"])
        project_path = workspace / "project.yml"
        if not project_path.is_file() or sha256_file(project_path) != initialized["project_sha256"]:
            raise ValueError(f"Project identity mismatch: {dataset_id}")
        harvest_path = workspace / "audit/harvest_manifest.json"
        processing_path = workspace / "audit/processing_manifest.json"
        quality_path = workspace / "quality/processing_report.json"
        if not all(path.is_file() for path in (harvest_path, processing_path, quality_path)):
            records.append(
                {
                    "dataset_id": dataset_id,
                    "priority": initialized["priority"],
                    "status": "awaiting_acquisition_or_processing",
                }
            )
            continue
        harvest = read_json(harvest_path)
        processing = read_json(processing_path)
        quality = read_json(quality_path)
        works_path = workspace / "canonical/works.parquet"
        authors_path = workspace / "canonical/authors.parquet"
        keyword_nodes_path = workspace / "canonical/visualization/keyword_occurrences.parquet"
        keyword_edges_path = workspace / "canonical/visualization/keyword_cooccurrence_edges.parquet"
        paths = (works_path, authors_path, keyword_nodes_path, keyword_edges_path)
        if not all(path.is_file() for path in paths):
            raise ValueError(f"Processed canonical files are incomplete: {dataset_id}")
        works = pd.read_parquet(works_path)
        authors = pd.read_parquet(authors_path)
        keyword_nodes = pd.read_parquet(keyword_nodes_path)
        keyword_edges = pd.read_parquet(keyword_edges_path)
        author_ids = authors["author_id"].fillna("").astype(str).str.casefold()
        checks = {
            "harvest_complete": harvest.get("status") == "complete"
            and int(harvest.get("unique_records") or -1)
            == int(harvest.get("planned_expected_records") or -2),
            "processing_complete": processing.get("status") == "complete",
            "processing_quality_passed": quality.get("passed") is True,
            "minimum_canonical_works": len(works) >= int(gates["minimum_canonical_works"]),
            "minimum_nonempty_publication_years": works["year"].dropna().nunique()
            >= int(gates["minimum_nonempty_publication_years"]),
            "minimum_large_keyword_nodes": len(keyword_nodes)
            >= int(gates["minimum_large_keyword_nodes"]),
            "minimum_large_keyword_edges": len(keyword_edges)
            >= int(gates["minimum_large_keyword_edges"]),
            "no_placeholder_author_identity": not author_ids.isin(
                {"", "none", "null", "openalex-author:none"}
            ).any(),
        }
        packet = _sample_packet(
            protocol_sha256=protocol_sha, dataset=dataset, works=works
        )
        packet_path = packet_root / packet["candidate_code"] / "query_review_packet.json"
        if packet_path.exists():
            existing_packet = read_json(packet_path)
            if _json_safe(existing_packet) == packet:
                if existing_packet != packet:
                    write_json(packet_path, packet)
            else:
                raise ValueError(f"Existing query packet drift: {dataset_id}")
        else:
            write_json(packet_path, packet)
        records.append(
            {
                "dataset_id": dataset_id,
                "priority": initialized["priority"],
                "candidate_code": packet["candidate_code"],
                "status": (
                    "awaiting_blind_query_relevance_review"
                    if all(checks.values())
                    else "failed_automatic_data_gate"
                ),
                "automatic_data_gate_passed": all(checks.values()),
                "checks": checks,
                "counts": {
                    "works": len(works),
                    "nonempty_years": int(works["year"].dropna().nunique()),
                    "keyword_nodes": len(keyword_nodes),
                    "keyword_edges": len(keyword_edges),
                },
                "source_hashes": {
                    str(path.resolve()): sha256_file(path) for path in paths
                },
                "query_review_packet": str(packet_path.resolve()),
                "query_review_packet_sha256": sha256_file(packet_path),
            }
        )
    processed = sum("automatic_data_gate_passed" in row for row in records)
    passed = sum(row.get("automatic_data_gate_passed") is True for row in records)
    target_topics = int(protocol["selection"]["target_topics"])
    priority_prefix_ready = all(
        row.get("automatic_data_gate_passed") is True
        for row in records[:target_topics]
    )
    return {
        "schema_version": 1,
        "status": (
            "awaiting_candidate_processing"
            if not priority_prefix_ready
            else "awaiting_blind_query_relevance_review"
        ),
        "confirmatory": True,
        "protocol_sha256": protocol_sha,
        "initialization_sha256": sha256_file(initialization_path),
        "candidates": len(records),
        "processed_candidates": processed,
        "automatic_gate_passes": passed,
        "selected_topics": [],
        "provider_responses": 0,
        "perturbation_outcomes": 0,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_candidates(
        args.protocol, args.freeze, args.initialization, args.packet_root
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
