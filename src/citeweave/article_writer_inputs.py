from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json


def _single_figure(pack: dict[str, Any]) -> dict[str, Any]:
    figures = pack.get("figures")
    if not isinstance(figures, list) or len(figures) != 1:
        raise ValueError("Each frozen evidence pack must contain exactly one figure")
    figure = figures[0]
    if not isinstance(figure, dict):
        raise TypeError("Figure record must be an object")
    path = Path(str(figure.get("path") or ""))
    if not path.is_file():
        raise ValueError(f"Frozen figure does not exist: {path}")
    if sha256_file(path) != figure.get("sha256"):
        raise ValueError(f"Frozen figure hash mismatch: {path}")
    return figure


def _contains_image_locator(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower()
            in {"image_path", "image_url", "image_uri", "data_uri", "image_bytes"}
            or _contains_image_locator(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_image_locator(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
    return False


def build_text_only_writer_inputs(
    source_manifest_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Derive writer-visible inputs while keeping the rendered figure evaluator-only."""
    source_manifest = read_json(source_manifest_path)
    if source_manifest.get("status") != "same_evidence_packs_ready":
        raise ValueError("Source writer packs must be ready before derivation")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for record in sorted(source_manifest.get("records", []), key=lambda row: row["dataset_id"]):
        source_pack_path = Path(record["pack"])
        if not source_pack_path.is_file():
            raise ValueError(f"Source writer pack does not exist: {source_pack_path}")
        if sha256_file(source_pack_path) != record.get("pack_sha256"):
            raise ValueError(f"Source writer pack hash mismatch: {source_pack_path}")
        source_pack = read_json(source_pack_path)
        figure = _single_figure(source_pack)
        figure_path = Path(figure["path"])

        writer_input = copy.deepcopy(source_pack)
        writer_input["schema_version"] = 2
        writer_input["status"] = "text_only_writer_input_frozen_before_articles"
        writer_input["source_writer_pack_sha256"] = record["pack_sha256"]
        writer_input.pop("source_benchmark", None)
        writer_input["condition_contract"] = {
            **writer_input.get("condition_contract", {}),
            "shared_inputs": [
                "writing_brief",
                "graph_phenomena",
                "representative_sources",
                "nonvisual_figure_metadata",
            ],
            "rendered_figure_during_drafting": False,
            "posthoc_figure_insertion": True,
        }
        writer_input.pop("figures", None)
        writer_input["nonvisual_figure_metadata"] = {
            key: value
            for key, value in figure.items()
            if key not in {"path", "sha256"}
        }
        writer_input["posthoc_figure_commitment"] = {
            "figure_sha256": figure["sha256"],
            "rule": (
                "The rendered figure is withheld from every writer during drafting and "
                "attached unchanged to every condition before blinded evaluation."
            ),
        }
        if _contains_image_locator(writer_input):
            raise ValueError("Writer-visible input still contains an image locator")

        topic_dir = output_dir / record["dataset_id"]
        writer_input_path = topic_dir / "writer_input.json"
        write_json(writer_input_path, writer_input)
        records.append(
            {
                "dataset_id": record["dataset_id"],
                "source_panel": record.get("source_panel"),
                "pack": str(writer_input_path.resolve()),
                "pack_sha256": sha256_file(writer_input_path),
                "source_writer_pack": str(source_pack_path.resolve()),
                "source_writer_pack_sha256": record["pack_sha256"],
                "evaluation_figure": str(figure_path.resolve()),
                "evaluation_figure_sha256": figure["sha256"],
                "rendered_figure_exposed_to_writer": False,
                "passed": True,
            }
        )
    if len(records) != 8:
        raise ValueError(f"Confirmatory article experiment requires 8 topics, got {len(records)}")
    manifest = {
        "schema_version": 2,
        "status": "text_only_same_evidence_writer_inputs_ready",
        "access_mode": "structured_graph_text_only_figure_inserted_posthoc",
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "datasets": len(records),
        "records": records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
