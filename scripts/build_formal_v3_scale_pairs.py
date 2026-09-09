from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file, write_json

ANCHOR_FIELDS = {
    "multi_hop_connector": ("source_label", "target_label"),
    "bridge_counterfactual": ("source_label", "target_label"),
    "community_role_contrast": (
        "dominant_representative",
        "outward_representative",
    ),
    "hub_removal_resilience": ("removed_hub",),
    "temporal_structural_shift": ("emerging_label", "established_label"),
}
SCALES = ("small", "medium", "large")


def build_pairs(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    keyword_tasks = [
        task
        for task in benchmark["tasks"]
        if task["network"] == "keyword_cooccurrence"
    ]
    pairs = []
    for task_type, fields in ANCHOR_FIELDS.items():
        by_scale = {
            task["scale"]: task
            for task in keyword_tasks
            if task["task_type"] == task_type
        }
        if set(by_scale) != set(SCALES):
            continue
        signatures = {
            scale: tuple(by_scale[scale]["answer"][field] for field in fields)
            for scale in SCALES
        }
        if len(set(signatures.values())) != 1:
            continue
        anchor = dict(zip(fields, signatures["small"], strict=True))
        digest = hashlib.sha256(
            json.dumps(
                [benchmark["dataset_id"], task_type, anchor],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        pairs.append(
            {
                "pair_id": f"SP{digest}",
                "dataset_id": benchmark["dataset_id"],
                "network": "keyword_cooccurrence",
                "task_type": task_type,
                "anchor_fields": list(fields),
                "anchor": anchor,
                "item_ids": {scale: by_scale[scale]["item_id"] for scale in SCALES},
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a frozen scale-pair artifact")
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    pairs = []
    benchmark_hashes = {}
    for record in construction["records"]:
        dataset_id = record["dataset_id"]
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        benchmark_hashes[dataset_id] = sha256_file(benchmark_path)
        pairs.extend(build_pairs(read_json(benchmark_path)))
    artifact = {
        "schema_version": 1,
        "status": "frozen_before_model_execution",
        "selection_rule": (
            "Include a keyword task family only when the registered semantic entity "
            "anchor fields are identical in small, medium, and large graphs."
        ),
        "anchor_fields_by_task_type": {
            key: list(value) for key, value in ANCHOR_FIELDS.items()
        },
        "scales": list(SCALES),
        "benchmark_hashes": benchmark_hashes,
        "groups": len(pairs),
        "items": len(pairs) * len(SCALES),
        "pairs": pairs,
        "model_outcomes_inspected": False,
    }
    write_json(args.output, artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
