from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from citeweave.author_benchmark_rebuild import rebuild_author_benchmark
from citeweave.bulk_acquisition import _pid_is_running
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-benchmark-root", type=Path, required=True)
    parser.add_argument("--original-workspace-root", type=Path, required=True)
    parser.add_argument("--corrected-workspace-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-pid", type=int)
    args = parser.parse_args()
    freeze = read_json(args.amendment_freeze)
    amendment_hash = sha256_file(args.amendment)
    if freeze["amendment_sha256"] != amendment_hash or any(
        sha256_file(Path(p)) != h for p, h in freeze["code_sha256"].items()
    ):
        raise SystemExit("Correction freeze/code changed")
    original = read_json(args.original_benchmark_root / "construction_manifest.json")
    protocol = yaml.safe_load(Path(original["protocol_path"]).read_text(encoding="utf-8"))
    records = []
    for row in sorted(original["records"], key=lambda r: r["dataset_id"]):
        topic = row["dataset_id"]
        original_benchmark = args.original_benchmark_root / topic / "benchmark.json"
        if sha256_file(original_benchmark) != row["benchmark_sha256"]:
            raise SystemExit(f"Original frozen benchmark changed: {topic}")
        corrected = args.corrected_workspace_root / topic
        while not (corrected / "identity_correction_verification.json").is_file():
            if not args.upstream_pid or not _pid_is_running(args.upstream_pid):
                raise SystemExit(f"Corrected input unavailable and upstream not running: {topic}")
            time.sleep(10)
        output = args.output_root / topic
        receipt_path = output / "author_identity_rebuild_receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                receipt["benchmark_sha256"] != sha256_file(output / "benchmark.json")
                or receipt["author_identity_correction_sha256"] != amendment_hash
            ):
                raise SystemExit("Existing rebuild receipt mismatch")
            if any(sha256_file(Path(p)) != h for p, h in receipt["source_sha256"].items()):
                raise SystemExit("Rebuild input identity drift")
        else:
            print(
                json.dumps({"topic": topic, "stage": "rebuilding_all_five_author_tasks"}),
                flush=True,
            )
            receipt = rebuild_author_benchmark(
                original_workspace=args.original_workspace_root / topic,
                corrected_workspace=corrected,
                original_benchmark=original_benchmark,
                output=output,
                author_task_types=tuple(protocol["benchmark"]["external_validity_task_types"]),
                amendment_hash=amendment_hash,
            )
        benchmark = read_json(output / "benchmark.json")
        records.append(
            {
                **row,
                "benchmark_sha256": receipt["benchmark_sha256"],
                "graph_scales": benchmark["scales"],
                "author_identity_correction_sha256": amendment_hash,
            }
        )
        write_json(
            args.output_root / "construction_manifest.json",
            {
                **original,
                "records": records,
                "status": "constructed_not_executed"
                if len(records) == len(original["records"])
                else "constructing",
                "author_identity_correction_sha256": amendment_hash,
                "original_construction_manifest_sha256": sha256_file(
                    args.original_benchmark_root / "construction_manifest.json"
                ),
            },
        )
        print(
            json.dumps(
                {
                    "topic": topic,
                    "stage": "rebuilt_not_promoted",
                    "unchanged": receipt["task_comparisons"]["unchanged_tasks"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
