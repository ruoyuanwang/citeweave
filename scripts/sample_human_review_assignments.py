from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from citeweave.io import read_json, write_json
from citeweave.review_sampling import build_sampled_review_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--factual-per-reviewer", type=int, default=80)
    parser.add_argument("--semantic-per-reviewer", type=int, default=50)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("Refusing to overwrite an existing sampled review round")
    source_manifest = read_json(args.source_root / "internal_manifest.json")
    sampled = build_sampled_review_manifest(
        source_manifest,
        packet_root=args.source_root,
        factual_per_reviewer=args.factual_per_reviewer,
        semantic_per_reviewer=args.semantic_per_reviewer,
        overlap_fraction=args.overlap_fraction,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True)
    selected = {
        layer: {
            packet_id
            for reviewer in sampled["reviewers"]
            for packet_id in sampled["assignments"][reviewer][layer]
        }
        for layer in ("factual", "semantic")
    }
    for layer, packet_ids in selected.items():
        destination = args.output_root / "packets" / layer
        destination.mkdir(parents=True)
        for packet_id in sorted(packet_ids):
            shutil.copy2(
                args.source_root / "packets" / layer / f"{packet_id}.json",
                destination / f"{packet_id}.json",
            )
    write_json(args.output_root / "internal_manifest.json", sampled)
    write_json(
        args.output_root / "reviewer_protocol.json",
        {
            "schema_version": 2,
            "reviewer_codes": sampled["reviewers"],
            "assignments": {
                reviewer: {
                    layer: len(sampled["assignments"][reviewer][layer])
                    for layer in ("factual", "semantic")
                }
                for reviewer in sampled["reviewers"]
            },
            "common_double_review": {
                layer: sampled["sampling"]["layers"][layer]["common_double_review"]
                for layer in ("factual", "semantic")
            },
            "instructions": [
                "Review independently; do not communicate before submission.",
                "Do not inspect internal_manifest.json; it contains hidden strata.",
                "Only common packets contribute to inter-reviewer agreement.",
                "Disagreements on common packets require independent adjudication.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "sampling": sampled["sampling"]["layers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
