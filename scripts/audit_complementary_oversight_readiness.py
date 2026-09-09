from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.oversight_readiness import assess_complementary_oversight_readiness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--reviewer-registry", type=Path)
    parser.add_argument("--capability-freeze", type=Path)
    parser.add_argument("--assignment-manifest", type=Path)
    parser.add_argument(
        "--complementary-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_001_complementary_oversight.yml"
        ),
    )
    parser.add_argument(
        "--cluster-inference-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_002_cluster_inference.yml"
        ),
    )
    parser.add_argument(
        "--factorial-analysis-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_003_factorial_cluster_inference.yml"
        ),
    )
    parser.add_argument(
        "--factorial-analysis-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_003_factorial_cluster_inference_freeze.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readiness = assess_complementary_oversight_readiness(
        packet_manifest_path=args.packet_manifest,
        reviewer_registry_path=args.reviewer_registry,
        capability_freeze_path=args.capability_freeze,
        assignment_manifest_path=args.assignment_manifest,
        complementary_amendment_path=args.complementary_amendment,
        cluster_inference_amendment_path=args.cluster_inference_amendment,
        factorial_analysis_amendment_path=args.factorial_analysis_amendment,
        factorial_analysis_freeze_path=args.factorial_analysis_freeze,
    )
    write_json(args.output, readiness)
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    raise SystemExit(0 if readiness["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
