from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_factorial_assignment import (
    build_factorial_oversight_assignment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--reviewer-registry", type=Path, required=True)
    parser.add_argument("--capability-freeze", type=Path, required=True)
    parser.add_argument("--combined-observations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--minimum-capability-lower-95", type=float, default=0.5)
    parser.add_argument("--reviewer-second-cost", type=float, default=0.002)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_004_multidimensional_calibration.yml"
        ),
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_004_multidimensional_calibration_freeze.json"
        ),
    )
    args = parser.parse_args()
    result = build_factorial_oversight_assignment(
        packet_manifest_path=args.packet_manifest,
        reviewer_registry_path=args.reviewer_registry,
        capability_freeze_path=args.capability_freeze,
        combined_observations_path=args.combined_observations,
        amendment_path=args.amendment,
        amendment_freeze_path=args.amendment_freeze,
        output_root=args.output_root,
        seed=args.seed,
        minimum_capability_lower_95=args.minimum_capability_lower_95,
        reviewer_second_cost=args.reviewer_second_cost,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
