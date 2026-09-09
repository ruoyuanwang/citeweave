from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.oversight_factorial_readiness_v2 import (
    assess_factorial_oversight_readiness_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--reviewer-registry", type=Path, required=True)
    parser.add_argument("--capability-freeze", type=Path, required=True)
    parser.add_argument("--combined-observations", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--previous-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_factorial_oversight_readiness_v2(
        packet_manifest_path=args.packet_manifest,
        reviewer_registry_path=args.reviewer_registry,
        capability_freeze_path=args.capability_freeze,
        combined_observations_path=args.combined_observations,
        assignment_root=args.assignment_root,
        amendment_path=args.amendment,
        amendment_freeze_path=args.amendment_freeze,
        previous_amendment_path=args.previous_amendment,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
