from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_randomization_analysis import analyze_oversight_randomization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=50_000)
    args = parser.parse_args()

    amendment_hash = sha256_file(args.amendment)
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    freeze = read_json(args.amendment_freeze)
    if (
        amendment.get("amendment_id")
        != "review_policy_amendment_006_finite_benchmark_randomization"
        or amendment.get("human_outcomes_inspected") is not False
        or amendment.get("heldout_outcomes_inspected") is not False
        or freeze.get("sha256") != amendment_hash
    ):
        raise SystemExit("Randomization-inference amendment is not frozen")
    implementations = amendment.get("implementation") or {}
    module_path = Path("src/citeweave/oversight_randomization_analysis.py")
    if implementations.get("analysis_module", {}).get("sha256") != sha256_file(
        module_path
    ):
        raise SystemExit("Randomization analysis module differs from amendment")
    if implementations.get("analysis_cli", {}).get("sha256") != sha256_file(
        Path(__file__)
    ):
        raise SystemExit("Randomization analysis CLI differs from amendment")

    outcomes = read_json(args.outcomes)
    packets = read_json(args.packet_manifest)
    assignment = read_json(args.assignment_manifest)
    if outcomes.get("status") != "factorial_oversight_outcomes_finalized":
        raise SystemExit("Held-out outcomes are not finalized")
    if outcomes.get("packet_manifest_sha256") != sha256_file(args.packet_manifest):
        raise SystemExit("Outcomes do not bind the packet manifest")
    if outcomes.get("assignment_manifest_sha256") != sha256_file(
        args.assignment_manifest
    ):
        raise SystemExit("Outcomes do not bind the assignment manifest")
    result = analyze_oversight_randomization(
        outcomes.get("records") or [],
        packets.get("records") or [],
        assignment,
        permutations=args.permutations,
    )
    result.update(
        {
            "outcomes_sha256": sha256_file(args.outcomes),
            "packet_manifest_sha256": sha256_file(args.packet_manifest),
            "assignment_manifest_sha256": sha256_file(args.assignment_manifest),
            "analysis_amendment_sha256": amendment_hash,
        }
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
