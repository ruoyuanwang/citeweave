from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_factorial_analysis import (
    OversightFactorialOutcome,
    analyze_oversight_factorial,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    amendment_hash = sha256_file(args.amendment)
    freeze = read_json(args.amendment_freeze)
    if freeze.get("sha256") != amendment_hash:
        raise SystemExit("Complementary-oversight amendment differs from its freeze")
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    implementations = amendment.get("implementations") or {}
    if implementations.get(Path(__file__).name) != sha256_file(Path(__file__)):
        raise SystemExit("Complementary-oversight analyzer differs from amendment")
    module_path = Path("src/citeweave/oversight_factorial_analysis.py")
    if implementations.get(module_path.name) != sha256_file(module_path):
        raise SystemExit("Complementary-oversight analysis module differs from amendment")
    payload = read_json(args.input)
    if payload.get("packet_manifest_sha256") != sha256_file(args.packet_manifest):
        raise SystemExit("Outcome file does not bind the frozen packet manifest")
    if payload.get("assignment_manifest_sha256") != sha256_file(
        args.assignment_manifest
    ):
        raise SystemExit("Outcome file does not bind the frozen assignment manifest")
    rows = [
        OversightFactorialOutcome(**row) for row in (payload.get("records") or [])
    ]
    result = analyze_oversight_factorial(rows)
    result.update(
        {
            "input_sha256": sha256_file(args.input),
            "packet_manifest_sha256": sha256_file(args.packet_manifest),
            "assignment_manifest_sha256": sha256_file(args.assignment_manifest),
            "analysis_amendment_sha256": amendment_hash,
        }
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
