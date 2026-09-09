from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_complementarity_analysis import (
    analyze_complementarity_cases,
    load_complementarity_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--final-outcomes", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite complementarity analysis: {args.output}")
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    freeze = read_json(args.amendment_freeze)
    if freeze.get("sha256") != sha256_file(args.amendment):
        raise RuntimeError("Complementarity amendment differs from its freeze")
    implementation = Path(amendment["implementation"]["analysis_module"]["path"])
    if sha256_file(implementation) != amendment["implementation"]["analysis_module"][
        "sha256"
    ]:
        raise RuntimeError("Complementarity implementation hash mismatch")
    cases = load_complementarity_cases(
        packet_manifest_path=args.packet_manifest,
        assignment_root=args.assignment_root,
        primary_validation_path=args.primary_validation,
        final_outcomes_path=args.final_outcomes,
    )
    result = analyze_complementarity_cases(cases)
    result["amendment_sha256"] = sha256_file(args.amendment)
    result["packet_manifest_sha256"] = sha256_file(args.packet_manifest)
    result["assignment_manifest_sha256"] = sha256_file(
        args.assignment_root / "assignment_manifest.json"
    )
    result["primary_validation_sha256"] = sha256_file(args.primary_validation)
    result["final_outcomes_sha256"] = sha256_file(args.final_outcomes)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
