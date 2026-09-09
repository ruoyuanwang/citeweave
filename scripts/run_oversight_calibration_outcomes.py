from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_calibration_outcomes import (
    finalize_calibration_observations,
    prepare_calibration_adjudication,
    validate_calibration_primary_returns,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-primary")
    validate.add_argument("--assignment-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    adjudicate = subparsers.add_parser("prepare-adjudication")
    adjudicate.add_argument("--assignment-root", type=Path, required=True)
    adjudicate.add_argument("--primary-validation", type=Path, required=True)
    adjudicate.add_argument("--output-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--assignment-root", type=Path, required=True)
    finalize.add_argument("--packet-root", type=Path, required=True)
    finalize.add_argument("--reviewer-roster", type=Path, required=True)
    finalize.add_argument("--primary-validation", type=Path, required=True)
    finalize.add_argument("--adjudication-root", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-primary":
        result = validate_calibration_primary_returns(
            args.assignment_root, output_path=args.output
        )
    elif args.command == "prepare-adjudication":
        result = prepare_calibration_adjudication(
            args.assignment_root,
            args.primary_validation,
            output_root=args.output_root,
        )
    else:
        result = finalize_calibration_observations(
            args.assignment_root,
            args.packet_root,
            args.reviewer_roster,
            args.primary_validation,
            args.adjudication_root,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
