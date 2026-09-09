from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_calibration_assignment import build_calibration_assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--reviewer-roster", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    result = build_calibration_assignment(
        packet_root=args.packet_root,
        reviewer_roster_path=args.reviewer_roster,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
