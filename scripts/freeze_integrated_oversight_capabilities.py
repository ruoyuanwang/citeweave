from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_capability_freeze import (
    freeze_integrated_reviewer_capabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-observations", type=Path, required=True)
    parser.add_argument("--calibration-observations", type=Path, required=True)
    parser.add_argument("--reviewer-roster", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_integrated_reviewer_capabilities(
        source_observations_path=args.source_observations,
        calibration_observations_path=args.calibration_observations,
        reviewer_roster_path=args.reviewer_roster,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
