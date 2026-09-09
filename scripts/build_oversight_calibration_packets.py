from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_calibration import (
    build_multidimensional_calibration_packets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases-per-issue-per-dataset", type=int, default=3)
    args = parser.parse_args()
    result = build_multidimensional_calibration_packets(
        prototype_manifest_path=args.prototype_manifest,
        output_root=args.output_root,
        cases_per_issue_per_dataset=args.cases_per_issue_per_dataset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
