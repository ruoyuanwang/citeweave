from __future__ import annotations

import argparse
from pathlib import Path

from citeweave.citecalibrator_benchmark import build_controlled_challenge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_controlled_challenge(
        pilot_cases_path=args.pilot_cases,
        output_dir=args.output_dir,
    )
    print(
        f"Built {manifest['cases']} controlled cases from "
        f"{manifest['phenomena']} graph phenomena"
    )


if __name__ == "__main__":
    main()
