from __future__ import annotations

import argparse
from pathlib import Path

from citeweave.citecalibrator_benchmark import build_pilot_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--robustness-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_pilot_benchmark(
        plan_path=args.plan,
        analysis_path=args.analysis,
        robustness_root=args.robustness_root,
        output_dir=args.output_dir,
    )
    print(
        f"Built {manifest['cases']} pilot cases from {manifest['articles']} development articles"
    )


if __name__ == "__main__":
    main()
