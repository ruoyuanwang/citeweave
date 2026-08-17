from __future__ import annotations

import argparse
from pathlib import Path

from citeweave.formal_statistics import (
    FormalStatisticsError,
    analyze_formal_experiment,
    write_formal_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the frozen multi-topic experiment with topic-cluster bootstrap "
            "confidence intervals and Holm-adjusted graph comparisons."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    try:
        summary = analyze_formal_experiment(
            args.manifest,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_formal_statistics(summary, args.output_dir)
    except FormalStatisticsError as exc:
        raise SystemExit(f"Formal statistics refused the inputs: {exc}") from exc
    print(
        f"Analyzed {summary['topic_clusters']} topic clusters; outputs written to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
