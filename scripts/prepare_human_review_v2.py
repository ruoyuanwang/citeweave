from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.review_packets_v2 import build_review_packets

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "cross_topic_panel_20260820",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "benchmarks",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "human_review_v2" / "cross_topic_panel_20260820",
    )
    args = parser.parse_args()
    manifest = build_review_packets(
        panel_root=args.panel_root,
        benchmark_root=args.benchmark_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "factual_packets": manifest["factual_packets"],
                "semantic_packets": manifest["semantic_packets"],
                "reviewers": manifest["reviewers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
