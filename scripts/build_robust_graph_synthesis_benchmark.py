from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.robust_graph_synthesis import build_benchmark_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-budget", type=int, default=20)
    args = parser.parse_args()
    result = build_benchmark_panel(
        args.source_root,
        args.output_root,
        record_budget=args.record_budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
