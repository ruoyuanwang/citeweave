from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.robust_graph_statistics import analyze_robust_graph_synthesis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_908)
    args = parser.parse_args()
    result = analyze_robust_graph_synthesis(
        args.construction_manifest,
        args.run_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
