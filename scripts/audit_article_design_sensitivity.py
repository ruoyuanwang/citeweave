from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_design_sensitivity import build_article_sensitivity_report
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    result = build_article_sensitivity_report(
        simulations=args.simulations, seed=args.seed
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
