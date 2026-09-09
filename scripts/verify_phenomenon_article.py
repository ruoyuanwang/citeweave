from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_verification import verify_phenomenon_article


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_phenomenon_article(
        article_path=args.article,
        cards_path=args.cards,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
