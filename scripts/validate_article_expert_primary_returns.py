from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_expert_collection import (
    validate_article_expert_primary_returns,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_article_expert_primary_returns(
        args.collection_manifest,
        args.protocol,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
