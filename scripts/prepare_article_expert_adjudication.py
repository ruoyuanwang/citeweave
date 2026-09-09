from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_expert_collection import (
    prepare_article_expert_adjudication,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--primary-collection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_article_expert_adjudication(
        args.primary_validation,
        args.primary_collection_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
