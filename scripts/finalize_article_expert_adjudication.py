from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_expert_collection import (
    finalize_article_expert_adjudication,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--adjudication-collection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_article_expert_adjudication(
        args.primary_validation,
        args.adjudication_collection_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
