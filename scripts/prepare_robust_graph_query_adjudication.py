from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.confirmation_query_review import build_query_review_adjudication


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_query_review_adjudication(
        args.primary_validation,
        args.collection_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
