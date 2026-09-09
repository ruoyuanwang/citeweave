from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.confirmation_query_review import finalize_query_relevance_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-validation", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--adjudication-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_query_relevance_selection(
        args.primary_validation,
        args.collection_manifest,
        adjudication_manifest_path=args.adjudication_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
