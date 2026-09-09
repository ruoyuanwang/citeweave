from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.confirmation_panel import build_confirmation_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--review-collection", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_confirmation_panel(
        args.protocol,
        args.protocol_freeze,
        args.selection,
        args.review_collection,
        args.candidate_audit,
        args.initialization,
        args.tokenizer_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
