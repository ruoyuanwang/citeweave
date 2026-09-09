from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_claim_review import validate_article_claim_review_protocol
from citeweave.article_review_revision import (
    apply_controlled_paragraph_revisions,
    compile_article_revision_worklist,
    prepare_article_adjudication,
    resolve_article_claim_reviews,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--machine-plan", type=Path, required=True)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/article_claim_review_amendment_001_integrity.yml"
        ),
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/article_claim_review_amendment_001_integrity_freeze.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    adjudicate = subparsers.add_parser("prepare-adjudication")
    adjudicate.add_argument("--primary-root", type=Path, required=True)
    adjudicate.add_argument("--output-root", type=Path, required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--primary-root", type=Path, required=True)
    resolve.add_argument("--adjudication-root", type=Path)
    resolve.add_argument("--output", type=Path, required=True)
    compile_parser = subparsers.add_parser("compile-worklist")
    compile_parser.add_argument("--primary-root", type=Path, required=True)
    compile_parser.add_argument("--resolved", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply-revisions")
    apply_parser.add_argument("--worklist", type=Path, required=True)
    apply_parser.add_argument("--replacements", type=Path, required=True)
    apply_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    validate_article_claim_review_protocol(
        args.protocol,
        args.protocol_freeze,
        args.machine_plan,
        args.amendment,
        args.amendment_freeze,
    )
    if args.command == "prepare-adjudication":
        result = prepare_article_adjudication(args.primary_root, output_root=args.output_root)
    elif args.command == "resolve":
        result = resolve_article_claim_reviews(
            args.primary_root,
            adjudication_root=args.adjudication_root,
            output_path=args.output,
        )
    elif args.command == "compile-worklist":
        result = compile_article_revision_worklist(
            args.primary_root,
            args.resolved,
            args.machine_plan,
            output_path=args.output,
        )
    else:
        result = apply_controlled_paragraph_revisions(
            args.worklist, args.replacements, output_root=args.output_root
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
