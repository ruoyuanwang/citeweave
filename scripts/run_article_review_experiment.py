from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_claim_review import validate_article_claim_review_protocol
from citeweave.article_review_experiment import (
    analyze_article_review_routing,
    analyze_revision_evaluation,
    build_article_review_routing_plan,
    build_revision_evaluation_packets,
)
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("experiments/article_quality_v2")
    parser.add_argument("--protocol", type=Path, default=base / "article_claim_review_protocol.yml")
    parser.add_argument(
        "--protocol-freeze", type=Path, default=base / "article_claim_review_protocol_freeze.json"
    )
    parser.add_argument(
        "--machine-plan", type=Path, default=base / "machine_generation_v1/plan.json"
    )
    parser.add_argument(
        "--amendment", type=Path, default=base / "article_claim_review_amendment_001_integrity.yml"
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=base / "article_claim_review_amendment_001_integrity_freeze.json",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("routing-plan")
    route.add_argument("--primary-root", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("routing-analysis")
    replay.add_argument("--routing-plan", type=Path, required=True)
    replay.add_argument("--resolved", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    packets = commands.add_parser("revision-packets")
    for flag in (
        "primary-root",
        "worklist",
        "replacements",
        "revision-manifest",
        "evaluator-roster",
        "output-root",
    ):
        packets.add_argument(f"--{flag}", type=Path, required=True)
    analysis = commands.add_parser("revision-analysis")
    analysis.add_argument("--evaluation-root", type=Path, required=True)
    analysis.add_argument("--resolved", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_article_claim_review_protocol(
        args.protocol,
        args.protocol_freeze,
        args.machine_plan,
        args.amendment,
        args.amendment_freeze,
    )
    if args.command == "routing-plan":
        result = build_article_review_routing_plan(args.primary_root, output_path=args.output)
    elif args.command == "routing-analysis":
        result = analyze_article_review_routing(
            args.routing_plan, args.resolved, output_path=args.output
        )
    elif args.command == "revision-packets":
        result = build_revision_evaluation_packets(
            args.primary_root,
            args.worklist,
            args.replacements,
            args.revision_manifest,
            args.evaluator_roster,
            output_root=args.output_root,
        )
    else:
        result = analyze_revision_evaluation(args.evaluation_root, args.resolved)
        if result["primary_test"]["topics"] != 8:
            raise ValueError("Formal revision-effect analysis requires exactly eight topics")
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
