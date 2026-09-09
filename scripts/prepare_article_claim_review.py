from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_claim_review import (
    assess_article_claim_review_readiness,
    build_article_claim_review_packets,
    build_article_reviewer_roster_template,
    validate_article_claim_review_protocol,
)
from citeweave.article_review_experiment import build_article_review_routing_plan
from citeweave.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-plan", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/article_claim_review_amendment_002_shared_ui_compatibility.yml"
        ),
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/article_claim_review_amendment_002_shared_ui_compatibility_freeze.json"
        ),
    )
    parser.add_argument("--initialize-roster", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--readiness-output", type=Path)
    args = parser.parse_args()
    validate_article_claim_review_protocol(
        args.protocol,
        args.protocol_freeze,
        args.machine_plan,
        args.amendment,
        args.amendment_freeze,
    )
    if args.initialize_roster:
        plan = read_json(args.machine_plan)
        topics = sorted({row["dataset_id"] for row in plan["cells"]})
        result = build_article_reviewer_roster_template(topics, output_path=args.roster)
    elif args.audit:
        result = assess_article_claim_review_readiness(
            args.machine_plan,
            args.roster,
            output_path=args.readiness_output,
        )
    else:
        result = build_article_claim_review_packets(
            args.machine_plan,
            args.roster,
            output_root=args.output_root,
        )
        routing_path = args.output_root / "routing_plan.json"
        if not routing_path.exists():
            build_article_review_routing_plan(args.output_root, output_path=routing_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
