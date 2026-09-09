from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.source_review_outcomes import (
    build_source_adjudicator_commitment,
    finalize_source_warmup,
    prepare_source_adjudication_panel,
    validate_source_outcome_protocol,
    validate_source_primary_returns,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("commit-adjudicators", "prepare-adjudication", "finalize"),
    )
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--reviewer-roster", type=Path)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--primary-validation", type=Path)
    parser.add_argument("--adjudication-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol.yml"
        ),
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol_freeze.json"
        ),
    )
    parser.add_argument(
        "--execution-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity.yml"
        ),
    )
    parser.add_argument(
        "--execution-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity_freeze.json"
        ),
    )
    parser.add_argument(
        "--outcome-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_002_closed_loop.yml"
        ),
    )
    parser.add_argument(
        "--outcome-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_002_closed_loop_freeze.json"
        ),
    )
    args = parser.parse_args()
    validate_source_outcome_protocol(
        protocol_path=args.protocol,
        protocol_freeze_path=args.protocol_freeze,
        execution_amendment_path=args.execution_amendment,
        execution_amendment_freeze_path=args.execution_amendment_freeze,
        outcome_amendment_path=args.outcome_amendment,
        outcome_amendment_freeze_path=args.outcome_amendment_freeze,
    )

    if args.stage == "commit-adjudicators":
        if args.reviewer_roster is None:
            raise SystemExit("--reviewer-roster is required for commit-adjudicators")
        result = build_source_adjudicator_commitment(
            args.assignment_root,
            args.reviewer_roster,
            output_path=args.commitment,
            seed=args.seed,
        )
    elif args.stage == "prepare-adjudication":
        if args.primary_validation is None or args.adjudication_root is None:
            raise SystemExit(
                "--primary-validation and --adjudication-root are required for "
                "prepare-adjudication"
            )
        validate_source_primary_returns(
            args.assignment_root,
            args.commitment,
            output_path=args.primary_validation,
        )
        result = prepare_source_adjudication_panel(
            args.assignment_root,
            args.commitment,
            args.primary_validation,
            output_root=args.adjudication_root,
        )
    else:
        if (
            args.primary_validation is None
            or args.adjudication_root is None
            or args.output_root is None
        ):
            raise SystemExit(
                "--primary-validation, --adjudication-root, and --output-root are "
                "required for finalize"
            )
        result = finalize_source_warmup(
            args.assignment_root,
            args.commitment,
            args.primary_validation,
            args.adjudication_root,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
