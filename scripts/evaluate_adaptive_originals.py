from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.adaptive_original_evaluation import (
    export_blind_batch,
    finalize_original_evaluation,
    import_blind_results,
    prepare_original_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, exchange, and score condition-blind evaluations of untouched "
            "pre-intervention candidates. This command never calls an API."
        )
    )
    parser.add_argument(
        "action",
        choices=["prepare", "export", "import-results", "finalize"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "formal_adaptive_original_evaluation",
    )
    parser.add_argument("--cases", type=Path)
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT / "experiments" / "human_references.yml",
    )
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--development-calibration", action="store_true")
    parser.add_argument("--rubric-version", default="adaptive-evaluation-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.action == "prepare":
        if args.cases is None:
            parser.error("prepare requires --cases")
        result = prepare_original_evaluation(
            cases_path=args.cases,
            reference_registry=args.references,
            output_root=args.output_root,
            experiment_mode=(
                "development_calibration"
                if args.development_calibration
                else "formal"
            ),
            rubric_version=args.rubric_version,
            seed=args.seed,
        )
    elif args.action == "export":
        if args.batch_root is None:
            parser.error("export requires --batch-root")
        result = export_blind_batch(args.output_root, args.batch_root)
    elif args.action == "import-results":
        if args.batch_root is None:
            parser.error("import-results requires --batch-root")
        result = import_blind_results(args.output_root, args.batch_root)
    else:
        result = finalize_original_evaluation(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
