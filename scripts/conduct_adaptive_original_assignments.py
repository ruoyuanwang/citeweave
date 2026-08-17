from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.adaptive_original_assignment_conductor import (
    AdaptiveOriginalAssignmentError,
    prepare_original_evaluation_assignment,
    prepare_semantic_audit_assignment,
    validate_and_import_original_evaluation,
    validate_and_import_semantic_audit,
    validate_original_evaluation_results,
    validate_semantic_audit_results,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or admit fail-closed Codex assignments for untouched-original "
            "evaluation and its independent semantic audit. This command never "
            "generates a judge result or calls a model/API."
        )
    )
    parser.add_argument(
        "action",
        choices=[
            "prepare-original",
            "validate-original",
            "import-original",
            "prepare-semantic",
            "validate-semantic",
            "import-semantic",
        ],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "formal_adaptive_original_evaluation",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=(
            ROOT / "experiments" / "formal_adaptive_original_evaluation_exchange"
        ),
    )
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--controller-manifest", type=Path)
    parser.add_argument("--assignment-root", type=Path, required=True)
    args = parser.parse_args()

    kwargs = {
        "batch_root": args.batch_root,
        "assignment_root": args.assignment_root,
    }
    try:
        if args.action == "prepare-original":
            result = prepare_original_evaluation_assignment(**kwargs)
        elif args.action == "validate-original":
            result = validate_original_evaluation_results(**kwargs)
        elif args.action == "import-original":
            result = validate_and_import_original_evaluation(
                output_root=args.output_root,
                **kwargs,
            )
        else:
            if args.audit_root is None or args.controller_manifest is None:
                parser.error(
                    f"{args.action} requires --audit-root and --controller-manifest"
                )
            semantic_kwargs = {
                **kwargs,
                "audit_root": args.audit_root,
                "controller_manifest_path": args.controller_manifest,
            }
            if args.action == "prepare-semantic":
                result = prepare_semantic_audit_assignment(**semantic_kwargs)
            elif args.action == "validate-semantic":
                result = validate_semantic_audit_results(**semantic_kwargs)
            else:
                result = validate_and_import_semantic_audit(**semantic_kwargs)
    except (
        AdaptiveOriginalAssignmentError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(f"Adaptive original assignment conductor refused: {exc}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
