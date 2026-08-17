from __future__ import annotations

import argparse
import json
from pathlib import Path

from exchange_adaptive_blind_packets import import_results

from citeweave.adaptive_assignment_conductor import (
    prepare_assignments,
    validate_and_import,
    validate_complete_results,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "experiments" / "formal_adaptive_review"
DEFAULT_BATCH_ROOT = ROOT / "experiments" / "formal_adaptive_exchange"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare isolated Codex Human-Proxy/Evaluation assignments and admit "
            "only an exact, fully validated adaptive blind-result batch. This "
            "conductor never calls a model and never creates Judge results."
        )
    )
    parser.add_argument(
        "action",
        choices=["prepare", "validate-results", "import-results"],
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--assignment-root", type=Path)
    args = parser.parse_args()
    batch_root = args.batch_root.resolve()
    assignment_root = (
        args.assignment_root.resolve()
        if args.assignment_root is not None
        else batch_root / "assignments"
    )
    try:
        if args.action == "prepare":
            result = prepare_assignments(
                batch_root=batch_root,
                assignment_root=assignment_root,
            )
        elif args.action == "validate-results":
            result = validate_complete_results(
                batch_root=batch_root,
                assignment_root=assignment_root,
            )
        else:
            result = validate_and_import(
                run_root=args.run_root.resolve(),
                batch_root=batch_root,
                assignment_root=assignment_root,
                importer=import_results,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"Adaptive assignment conductor refused the exchange: {exc}"
        ) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
