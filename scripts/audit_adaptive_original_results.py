from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.adaptive_semantic_audit import (
    import_semantic_audit_exchange,
    prepare_semantic_audit_exchange,
    validate_semantic_audit_archive,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and validate a condition-blind, read-only semantic audit of "
            "adaptive original-evaluation packet/result pairs. No API is called."
        )
    )
    parser.add_argument("action", choices=["prepare", "import-results", "validate"])
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=ROOT / "experiments" / "formal_adaptive_original_evaluation_exchange",
    )
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--controller-manifest", type=Path)
    args = parser.parse_args()

    if args.action in {"prepare", "import-results"}:
        if args.audit_root is None or args.controller_manifest is None:
            parser.error(
                f"{args.action} requires --audit-root and --controller-manifest"
            )
        if args.action == "prepare":
            result = prepare_semantic_audit_exchange(
                batch_root=args.batch_root,
                audit_root=args.audit_root,
                controller_manifest_path=args.controller_manifest,
            )
        else:
            result = import_semantic_audit_exchange(
                batch_root=args.batch_root,
                audit_root=args.audit_root,
                controller_manifest_path=args.controller_manifest,
            )
    else:
        result = validate_semantic_audit_archive(args.batch_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
