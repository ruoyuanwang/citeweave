from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_balanced_panel import (
    build_balanced_heldout_selection,
    materialize_balanced_heldout_packets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--base-selection", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=20260908)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--selection", type=Path, required=True)
    materialize.add_argument("--terminal-audit", type=Path, action="append", required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        result = build_balanced_heldout_selection(
            base_selection_path=args.base_selection,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        result = materialize_balanced_heldout_packets(
            selection_path=args.selection,
            terminal_audit_paths=args.terminal_audit,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
