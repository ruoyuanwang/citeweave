from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.oversight_heldout_packets import (
    freeze_heldout_case_selection,
    materialize_heldout_factorial_packets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("freeze-selection")
    select.add_argument("--primary-construction", type=Path, required=True)
    select.add_argument("--replication-construction", type=Path, required=True)
    select.add_argument("--primary-benchmark-root", type=Path, required=True)
    select.add_argument("--replication-benchmark-root", type=Path, required=True)
    select.add_argument("--primary-results-root", type=Path, required=True)
    select.add_argument("--replication-results-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=20260907)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--selection", type=Path, required=True)
    materialize.add_argument("--terminal-audit", type=Path, action="append", required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-selection":
        result = freeze_heldout_case_selection(
            primary_construction_manifest_path=args.primary_construction,
            replication_construction_manifest_path=args.replication_construction,
            primary_benchmark_root=args.primary_benchmark_root,
            replication_benchmark_root=args.replication_benchmark_root,
            primary_results_root=args.primary_results_root,
            replication_results_root=args.replication_results_root,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        result = materialize_heldout_factorial_packets(
            selection_path=args.selection,
            terminal_audit_paths=args.terminal_audit,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
