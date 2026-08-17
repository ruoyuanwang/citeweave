from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.harvest_repair import (
    HarvestRepairError,
    inspect_openalex_terminal_cursor_repair,
    repair_openalex_terminal_cursors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed repair for OpenAlex slices that were incorrectly marked complete "
            "after a full page with a non-null next cursor. The default is read-only."
        )
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expected-project-id", required=True)
    parser.add_argument(
        "--slice",
        action="append",
        required=True,
        dest="slices",
        help="Exact slice ID to reopen; repeat for every eligible slice.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the validated repair and write immutable audit artifacts.",
    )
    args = parser.parse_args()
    try:
        if args.execute:
            result = repair_openalex_terminal_cursors(
                args.workspace,
                expected_project_id=args.expected_project_id,
                slice_ids=tuple(args.slices),
            )
        else:
            result = inspect_openalex_terminal_cursor_repair(
                args.workspace,
                expected_project_id=args.expected_project_id,
                slice_ids=tuple(args.slices),
            )
    except HarvestRepairError as exc:
        raise SystemExit(f"OpenAlex terminal-cursor repair refused: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
