from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.panel_terminal_audit import audit_panel_terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--maximum-parse-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_panel_terminal(
        args.plan, maximum_parse_attempts=args.maximum_parse_attempts
    )
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] == "invalid":
        raise SystemExit("Formal panel terminal audit found an integrity failure")


if __name__ == "__main__":
    main()
