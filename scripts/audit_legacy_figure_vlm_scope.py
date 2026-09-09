from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.legacy_figure_vlm_audit import audit_legacy_figure_vlm_scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resolved-judgments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_legacy_figure_vlm_scope(
        args.formal_runs_root,
        args.run_id,
        args.resolved_judgments,
    )
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["status"] != "scope_confirmed":
        raise SystemExit("Legacy Figure/VLM scope audit did not confirm the expected scope")


if __name__ == "__main__":
    main()
