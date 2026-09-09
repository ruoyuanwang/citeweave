from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.author_graph_sensitivity import audit_author_graph_sensitivity
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, action="append", required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.workspace_root:
        for workspace in sorted(root.iterdir()):
            corrected = args.correction_root / workspace.name
            if not (corrected / "repair_receipt.json").is_file():
                continue
            benchmark = args.benchmark_root / workspace.name / "benchmark.json"
            row = audit_author_graph_sensitivity(
                workspace, corrected, benchmark if benchmark.is_file() else None
            )
            rows.append(row)
            print(
                json.dumps(
                    {"dataset_id": row["dataset_id"], "task_comparisons": row["task_comparisons"]}
                ),
                flush=True,
            )
    write_json(args.output, {"schema_version": 1, "datasets": rows})


if __name__ == "__main__":
    main()
