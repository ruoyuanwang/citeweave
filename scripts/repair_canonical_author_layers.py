from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.canonical_author_repair import repair_author_layer
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.workspace_root:
        for workspace in sorted(root.iterdir()):
            if not (workspace / "canonical" / "authorships.parquet").is_file():
                continue
            row = repair_author_layer(workspace, args.output_root / workspace.name)
            rows.append(row)
            print(
                json.dumps({"dataset_id": row["dataset_id"], "graphs": row["graph_statistics"]}),
                flush=True,
            )
    write_json(args.output_root / "manifest.json", {"schema_version": 1, "datasets": rows})


if __name__ == "__main__":
    main()
