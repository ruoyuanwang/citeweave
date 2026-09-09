from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.canonical_identity_audit import audit_canonical_identities
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for root in args.workspace_root:
        for workspace in sorted(root.iterdir()):
            if not (workspace / "canonical" / "authorships.parquet").is_file():
                continue
            result = audit_canonical_identities(workspace, args.benchmark_root)
            records.append(result)
            print(
                json.dumps(
                    {
                        "dataset_id": result["dataset_id"],
                        "status": result["status"],
                        "raw_distinct_names_merged": result["raw_distinct_names_merged"],
                        "coauthor_edges_incident_to_placeholder": result[
                            "coauthor_edges_incident_to_placeholder"
                        ],
                    }
                ),
                flush=True,
            )
    write_json(args.output, {"schema_version": 1, "datasets": records})


if __name__ == "__main__":
    main()
