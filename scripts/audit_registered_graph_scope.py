from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.graph_scope_audit import audit_registered_graph_scope
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, action="append", required=True)
    parser.add_argument("--workspace-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_registered_graph_scope(args.benchmark_root, args.workspace_root)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"datasets", "benchmark_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
