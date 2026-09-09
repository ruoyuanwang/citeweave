from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.graph_necessity_audit import audit_graph_necessity_benchmarks
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_graph_necessity_benchmarks(args.benchmark_root)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit("Graph-necessity audit failed")


if __name__ == "__main__":
    main()
