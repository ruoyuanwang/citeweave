from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import read_json, write_json
from citeweave.operator_verification import audit_fault_injection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_fault_injection(
        read_json(args.benchmark),
        workspace=args.workspace.resolve(),
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
