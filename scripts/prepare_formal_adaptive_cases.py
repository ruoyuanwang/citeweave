from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.formal_adaptive_review import assemble_formal_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze formal report and graph outputs as adaptive-review cases."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = assemble_formal_cases(args.spec.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "cases": len(cases),
                "output": str(args.output.resolve()),
                "datasets": list(dict.fromkeys(case.dataset_id for case in cases)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
