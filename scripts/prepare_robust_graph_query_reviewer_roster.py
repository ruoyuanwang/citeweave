from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.confirmation_query_review import (
    build_query_reviewer_roster_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_query_reviewer_roster_template(
        args.protocol, args.candidate_audit, output_path=args.output
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
