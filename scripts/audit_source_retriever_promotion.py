from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.source_retriever_promotion import (
    SourceRetrievalReplayRow,
    audit_source_retriever_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        SourceRetrievalReplayRow(
            **{
                **row,
                "training_datasets": tuple(row["training_datasets"]),
            }
        )
        for row in (
            json.loads(line)
            for line in args.replay.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    report = audit_source_retriever_promotion(rows)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
