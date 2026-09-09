from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-input-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()
    if read_json(args.protocol_freeze).get("sha256") != sha256_file(args.protocol):
        raise SystemExit("Article-compiler protocol differs from freeze")
    datasets = []
    for dataset_id in args.dataset:
        writer_input = args.writer_input_root / dataset_id / "writer_input.json"
        if read_json(writer_input).get("dataset_id") != dataset_id:
            raise ValueError(f"Writer-input dataset mismatch: {dataset_id}")
        datasets.append(
            {
                "dataset_id": dataset_id,
                "writer_input": str(writer_input.resolve()),
                "writer_input_sha256": sha256_file(writer_input),
                "output_dir": str((args.output_root / dataset_id).resolve()),
            }
        )
    plan = {
        "schema_version": 1,
        "status": "post_result_article_compiler_development_plan",
        "confirmatory_replacement": False,
        "protocol_sha256": sha256_file(args.protocol),
        "datasets": datasets,
        "section_calls_per_dataset": 7,
        "maximum_provider_calls": 7 * len(datasets),
    }
    write_json(args.output_plan, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
