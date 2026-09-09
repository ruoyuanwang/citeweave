from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_generation_v3_1 import (
    PILOT_CONDITIONS,
    PROMPT_VERSION,
    build_checkpoint_article_request,
)
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
        raise SystemExit("Article-v3.1 pilot protocol differs from freeze")
    cells = []
    for dataset_id in args.dataset:
        writer_input_path = args.writer_input_root / dataset_id / "writer_input.json"
        writer_input = read_json(writer_input_path)
        if writer_input.get("dataset_id") != dataset_id:
            raise ValueError(f"Writer-input dataset mismatch: {dataset_id}")
        for condition in PILOT_CONDITIONS:
            output_dir = args.output_root / dataset_id / condition
            request, argument_plan = build_checkpoint_article_request(
                writer_input, condition=condition
            )
            request_path = output_dir / "request.json"
            write_json(request_path, request)
            argument_plan_path = None
            argument_plan_sha256 = None
            if argument_plan is not None:
                argument_plan_path = output_dir / "argument_plan.json"
                write_json(argument_plan_path, argument_plan)
                argument_plan_sha256 = sha256_file(argument_plan_path)
            cells.append(
                {
                    "cell_id": f"{dataset_id}:{condition}",
                    "dataset_id": dataset_id,
                    "condition": condition,
                    "writer_input": str(writer_input_path.resolve()),
                    "writer_input_sha256": sha256_file(writer_input_path),
                    "request": str(request_path.resolve()),
                    "request_sha256": sha256_file(request_path),
                    "argument_plan": str(argument_plan_path.resolve())
                    if argument_plan_path
                    else None,
                    "argument_plan_sha256": argument_plan_sha256,
                    "output_dir": str(output_dir.resolve()),
                    "generation_requests_allowed": 1,
                }
            )
    plan = {
        "schema_version": 1,
        "status": "post_result_development_pilot_plan",
        "confirmatory_replacement": False,
        "prompt_version": PROMPT_VERSION,
        "protocol_sha256": sha256_file(args.protocol),
        "datasets": list(args.dataset),
        "conditions": list(PILOT_CONDITIONS),
        "cells": cells,
    }
    write_json(args.output_plan, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
