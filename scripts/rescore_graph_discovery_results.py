from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.graph_discovery import score_discovery_response
from citeweave.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = read_json(args.benchmark)
    tasks = {task["item_id"]: task for task in benchmark["tasks"]}
    payload = read_json(args.results)
    records = []
    for record in payload["records"]:
        if record.get("status", "complete") != "complete":
            continue
        rescored = dict(record)
        rescored["score"] = score_discovery_response(
            tasks[record["item_id"]], record["response"]
        )
        records.append(rescored)
    conditions = sorted({record["condition"] for record in records})
    summary = {}
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        summary[condition] = {
            "items": len(subset),
            "answer_accuracy": sum(record["score"]["answer_exact"] for record in subset)
            / len(subset),
            "mean_evidence_f1": sum(record["score"]["evidence_f1"] for record in subset)
            / len(subset),
            "by_scale": {
                scale: {
                    "items": len(scale_records),
                    "answer_accuracy": sum(
                        record["score"]["answer_exact"] for record in scale_records
                    )
                    / len(scale_records),
                }
                for scale in sorted({record["scale"] for record in subset})
                if (scale_records := [record for record in subset if record["scale"] == scale])
            },
            "by_task_type": {
                task_type: {
                    "items": len(task_records),
                    "answer_accuracy": sum(
                        record["score"]["answer_exact"] for record in task_records
                    )
                    / len(task_records),
                }
                for task_type in sorted({record["task_type"] for record in subset})
                if (
                    task_records := [
                        record for record in subset if record["task_type"] == task_type
                    ]
                )
            },
        }
    output = {
        "schema_version": 1,
        "benchmark": str(args.benchmark.resolve()),
        "raw_results": str(args.results.resolve()),
        "records": records,
        "summary": summary,
    }
    write_json(args.output, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
