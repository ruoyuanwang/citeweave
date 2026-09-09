from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from citeweave.citecalibrator_benchmark import deterministic_rule_judge
from citeweave.io import sha256_file, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = _read_jsonl(args.cases)
    predictions = []
    for case in cases:
        started = perf_counter()
        prediction = deterministic_rule_judge(case)
        prediction["latency_seconds"] = perf_counter() - started
        predictions.append(prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "predictions.jsonl"
    write_jsonl(output, predictions)
    counts: dict[str, int] = {}
    for row in predictions:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "citecalibrator_rule_baseline_complete",
            "condition": "deterministic_rules_v1",
            "cases": len(cases),
            "action_counts": counts,
            "input_cases_sha256": sha256_file(args.cases),
            "predictions": str(output.resolve()),
            "predictions_sha256": sha256_file(output),
        },
    )
    print(f"Scored {len(cases)} cases: {counts}")


if __name__ == "__main__":
    main()
