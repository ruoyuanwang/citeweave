from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.citecalibrator_benchmark import score_predictions, summarize_gold
from citeweave.io import write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {row["case_id"]: row for row in _read_jsonl(args.cases)}
    gold = _read_jsonl(args.gold)
    enriched_gold = [{**cases[row["case_id"]], **row} for row in gold]
    splits = sorted({str(row.get("benchmark_split")) for row in cases.values()})
    natural_prevalence_permitted = all(
        row.get("eligibility", {}).get("eligible_for_natural_prevalence") is True
        for row in cases.values()
    )
    result = {
        "schema_version": 1,
        "status": "citecalibrator_need_assessment_analyzed",
        "benchmark_splits": splits,
        "natural_prevalence_interpretation_permitted": natural_prevalence_permitted,
        "gold": summarize_gold(enriched_gold),
        "conditions": {},
    }
    for path in args.predictions:
        rows = _read_jsonl(path)
        condition = str(rows[0].get("condition") or path.parent.name) if rows else path.parent.name
        result["conditions"][condition] = score_predictions(gold, rows)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
