from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from citeweave.report_conditions import ReportRunConfig, run_report_condition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run immutable formal report-generation experiment conditions."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/formal_reports"))
    parser.add_argument(
        "--condition",
        action="append",
        choices=["structured_one_shot", "citeweave_full"],
        required=True,
    )
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = []
    for condition in dict.fromkeys(args.condition):
        results.append(
            run_report_condition(
                evidence_path=args.evidence,
                output_root=args.output_root,
                config=ReportRunConfig(
                    dataset_id=args.dataset_id,
                    condition=condition,
                    model=args.model,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                ),
                resume=args.resume,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
