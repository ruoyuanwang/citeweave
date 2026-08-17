from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.experiment_benchmark import (
    load_json_records,
    save_benchmark_result,
    score_graph_predictions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore saved LLM predictions without calling the model again."
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--condition", action="append", default=["graph_rag", "no_graph"]
    )
    args = parser.parse_args()
    safe_model = args.model.replace("/", "_")

    for dataset in args.dataset:
        run_dir = REPOSITORY_ROOT / "experiments" / "runs" / dataset
        qa_path = (
            REPOSITORY_ROOT
            / "experiments"
            / "workspaces"
            / dataset
            / "evidence"
            / "graph_qa_benchmark.json"
        )
        items = load_json_records(qa_path)
        for condition in dict.fromkeys(args.condition):
            output = run_dir / f"llm_{condition}_{safe_model}.json"
            payload = json.loads(output.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") or {}
            predictions = [
                record["parsed_prediction"]
                for record in metadata.get("raw_records", [])
            ]
            result = score_graph_predictions(items, predictions)
            save_benchmark_result(
                result,
                output,
                condition=payload.get("condition", condition),
                metadata=metadata,
            )
            metrics = result["metrics"]
            print(
                f"{dataset} {condition}: "
                f"accuracy={metrics['exact_answer_accuracy']:.3f}, "
                f"UCR={metrics['unsupported_claim_rate']:.3f}, "
                f"format_failure={metrics['format_failure_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
