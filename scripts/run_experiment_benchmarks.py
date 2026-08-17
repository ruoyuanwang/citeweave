from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from citeweave.experiment_benchmark import (
    load_json_records,
    no_context_abstain_predictions,
    no_context_forced_predictions,
    oracle_graph_predictions,
    save_benchmark_result,
    score_graph_predictions,
)
from citeweave.experiment_validation import (
    validate_experiment_project,
    write_validation_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score deterministic CiteWeave benchmarks.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    registry = yaml.safe_load(
        (REPOSITORY_ROOT / "experiments" / "datasets.yml").read_text(encoding="utf-8")
    )
    entry = next(item for item in registry["datasets"] if item["id"] == args.dataset)
    workspace = REPOSITORY_ROOT / "experiments" / "workspaces" / args.dataset
    run_dir = REPOSITORY_ROOT / "experiments" / "runs" / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_experiment_project(
        workspace,
        year_from=entry["year_from"],
        year_to=entry["year_to"],
    )
    write_validation_report(validation, run_dir / "pipeline_validity.json")

    items = load_json_records(workspace / "evidence" / "graph_qa_benchmark.json")
    predictions = oracle_graph_predictions(items)
    score = score_graph_predictions(items, predictions)
    save_benchmark_result(
        score,
        run_dir / "graph_oracle.json",
        condition="structured_graph_oracle",
        metadata={
            "dataset_id": args.dataset,
            "purpose": "Benchmark self-test and upper-bound reference; not an LLM result.",
        },
    )
    for condition, baseline_predictions in (
        ("no_context_abstain_stress_test", no_context_abstain_predictions(items)),
        ("no_context_forced_answer_stress_test", no_context_forced_predictions(items)),
    ):
        baseline_score = score_graph_predictions(items, baseline_predictions)
        save_benchmark_result(
            baseline_score,
            run_dir / f"{condition}.json",
            condition=condition,
            metadata={
                "dataset_id": args.dataset,
                "purpose": (
                    "Deterministic metric stress test; not an LLM ablation result."
                ),
            },
        )
    print(
        f"{args.dataset}: pipeline={'PASS' if validation['passed'] else 'FAIL'}, "
        f"graph_oracle_accuracy={score['metrics']['exact_answer_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
