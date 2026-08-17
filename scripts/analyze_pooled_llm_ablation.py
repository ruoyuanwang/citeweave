from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from citeweave.ablation_analysis import analyze_paired_ablation
from citeweave.io import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = [
    "crispr_editing_2018_2020",
    "quantum_machine_learning_2019_2021",
    "climate_adaptation_2018_2020",
]


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool untouched-topic LLM ablations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    datasets = args.dataset or DEFAULT_DATASETS
    safe_model = args.model.replace("/", "_")
    graph_rows: list[dict[str, Any]] = []
    no_graph_rows: list[dict[str, Any]] = []
    topic_results: dict[str, Any] = {}
    topic_p_values: dict[str, float] = {}

    for dataset in datasets:
        run_dir = REPOSITORY_ROOT / "experiments" / "runs" / dataset
        graph = json.loads(
            (run_dir / f"llm_graph_rag_{safe_model}.json").read_text(encoding="utf-8")
        )
        no_graph = json.loads(
            (run_dir / f"llm_no_graph_{safe_model}.json").read_text(encoding="utf-8")
        )
        for row in graph["rows"]:
            row["dataset_id"] = dataset
        for row in no_graph["rows"]:
            row["dataset_id"] = dataset
        graph_rows.extend(graph["rows"])
        no_graph_rows.extend(no_graph["rows"])
        topic = analyze_paired_ablation(
            graph,
            no_graph,
            bootstrap_samples=args.bootstrap_samples,
            seed=42,
        )
        topic_results[dataset] = topic
        topic_p_values[dataset] = topic["mcnemar_exact"]["two_sided_p_value"]

    pooled = analyze_paired_ablation(
        {"condition": "graph_rag", "rows": graph_rows},
        {"condition": "no_graph", "rows": no_graph_rows},
        bootstrap_samples=args.bootstrap_samples,
        seed=42,
    )
    result = {
        "analysis_version": 1,
        "model": args.model,
        "datasets": datasets,
        "development_prompt_topic_excluded": "rag_graph_2022_2025",
        "pooled": pooled,
        "by_dataset": topic_results,
        "holm_adjusted_topic_p_values": _holm_adjust(topic_p_values),
    }
    output = (
        REPOSITORY_ROOT
        / "experiments"
        / "runs"
        / f"llm_ablation_pooled_{safe_model}.json"
    )
    write_json(output, result)
    effects = pooled["effects"]
    mcnemar = pooled["mcnemar_exact"]
    stratified_accuracy = effects[
        "accuracy_difference_topic_stratified_bootstrap_95_ci"
    ]
    stratified_ucr = effects[
        "unsupported_claim_rate_reduction_topic_stratified_bootstrap_95_ci"
    ]
    lines = [
        "# Pooled LLM Graph-Grounding Ablation",
        "",
        f"- Model: `{args.model}`",
        f"- Topics: {len(datasets)}",
        f"- Paired items: {pooled['paired_items']}",
        "- Prompt-development topic excluded from pooling: `rag_graph_2022_2025`",
        (
            f"- Accuracy: graph {pooled['graph_metrics']['accuracy']:.3f}; "
            f"no graph {pooled['no_graph_metrics']['accuracy']:.3f}"
        ),
        (
            f"- Accuracy difference: {effects['accuracy_difference']:.3f} "
            f"(topic-stratified bootstrap 95% CI {stratified_accuracy[0]:.3f} "
            f"to {stratified_accuracy[1]:.3f})"
        ),
        (
            f"- Unsupported-claim rate: graph "
            f"{pooled['graph_metrics']['unsupported_claim_rate']:.3f}; "
            f"no graph {pooled['no_graph_metrics']['unsupported_claim_rate']:.3f}"
        ),
        (
            f"- Statement-claim coverage: graph "
            f"{pooled['graph_metrics']['statement_claim_coverage']:.3f}; "
            f"no graph {pooled['no_graph_metrics']['statement_claim_coverage']:.3f}"
        ),
        (
            f"- Format-failure rate: graph "
            f"{pooled['graph_metrics']['format_failure_rate']:.3f}; "
            f"no graph {pooled['no_graph_metrics']['format_failure_rate']:.3f}"
        ),
        (
            f"- Structured unsupported-answer rate: graph "
            f"{pooled['graph_metrics']['structured_unsupported_answer_rate']:.3f}; "
            f"no graph "
            f"{pooled['no_graph_metrics']['structured_unsupported_answer_rate']:.3f}"
        ),
        (
            f"- UCR reduction: {effects['unsupported_claim_rate_reduction']:.3f} "
            f"(topic-stratified bootstrap 95% CI {stratified_ucr[0]:.3f} "
            f"to {stratified_ucr[1]:.3f})"
        ),
        (
            f"- Exact pooled McNemar test: graph-only correct "
            f"{mcnemar['graph_only_correct']}, no-graph-only correct "
            f"{mcnemar['no_graph_only_correct']}, "
            f"p={mcnemar['two_sided_p_value']:.6g}"
        ),
        "",
        "## Holm-adjusted topic tests",
        "",
        "| Dataset | Adjusted p |",
        "|---|---:|",
        *[
            f"| {dataset} | {value:.6g} |"
            for dataset, value in result["holm_adjusted_topic_p_values"].items()
        ],
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
