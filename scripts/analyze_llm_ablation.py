from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.ablation_analysis import analyze_paired_ablation
from citeweave.io import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired graph/no-graph runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    run_dir = REPOSITORY_ROOT / "experiments" / "runs" / args.dataset
    safe_model = args.model.replace("/", "_")
    graph_path = run_dir / f"llm_graph_rag_{safe_model}.json"
    no_graph_path = run_dir / f"llm_no_graph_{safe_model}.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    no_graph = json.loads(no_graph_path.read_text(encoding="utf-8"))
    result = analyze_paired_ablation(
        graph,
        no_graph,
        bootstrap_samples=args.bootstrap_samples,
        seed=42,
    )
    result["dataset_id"] = args.dataset
    result["model"] = args.model
    result["graph_run"] = str(graph_path.relative_to(REPOSITORY_ROOT))
    result["no_graph_run"] = str(no_graph_path.relative_to(REPOSITORY_ROOT))
    output = run_dir / f"llm_ablation_{safe_model}.json"
    write_json(output, result)
    effects = result["effects"]
    mcnemar = result["mcnemar_exact"]
    lines = [
        f"# LLM Graph-Grounding Ablation: {args.dataset}",
        "",
        f"- Model: `{args.model}`",
        f"- Paired items: {result['paired_items']}",
        (
            f"- Accuracy: graph {result['graph_metrics']['accuracy']:.3f}; "
            f"no graph {result['no_graph_metrics']['accuracy']:.3f}"
        ),
        (
            f"- Accuracy difference: {effects['accuracy_difference']:.3f} "
            f"(paired bootstrap 95% CI "
            f"{effects['accuracy_difference_bootstrap_95_ci'][0]:.3f} to "
            f"{effects['accuracy_difference_bootstrap_95_ci'][1]:.3f})"
        ),
        (
            f"- Unsupported-claim rate: graph "
            f"{result['graph_metrics']['unsupported_claim_rate']:.3f}; "
            f"no graph {result['no_graph_metrics']['unsupported_claim_rate']:.3f}"
        ),
        (
            f"- Statement-claim coverage: graph "
            f"{result['graph_metrics']['statement_claim_coverage']:.3f}; "
            f"no graph {result['no_graph_metrics']['statement_claim_coverage']:.3f}"
        ),
        (
            f"- Format-failure rate: graph "
            f"{result['graph_metrics']['format_failure_rate']:.3f}; "
            f"no graph {result['no_graph_metrics']['format_failure_rate']:.3f}"
        ),
        (
            f"- Structured unsupported-answer rate: graph "
            f"{result['graph_metrics']['structured_unsupported_answer_rate']:.3f}; "
            f"no graph "
            f"{result['no_graph_metrics']['structured_unsupported_answer_rate']:.3f}"
        ),
        (
            f"- UCR reduction: {effects['unsupported_claim_rate_reduction']:.3f} "
            f"(paired bootstrap 95% CI "
            f"{effects['unsupported_claim_rate_reduction_bootstrap_95_ci'][0]:.3f} "
            f"to {effects['unsupported_claim_rate_reduction_bootstrap_95_ci'][1]:.3f})"
        ),
        (
            f"- Exact McNemar test: graph-only correct "
            f"{mcnemar['graph_only_correct']}, no-graph-only correct "
            f"{mcnemar['no_graph_only_correct']}, "
            f"p={mcnemar['two_sided_p_value']:.6g}"
        ),
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
