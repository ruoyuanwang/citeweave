from citeweave.ablation_analysis import analyze_paired_ablation


def _result(condition, correct):
    return {
        "condition": condition,
        "rows": [
            {
                "item_id": f"i{index}",
                "dataset_id": "topic",
                "network": "citation",
                "correct": value,
                "abstained": False,
                "answerable": True,
                "schema_valid": value,
                "statement_claim": True,
                "statement_supported": value,
                "evidence_valid": value,
            }
            for index, value in enumerate(correct)
        ],
    }


def test_paired_ablation_reports_positive_graph_effect():
    result = analyze_paired_ablation(
        _result("graph_rag", [True, True, True, False]),
        _result("no_graph", [False, False, True, False]),
        bootstrap_samples=200,
        seed=7,
    )

    assert result["effects"]["accuracy_difference"] == 0.5
    assert result["effects"]["unsupported_claim_rate_reduction"] == 0.5
    assert result["mcnemar_exact"]["graph_only_correct"] == 2
    assert result["mcnemar_exact"]["no_graph_only_correct"] == 0
    assert result["graph_metrics"]["structured_unsupported_answer_rate"] == 0.25
    assert result["graph_metrics"]["format_failure_rate"] == 0.25
