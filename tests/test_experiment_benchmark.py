from citeweave.experiment_benchmark import (
    no_context_abstain_predictions,
    no_context_forced_predictions,
    oracle_graph_predictions,
    score_graph_predictions,
)


def _items():
    return [
            {
            "item_id": "d:G001",
            "dataset_id": "d",
            "network": "keyword_cooccurrence",
            "task_type": "network_size",
            "answerable": True,
            "gold_answer": {"nodes": 10, "edges": 12},
            "gold_statement": "The network has 10 nodes and 12 edges.",
            "gold_evidence_nodes": ["fact:G001"],
            "gold_evidence_edges": [],
        },
        {
            "item_id": "d:U",
            "dataset_id": "d",
            "network": "keyword_cooccurrence",
            "task_type": "unanswerable_false_premise",
            "answerable": False,
            "gold_answer": None,
            "gold_statement": "The node is absent.",
            "gold_evidence_nodes": ["network:keyword_cooccurrence"],
            "gold_evidence_edges": [],
        },
    ]


def test_oracle_predictions_pass_all_graph_metrics():
    items = _items()
    result = score_graph_predictions(items, oracle_graph_predictions(items))

    assert result["metrics"]["exact_answer_accuracy"] == 1.0
    assert result["metrics"]["schema_valid_rate"] == 1.0
    assert result["metrics"]["unsupported_claim_rate"] == 0.0
    assert result["metrics"]["abstention_f1"] == 1.0


def test_no_context_stress_baselines_expose_safety_coverage_tradeoff():
    items = _items()
    abstain = score_graph_predictions(items, no_context_abstain_predictions(items))
    forced = score_graph_predictions(items, no_context_forced_predictions(items))

    assert abstain["metrics"]["unsupported_claim_rate"] == 0.0
    assert abstain["metrics"]["exact_answer_accuracy"] == 0.5
    assert abstain["metrics"]["evidence_path_validity"] == 0.0
    assert abstain["metrics"]["evidence_precision"] == 0.0
    assert forced["metrics"]["unsupported_claim_rate"] == 1.0
    assert forced["metrics"]["exact_answer_accuracy"] == 0.0
    assert forced["metrics"]["evidence_path_validity"] == 0.0
    assert forced["metrics"]["evidence_precision"] == 0.0


def test_wrong_answer_and_false_premise_increase_unsupported_claim_rate():
    items = _items()
    predictions = [
        {
            "item_id": "d:G001",
            "abstain": False,
            "answer": {"nodes": 11, "edges": 12},
            "evidence_nodes": [],
                "evidence_edges": [],
                "statement": "The network has 11 nodes and 12 edges.",
            },
        {
            "item_id": "d:U",
            "abstain": False,
            "answer": {"cluster": 3},
            "evidence_nodes": ["invented"],
                "evidence_edges": [],
                "statement": "The absent node belongs to cluster 3.",
            },
    ]
    result = score_graph_predictions(items, predictions)

    assert result["metrics"]["exact_answer_accuracy"] == 0.0
    assert result["metrics"]["unsupported_claim_rate"] == 1.0
    assert result["metrics"]["structured_unsupported_answer_rate"] == 1.0
    assert result["metrics"]["evidence_path_validity"] == 0.0
    assert result["metrics"]["abstention_f1"] == 0.0


def test_network_size_accepts_figure_manifest_links_vocabulary():
    items = [
        {
            "item_id": "d:size",
            "dataset_id": "d",
            "network": "coauthorship",
            "task_type": "network_size",
            "answerable": True,
            "gold_answer": {"nodes": 60, "links": 127},
        }
    ]
    predictions = [
        {
            "item_id": "d:size",
            "abstain": False,
            "answer": {"nodes": 60, "links": 127},
        }
    ]

    result = score_graph_predictions(items, predictions)

    assert result["metrics"]["exact_answer_accuracy"] == 1.0
