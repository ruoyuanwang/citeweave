from __future__ import annotations

from scripts.analyze_formal_v3_complexity_extension import analyze_complexity_factorial


def test_complexity_and_scale_interactions_are_clustered_by_dataset() -> None:
    tasks = []
    records = []
    conditions = (
        "flat_hybrid",
        "flat_neural_dense",
        "graph_hierarchical_retrieval_v2",
        "graph_program",
    )
    for dataset_index in range(8):
        dataset = f"d{dataset_index}"
        for scale in ("small", "medium", "large"):
            for pair_index, (simple_type, complex_type) in enumerate(
                (
                    ("direct_edge_lookup", "bridge_counterfactual"),
                    ("node_attribute_lookup", "hub_removal_resilience"),
                )
            ):
                simple_id = f"{dataset}:{scale}:simple:{pair_index}"
                complex_id = f"{dataset}:{scale}:complex:{pair_index}"
                tasks.extend(
                    [
                        {
                            "item_id": simple_id,
                            "dataset_id": dataset,
                            "scale": scale,
                            "task_type": simple_type,
                            "complexity": 1,
                            "matched_complex_item_id": complex_id,
                        },
                        {
                            "item_id": complex_id,
                            "dataset_id": dataset,
                            "scale": scale,
                            "task_type": complex_type,
                            "complexity": 4,
                        },
                    ]
                )
                for item_id, band in ((simple_id, "simple"), (complex_id, "complex")):
                    for condition in conditions:
                        correct = (
                            condition
                            in ("graph_program", "graph_hierarchical_retrieval_v2")
                            and band == "complex"
                        )
                        if scale == "small" and band == "complex":
                            correct = False
                        records.append(
                            {
                                "item_id": item_id,
                                "condition": condition,
                                "status": "complete",
                                "score": {
                                    "answer_exact": correct,
                                    "evidence_f1": float(correct),
                                },
                            }
                        )
    result = analyze_complexity_factorial(tasks, records, bootstrap_samples=100)
    assert result["datasets"] == 8
    assert result["matched_pairs"] == 48
    assert result["registered_contrasts"]["C1_complexity_selectivity"][
        "exact_cluster_signflip"
    ]["minimum_attainable_p"] == 1 / 256
    assert result["registered_contrasts"][
        "C2_scale_amplification_of_complexity_selectivity"
    ]["cluster_bootstrap"]["estimate"] == 1.0
    assert result["status"] == "confirmatory_analysis_complete"
    assert result["registered_contrasts"]["C1_complexity_selectivity"][
        "multiplicity"
    ]["holm_adjusted_p_value"] == 2 / 256
    assert result["secondary"]["C3_graph_program_advantage_on_simple_tasks"][
        "cluster_tost"
    ]["equivalent_at_0_05"] is True
    assert result["mixed_logistic"]["observations"] == 192
    assert result["cluster_validity_checks"][
        "E1_graph_program_over_flat_hybrid_complex_3_to_5"
    ]["exact_cluster_signflip"]["p_value_one_sided"] == 1 / 256
    assert result["cluster_validity_checks"][
        "E2_hierarchical_over_flat_hybrid_global_counterfactual"
    ]["multiplicity"]["holm_adjusted_p_value"] == 2 / 256
