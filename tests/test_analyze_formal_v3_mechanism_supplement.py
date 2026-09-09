from __future__ import annotations

from scripts.analyze_formal_v3_mechanism_supplement import (
    analyze_mechanism_supplement,
)


def test_mechanism_analysis_uses_eight_equal_dataset_clusters() -> None:
    tasks = []
    records = []
    exposure = []
    task_types = (
        "direct_edge_lookup",
        "node_attribute_lookup",
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    complex_types = set(task_types[2:])
    for dataset_index in range(8):
        dataset_id = f"d{dataset_index}"
        for scale in ("small", "medium", "large"):
            for task_type in task_types:
                item_id = f"{dataset_id}:{scale}:{task_type}"
                tasks.append(
                    {
                        "dataset_id": dataset_id,
                        "item_id": item_id,
                        "task_type": task_type,
                    }
                )
                if task_type in complex_types:
                    exposure.append(
                        {
                            "item_id": item_id,
                            "derivability": {
                                "operator_trace": {
                                    "fully_derivable": task_type
                                    != "bridge_counterfactual"
                                }
                            },
                        }
                    )
                for condition in (
                    "graph_program",
                    "flat_program",
                    "operator_only",
                ):
                    operator_bridge = (
                        condition == "operator_only"
                        and task_type == "bridge_counterfactual"
                    )
                    records.append(
                        {
                            "item_id": item_id,
                            "condition": condition,
                            "status": "complete",
                            "score": {
                                "answer_exact": not operator_bridge,
                                "evidence_f1": 0.5
                                if condition == "operator_only"
                                else 1.0,
                            },
                        }
                    )
    result = analyze_mechanism_supplement(
        tasks=tasks,
        result_records=records,
        exposure_records=exposure,
        bootstrap_samples=100,
    )
    assert result["logical_cells"] == 504
    assert result["M1_same_computation_representation"]["equivalence"][
        "equivalent_at_0_05"
    ]
    assert result["M2_raw_provenance_value"]["answer_accuracy_noninferiority"][
        "noninferior_at_0_05"
    ]
    assert result["M3_incomplete_trace_selectivity"]["estimate"] == 1.0
    assert result["holm_family"]["M3_trace_incomplete_selectivity"][
        "reject_at_0_05"
    ]
