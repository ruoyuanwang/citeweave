from __future__ import annotations

from scripts.build_formal_v3_complexity_extension import _validate_matches


def test_anchor_match_validator_accepts_edge_and_hub_controls() -> None:
    tasks = []
    for scale in ("small", "medium", "large"):
        edge_parent = f"d:n:{scale}:bridge_counterfactual"
        hub_parent = f"d:n:{scale}:hub_removal_resilience"
        tasks.extend(
            [
                {
                    "item_id": edge_parent,
                    "complexity": 4,
                    "evidence_ids": ["n1", "n2", "e1"],
                },
                {
                    "item_id": hub_parent,
                    "complexity": 4,
                    "evidence_ids": ["hub", "replacement"],
                },
                {
                    "item_id": f"d:n:{scale}:direct_edge_lookup",
                    "task_type": "direct_edge_lookup",
                    "complexity": 1,
                    "matched_complex_item_id": edge_parent,
                    "evidence_ids": ["n1", "n2", "e1"],
                },
                {
                    "item_id": f"d:n:{scale}:node_attribute_lookup",
                    "task_type": "node_attribute_lookup",
                    "complexity": 1,
                    "matched_complex_item_id": hub_parent,
                    "evidence_ids": ["hub"],
                },
            ]
        )
    _validate_matches(tasks)
