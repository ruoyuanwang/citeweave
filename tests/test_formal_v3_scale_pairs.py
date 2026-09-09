from __future__ import annotations

from scripts.build_formal_v3_scale_pairs import build_pairs


def _task(scale: str, source: str, target: str) -> dict:
    return {
        "item_id": f"d:keyword:{scale}:multi",
        "network": "keyword_cooccurrence",
        "scale": scale,
        "task_type": "multi_hop_connector",
        "answer": {"source_label": source, "target_label": target},
    }


def test_scale_pair_requires_identical_registered_anchors() -> None:
    benchmark = {
        "dataset_id": "d",
        "tasks": [
            _task("small", "a", "b"),
            _task("medium", "a", "b"),
            _task("large", "a", "b"),
        ],
    }
    pairs = build_pairs(benchmark)
    assert len(pairs) == 1
    assert pairs[0]["anchor"] == {"source_label": "a", "target_label": "b"}
    benchmark["tasks"][-1]["answer"]["target_label"] = "c"
    assert build_pairs(benchmark) == []
