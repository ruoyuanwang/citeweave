import json

import pandas as pd

from citeweave.graph_suite import aggregate_graph_suite, load_graph_suite_spec


def _record(topic: str, mode: str, repeat: int, support: float) -> dict:
    return {
        "topic_id": topic,
        "project_id": topic,
        "case_id": f"case-{topic}",
        "mode": mode,
        "repeat": repeat,
        "status": "complete",
        "metrics": {
            "verified_slot_coverage": support,
            "edge_hallucination_rate": 1 - support,
            "claim_support_rate": support,
            "path_validity_rate": support,
            "verified_complex_claims": support * 5,
            "abstention_rate": 0.0,
        },
        "usage": {"prompt_tokens": 100 + repeat},
    }


def test_suite_aggregation_uses_graph_as_statistical_unit(tmp_path):
    records = []
    values = {
        "topic-a": {"vlm": 0.2, "flat_kg": 0.5, "graph_rag": 0.8},
        "topic-b": {"vlm": 0.4, "flat_kg": 0.6, "graph_rag": 1.0},
    }
    for topic, modes in values.items():
        for mode, support in modes.items():
            records.extend(_record(topic, mode, repeat, support) for repeat in (1, 2, 3))
    (tmp_path / "records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    result = aggregate_graph_suite(tmp_path, seed=7)

    assert result["independent_topics"] == 2
    topic_mode = pd.read_csv(tmp_path / "topic_mode_summary.csv")
    assert len(topic_mode) == 6
    assert set(topic_mode["successful_repeats"]) == {3}
    aggregate = pd.read_csv(tmp_path / "aggregate_summary.csv")
    row = aggregate[
        (aggregate["mode"] == "graph_rag")
        & (aggregate["metric"] == "claim_support_rate")
    ].iloc[0]
    assert row["graphs"] == 2
    assert row["mean"] == 0.9


def test_formal_suite_spec_is_valid():
    spec = load_graph_suite_spec(
        __import__("pathlib").Path("experiments/formal_graph_suite.yml")
    )

    assert len(spec["topics"]) == 12
    assert spec["minimum_topics"] == 10
    assert spec["year_to"] == 2025
    assert all(topic["id"] != "pilot-system-graphrag" for topic in spec["topics"])
