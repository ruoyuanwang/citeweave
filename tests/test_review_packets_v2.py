from __future__ import annotations

from pathlib import Path

from src.citeweave.io import read_json, write_json
from src.citeweave.review_packets_v2 import build_review_packets


def test_builds_blinded_two_layer_review_packets(tmp_path: Path) -> None:
    panel_root = tmp_path / "panel"
    benchmark_root = tmp_path / "benchmarks"
    topic = "topic-a"
    (panel_root / topic).mkdir(parents=True)
    (benchmark_root / topic).mkdir(parents=True)
    task = {
        "item_id": "topic-a:network:large:task",
        "task_type": "temporal_structural_shift",
        "complexity": 5,
        "question": "What changed?",
        "answer": {"value": 3},
        "answer_alternatives": [],
        "evidence_ids": ["e1"],
        "interpretation_contract": {"forbidden": ["causality"]},
        "contexts": {
            "flat_retrieval": {"representation": "flat", "rows": []},
            "flat_tfidf": {"representation": "flat_tfidf", "rows": []},
            "flat_bm25": {"representation": "flat_bm25", "rows": []},
            "flat_program": {
                "representation": "flat_program",
                "rows": [],
                "derived_rows": [],
            },
            "graph_retrieval": {"representation": "graph", "nodes": [], "edges": []},
            "graph_query_retrieval": {
                "representation": "query_graph",
                "nodes": [],
                "edges": [],
            },
            "operator_only": {
                "representation": "operator_only",
                "operator_trace": [],
            },
            "graph_program": {
                "representation": "program",
                "nodes": [],
                "edges": [],
                "operator_trace": [],
            },
        },
    }
    write_json(benchmark_root / topic / "benchmark.json", {"tasks": [task]})
    response = {
        "abstain": False,
        "answer": {"value": 3},
        "evidence_ids": ["e1"],
        "phenomenon": "A descriptive change.",
        "alternative_explanation": "Sampling may explain it.",
        "limitation": "This is corpus dependent.",
    }
    write_json(
        panel_root / topic / "results.json",
        {
            "records": [
                {
                    "item_id": task["item_id"],
                    "condition": "graph_program",
                    "status": "complete",
                    "response": response,
                }
            ]
        },
    )
    manifest = build_review_packets(
        panel_root=panel_root,
        benchmark_root=benchmark_root,
        output_root=tmp_path / "review",
    )
    assert manifest["factual_packets"] == 1
    assert manifest["semantic_packets"] == 1
    factual_id = manifest["records"][0]["factual_packet_id"]
    packet = read_json(tmp_path / "review" / "packets" / "factual" / f"{factual_id}.json")
    assert "condition" not in packet
    assert "gold" not in packet
    assert "representation" not in packet["visible_evidence"]["bundle"]
