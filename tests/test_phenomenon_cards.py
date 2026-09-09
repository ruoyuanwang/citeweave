from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.citeweave.graph_discovery import build_discovery_benchmark
from src.citeweave.io import write_json
from src.citeweave.phenomenon_cards import build_phenomenon_cards


def test_builds_verified_cards_with_literature_dependencies(tmp_path: Path) -> None:
    workspace = tmp_path / "topic"
    root = workspace / "canonical" / "visualization"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "keyword": [f"k{index}" for index in range(8)],
            "keyword_type": ["test"] * 8,
            "occurrences": [100, 90, 80, 70, 60, 50, 40, 30],
        }
    ).to_parquet(root / "keyword_occurrences.parquet", index=False)
    pd.DataFrame(
        {
            "source_id": ["k0", "k1", "k2", "k0", "k4", "k5", "k6", "k3"],
            "target_id": ["k1", "k2", "k3", "k2", "k5", "k6", "k7", "k4"],
            "weight": [9, 8, 7, 6, 9, 8, 7, 1],
        }
    ).to_parquet(root / "keyword_cooccurrence_edges.parquet", index=False)
    pd.DataFrame(
        {
            "work_id": ["w1"],
            "doi": ["10.1/example"],
            "title": ["A study of k0"],
            "abstract": ["This study describes k0."],
            "year": [2024],
            "cited_by_count": [5],
        }
    ).to_parquet(workspace / "canonical" / "works.parquet", index=False)
    pd.DataFrame(
        {"work_id": ["w1"], "keyword": ["k0"], "keyword_type": ["test"], "score": [1.0]}
    ).to_parquet(workspace / "canonical" / "keywords.parquet", index=False)
    benchmark = build_discovery_benchmark(
        workspace,
        tmp_path / "benchmark",
        scales=("small",),
        networks=("keyword_cooccurrence",),
    )
    records = []
    for task in benchmark["tasks"]:
        records.append(
            {
                "item_id": task["item_id"],
                "condition": "graph_program",
                "status": "complete",
                "response": {
                    "answer": task["answer"],
                    "evidence_ids": task["evidence_ids"],
                    "phenomenon": "A structural pattern.",
                    "alternative_explanation": "Corpus coverage.",
                    "limitation": "Descriptive only.",
                },
            }
        )
    results_path = tmp_path / "results.json"
    write_json(results_path, {"records": records})
    payload = build_phenomenon_cards(
        benchmark_path=tmp_path / "benchmark" / "benchmark.json",
        results_path=results_path,
        workspace=workspace,
        output_path=tmp_path / "cards.json",
    )
    assert payload["cards"]
    assert all(card["complexity"] > 1 for card in payload["cards"])
    assert all(card["dependency_record"]["operator_replay_valid"] for card in payload["cards"])
