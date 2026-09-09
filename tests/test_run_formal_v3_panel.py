from __future__ import annotations

import json
from pathlib import Path

import yaml

from citeweave.io import sha256_file
from scripts.run_formal_v3_panel import build_panel_plan


def test_panel_plan_counts_all_registered_conditions(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.yml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "conditions": {
                    "primary": ["flat_bm25", "flat_hybrid"],
                    "diagnostic": ["flat_program"],
                }
            }
        ),
        encoding="utf-8",
    )
    neural_path = tmp_path / "neural.yml"
    neural_path.write_text(
        yaml.safe_dump(
            {
                "changes": {
                    "neural_dense_baseline": {"condition_name": "flat_neural_dense"}
                }
            }
        ),
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    dataset_dir = benchmark_root / "topic"
    dataset_dir.mkdir(parents=True)
    benchmark_path = dataset_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps({"tasks": [{"item_id": "a"}, {"item_id": "b"}]}),
        encoding="utf-8",
    )
    (benchmark_root / "construction_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset_id": "topic",
                        "benchmark_sha256": sha256_file(benchmark_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps({"status": "blocked", "blocking_reasons": ["missing"]}),
        encoding="utf-8",
    )

    plan = build_panel_plan(
        protocol_path=protocol_path,
        neural_amendment_path=neural_path,
        benchmark_root=benchmark_root,
        results_root=tmp_path / "runs",
        readiness_path=readiness_path,
    )

    assert plan["status"] == "blocked"
    assert plan["conditions"] == [
        "flat_bm25",
        "flat_hybrid",
        "flat_program",
        "flat_neural_dense",
    ]
    assert plan["tasks"] == 2
    assert plan["calls"] == 8
