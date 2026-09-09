from __future__ import annotations

import json
from pathlib import Path

import yaml

from citeweave.io import sha256_file
from scripts.run_formal_v3_replication_panel import build_replication_plan


def test_replication_plan_counts_two_condition_calls(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yml"
    protocol.write_text(
        yaml.safe_dump({"conditions": {"confirmatory": ["flat_hybrid", "graph_program"]}}),
        encoding="utf-8",
    )
    root = tmp_path / "benchmarks"
    topic = root / "topic"
    topic.mkdir(parents=True)
    benchmark = topic / "benchmark.json"
    benchmark.write_text(json.dumps({"tasks": [{"item_id": "a"}] * 15}), encoding="utf-8")
    (root / "construction_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset_id": "topic",
                        "benchmark_sha256": sha256_file(benchmark),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    plan = build_replication_plan(
        protocol_path=protocol,
        benchmark_root=root,
        results_root=tmp_path / "runs",
        readiness_path=readiness,
    )
    assert plan["tasks"] == 15
    assert plan["calls"] == 30
    assert plan["status"] == "blocked"
