from __future__ import annotations

import json
from pathlib import Path

from citeweave.io import sha256_file, write_json
from scripts.run_formal_v3_complexity_extension_panel import build_extension_plan


def test_extension_plan_has_312_incremental_calls(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    records = []
    for index in range(8):
        dataset_id = f"d{index}"
        tasks = []
        for scale in ("small", "medium", "large"):
            for task_type in (
                "direct_edge_lookup",
                "node_attribute_lookup",
                "multi_hop_connector",
                "bridge_counterfactual",
                "community_role_contrast",
                "hub_removal_resilience",
                "temporal_structural_shift",
            ):
                tasks.append(
                    {
                        "item_id": f"{dataset_id}:{scale}:{task_type}",
                        "task_type": task_type,
                    }
                )
        path = benchmark_root / dataset_id / "benchmark.json"
        write_json(path, {"dataset_id": dataset_id, "tasks": tasks})
        records.append(
            {
                "dataset_id": dataset_id,
                "source_panel": "primary" if index < 4 else "replication",
                "benchmark_sha256": sha256_file(path),
            }
        )
    write_json(
        benchmark_root / "construction_manifest.json",
        {"protocol_sha256": "p", "records": records},
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "status": "blocked",
                "execution_amendment_sha256": "a",
                "blocking_reasons": ["tokenizer"],
            }
        ),
        encoding="utf-8",
    )
    plan = build_extension_plan(
        benchmark_root=benchmark_root,
        execution_root=tmp_path / "runs",
        readiness_path=readiness,
    )
    assert plan["status"] == "blocked"
    assert len(plan["jobs"]) == 12
    assert plan["reused_cells"] == 360
    assert plan["incremental_calls"] == 312
