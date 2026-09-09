from __future__ import annotations

from pathlib import Path

from citeweave.io import sha256_file, write_json
from scripts.run_formal_v3_mechanism_supplement_panel import (
    COMPLEX_TASKS,
    SIMPLE_TASKS,
    build_mechanism_supplement_plan,
)


def test_mechanism_supplement_plan_has_216_incremental_calls(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    records = []
    for dataset_index in range(8):
        dataset_id = f"D{dataset_index}"
        tasks = []
        for task_type in SIMPLE_TASKS:
            for scale in ("small", "medium", "large"):
                tasks.append(
                    {
                        "item_id": f"{dataset_id}:{scale}:{task_type}",
                        "task_type": task_type,
                        "contexts": {"flat_program": {}, "operator_only": {}},
                    }
                )
        for task_type in COMPLEX_TASKS:
            for scale in ("small", "medium", "large"):
                tasks.append(
                    {
                        "item_id": f"{dataset_id}:{scale}:{task_type}",
                        "task_type": task_type,
                        "contexts": {"flat_program": {}, "operator_only": {}},
                    }
                )
        benchmark = benchmark_root / dataset_id / "benchmark.json"
        write_json(benchmark, {"dataset_id": dataset_id, "tasks": tasks})
        records.append(
            {
                "dataset_id": dataset_id,
                "source_panel": "primary" if dataset_index < 4 else "replication",
                "benchmark_sha256": sha256_file(benchmark),
            }
        )
    write_json(
        benchmark_root / "construction_manifest.json",
        {"datasets": 8, "tasks": 168, "records": records},
    )
    protocol = Path(
        "experiments/graph_discovery_v2/"
        "formal_v3_mechanism_supplement_protocol.yml"
    )
    freeze = Path(
        "experiments/graph_discovery_v2/"
        "formal_v3_mechanism_supplement_protocol_freeze.json"
    )
    readiness = tmp_path / "readiness.json"
    write_json(
        readiness,
        {
            "status": "ready",
            "supplement_protocol_sha256": sha256_file(protocol),
            "conditions": ["graph_program", "flat_program", "operator_only"],
            "cell_identity_manifest": [{} for _ in range(336)],
        },
    )
    plan = build_mechanism_supplement_plan(
        protocol_path=protocol,
        protocol_freeze_path=freeze,
        benchmark_root=benchmark_root,
        execution_root=tmp_path / "runs",
        readiness_path=readiness,
    )
    assert plan["status"] == "ready_to_execute"
    assert plan["logical_cells"] == 504
    assert plan["reused_cells"] == 288
    assert plan["incremental_calls"] == 216
    assert len(plan["jobs"]) == 12
