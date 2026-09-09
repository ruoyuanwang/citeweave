from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.citeweave.graph_discovery import build_discovery_benchmark
from src.citeweave.operator_verification import (
    audit_fault_injection,
    corrupt_operator_trace,
    verify_benchmark_operator_replay,
)


def _workspace(tmp_path: Path) -> Path:
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
    return workspace


def test_operator_replay_accepts_clean_and_rejects_corruption(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    benchmark = build_discovery_benchmark(
        workspace,
        tmp_path / "out",
        scales=("small",),
        networks=("keyword_cooccurrence",),
    )
    clean = verify_benchmark_operator_replay(benchmark, workspace=workspace)
    assert clean["invalid_tasks"] == 0
    corrupted = {
        **benchmark,
        "tasks": [corrupt_operator_trace(benchmark["tasks"][0])],
    }
    invalid = verify_benchmark_operator_replay(corrupted, workspace=workspace)
    assert invalid["invalid_tasks"] == 1
    assert "operator_trace" in invalid["records"][0]["mismatches"]


def test_fault_injection_detects_every_modified_trace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    benchmark = build_discovery_benchmark(
        workspace,
        tmp_path / "out",
        scales=("small",),
        networks=("keyword_cooccurrence",),
    )
    audit = audit_fault_injection(benchmark, workspace=workspace)
    assert audit["clean_false_rejections"] == 0
    assert audit["detection_rate"] == 1.0
