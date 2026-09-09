from __future__ import annotations

import importlib.util
from pathlib import Path

from src.citeweave.io import read_json, write_json


def _module():
    path = Path(__file__).parents[1] / "scripts" / "finalize_human_review.py"
    spec = importlib.util.spec_from_file_location("finalize_human_review", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finalizer_refuses_to_compile_when_reviewers_disagree(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "packets"
    for layer in ("factual", "semantic"):
        (root / "packets" / layer).mkdir(parents=True)
    reviewers = ["A", "B"]
    manifest = {
        "reviewers": reviewers,
        "assignments": {
            reviewer: {"factual": ["F1"], "semantic": []} for reviewer in reviewers
        },
    }
    write_json(root / "internal_manifest.json", manifest)
    write_json(root / "packets" / "factual" / "F1.json", {"packet_id": "F1"})
    for reviewer, correct in zip(reviewers, (True, False)):
        write_json(
            root / "returns" / f"{reviewer}.json",
            {
                "reviewer_code": reviewer,
                "results": [
                    {
                        "packet_id": "F1",
                        "reviewer_code": reviewer,
                        "answer_correct": correct,
                        "evidence_sufficient": True,
                        "action": "accept" if correct else "rewrite",
                        "rationale": "Independent judgment.",
                        "review_seconds": 4.0,
                    }
                ],
            },
        )
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_human_review.py",
            "--packet-root",
            str(root),
            "--output",
            str(output),
        ],
    )
    _module().main()
    summary = read_json(output / "review_summary.json")
    assert summary["adjudication_required"] == 1
    assert summary["policy_compilation_unlocked"] is False
