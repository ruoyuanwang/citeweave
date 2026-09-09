from __future__ import annotations

import json
from pathlib import Path

from citeweave.legacy_figure_vlm_audit import audit_legacy_figure_vlm_scope


def _write_score(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def _row(item_id: str, task_type: str, explanation: str, answer: dict) -> dict:
    return {
        "item_id": item_id,
        "task_type": task_type,
        "correct": True,
        "prediction": {"explanation": explanation, "answer": answer},
    }


def test_confirms_subtitle_only_visual_scope(tmp_path: Path) -> None:
    run_root = tmp_path / "topic" / "run"
    visual_id = "topic:network:network_size"
    _write_score(
        run_root / "figure_vlm" / "score.json",
        [
            _row(
                visual_id,
                "network_size",
                "The visible subtitle states 10 nodes and 12 links.",
                {"nodes": 10, "links": 12},
            )
        ],
    )
    _write_score(
        run_root / "graph_rag" / "score.json",
        [
            _row(visual_id, "network_size", "Graph facts.", {"nodes": 10, "links": 12}),
            _row("topic:network:path", "multi_hop_connector", "Path.", {"hops": 2}),
        ],
    )
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        json.dumps({"sample_id": visual_id, "preference": "tie"}) + "\n",
        encoding="utf-8",
    )

    report = audit_legacy_figure_vlm_scope(tmp_path, "run", judgments)

    assert report["status"] == "scope_confirmed"
    assert report["summary"]["figure_items"] == 1
    assert report["summary"]["graph_items_without_visual_counterpart"] == 1
    assert report["summary"]["figure_exact_accuracy"] == 1.0
    assert report["summary"]["graph_exact_accuracy_on_overlap"] == 1.0
    assert report["summary"]["formal_judged_task_type_counts"] == {"network_size": 1}


def test_rejects_claim_that_non_subtitle_task_is_same_scope(tmp_path: Path) -> None:
    run_root = tmp_path / "topic" / "run"
    _write_score(
        run_root / "figure_vlm" / "score.json",
        [_row("topic:path", "multi_hop_connector", "I inferred a path.", {"hops": 2})],
    )
    _write_score(
        run_root / "graph_rag" / "score.json",
        [_row("topic:path", "multi_hop_connector", "Graph path.", {"hops": 2})],
    )

    report = audit_legacy_figure_vlm_scope(tmp_path, "run")

    assert report["status"] == "scope_not_confirmed"
    assert report["summary"]["all_figure_items_are_network_size"] is False
    assert report["summary"]["all_figure_answers_read_printed_subtitles"] is False
