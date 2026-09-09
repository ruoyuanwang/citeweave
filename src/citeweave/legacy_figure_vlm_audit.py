from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file


def _score_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise TypeError(f"score rows must be a list: {path}")
    return rows


def _reads_printed_subtitle(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction") or {}
    explanation = str(prediction.get("explanation") or "").casefold()
    answer = prediction.get("answer") or {}
    return (
        "subtitle" in explanation
        and "node" in explanation
        and "link" in explanation
        and isinstance(answer, dict)
        and {"nodes", "links"}.issubset(answer)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def audit_legacy_figure_vlm_scope(
    formal_runs_root: Path,
    run_id: str,
    resolved_judgments_path: Path | None = None,
) -> dict[str, Any]:
    """Audit what the legacy Figure/VLM panel actually measured.

    The audit is descriptive and does not rescore answers. It verifies whether the
    visual condition covered structural reasoning tasks or merely transcribed values
    printed in figure subtitles.
    """

    dataset_rows: list[dict[str, Any]] = []
    figure_task_types: Counter[str] = Counter()
    graph_task_types: Counter[str] = Counter()
    figure_ids: set[str] = set()
    graph_ids: set[str] = set()
    task_type_by_id: dict[str, str] = {}
    figure_correct_by_id: dict[str, bool] = {}
    graph_correct_by_id: dict[str, bool] = {}
    subtitle_reads = 0
    source_files: list[dict[str, str]] = []

    for dataset_dir in sorted(path for path in formal_runs_root.iterdir() if path.is_dir()):
        run_root = dataset_dir / run_id
        figure_score = run_root / "figure_vlm" / "score.json"
        if not figure_score.exists():
            continue
        graph_score = run_root / "graph_rag" / "score.json"
        if not graph_score.exists():
            raise FileNotFoundError(f"missing paired graph score: {graph_score}")

        figure_rows = _score_rows(figure_score)
        graph_rows = _score_rows(graph_score)
        dataset_figure_ids = {str(row["item_id"]) for row in figure_rows}
        dataset_graph_ids = {str(row["item_id"]) for row in graph_rows}
        dataset_subtitle_reads = sum(_reads_printed_subtitle(row) for row in figure_rows)

        for row in figure_rows:
            item_id = str(row["item_id"])
            task_type = str(row["task_type"])
            figure_ids.add(item_id)
            figure_task_types[task_type] += 1
            task_type_by_id[item_id] = task_type
            figure_correct_by_id[item_id] = row.get("correct") is True
        for row in graph_rows:
            item_id = str(row["item_id"])
            task_type = str(row["task_type"])
            graph_ids.add(item_id)
            graph_task_types[task_type] += 1
            task_type_by_id.setdefault(item_id, task_type)
            graph_correct_by_id[item_id] = row.get("correct") is True

        subtitle_reads += dataset_subtitle_reads
        dataset_rows.append(
            {
                "dataset_id": dataset_dir.name,
                "figure_items": len(figure_rows),
                "figure_task_types": dict(
                    sorted(Counter(str(row["task_type"]) for row in figure_rows).items())
                ),
                "direct_printed_subtitle_reads": dataset_subtitle_reads,
                "graph_items": len(graph_rows),
                "graph_only_items": len(dataset_graph_ids - dataset_figure_ids),
                "overlap_items": len(dataset_graph_ids & dataset_figure_ids),
            }
        )
        source_files.extend(
            [
                {"path": str(figure_score), "sha256": sha256_file(figure_score)},
                {"path": str(graph_score), "sha256": sha256_file(graph_score)},
            ]
        )

    if not dataset_rows:
        raise FileNotFoundError(
            f"no Figure/VLM scores found under {formal_runs_root} for run {run_id}"
        )

    judgment_rows: list[dict[str, Any]] = []
    judgment_task_types: Counter[str] = Counter()
    preferences: Counter[str] = Counter()
    unknown_judgment_ids: list[str] = []
    if resolved_judgments_path is not None:
        judgment_rows = _read_jsonl(resolved_judgments_path)
        source_files.append(
            {
                "path": str(resolved_judgments_path),
                "sha256": sha256_file(resolved_judgments_path),
            }
        )
        for row in judgment_rows:
            sample_id = str(row.get("sample_id") or "")
            task_type = task_type_by_id.get(sample_id)
            if task_type is None:
                unknown_judgment_ids.append(sample_id)
            else:
                judgment_task_types[task_type] += 1
            preferences[str(row.get("preference") or "missing")] += 1

    all_visual_network_size = set(figure_task_types) == {"network_size"}
    all_visual_direct_subtitle = subtitle_reads == len(figure_ids)
    all_judged_network_size = (
        bool(judgment_rows)
        and not unknown_judgment_ids
        and set(judgment_task_types) == {"network_size"}
    )
    scope_confirmed = (
        all_visual_network_size
        and all_visual_direct_subtitle
        and (resolved_judgments_path is None or all_judged_network_size)
    )
    overlap_ids = figure_ids & graph_ids
    figure_correct = sum(figure_correct_by_id.values())
    graph_overlap_correct = sum(graph_correct_by_id[item_id] for item_id in overlap_ids)
    both_correct = sum(
        figure_correct_by_id[item_id] and graph_correct_by_id[item_id]
        for item_id in overlap_ids
    )

    return {
        "schema_version": 1,
        "status": "scope_confirmed" if scope_confirmed else "scope_not_confirmed",
        "run_id": run_id,
        "summary": {
            "datasets_with_figure_vlm": len(dataset_rows),
            "figure_items": len(figure_ids),
            "figure_task_type_counts": dict(sorted(figure_task_types.items())),
            "direct_printed_subtitle_reads": subtitle_reads,
            "figure_correct_items": figure_correct,
            "figure_exact_accuracy": figure_correct / len(figure_ids),
            "all_figure_items_are_network_size": all_visual_network_size,
            "all_figure_answers_read_printed_subtitles": all_visual_direct_subtitle,
            "graph_items": len(graph_ids),
            "graph_task_type_counts": dict(sorted(graph_task_types.items())),
            "figure_graph_overlap_items": len(overlap_ids),
            "graph_correct_on_overlap_items": graph_overlap_correct,
            "graph_exact_accuracy_on_overlap": graph_overlap_correct / len(overlap_ids),
            "both_conditions_correct_on_overlap_items": both_correct,
            "graph_items_without_visual_counterpart": len(graph_ids - figure_ids),
            "formal_judged_pairs": len(judgment_rows),
            "formal_judged_task_type_counts": dict(sorted(judgment_task_types.items())),
            "formal_preferences": dict(sorted(preferences.items())),
            "all_formal_judgments_are_network_size": all_judged_network_size,
            "unknown_formal_judgment_ids": unknown_judgment_ids,
        },
        "datasets": dataset_rows,
        "interpretation": {
            "supports": (
                "The legacy visual panel measured transcription of node/link counts "
                "printed in figure subtitles and exhibited a ceiling on that narrow task."
            ),
            "does_not_support": (
                "It does not estimate Figure/VLM performance on node ranking, edge selection, "
                "community membership, multi-hop paths, counterfactual deletion, temporal-"
                "structural joins, or graph-level synthesis, and therefore cannot establish "
                "equivalence between visual input and structural GraphRAG."
            ),
        },
        "source_files": source_files,
    }
