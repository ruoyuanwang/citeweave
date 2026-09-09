from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .graph_discovery import score_discovery_response
from .io import read_json, write_json
from .operator_verification import verify_benchmark_operator_replay


def _card_id(dataset_id: str, item_id: str) -> str:
    digest = hashlib.sha256(f"phenomenon-card-v1|{item_id}".encode()).hexdigest()[:16]
    return f"PC-{dataset_id[:12]}-{digest}"


def _answer_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_answer_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_answer_strings(item) for item in value), set())
    return set()


def _representative_works(workspace: Path, answer: dict[str, Any]) -> list[dict[str, Any]]:
    works = pd.read_parquet(workspace / "canonical" / "works.parquet")
    keywords = pd.read_parquet(workspace / "canonical" / "keywords.parquet")
    labels = {value.casefold(): value for value in _answer_strings(answer)}
    matched = keywords[keywords["keyword"].astype(str).str.casefold().isin(labels)].copy()
    if matched.empty:
        return []
    matched["matched_keyword"] = matched["keyword"].astype(str)
    joined = matched.merge(works, on="work_id", how="inner")
    joined = joined.sort_values(
        ["cited_by_count", "year", "work_id"],
        ascending=[False, False, True],
        na_position="last",
    )
    records = []
    seen: set[str] = set()
    for row in joined.to_dict("records"):
        work_id = str(row["work_id"])
        if work_id in seen:
            continue
        seen.add(work_id)
        abstract = str(row.get("abstract") or "").strip()
        records.append(
            {
                "work_id": work_id,
                "matched_keyword": str(row["matched_keyword"]),
                "title": str(row.get("title") or "Untitled"),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "doi": str(row.get("doi") or "") or None,
                "cited_by_count": int(row.get("cited_by_count") or 0),
                "abstract_excerpt": abstract[:1200],
            }
        )
        if len(records) >= 3:
            break
    return records


def build_phenomenon_cards(
    *,
    benchmark_path: Path,
    results_path: Path,
    workspace: Path,
    output_path: Path,
    condition: str = "graph_program",
) -> dict[str, Any]:
    benchmark = read_json(benchmark_path)
    replay = verify_benchmark_operator_replay(benchmark, workspace=workspace)
    if replay["invalid_tasks"]:
        raise ValueError("Operator replay failed; phenomenon cards cannot be trusted")
    tasks = {task["item_id"]: task for task in benchmark["tasks"]}
    results = read_json(results_path)
    cards = []
    for record in results["records"]:
        if record.get("status", "complete") != "complete" or record["condition"] != condition:
            continue
        task = tasks[record["item_id"]]
        if int(task.get("complexity") or 0) <= 1:
            continue
        score = score_discovery_response(task, record["response"])
        if not score["answer_exact"]:
            continue
        response = record["response"]
        counterfactual = [
            step
            for step in task["operator_trace"]
            if any(token in step["operator"] for token in ("delete", "recompute"))
        ]
        works = _representative_works(workspace, task["answer"])
        cards.append(
            {
                "card_id": _card_id(benchmark["dataset_id"], task["item_id"]),
                "dataset_id": benchmark["dataset_id"],
                "item_id": task["item_id"],
                "task_type": task["task_type"],
                "complexity": task["complexity"],
                "question": task["question"],
                "verified_answer": task["answer"],
                "phenomenon": response.get("phenomenon"),
                "counterfactual_result": counterfactual,
                "alternative_explanation": response.get("alternative_explanation"),
                "limitation": response.get("limitation"),
                "interpretation_contract": task["interpretation_contract"],
                "operator_trace": task["operator_trace"],
                "evidence_ids": task["evidence_ids"],
                "representative_works": works,
                "dependency_record": {
                    "operator_replay_valid": True,
                    "source_work_ids": [work["work_id"] for work in works],
                    "evidence_ids": task["evidence_ids"],
                },
            }
        )
    task_types = {card["task_type"] for card in cards}
    gates = {
        "minimum_four_cards": len(cards) >= 4,
        "has_counterfactual": bool(
            task_types & {"bridge_counterfactual", "hub_removal_resilience"}
        ),
        "has_global_structure": "community_role_contrast" in task_types,
        "has_temporal_structure": "temporal_structural_shift" in task_types,
        "all_operator_replays_valid": replay["invalid_tasks"] == 0,
        "cards_with_representative_works": sum(
            bool(card["representative_works"]) for card in cards
        ),
    }
    payload = {
        "schema_version": 1,
        "dataset_id": benchmark["dataset_id"],
        "condition": condition,
        "quality_gates": gates,
        "passed": all(value for key, value in gates.items() if key != "cards_with_representative_works")
        and gates["cards_with_representative_works"] >= 3,
        "cards": cards,
    }
    write_json(output_path, payload)
    return payload
