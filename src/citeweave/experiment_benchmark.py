from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .io import write_json


def _float_equal(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _score_answer(item: dict[str, Any], prediction: dict[str, Any]) -> bool:
    if not item["answerable"]:
        return bool(prediction.get("abstain"))
    if prediction.get("abstain"):
        return False
    gold = item["gold_answer"] or {}
    answer = prediction.get("answer") or {}
    task = item["task_type"]
    if task == "network_size":
        edge_key = "links" if "links" in gold else "edges"
        return (
            answer.get("nodes") == gold.get("nodes")
            and answer.get(edge_key) == gold.get(edge_key)
        )
    if task == "highest_weighted_degree":
        return str(answer.get("id")) == str(gold.get("id")) and _float_equal(
            answer.get("weighted_degree"), gold.get("weighted_degree")
        )
    if task == "cluster_count":
        return answer.get("clusters") == gold.get("clusters")
    if task == "strongest_edge":
        predicted_pair = {str(answer.get("source")), str(answer.get("target"))}
        gold_pair = {str(gold.get("source")), str(gold.get("target"))}
        return predicted_pair == gold_pair and _float_equal(
            answer.get("weight"), gold.get("weight")
        )
    return answer == gold


def _evidence_counts(
    gold: list[str],
    predicted: list[str],
) -> tuple[int, int, int]:
    gold_set = set(gold)
    predicted_set = set(predicted)
    return (
        len(gold_set & predicted_set),
        len(predicted_set - gold_set),
        len(gold_set - predicted_set),
    )


def _normalize_statement(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" .;:")


def _statement_support(
    item: dict[str, Any],
    prediction: dict[str, Any],
    *,
    abstained: bool,
) -> bool | None:
    if abstained:
        return None
    statement = str(prediction.get("statement") or "").strip()
    if not statement or statement.startswith(("{", "[", "```")):
        return None
    if not item["answerable"]:
        return False
    return _normalize_statement(statement) == _normalize_statement(item["gold_statement"])


def score_graph_predictions(
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_by_id = {
        str(prediction["item_id"]): prediction
        for prediction in predictions
        if prediction.get("item_id") is not None
    }
    rows: list[dict[str, Any]] = []
    evidence_tp = evidence_fp = evidence_fn = 0
    answered = incorrect_answered = 0
    statement_claims = unsupported_statement_claims = 0
    abstention = Counter()

    for item in items:
        prediction = prediction_by_id.get(item["item_id"], {})
        schema_valid = bool(
            prediction
            and isinstance(prediction.get("abstain"), bool)
            and (
                prediction["abstain"]
                or isinstance(prediction.get("answer"), dict)
            )
        )
        correct = schema_valid and _score_answer(item, prediction)
        predicted_abstain = bool(prediction.get("abstain")) if schema_valid else False
        statement_supported = _statement_support(
            item,
            prediction,
            abstained=predicted_abstain,
        )
        if statement_supported is not None:
            statement_claims += 1
            unsupported_statement_claims += int(not statement_supported)
        if item["answerable"]:
            if predicted_abstain:
                abstention["fn"] += 1
            else:
                abstention["tn"] += 1
                answered += 1
                incorrect_answered += int(not correct)
        else:
            if predicted_abstain:
                abstention["tp"] += 1
            else:
                abstention["fp"] += 1
                answered += 1
                incorrect_answered += 1

        node_counts = _evidence_counts(
            item.get("gold_evidence_nodes") or [],
            prediction.get("evidence_nodes") or [],
        )
        edge_counts = _evidence_counts(
            item.get("gold_evidence_edges") or [],
            prediction.get("evidence_edges") or [],
        )
        evidence_tp += node_counts[0] + edge_counts[0]
        evidence_fp += node_counts[1] + edge_counts[1]
        evidence_fn += node_counts[2] + edge_counts[2]
        predicted_evidence_count = len(prediction.get("evidence_nodes") or []) + len(
            prediction.get("evidence_edges") or []
        )
        evidence_valid = (
            None
            if predicted_abstain
            else (
                predicted_evidence_count > 0
                and node_counts[1] == 0
                and edge_counts[1] == 0
            )
        )
        rows.append(
            {
                "item_id": item["item_id"],
                "dataset_id": item["dataset_id"],
                "network": item["network"],
                "task_type": item["task_type"],
                "answerable": item["answerable"],
                "schema_valid": schema_valid,
                "correct": correct,
                "abstained": predicted_abstain,
                "evidence_valid": evidence_valid,
                "statement_claim": statement_supported is not None,
                "statement_supported": statement_supported,
                "prediction": prediction,
            }
        )

    total = len(items)
    correct_count = sum(row["correct"] for row in rows)
    schema_count = sum(row["schema_valid"] for row in rows)
    evidence_precision = (
        evidence_tp / (evidence_tp + evidence_fp)
        if evidence_tp + evidence_fp
        else (0.0 if evidence_fn else 1.0)
    )
    evidence_recall = (
        evidence_tp / (evidence_tp + evidence_fn) if evidence_tp + evidence_fn else 1.0
    )
    abstention_precision = (
        abstention["tp"] / (abstention["tp"] + abstention["fp"])
        if abstention["tp"] + abstention["fp"]
        else 0.0
    )
    abstention_recall = (
        abstention["tp"] / (abstention["tp"] + abstention["fn"])
        if abstention["tp"] + abstention["fn"]
        else 0.0
    )
    abstention_f1 = (
        2 * abstention_precision * abstention_recall
        / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall
        else 0.0
    )
    structured_unsupported_answer_rate = (
        incorrect_answered / answered if answered else 0.0
    )
    unsupported_claim_rate = (
        unsupported_statement_claims / statement_claims if statement_claims else 0.0
    )
    answerable_items = sum(bool(item["answerable"]) for item in items)
    return {
        "benchmark_version": 1,
        "items": total,
        "predictions_received": len(prediction_by_id),
        "metrics": {
            "exact_answer_accuracy": correct_count / total if total else 0.0,
            "schema_valid_rate": schema_count / total if total else 0.0,
            "unsupported_claim_rate": unsupported_claim_rate,
            "structured_unsupported_answer_rate": structured_unsupported_answer_rate,
            "atomic_statement_precision": (
                1 - unsupported_claim_rate if statement_claims else None
            ),
            "statement_claim_coverage": (
                statement_claims / answerable_items if answerable_items else 0.0
            ),
            "format_failure_rate": 1 - schema_count / total if total else 0.0,
            "evidence_path_validity": _evidence_validity_rate(rows),
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "abstention_precision": abstention_precision,
            "abstention_recall": abstention_recall,
            "abstention_f1": abstention_f1,
        },
        "counts": {
            "correct": correct_count,
            "answered": answered,
            "incorrect_answered": incorrect_answered,
            "statement_claims": statement_claims,
            "unsupported_statement_claims": unsupported_statement_claims,
            "evidence_tp": evidence_tp,
            "evidence_fp": evidence_fp,
            "evidence_fn": evidence_fn,
            "evidence_path_eligible_answers": sum(
                row["evidence_valid"] is not None for row in rows
            ),
            "abstention": dict(abstention),
        },
        "by_task": _aggregate_rows(rows, "task_type"),
        "by_network": _aggregate_rows(rows, "network"),
        "rows": rows,
    }


def _aggregate_rows(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        name: {
            "items": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "schema_valid_rate": sum(row["schema_valid"] for row in group) / len(group),
            "evidence_path_validity": _evidence_validity_rate(group),
        }
        for name, group in sorted(groups.items())
    }


def _evidence_validity_rate(rows: list[dict[str, Any]]) -> float:
    eligible = [row for row in rows if row["evidence_valid"] is not None]
    return (
        sum(bool(row["evidence_valid"]) for row in eligible) / len(eligible)
        if eligible
        else 0.0
    )


def oracle_graph_predictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item["item_id"],
            "abstain": not item["answerable"],
            "answer": item["gold_answer"] if item["answerable"] else None,
            "evidence_nodes": item["gold_evidence_nodes"],
            "evidence_edges": item["gold_evidence_edges"],
            "statement": item["gold_statement"],
        }
        for item in items
    ]


def no_context_abstain_predictions(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item["item_id"],
            "abstain": True,
            "answer": None,
            "evidence_nodes": [],
            "evidence_edges": [],
            "statement": "Insufficient structured evidence.",
        }
        for item in items
    ]


def no_context_forced_predictions(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    default_answers = {
        "network_size": {"nodes": 0, "edges": 0},
        "highest_weighted_degree": {"id": "unknown", "weighted_degree": 0},
        "cluster_count": {"clusters": 0},
        "strongest_edge": {"source": "unknown", "target": "unknown", "weight": 0},
        "unanswerable_false_premise": {
            "cluster": "unknown",
            "node": "CiteWeave absent benchmark node",
        },
    }
    return [
        {
            "item_id": item["item_id"],
            "abstain": False,
            "answer": default_answers[item["task_type"]],
            "evidence_nodes": [],
            "evidence_edges": [],
            "statement": "An answer was forced without graph context.",
        }
        for item in items
    ]


def save_benchmark_result(
    result: dict[str, Any],
    output: Path,
    *,
    condition: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "condition": condition,
        "metadata": metadata or {},
        **result,
    }
    write_json(output, payload)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON array")
    return payload
