from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .graph_discovery import score_discovery_response
from .io import read_json, write_json


def _packet_id(dataset_id: str, item_id: str, condition: str, layer: str) -> str:
    digest = hashlib.sha256(
        f"review-v2|{dataset_id}|{item_id}|{condition}|{layer}".encode()
    ).hexdigest()[:20]
    return f"HRP{digest}"


def _sanitized_context(task: dict[str, Any], condition: str) -> dict[str, Any]:
    if condition == "no_reference":
        return {"evidence_available": False, "bundle": None}
    context = dict(task["contexts"][condition])
    context.pop("representation", None)
    return {"evidence_available": True, "bundle": context}


def build_review_packets(
    *,
    panel_root: Path,
    benchmark_root: Path,
    output_root: Path,
    reviewer_codes: tuple[str, str] = ("REVIEWER-A", "REVIEWER-B"),
) -> dict[str, Any]:
    factual_dir = output_root / "packets" / "factual"
    semantic_dir = output_root / "packets" / "semantic"
    factual_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    factual_ids: list[str] = []
    semantic_ids: list[str] = []

    for topic_dir in sorted(path for path in panel_root.iterdir() if path.is_dir()):
        dataset_id = topic_dir.name
        benchmark = read_json(benchmark_root / dataset_id / "benchmark.json")
        tasks = {task["item_id"]: task for task in benchmark["tasks"]}
        results = read_json(topic_dir / "results.json")
        for record in results["records"]:
            if record.get("status", "complete") != "complete":
                continue
            task = tasks[record["item_id"]]
            condition = record["condition"]
            score = score_discovery_response(task, record["response"])
            factual_id = _packet_id(dataset_id, task["item_id"], condition, "factual")
            factual_packet = {
                "schema_version": 1,
                "packet_id": factual_id,
                "review_layer": "factual",
                "dataset_id": dataset_id,
                "task_type": task["task_type"],
                "complexity": task["complexity"],
                "question": task["question"],
                "candidate": {
                    "abstain": record["response"].get("abstain"),
                    "answer": record["response"].get("answer"),
                    "evidence_ids": record["response"].get("evidence_ids"),
                },
                "visible_evidence": _sanitized_context(task, condition),
                "instructions": [
                    "Judge whether every answer field is supported by visible evidence.",
                    "Do not judge prose style or scientific importance in this layer.",
                    "Record unsupported fields separately from missing evidence.",
                    "Do not infer the experimental condition.",
                ],
                "response_schema": {
                    "reviewer_code": "string",
                    "answer_correct": "boolean",
                    "evidence_sufficient": "boolean",
                    "unsupported_fields": "list[string]",
                    "action": "accept|rewrite|reject_evidence|abstain",
                    "correction": "object|null",
                    "rationale": "string",
                    "review_seconds": "number",
                },
            }
            write_json(factual_dir / f"{factual_id}.json", factual_packet)
            factual_ids.append(factual_id)
            semantic_id = None
            if score["answer_exact"]:
                semantic_id = _packet_id(dataset_id, task["item_id"], condition, "semantic")
                semantic_packet = {
                    "schema_version": 1,
                    "packet_id": semantic_id,
                    "review_layer": "semantic_interpretation",
                    "dataset_id": dataset_id,
                    "task_type": task["task_type"],
                    "complexity": task["complexity"],
                    "verified_structured_answer": record["response"].get("answer"),
                    "candidate": {
                        "phenomenon": record["response"].get("phenomenon"),
                        "alternative_explanation": record["response"].get(
                            "alternative_explanation"
                        ),
                        "limitation": record["response"].get("limitation"),
                    },
                    "interpretation_contract": task["interpretation_contract"],
                    "instructions": [
                        "The structured answer already passed a deterministic verifier.",
                        "Judge calibration, alternative explanations, domain usefulness, and overreach.",
                        "Mark the smallest span requiring revision and provide a guarded correction.",
                        "A correction guard must state when it may transfer to future claims.",
                    ],
                    "response_schema": {
                        "reviewer_code": "string",
                        "calibrated": "boolean",
                        "phenomenon_value": "1|2|3|4|5",
                        "alternative_adequate": "boolean",
                        "limitation_adequate": "boolean",
                        "action": "accept|rewrite|reject_claim|abstain",
                        "target_span": "string|null",
                        "replacement": "string|null",
                        "guard": "object",
                        "rationale": "string",
                        "review_seconds": "number",
                    },
                }
                write_json(semantic_dir / f"{semantic_id}.json", semantic_packet)
                semantic_ids.append(semantic_id)
            manifest_records.append(
                {
                    "dataset_id": dataset_id,
                    "item_id": task["item_id"],
                    "task_type": task["task_type"],
                    "complexity": task["complexity"],
                    "condition": condition,
                    "factual_packet_id": factual_id,
                    "semantic_packet_id": semantic_id,
                    "deterministic_answer_exact": score["answer_exact"],
                    "deterministic_evidence_f1": score["evidence_f1"],
                }
            )

    assignments = {
        reviewer: {
            "factual": sorted(
                factual_ids,
                key=lambda packet_id: hashlib.sha256(
                    f"{reviewer}|{packet_id}".encode()
                ).hexdigest(),
            ),
            "semantic": sorted(
                semantic_ids,
                key=lambda packet_id: hashlib.sha256(
                    f"{reviewer}|{packet_id}".encode()
                ).hexdigest(),
            ),
        }
        for reviewer in reviewer_codes
    }
    manifest = {
        "schema_version": 1,
        "blinding": {
            "condition_hidden_from_packets": True,
            "gold_answer_hidden_from_factual_packets": True,
            "deterministic_pass_required_for_semantic_layer": True,
        },
        "reviewers": list(reviewer_codes),
        "factual_packets": len(factual_ids),
        "semantic_packets": len(semantic_ids),
        "double_review": True,
        "adjudicate_disagreements": True,
        "records": manifest_records,
        "assignments": assignments,
    }
    write_json(output_root / "internal_manifest.json", manifest)
    write_json(
        output_root / "reviewer_protocol.json",
        {
            "schema_version": 1,
            "factual_packets_per_reviewer": len(factual_ids),
            "semantic_packets_per_reviewer": len(semantic_ids),
            "reviewer_codes": list(reviewer_codes),
            "instructions": [
                "Review independently; do not communicate before submission.",
                "Use a timer and record active review seconds for every packet.",
                "Do not open internal_manifest.json; it contains condition labels.",
                "An adjudicator reviews only disagreements after both returns are frozen.",
            ],
        },
    )
    return manifest
