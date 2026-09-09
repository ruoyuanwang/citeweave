from __future__ import annotations

import hashlib
import json
from typing import Any

from .token_budget import (
    CommandTokenizer,
    canonical_context_json,
    truncate_context_to_token_budget,
)

PROMPT_VERSION = "graph-discovery-v2-strong-controls-20260820"


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def task_payload_sha256(task: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in task.items()
            if key not in {"contexts", "context_hashes"}
        }
    )


def build_messages(
    task: dict[str, Any],
    condition: str,
    *,
    context_override: dict[str, Any] | None = None,
    tokenizer: CommandTokenizer | None = None,
    token_budget: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    system = (
        "You are evaluating a bibliometric graph, not describing a picture. "
        "Use only supplied material. Perform the requested multi-step analysis and return "
        "one JSON object with exactly these keys: item_id, abstain, answer, phenomenon, "
        "evidence_ids, alternative_explanation, limitation. The answer object must use "
        "exactly the requested fields and values. Cite at most eight evidence_ids and keep "
        "each prose field to at most three sentences. A structural association is not causality, "
        "quality, author intent, or real-world information flow. Abstain when evidence is "
        "insufficient."
    )
    user = (
        f"ITEM_ID: {task['item_id']}\n"
        f"TASK_TYPE: {task['task_type']}\n"
        f"COMPLEXITY_LEVEL: {task['complexity']}\n"
        f"QUESTION: {task['question']}\n"
        f"ANSWER_FIELDS: {json.dumps(list(task['answer']), ensure_ascii=False)}\n"
    )
    context_audit: dict[str, Any] = {
        "context_characters": 0,
        "full_context_tokens": 0,
        "budgeted_context_tokens": 0,
        "context_token_budget": token_budget,
        "retained_records": {},
        "original_records": {},
        "context_sha256": None,
    }
    if condition != "no_reference":
        context = context_override or task["contexts"][condition]
        if tokenizer is not None and token_budget is not None:
            budgeted = truncate_context_to_token_budget(
                context,
                count_tokens=tokenizer.count,
                token_budget=token_budget,
            )
            context = budgeted.context
            context_audit.update(
                {
                    "full_context_tokens": budgeted.full_tokens,
                    "budgeted_context_tokens": budgeted.budgeted_tokens,
                    "retained_records": budgeted.retained_records,
                    "original_records": budgeted.original_records,
                    "context_sha256": budgeted.context_sha256,
                }
            )
        rendered = canonical_context_json(context)
        context_audit["context_characters"] = len(rendered)
        user += "CONTEXT:\n" + rendered
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], context_audit
