from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def query_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "topic": entry["topic"],
        "source": entry["source"],
        "keywords": entry["keywords"],
        "query_mode": entry["query_mode"],
        "search_expression": entry.get("search_expression"),
        "search_scope": entry.get("search_scope", "fulltext"),
        "year_from": entry["year_from"],
        "year_to": entry["year_to"],
        "document_types": entry.get("document_types", []),
        "language": entry.get("language"),
        "max_records": entry.get("max_records"),
    }


def query_set_sha256(datasets: list[dict[str, Any]]) -> str:
    ordered = [query_payload(item) for item in sorted(datasets, key=lambda row: row["id"])]
    return sha256_json(ordered)


def verify_frozen_query_registry(registry: dict[str, Any]) -> None:
    if registry.get("status") != "frozen":
        raise ValueError("Formal registry is not frozen by Query Judge.")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Formal registry contains no datasets.")
    attestation = registry.get("query_freeze_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("Formal registry lacks query_freeze_attestation.")  # noqa: TRY004
    expected_set_hash = query_set_sha256(datasets)
    if attestation.get("query_set_sha256") != expected_set_hash:
        raise ValueError("Frozen query-set hash does not match the current registry.")
    accepted_ids = sorted(attestation.get("accepted_ids") or [])
    actual_ids = sorted(item["id"] for item in datasets)
    if accepted_ids != actual_ids:
        raise ValueError("Query Judge accepted_ids do not match the formal datasets.")
    for item in datasets:
        if item.get("query_status") != "frozen":
            raise ValueError(f"{item['id']}: query_status is not frozen")
        if item.get("query_sha256") != sha256_json(query_payload(item)):
            raise ValueError(f"{item['id']}: frozen query hash mismatch")
        if item.get("max_records") is not None:
            raise ValueError(f"{item['id']}: formal max_records must be null")
