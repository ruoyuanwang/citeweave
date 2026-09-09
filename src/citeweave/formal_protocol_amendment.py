from __future__ import annotations

from typing import Any


def effective_query_datasets(
    protocol: dict[str, Any],
    *,
    amendment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply a narrowly scoped, pre-acquisition query-only amendment."""
    datasets = [dict(row) for row in protocol["datasets"]]
    if amendment is None:
        return datasets
    overrides = amendment.get("changes", {}).get("datasets", {})
    known = {row["id"] for row in datasets}
    if set(overrides) - known:
        raise ValueError("Query amendment refers to unknown dataset IDs")
    for row in datasets:
        change = overrides.get(row["id"])
        if change:
            permitted = {"keywords", "query_mode", "search_expression"}
            unexpected = set(change) - permitted - {"rationale"}
            if unexpected:
                raise ValueError(f"Query amendment changes forbidden fields: {unexpected}")
            row.update({key: value for key, value in change.items() if key in permitted})
    return datasets
