from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, sha256_file


def _plan_units(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if "datasets" in plan:
        return [
            {
                **row,
                "unit_id": row["dataset_id"],
                "conditions": list(plan["conditions"]),
                "task_types": None,
            }
            for row in plan["datasets"]
        ]
    return [
        {
            **row,
            "unit_id": row["job_id"],
        }
        for row in plan.get("jobs") or []
    ]


def audit_panel_terminal(
    plan_path: Path, *, maximum_parse_attempts: int = 3
) -> dict[str, Any]:
    if maximum_parse_attempts < 1:
        raise ValueError("maximum_parse_attempts must be positive")
    plan = read_json(plan_path)
    reasons: list[str] = []
    retry_cells: list[dict[str, Any]] = []
    terminal_parse_failures: list[dict[str, Any]] = []
    complete_cells = 0
    expected_cells = 0
    units = []
    for unit in _plan_units(plan):
        benchmark_path = Path(unit["benchmark"])
        benchmark = read_json(benchmark_path)
        tasks = [
            task
            for task in benchmark["tasks"]
            if unit.get("task_types") is None
            or task["task_type"] in set(unit["task_types"])
        ]
        expected = {
            (task["item_id"], condition)
            for task in tasks
            for condition in unit["conditions"]
        }
        expected_cells += len(expected)
        results_path = Path(unit["output"]) / "results.json"
        unit_reasons = []
        if not results_path.is_file():
            retry_cells.extend(
                {"unit_id": unit["unit_id"], "item_id": item, "condition": condition}
                for item, condition in sorted(expected)
            )
            units.append(
                {
                    "unit_id": unit["unit_id"],
                    "status": "missing_results",
                    "expected_cells": len(expected),
                    "results_path": str(results_path),
                }
            )
            continue
        result = read_json(results_path)
        manifest = result.get("manifest") or {}
        if manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
            unit_reasons.append("benchmark_hash_mismatch")
        if set(manifest.get("conditions") or []) != set(unit["conditions"]):
            unit_reasons.append("condition_manifest_mismatch")
        records = result.get("records") or []
        indexed = {
            (row.get("item_id"), row.get("condition")): row for row in records
        }
        if len(indexed) != len(records):
            unit_reasons.append("duplicate_final_cells")
        if set(indexed) - expected:
            unit_reasons.append("unexpected_final_cells")
        for item_id, condition in sorted(expected):
            row = indexed.get((item_id, condition))
            if row is None:
                retry_cells.append(
                    {
                        "unit_id": unit["unit_id"],
                        "item_id": item_id,
                        "condition": condition,
                        "reason": "missing",
                    }
                )
                continue
            status = row.get("status", "complete")
            if status == "complete":
                complete_cells += 1
            elif status == "failed_parse":
                attempt = int(row.get("attempt", 1))
                cell = {
                    "unit_id": unit["unit_id"],
                    "item_id": item_id,
                    "condition": condition,
                    "attempt": attempt,
                }
                if attempt >= maximum_parse_attempts:
                    terminal_parse_failures.append(cell)
                else:
                    retry_cells.append({**cell, "reason": "parse_retry_permitted"})
            else:
                unit_reasons.append(
                    f"invalid_final_status:{item_id}:{condition}:{status}"
                )
        reasons.extend(f"{unit['unit_id']}:{reason}" for reason in unit_reasons)
        units.append(
            {
                "unit_id": unit["unit_id"],
                "status": "invalid" if unit_reasons else "audited",
                "expected_cells": len(expected),
                "observed_cells": len(indexed),
                "results_path": str(results_path),
                "results_sha256": sha256_file(results_path),
                "reasons": unit_reasons,
            }
        )
    if reasons:
        status = "invalid"
    elif retry_cells:
        status = "needs_retry"
    else:
        status = "terminal"
    return {
        "schema_version": 1,
        "status": status,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "maximum_parse_attempts": maximum_parse_attempts,
        "expected_cells": expected_cells,
        "complete_cells": complete_cells,
        "terminal_parse_failure_cells": len(terminal_parse_failures),
        "retry_cells": retry_cells,
        "terminal_parse_failures": terminal_parse_failures,
        "integrity_reasons": reasons,
        "units": units,
    }
