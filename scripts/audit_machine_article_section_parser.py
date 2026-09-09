from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from citeweave.article_generation_diagnostic import (
    assess_generated_article_hierarchically,
)
from citeweave.io import read_json, sha256_file, write_json


def _validate_amendment(amendment_path: Path, freeze_path: Path) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    if freeze.get("sha256") != sha256_file(amendment_path):
        raise RuntimeError("Article parser diagnostic amendment differs from freeze")
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    for label, artifact in (amendment.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Article parser diagnostic implementation mismatch: {label}")
    return amendment


def _changed_gates(primary: dict[str, Any], secondary: dict[str, Any]) -> list[str]:
    primary_gates = primary.get("quality_gates") or {}
    secondary_gates = secondary.get("quality_gates") or {}
    return sorted(
        key
        for key in set(primary_gates) | set(secondary_gates)
        if primary_gates.get(key) != secondary_gates.get(key)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    amendment = _validate_amendment(args.amendment, args.amendment_freeze)
    plan = read_json(args.plan)
    records: list[dict[str, Any]] = []
    gate_failures: dict[str, Counter[str]] = defaultdict(Counter)
    for cell in plan.get("cells") or []:
        output_dir = Path(cell["output_dir"])
        draft_path = output_dir / "draft.md"
        primary_path = output_dir / "draft_assessment.json"
        execution_path = output_dir / "execution_record.json"
        writer_input_path = Path(cell["writer_input"])
        for path in (draft_path, primary_path, execution_path, writer_input_path):
            if not path.is_file():
                raise RuntimeError(f"Required immutable artifact is missing: {path}")
        execution = read_json(execution_path)
        if execution.get("draft_sha256") != sha256_file(draft_path):
            raise RuntimeError(f"Draft hash mismatch: {cell['cell_id']}")
        primary = read_json(primary_path)
        secondary = assess_generated_article_hierarchically(
            draft_path.read_text(encoding="utf-8"),
            writer_input=read_json(writer_input_path),
        )
        condition = cell["condition"]
        for gate, passed in secondary["quality_gates"].items():
            if not passed:
                gate_failures[condition][gate] += 1
        records.append(
            {
                "cell_id": cell["cell_id"],
                "dataset_id": cell["dataset_id"],
                "condition": condition,
                "draft_sha256": sha256_file(draft_path),
                "primary_assessment_sha256": sha256_file(primary_path),
                "primary_passed": bool(primary.get("passed")),
                "secondary_passed_all_recomputed_gates": secondary[
                    "passed_all_recomputed_gates"
                ],
                "changed_gates": _changed_gates(primary, secondary),
                "secondary_assessment": secondary,
            }
        )
    by_condition: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in records}):
        selected = [row for row in records if row["condition"] == condition]
        by_condition[condition] = {
            "n": len(selected),
            "primary_passed": sum(row["primary_passed"] for row in selected),
            "secondary_passed_all_recomputed_gates": sum(
                row["secondary_passed_all_recomputed_gates"] for row in selected
            ),
            "secondary_gate_failure_counts": dict(gate_failures[condition]),
            "word_counts": [
                row["secondary_assessment"]["word_count"] for row in selected
            ],
        }
    report = {
        "schema_version": 1,
        "status": "post_generation_secondary_parser_diagnostic_complete",
        "interpretation": {
            "primary_execution_outcomes_preserved": True,
            "confirmatory_reclassification_prohibited": True,
            "permitted_use": "diagnose parser-induced versus substantive gate failures",
        },
        "plan_sha256": sha256_file(args.plan),
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256_file(args.amendment),
        "cells": len(records),
        "by_condition": by_condition,
        "records": records,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
