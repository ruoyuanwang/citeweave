from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.article_budget_optimizer import optimize_article_word_budget
from citeweave.article_generation_diagnostic import (
    assess_generated_article_hierarchically,
)
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    args = parser.parse_args()
    if read_json(args.protocol_freeze).get("sha256") != sha256_file(args.protocol):
        raise RuntimeError("Budget-optimizer protocol differs from freeze")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Budget-optimizer implementation mismatch: {label}")
    plan = read_json(args.plan)
    if read_json(args.plan_freeze).get("sha256") != sha256_file(args.plan):
        raise RuntimeError("Budget-optimizer plan differs from freeze")
    records = []
    for row in plan["datasets"]:
        source = Path(row["source_draft"])
        writer_input = Path(row["writer_input"])
        if sha256_file(source) != row["source_draft_sha256"]:
            raise RuntimeError(f"Source draft mismatch: {row['dataset_id']}")
        if sha256_file(writer_input) != row["writer_input_sha256"]:
            raise RuntimeError(f"Writer input mismatch: {row['dataset_id']}")
        optimized, optimization = optimize_article_word_budget(
            source.read_text(encoding="utf-8")
        )
        output_dir = Path(row["output_dir"])
        draft_path = output_dir / "draft.md"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(optimized, encoding="utf-8")
        optimization_path = output_dir / "optimization_audit.json"
        write_json(optimization_path, optimization)
        assessment = assess_generated_article_hierarchically(
            optimized, writer_input=read_json(writer_input)
        )
        assessment_path = output_dir / "draft_assessment.json"
        write_json(assessment_path, assessment)
        passed = bool(assessment["passed_all_recomputed_gates"])
        execution = {
            "schema_version": 1,
            "dataset_id": row["dataset_id"],
            "status": "budget_optimized_quality_gate_passed"
            if passed
            else "budget_optimized_quality_gate_failed",
            "development_only": True,
            "source_draft_sha256": row["source_draft_sha256"],
            "draft_sha256": sha256_file(draft_path),
            "optimization_audit_sha256": sha256_file(optimization_path),
            "draft_assessment_sha256": sha256_file(assessment_path),
        }
        write_json(output_dir / "execution_record.json", execution)
        records.append({**execution, "optimization": optimization})
    report = {
        "schema_version": 1,
        "status": "deterministic_budget_optimization_complete",
        "articles": len(records),
        "passed": sum(
            row["status"] == "budget_optimized_quality_gate_passed"
            for row in records
        ),
        "records": records,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
