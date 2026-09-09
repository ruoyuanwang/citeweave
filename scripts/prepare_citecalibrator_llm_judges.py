from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from citeweave.io import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(args.cases) != protocol["input"]["source_cases_sha256"]:
        raise ValueError("Cases differ from the frozen judge protocol")
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != protocol["input"]["expected_cases"]:
        raise ValueError("Case count differs from the frozen judge protocol")
    if any("condition" in row for row in cases):
        raise ValueError("Reviewer-visible generation condition detected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for judge_id in ("judge_a", "judge_b"):
        judge_root = args.output_root / judge_id
        input_root = judge_root / "input"
        output_root = judge_root / "output"
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.cases, input_root / "cases.jsonl")
        shutil.copyfile(args.protocol, input_root / "judge_protocol.yml")
        shutil.copyfile(args.rubric, input_root / "judge_rubric.md")
        assignment = {
            "schema_version": 1,
            "judge_id": judge_id,
            "status": "ready_for_isolated_judging",
            "expected_cases": len(cases),
            "allowed_read_paths": [
                str((input_root / "cases.jsonl").resolve()),
                str((input_root / "judge_protocol.yml").resolve()),
                str((input_root / "judge_rubric.md").resolve()),
            ],
            "required_output": str((output_root / "reviews.jsonl").resolve()),
            "forbidden_cross_judge_access": True,
            "inherit_parent_context": False,
        }
        write_json(judge_root / "assignment.json", assignment)
        records.append(
            {
                "judge_id": judge_id,
                "assignment": str((judge_root / "assignment.json").resolve()),
                "cases_sha256": sha256_file(input_root / "cases.jsonl"),
                "protocol_sha256": sha256_file(input_root / "judge_protocol.yml"),
                "rubric_sha256": sha256_file(input_root / "judge_rubric.md"),
            }
        )
    if records[0]["cases_sha256"] != records[1]["cases_sha256"]:
        raise RuntimeError("Judge inputs are not identical")
    manifest = {
        "schema_version": 1,
        "status": "two_clean_isolated_judge_assignments_ready",
        "source_cases_sha256": sha256_file(args.cases),
        "cases": len(cases),
        "judges": records,
        "context_inheritance": False,
        "cross_judge_visibility": False,
    }
    write_json(args.output_root / "assignment_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
