from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.feedback_repair_trial import (
    RepairTrialOutcome,
    analyze_feedback_repair_trial,
)
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-freeze", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/human_review_v2/feedback_repair_trial_protocol.yml"),
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/feedback_repair_trial_protocol_freeze.json"
        ),
    )
    args = parser.parse_args()
    protocol_hash = sha256_file(args.protocol)
    if read_json(args.protocol_freeze).get("sha256") != protocol_hash:
        raise SystemExit("Feedback-repair protocol differs from its freeze")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Feedback-repair implementation mismatch: {label}")
    plan_hash = sha256_file(args.plan)
    plan_freeze = read_json(args.plan_freeze)
    if plan_freeze.get("sha256") != plan_hash:
        raise SystemExit("Feedback-repair plan differs from its pre-outcome freeze")
    payload = read_json(args.input)
    if payload.get("plan_sha256") != plan_hash:
        raise SystemExit("Feedback-repair outcomes do not bind the frozen plan")
    rows = []
    for row in payload.get("resolved_outcomes") or []:
        rows.append(
            RepairTrialOutcome(
                **{
                    **row,
                    "primary_reviewer_ids": tuple(row["primary_reviewer_ids"]),
                }
            )
        )
    result = analyze_feedback_repair_trial(rows)
    result["protocol_sha256"] = protocol_hash
    result["plan_sha256"] = plan_hash
    result["input_sha256"] = sha256_file(args.input)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
