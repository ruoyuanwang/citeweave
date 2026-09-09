from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.feedback_repair_trial import (
    assess_feedback_repair_trial_readiness,
    build_feedback_repair_trial_plan,
)
from citeweave.io import read_json, sha256_file, write_json


def _validate_protocol(protocol_path: Path, freeze_path: Path) -> str:
    protocol_hash = sha256_file(protocol_path)
    freeze = read_json(freeze_path)
    if freeze.get("sha256") != protocol_hash:
        raise SystemExit("Feedback-repair protocol differs from its freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Feedback-repair implementation mismatch: {label}")
    return protocol_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--cases-per-topic", type=int, default=12)
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
    protocol_hash = _validate_protocol(args.protocol, args.protocol_freeze)
    if args.audit:
        result = assess_feedback_repair_trial_readiness(
            args.worklist, cases_per_topic=args.cases_per_topic
        )
        result["protocol_sha256"] = protocol_hash
        write_json(args.output, result)
    else:
        result = build_feedback_repair_trial_plan(
            args.worklist,
            output_path=args.output,
            cases_per_topic=args.cases_per_topic,
        )
        result["protocol_sha256"] = protocol_hash
        write_json(args.output, result)
        write_json(
            args.output.with_suffix(".freeze.json"),
            {
                "schema_version": 1,
                "sha256": sha256_file(args.output),
                "provider_outcomes_present": False,
                "human_evaluation_outcomes_present": False,
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.audit and result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
