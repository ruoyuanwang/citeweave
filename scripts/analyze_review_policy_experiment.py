from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.review_policy_experiment import (
    ReviewPolicyOutcome,
    analyze_review_policy_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_002_cluster_inference.yml"
        ),
    )
    parser.add_argument(
        "--analysis-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "review_policy_amendment_002_cluster_inference_freeze.json"
        ),
    )
    args = parser.parse_args()
    payload = read_json(args.input)
    protocol_hash = sha256_file(args.protocol)
    freeze = read_json(args.freeze)
    if freeze.get("sha256") != protocol_hash:
        raise SystemExit("Review-policy protocol hash differs from its freeze artifact")
    if payload.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Review-policy input does not match the frozen protocol")
    amendment_hash = sha256_file(args.analysis_amendment)
    amendment_freeze = read_json(args.analysis_amendment_freeze)
    if amendment_freeze.get("sha256") != amendment_hash:
        raise SystemExit("Review-policy analysis amendment differs from freeze")
    amendment = yaml.safe_load(args.analysis_amendment.read_text(encoding="utf-8"))
    if amendment.get("base_protocol_sha256") != protocol_hash:
        raise SystemExit("Review-policy analysis amendment protocol mismatch")
    implementations = amendment.get("implementations") or {}
    if implementations.get(Path(__file__).name) != sha256_file(Path(__file__)):
        raise SystemExit("Review-policy analysis script differs from amendment")
    module_path = Path("src/citeweave/review_policy_experiment.py")
    if implementations.get(module_path.name) != sha256_file(module_path):
        raise SystemExit("Review-policy analysis module differs from amendment")
    records = []
    for row in payload.get("records") or []:
        normalized = dict(row)
        normalized["active_feedback_ids"] = tuple(
            normalized.get("active_feedback_ids") or []
        )
        records.append(ReviewPolicyOutcome(**normalized))
    result = analyze_review_policy_panel(records)
    result["protocol_sha256"] = protocol_hash
    result["analysis_amendment_sha256"] = amendment_hash
    result["input_sha256"] = sha256_file(args.input)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
