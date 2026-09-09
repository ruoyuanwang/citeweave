from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.article_expert_evaluation import (
    ClaimExpertRating,
    HolisticExpertRating,
    PairwiseExpertPreference,
    analyze_article_expert_panel,
)
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--analysis-amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/expert_evaluation_analysis_amendment_003.yml"
        ),
    )
    parser.add_argument(
        "--analysis-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_analysis_amendment_003_freeze.json"
        ),
    )
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    freeze = read_json(args.freeze)
    if freeze.get("sha256") != protocol_hash:
        raise SystemExit("Article expert-evaluation protocol differs from freeze artifact")
    amendment_hash = sha256_file(args.analysis_amendment)
    amendment_freeze = read_json(args.analysis_amendment_freeze)
    if amendment_freeze.get("sha256") != amendment_hash:
        raise SystemExit("Article analysis amendment differs from freeze artifact")
    amendment = yaml.safe_load(args.analysis_amendment.read_text(encoding="utf-8"))
    if amendment.get("base_protocol_sha256") != protocol_hash:
        raise SystemExit("Article analysis amendment protocol hash mismatch")
    for label, artifact in (amendment.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Article analysis implementation mismatch: {label}")
    payload = read_json(args.input)
    if payload.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Expert-rating input does not match the frozen protocol")

    holistic = [HolisticExpertRating(**row) for row in payload.get("holistic_ratings", [])]
    claims = [ClaimExpertRating(**row) for row in payload.get("claim_ratings", [])]
    preferences = [
        PairwiseExpertPreference(**row)
        for row in payload.get("forced_pairwise_preferences", [])
    ]
    result = analyze_article_expert_panel(
        holistic,
        claims,
        preferences,
        bootstrap_samples=args.bootstrap_samples,
    )
    result["protocol_sha256"] = protocol_hash
    result["analysis_amendment_sha256"] = amendment_hash
    result["input_sha256"] = sha256_file(args.input)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
