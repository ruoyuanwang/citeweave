from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.article_expert_evaluation import ClaimExpertRating
from citeweave.article_quality_controls import analyze_length_normalized_claim_quality
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_004_length_parity_and_density.yml"
        ),
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_004_length_parity_and_density_freeze.json"
        ),
    )
    args = parser.parse_args()

    amendment_hash = sha256_file(args.amendment)
    amendment_freeze = read_json(args.amendment_freeze)
    if amendment_freeze.get("sha256") != amendment_hash:
        raise SystemExit("Article quality-control amendment differs from freeze")
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    for label, artifact in (amendment.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Article quality-control implementation mismatch: {label}")

    manifest = read_json(args.packet_manifest)
    controls = manifest.get("article_quality_controls") or {}
    if controls.get("status") != "pre_review_controls_passed":
        raise SystemExit("Packet manifest lacks passed pre-review quality controls")
    ratings = read_json(args.ratings)
    result = analyze_length_normalized_claim_quality(
        [ClaimExpertRating(**row) for row in ratings.get("claim_ratings", [])],
        controls.get("articles") or [],
    )
    result["packet_manifest_sha256"] = sha256_file(args.packet_manifest)
    result["ratings_sha256"] = sha256_file(args.ratings)
    result["amendment_sha256"] = amendment_hash
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
