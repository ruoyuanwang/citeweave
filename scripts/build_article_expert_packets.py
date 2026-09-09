from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.article_expert_packets import build_article_expert_packets
from citeweave.io import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--evaluator-roster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--quality-control-amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_005_role_separation.yml"
        ),
    )
    parser.add_argument(
        "--quality-control-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_005_role_separation_freeze.json"
        ),
    )
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=Path("experiments/article_quality_v2/expert_evaluation_protocol.yml"),
    )
    parser.add_argument(
        "--previous-amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_004_length_parity_and_density.yml"
        ),
    )
    parser.add_argument(
        "--previous-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_004_length_parity_and_density_freeze.json"
        ),
    )
    args = parser.parse_args()
    amendment_hash = sha256_file(args.quality_control_amendment)
    freeze = read_json(args.quality_control_amendment_freeze)
    if freeze.get("sha256") != amendment_hash:
        raise SystemExit("Article quality-control amendment differs from freeze")
    amendment = yaml.safe_load(
        args.quality_control_amendment.read_text(encoding="utf-8")
    )
    if amendment.get("base_protocol_sha256") != sha256_file(args.base_protocol):
        raise SystemExit("Article expert amendment base-protocol chain mismatch")
    previous_hash = sha256_file(args.previous_amendment)
    previous_freeze = read_json(args.previous_amendment_freeze)
    if previous_freeze.get("sha256") != previous_hash:
        raise SystemExit("Previous article expert amendment differs from its freeze")
    if amendment.get("previous_amendment_sha256") != previous_hash:
        raise SystemExit("Article expert amendment previous-amendment chain mismatch")
    for label, artifact in (amendment.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Article quality-control implementation mismatch: {label}")
    manifest = build_article_expert_packets(
        args.intake,
        args.evaluator_roster,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    manifest["quality_control_amendment_sha256"] = amendment_hash
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
