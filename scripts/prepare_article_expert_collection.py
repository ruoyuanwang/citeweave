from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.article_expert_collection import build_article_expert_collection
from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_007_blinded_evidence_collection.yml"
        ),
    )
    parser.add_argument(
        "--amendment-freeze",
        type=Path,
        default=Path(
            "experiments/article_quality_v2/"
            "expert_evaluation_amendment_007_blinded_evidence_collection_freeze.json"
        ),
    )
    args = parser.parse_args()
    amendment_hash = sha256_file(args.amendment)
    freeze = read_json(args.amendment_freeze)
    if freeze.get("sha256") != amendment_hash:
        raise SystemExit("Expert collection amendment differs from its freeze")
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    for label, artifact in amendment.get("implementation", {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"Expert collection implementation mismatch: {label}")
    manifest = build_article_expert_collection(
        args.intake,
        args.packet_manifest,
        output_dir=args.output_dir,
    )
    manifest["amendment_sha256"] = amendment_hash
    write_json(args.output_dir / "collection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
