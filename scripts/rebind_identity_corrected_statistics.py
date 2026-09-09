"""Rebind unchanged scale pairs/statistics to corrected benchmark file hashes."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json


def rebound_statistics(original: dict, *, pair_sha256: str, correction_hash: str) -> dict:
    result = copy.deepcopy(original)
    result["amendment_id"] = "formal_v3_statistics_identity_rebound"
    result["scale_pair_artifact"]["path"] = "scale_pairs.json"
    result["scale_pair_artifact"]["sha256"] = pair_sha256
    # Statistical dependency chain remains the original graph/neural amendments;
    # data identity correction is a separate provenance binding, not a new test.
    result["identity_correction_amendment_sha256"] = correction_hash
    result["reason"] = [
        *original["reason"],
        "Only benchmark identity bindings changed after the prospective author correction; pair membership, hypotheses, estimands, tests and budgets are unchanged.",
    ]
    assert result["primary_hypotheses"] == original["primary_hypotheses"]
    assert result["statistics"] == original["statistics"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--original-pairs", type=Path, required=True)
    parser.add_argument("--original-statistics", type=Path, required=True)
    parser.add_argument("--original-statistics-freeze", type=Path, required=True)
    parser.add_argument("--correction-amendment", type=Path, required=True)
    parser.add_argument("--correction-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("Refusing to overwrite a rebound statistics freeze")
    correction_hash = sha256_file(args.correction_amendment)
    if read_json(args.correction_freeze)["amendment_sha256"] != correction_hash:
        raise SystemExit("Correction amendment freeze mismatch")
    original = yaml.safe_load(args.original_statistics.read_text(encoding="utf-8"))
    original_freeze = read_json(args.original_statistics_freeze)
    if original_freeze["amendment_sha256"] != sha256_file(args.original_statistics):
        raise SystemExit("Original statistics freeze mismatch")
    if original["scale_pair_artifact"]["sha256"] != sha256_file(args.original_pairs):
        raise SystemExit("Original scale-pair freeze mismatch")
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    if construction["status"] != "constructed_not_executed" or len(construction["records"]) != 4:
        raise SystemExit("All four corrected primary benchmarks are required")
    for name in (
        "formal_v3_runs",
        "formal_v3_replication_runs",
        "formal_v3_complexity_extension_runs",
    ):
        if any(p.is_file() for p in (Path("experiments/graph_discovery_v2") / name).rglob("*")):
            raise SystemExit(
                "Cannot rebind prospective statistics after a formal result file exists"
            )
    spec = importlib.util.spec_from_file_location(
        "registered_scale_pairs", Path(__file__).with_name("build_formal_v3_scale_pairs.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = read_json(args.original_pairs)
    expected_pairs = {row["pair_id"]: row for row in pairs["pairs"]}
    actual_pairs = {}
    hashes = {}
    for record in construction["records"]:
        topic = record["dataset_id"]
        path = args.benchmark_root / topic / "benchmark.json"
        benchmark = read_json(path)
        if sha256_file(path) != record["benchmark_sha256"]:
            raise SystemExit("Corrected benchmark hash mismatch")
        if benchmark["formal_v3"]["original_benchmark_sha256"] != pairs["benchmark_hashes"][topic]:
            raise SystemExit("Incorrect old-to-new benchmark mapping")
        if benchmark["formal_v3"]["author_identity_correction_sha256"] != correction_hash:
            raise SystemExit("Incorrect correction identity")
        actual_pairs.update({r["pair_id"]: r for r in module.build_pairs(benchmark)})
        hashes[topic] = sha256_file(path)
    if actual_pairs != expected_pairs or len(actual_pairs) != 14:
        raise SystemExit("Pair membership/anchors changed; cannot identity-rebind")
    rebound_pairs = copy.deepcopy(pairs)
    rebound_pairs["benchmark_hashes"] = hashes
    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "scale_pairs.json", rebound_pairs)
    stats = rebound_statistics(
        original,
        pair_sha256=sha256_file(args.output_root / "scale_pairs.json"),
        correction_hash=correction_hash,
    )
    stats_path = args.output_root / "statistics.yml"
    stats_path.write_text(
        yaml.safe_dump(stats, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    now = datetime.now(UTC).isoformat()
    freeze = {
        **original_freeze,
        "amendment_id": stats["amendment_id"],
        "amendment_sha256": sha256_file(stats_path),
        "frozen_at_utc": now,
        "prior_amendment_sha256": stats["prior_amendments"],
        "parent_statistics_amendment_sha256": sha256_file(args.original_statistics),
        "identity_correction_amendment_sha256": correction_hash,
        "note": "Identity rebinding only; original statistical decisions remain unchanged. No formal outcome file exists.",
    }
    write_json(args.output_root / "statistics_freeze.json", freeze)
    write_json(
        args.output_root / "binding_receipt.json",
        {
            "schema_version": 1,
            "status": "identity_rebound_before_model_execution",
            "created_at_utc": now,
            "correction_amendment_sha256": correction_hash,
            "unchanged_pair_groups": len(actual_pairs),
            "pair_membership_and_anchors_exactly_equal": True,
            "hypotheses_and_statistics_exactly_equal": True,
            "old_benchmark_hashes": pairs["benchmark_hashes"],
            "new_benchmark_hashes": hashes,
            "old_statistics_sha256": sha256_file(args.original_statistics),
            "new_statistics_sha256": sha256_file(stats_path),
            "old_pair_artifact_sha256": sha256_file(args.original_pairs),
            "new_pair_artifact_sha256": sha256_file(args.output_root / "scale_pairs.json"),
        },
    )
    print(json.dumps({"status": "identity_rebound", "unchanged_pairs": len(actual_pairs)}))


if __name__ == "__main__":
    main()
