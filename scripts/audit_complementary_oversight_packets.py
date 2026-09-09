from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file, write_json


def audit_packets(*, packet_root: Path) -> dict[str, Any]:
    manifest_path = packet_root / "manifest.json"
    manifest = read_json(manifest_path)
    reasons: list[str] = []
    records = manifest.get("records") or []
    case_ids = [str(row.get("case_id")) for row in records]
    if len(case_ids) != len(set(case_ids)):
        reasons.append("duplicate_case_id")
    observed_datasets = set()
    observed_task_types: dict[str, int] = {}
    for row in records:
        case_id = str(row["case_id"])
        observed_datasets.add(str(row["dataset_id"]))
        task_type = str(row["task_type"])
        observed_task_types[task_type] = observed_task_types.get(task_type, 0) + 1
        source = Path(row["source_pack"])
        if not source.is_file() or sha256_file(source) != row["source_pack_sha256"]:
            reasons.append(f"{case_id}:source_pack_hash_mismatch")
        public_path = packet_root / row["public_path"]
        internal_path = packet_root / row["internal_path"]
        if (
            not public_path.is_file()
            or sha256_file(public_path) != row["public_sha256"]
        ):
            reasons.append(f"{case_id}:public_packet_hash_mismatch")
            continue
        if (
            not internal_path.is_file()
            or sha256_file(internal_path) != row["internal_sha256"]
        ):
            reasons.append(f"{case_id}:internal_packet_hash_mismatch")
            continue
        public = read_json(public_path)
        internal = read_json(internal_path)
        serialized_public = json.dumps(public, ensure_ascii=False, sort_keys=True)
        if any(
            forbidden in serialized_public
            for forbidden in ("role_map", "generation_condition", "gold_answer")
        ):
            reasons.append(f"{case_id}:public_role_or_outcome_leakage")
        if not public.get("evidence_set_a") or not public.get("evidence_set_b"):
            reasons.append(f"{case_id}:two_sided_evidence_missing")
        if set((internal.get("role_map") or {}).values()) != {
            "support",
            "challenge",
        }:
            reasons.append(f"{case_id}:invalid_internal_role_map")
        canonical_public_sha256 = hashlib.sha256(
            json.dumps(
                public,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if (
            internal.get("public_packet_canonical_sha256")
            != canonical_public_sha256
        ):
            reasons.append(f"{case_id}:internal_public_hash_mismatch")
        required_response = {
            "verdict",
            "decisive_evidence_ids",
            "invalid_operator_steps",
            "failure_mode",
            "minimal_rewrite",
            "rationale",
        }
        if set(public.get("response_schema") or {}) != required_response:
            reasons.append(f"{case_id}:response_schema_mismatch")
    checks = {
        "status_pre_review": manifest.get("status")
        == "packets_built_before_real_review",
        "human_outcomes_absent": manifest.get("human_outcomes_inspected") is False,
        "dataset_count_matches": len(observed_datasets) == manifest.get("datasets"),
        "case_count_matches": len(records) == manifest.get("cases"),
        "task_type_counts_match": dict(sorted(observed_task_types.items()))
        == manifest.get("task_type_counts"),
        "all_artifacts_valid": not reasons,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reasons": list(dict.fromkeys(reasons)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "datasets": len(observed_datasets),
        "cases": len(records),
        "task_type_counts": dict(sorted(observed_task_types.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_packets(packet_root=args.packet_root)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
