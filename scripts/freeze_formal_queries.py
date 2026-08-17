from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from citeweave.formal_protocol import (
    canonical_json,
    query_payload,
    query_set_sha256,
    sha256_json,
    verify_frozen_query_registry,
)
from citeweave.io import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets.yml"
DEFAULT_ARCHIVE = ROOT / "experiments" / "query_judgments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "query_judge_pending":
        raise SystemExit("Registry must be in query_judge_pending state.")
    packet = json.loads(args.packet.read_bytes())
    packet_hash = packet.get("packet_sha256")
    packet_without_hash = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet_hash != sha256_json(packet_without_hash):
        raise SystemExit("Query Judge packet hash is invalid.")
    if packet.get("query_set_sha256") != query_set_sha256(registry["datasets"]):
        raise SystemExit("Query Judge packet does not match the current registry.")
    judgment = json.loads(args.judgment.read_bytes())
    judgment_bytes = (canonical_json(judgment) + "\n").encode("utf-8")
    if judgment.get("judge_role") != "codex_query_judge":
        raise SystemExit("Judgment must declare judge_role=codex_query_judge.")
    if not judgment.get("rubric_version"):
        raise SystemExit("Judgment lacks rubric_version.")
    if not judgment.get("prompt_sha256"):
        raise SystemExit("Judgment lacks prompt_sha256.")
    if judgment["prompt_sha256"] != packet.get("prompt_sha256"):
        raise SystemExit("Judgment prompt hash does not match the Query Judge packet.")
    if judgment.get("packet_sha256") != packet_hash:
        raise SystemExit("Judgment packet hash does not match.")
    decisions = judgment.get("decisions")
    if not isinstance(decisions, list):
        raise SystemExit("Judgment decisions must be a list.")
    by_id = {item.get("id"): item for item in decisions if isinstance(item, dict)}
    datasets = registry["datasets"]
    expected_ids = {item["id"] for item in datasets}
    if set(by_id) != expected_ids:
        raise SystemExit("Judgment IDs do not exactly match the formal datasets.")
    rejected = {
        item_id: value.get("decision")
        for item_id, value in by_id.items()
        if value.get("decision") != "accept"
    }
    if rejected:
        raise SystemExit(f"All queries must be accepted before freezing: {rejected}")
    if any(not str(value.get("reason") or "").strip() for value in by_id.values()):
        raise SystemExit("Every Query Judge decision requires a reason.")

    for item in datasets:
        item["query_status"] = "frozen"
        item["query_sha256"] = sha256_json(query_payload(item))
    judgment_sha = sha256_json(judgment)
    registry["status"] = "frozen"
    registry["query_freeze_attestation"] = {
        "attestation_version": 1,
        "judged_at": datetime.now(UTC).isoformat(),
        "judge_role": "codex_query_judge",
        "rubric_version": judgment["rubric_version"],
        "judge_prompt_sha256": judgment["prompt_sha256"],
        "judge_packet_sha256": packet_hash,
        "judgment_sha256": judgment_sha,
        "generator_response_sha256": registry["generator"]["response_sha256"],
        "human_reference_registry_sha256": registry[
            "human_reference_registry_sha256"
        ],
        "query_set_sha256": query_set_sha256(datasets),
        "accepted_ids": sorted(expected_ids),
    }
    verify_frozen_query_registry(registry)

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive_dir / f"{judgment_sha}.json"
    if archive_path.exists() and archive_path.read_bytes() != judgment_bytes:
        raise SystemExit(f"Hash collision or altered judgment archive: {archive_path}")
    if not archive_path.exists():
        atomic_write_bytes(archive_path, judgment_bytes)
    payload = yaml.safe_dump(registry, allow_unicode=True, sort_keys=False).encode("utf-8")
    atomic_write_bytes(args.registry, payload)
    print(args.registry)


if __name__ == "__main__":
    main()
