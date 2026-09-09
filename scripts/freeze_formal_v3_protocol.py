from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing protocol freeze attestation")
    raw = args.protocol.read_bytes()
    parsed = yaml.safe_load(raw)
    if parsed.get("status") != "preregistered_data_pending":
        raise SystemExit("Protocol must be frozen before acquisition and model execution")
    datasets = parsed.get("datasets") or []
    if len(datasets) < 4 or len({row["domain"] for row in datasets}) < 4:
        raise SystemExit("At least four new datasets from four domains are required")
    excluded = set(parsed["contamination_policy"]["excluded_existing_topics"])
    overlap = excluded & {row["id"] for row in datasets}
    if overlap:
        raise SystemExit(f"Contaminated dataset IDs cannot be locked: {sorted(overlap)}")
    attestation = {
        "schema_version": 1,
        "status": "frozen_data_pending",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "dataset_ids": [row["id"] for row in datasets],
        "domains": [row["domain"] for row in datasets],
        "expected_total_tasks_if_all_eligible": parsed["benchmark"][
            "expected_total_tasks_if_all_eligible"
        ],
        "primary_hypotheses": [
            row["id"] for row in parsed["hypotheses"] if row.get("primary")
        ],
        "note": (
            "This attests design freeze only. It is not evidence that data were acquired, "
            "eligible, executed, reviewed, or statistically confirmed."
        ),
    }
    write_json(args.output, attestation)
    print(json.dumps(attestation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
