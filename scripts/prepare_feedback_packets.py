from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeweave.judge_protocol import (
    canonical_json,
    prepare_feedback_packet,
    scan_condition_leaks,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare anonymous, evidence-bounded Human Proxy packets."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--rubric-version", default="feedback-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    packet_path = args.output_dir / "feedback_packets.jsonl"
    map_path = args.output_dir / "secret_feedback_map.json"
    manifest_path = args.output_dir / "feedback_manifest.json"
    existing = [path for path in (packet_path, map_path, manifest_path) if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing feedback artifacts: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packets = []
    secret_map = []
    for record in _read_jsonl(args.input):
        risk_notice = record.get("risk_notice")
        if not isinstance(risk_notice, dict):
            raise SystemExit(
                "Every Human Proxy record requires a visible risk_notice object."
            )
        packet = prepare_feedback_packet(
            record,
            condition=args.condition,
            rubric_version=args.rubric_version,
            seed=args.seed,
            risk_notice=risk_notice,
        )
        leaks = scan_condition_leaks(packet.model_dump(mode="json"), [args.condition])
        if leaks:
            raise SystemExit(
                f"Condition-name leakage in feedback packet {packet.packet_id}: {leaks}"
            )
        packets.append(packet.model_dump(mode="json"))
        secret_map.append(
            {
                "packet_id": packet.packet_id,
                "sample_id": packet.sample_id,
                "condition": args.condition,
            }
        )

    packet_path.write_text(
        "".join(canonical_json(packet) + "\n" for packet in packets),
        encoding="utf-8",
    )
    map_path.write_text(
        json.dumps(secret_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "rubric_version": args.rubric_version,
                "seed": args.seed,
                "packets": len(packets),
                "created_at": datetime.now(UTC).isoformat(),
                "input": str(args.input),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(packets)} feedback packets in {args.output_dir}")


if __name__ == "__main__":
    main()
