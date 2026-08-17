from __future__ import annotations

import argparse
from pathlib import Path

from citeweave.judge_protocol import (
    FeedbackPacket,
    FeedbackResult,
    canonical_json,
    to_feedback_memory_record,
)


def _read_jsonl(path: Path, model: type[FeedbackPacket | FeedbackResult]):
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid record at {path}:{line_number}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Human Proxy feedback and create append-ready policy-memory records."
        )
    )
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite feedback memory records: {args.output}")
    packet_records = _read_jsonl(args.packets, FeedbackPacket)
    result_records = _read_jsonl(args.results, FeedbackResult)
    packets = {packet.packet_id: packet for packet in packet_records}
    results = {result.packet_id: result for result in result_records}
    if len(packets) != len(packet_records):
        raise SystemExit("Duplicate FeedbackPacket packet_id.")
    if len(results) != len(result_records):
        raise SystemExit("Duplicate FeedbackResult packet_id.")
    if set(packets) != set(results):
        raise SystemExit("Feedback packets and results must have identical packet IDs.")

    memory_records = [
        to_feedback_memory_record(results[packet_id], packets[packet_id])
        for packet_id in sorted(packets)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            canonical_json(record.model_dump(mode="json")) + "\n"
            for record in memory_records
        ),
        encoding="utf-8",
    )
    print(f"Validated {len(memory_records)} feedback records into {args.output}")


if __name__ == "__main__":
    main()
