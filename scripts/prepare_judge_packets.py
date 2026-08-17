from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeweave.judge_protocol import (
    canonical_json,
    prepare_blind_pair,
    prepare_dual_evidence_blind_pair,
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blinded A/B LLM-as-Judge packets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-a", required=True)
    parser.add_argument("--condition-b", required=True)
    parser.add_argument("--rubric-version", default="judge-v1")
    parser.add_argument(
        "--evidence-mode",
        choices=["shared", "paired"],
        default="shared",
        help="Use paired for system-versus-human reports with different underlying corpora.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.condition_a == args.condition_b:
        raise SystemExit("The two conditions must be distinct.")
    output_files = [
        args.output_dir / "eval_a_packets.jsonl",
        args.output_dir / "eval_b_packets.jsonl",
        args.output_dir / "secret_blind_map.json",
        args.output_dir / "manifest.json",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing judge artifacts: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packets_a = []
    packets_b = []
    mappings = []
    for record in _read_jsonl(args.input):
        prepare = (
            prepare_dual_evidence_blind_pair
            if args.evidence_mode == "paired"
            else prepare_blind_pair
        )
        packet_a, packet_b, mapping = prepare(
            record,
            condition_a=args.condition_a,
            condition_b=args.condition_b,
            rubric_version=args.rubric_version,
            seed=args.seed,
        )
        for packet in (packet_a, packet_b):
            leaks = scan_condition_leaks(
                packet.model_dump(mode="json"),
                [args.condition_a, args.condition_b],
            )
            if leaks:
                raise SystemExit(
                    f"Condition-name leakage in packet {packet.packet_id}: {leaks}"
                )
        packets_a.append(packet_a.model_dump(mode="json"))
        packets_b.append(packet_b.model_dump(mode="json"))
        mappings.append(mapping.model_dump(mode="json"))

    _write_jsonl(output_files[0], packets_a)
    _write_jsonl(output_files[1], packets_b)
    output_files[2].write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_files[3].write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "rubric_version": args.rubric_version,
                "seed": args.seed,
                "evidence_mode": args.evidence_mode,
                "packets": len(packets_a),
                "created_at": datetime.now(UTC).isoformat(),
                "input": str(args.input),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(packets_a)} paired blind packets in {args.output_dir}")


if __name__ == "__main__":
    main()
