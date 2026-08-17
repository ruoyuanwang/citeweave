from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from citeweave.judge_protocol import (
    BlindMap,
    BlindPacket,
    JudgeResult,
    aggregate_resolved_results,
    build_adjudication_packet,
    canonical_json,
    detect_conflicts,
    resolve_packet_results,
    scan_condition_leaks,
    validate_blind_packet,
    validate_judge_result,
)


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid record at {path}:{line_number}") from exc
    return records


def _index_unique(records: list[Any], field: str) -> dict[str, Any]:
    result = {}
    for record in records:
        key = str(getattr(record, field))
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = record
    return result


def _load_maps(path: Path) -> dict[str, BlindMap]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise TypeError("Blind-map file must contain a JSON array")
    return _index_unique([BlindMap.model_validate(value) for value in values], "packet_id")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, adjudicate, decode, and aggregate blind Judge results."
    )
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--packets-a", type=Path, required=True)
    parser.add_argument("--packets-b", type=Path, required=True)
    parser.add_argument("--blind-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    args = parser.parse_args()

    output = args.output_dir / "judge_metrics.json"
    resolved_output = args.output_dir / "resolved_judgments.jsonl"
    conflict_output = args.output_dir / "adjudication_packets.jsonl"
    existing = [path for path in (output, resolved_output) if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite completed scoring artifacts: {existing}")

    judge_a = _index_unique(_read_jsonl(args.judge_a, JudgeResult), "packet_id")
    judge_b = _index_unique(_read_jsonl(args.judge_b, JudgeResult), "packet_id")
    packets_a = _index_unique(_read_jsonl(args.packets_a, BlindPacket), "packet_id")
    packets_b = _index_unique(_read_jsonl(args.packets_b, BlindPacket), "packet_id")
    mappings = _load_maps(args.blind_map)
    adjudications = (
        _index_unique(_read_jsonl(args.adjudications, JudgeResult), "packet_id")
        if args.adjudications
        else {}
    )
    expected = set(mappings)
    observed_sets = {
        "judge-a": set(judge_a),
        "judge-b": set(judge_b),
        "packets-a": set(packets_a),
        "packets-b": set(packets_b),
    }
    if any(observed != expected for observed in observed_sets.values()):
        raise SystemExit(
            "Judge results, both packet sets, and blind map must have identical packet IDs."
        )
    if adjudications and not set(adjudications) <= expected:
        raise SystemExit("--adjudications contains packet IDs outside this comparison.")
    for packet_id in sorted(expected):
        packet_a = packets_a[packet_id]
        packet_b = packets_b[packet_id]
        mapping = mappings[packet_id]
        try:
            validate_blind_packet(packet_a)
            validate_blind_packet(packet_b)
            if packet_a.judge_id != "eval_a" or packet_b.judge_id != "eval_b":
                raise ValueError("packet files contain the wrong Judge identity")
            if (
                packet_a.sample_id != mapping.sample_id
                or packet_b.sample_id != mapping.sample_id
            ):
                raise ValueError("blind packet and secret map sample IDs differ")
            validate_judge_result(
                judge_a[packet_id],
                packet_a,
                expected_judge_id="eval_a",
            )
            validate_judge_result(
                judge_b[packet_id],
                packet_b,
                expected_judge_id="eval_b",
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid blind Judge exchange for {packet_id}: {exc}") from exc

    pending = []
    conflict_ids: set[str] = set()
    for packet_id in sorted(expected):
        conflicts = detect_conflicts(judge_a[packet_id], judge_b[packet_id], mappings[packet_id])
        if conflicts:
            conflict_ids.add(packet_id)
        if conflicts and packet_id not in adjudications:
            packet = build_adjudication_packet(
                packets_a[packet_id],
                judge_a[packet_id],
                judge_b[packet_id],
                mappings[packet_id],
            )
            condition_names = sorted(
                {
                    mappings[packet_id].assignments["eval_a"].A,
                    mappings[packet_id].assignments["eval_a"].B,
                }
            )
            leaks = scan_condition_leaks(packet, condition_names)
            if leaks:
                raise SystemExit(
                    f"Condition-name leakage in adjudication packet {packet_id}: {leaks}"
                )
            pending.append(packet)
    unexpected_adjudications = set(adjudications) - conflict_ids
    if unexpected_adjudications:
        raise SystemExit(
            "Adjudications are allowed only for detected conflicts; unexpected packet IDs: "
            f"{sorted(unexpected_adjudications)}"
        )
    if pending:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(conflict_output, pending)
        raise SystemExit(
            f"{len(pending)} packets require blind adjudication; "
            f"packets were written to {conflict_output}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = []
    for packet_id in sorted(expected):
        adjudication = adjudications.get(packet_id)
        if adjudication is not None:
            try:
                validate_judge_result(
                    adjudication,
                    packets_a[packet_id],
                    expected_judge_id="adjudicator",
                )
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid blind adjudication for {packet_id}: {exc}"
                ) from exc
        resolved.append(
            resolve_packet_results(
                judge_a[packet_id],
                judge_b[packet_id],
                mappings[packet_id],
                adjudication=adjudication,
            )
        )
    _write_jsonl(resolved_output, resolved)
    output.write_text(
        json.dumps(aggregate_resolved_results(resolved), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    scoring_manifest = {
        "schema_version": 1,
        "role": "read_only_blind_evaluation_and_adjudication",
        "judges_may_modify_pipeline_or_source_artifacts": False,
        "input_sha256": {
            "judge_a": _sha256(args.judge_a),
            "judge_b": _sha256(args.judge_b),
            "packets_a": _sha256(args.packets_a),
            "packets_b": _sha256(args.packets_b),
            "blind_map": _sha256(args.blind_map),
            **(
                {"adjudications": _sha256(args.adjudications)}
                if args.adjudications
                else {}
            ),
        },
        "packet_count": len(resolved),
        "conflict_count": len(conflict_ids),
    }
    (args.output_dir / "scoring_manifest.json").write_text(
        json.dumps(scoring_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(resolved)} blind packets; metrics saved to {output}")


if __name__ == "__main__":
    main()
