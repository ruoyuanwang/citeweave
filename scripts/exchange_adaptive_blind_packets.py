from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from citeweave.formal_adaptive_review import (
    FORMAL_CONDITIONS,
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    validate_evaluation_result,
)
from citeweave.io import atomic_write_bytes, read_json, sha256_file, write_json
from citeweave.judge_protocol import (
    FeedbackPacket,
    FeedbackResult,
    validate_feedback_result,
)

PacketKind = Literal["feedback", "evaluation"]


def _pending_packets(run_root: Path) -> dict[str, tuple[PacketKind, Path, Path]]:
    pending: dict[str, tuple[PacketKind, Path, Path]] = {}
    for condition in FORMAL_CONDITIONS:
        condition_root = run_root / condition
        state = read_json(condition_root / "state.json")
        for record in state["records"]:
            if record["status"] == "awaiting_feedback":
                kind: PacketKind = "feedback"
                relative = Path(record["feedback_packet_path"])
                packet_id = str(record["feedback_packet_id"])
            elif record["status"] == "awaiting_evaluation":
                kind = "evaluation"
                relative = Path(record["evaluation_packet_path"])
                packet_id = str(record["evaluation_packet_id"])
            else:
                continue
            if packet_id in pending:
                raise ValueError(f"Duplicate pending packet ID: {packet_id}")
            source = condition_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Pending packet is missing: {source}")
            pending[packet_id] = (kind, source, condition_root)
    return pending


def export_batch(run_root: Path, batch_root: Path) -> dict[str, Any]:
    if batch_root.exists():
        raise FileExistsError(f"Refusing to overwrite blind batch: {batch_root}")
    pending = _pending_packets(run_root)
    if not pending:
        raise ValueError("No pending adaptive packets are available for export.")
    rows = []
    for packet_id, (kind, source, _condition_root) in sorted(pending.items()):
        raw = source.read_bytes()
        model = FeedbackPacket if kind == "feedback" else AdaptiveEvaluationPacket
        model.model_validate_json(raw)
        target = batch_root / "packets" / kind / f"{packet_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, raw)
        rows.append(
            {
                "packet_id": packet_id,
                "kind": kind,
                "packet_sha256": sha256_file(target),
            }
        )
    manifest = {
        "schema_version": 1,
        "blind_exchange": True,
        "condition_identity_visible": False,
        "packets": rows,
        "result_directories": {
            "feedback": "results/feedback",
            "evaluation": "results/evaluation",
        },
    }
    write_json(batch_root / "manifest.json", manifest)
    return manifest


def import_results(
    run_root: Path,
    batch_root: Path,
    *,
    require_complete: bool = False,
) -> dict[str, int]:
    manifest = read_json(batch_root / "manifest.json")
    pending = _pending_packets(run_root)
    imported = {"feedback": 0, "evaluation": 0}
    expected: dict[str, tuple[str, str]] = {}
    for row in manifest["packets"]:
        packet_id = str(row["packet_id"])
        kind = str(row["kind"])
        if packet_id in expected:
            raise ValueError(f"Duplicate blind-batch packet ID: {packet_id}")
        if kind not in {"feedback", "evaluation"}:
            raise ValueError(f"Unexpected blind-batch packet kind: {kind}")
        expected[packet_id] = (kind, str(row["packet_sha256"]))
    if set(expected) - set(pending):
        raise ValueError("Blind batch contains packets that are no longer pending.")
    if require_complete and set(expected) != set(pending):
        raise ValueError("Complete blind import requires exact pending-packet coverage.")

    expected_results = {
        (batch_root / "results" / kind / f"{packet_id}.json").resolve()
        for packet_id, (kind, _packet_sha) in expected.items()
    }
    if require_complete:
        actual_results = (
            {
                path.resolve()
                for path in (batch_root / "results").rglob("*")
                if path.is_file()
            }
            if (batch_root / "results").exists()
            else set()
        )
        if actual_results != expected_results:
            raise ValueError(
                "Complete blind import requires exact result-file coverage."
            )

    # Preflight every available result and every immutable destination before
    # writing any inbox file.  This prevents a late schema/hash/conflict failure
    # from producing a partially imported logical batch.
    validated: list[tuple[PacketKind, Path, bytes]] = []
    for packet_id, (kind_raw, packet_sha) in expected.items():
        kind: PacketKind = kind_raw  # type: ignore[assignment]
        pending_kind, packet_path, condition_root = pending[packet_id]
        if kind != pending_kind or sha256_file(packet_path) != packet_sha:
            raise ValueError(f"Pending packet contract changed: {packet_id}")
        result_path = batch_root / "results" / kind / f"{packet_id}.json"
        if not result_path.is_file():
            if require_complete:
                raise ValueError(f"Complete blind import is missing: {packet_id}")
            continue
        if kind == "feedback":
            packet = FeedbackPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            result = FeedbackResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            validate_feedback_result(result, packet)
        else:
            packet = AdaptiveEvaluationPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            result = AdaptiveEvaluationResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            validate_evaluation_result(result, packet)
        inbox = condition_root / "inbox" / kind / f"{packet_id}.json"
        raw = result_path.read_bytes()
        if inbox.exists() and inbox.read_bytes() != raw:
            raise RuntimeError(f"Conflicting immutable adaptive result: {inbox}")
        validated.append((kind, inbox, raw))

    for kind, inbox, raw in validated:
        if not inbox.exists():
            inbox.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(inbox, raw)
        imported[kind] += 1
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export or import condition-blind adaptive-review packets."
    )
    parser.add_argument("action", choices=["export", "import-results"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    result = (
        export_batch(args.run_root.resolve(), args.batch_root.resolve())
        if args.action == "export"
        else import_results(args.run_root.resolve(), args.batch_root.resolve())
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
