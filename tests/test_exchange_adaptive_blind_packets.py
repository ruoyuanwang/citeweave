from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from citeweave.judge_protocol import prepare_feedback_packet

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "exchange_adaptive_blind_packets.py"
)
SPEC = importlib.util.spec_from_file_location("exchange_adaptive_blind_packets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_condition_blind_exchange_exports_and_routes_valid_feedback(tmp_path: Path):
    run_root = tmp_path / "run"
    packet = prepare_feedback_packet(
        {
            "sample_id": "sample-1",
            "canonical_evidence": [{"evidence_id": "E001", "statement": "Fact."}],
            "candidates": {"hidden-condition": "Candidate fact."},
        },
        condition="hidden-condition",
        rubric_version="feedback-v2",
        seed=42,
        risk_notice={"severity": "high", "message": "Review the visible claim."},
    )
    for condition in ("always_review", "static_review", "adaptive_review"):
        condition_root = run_root / condition
        condition_root.mkdir(parents=True)
        if condition == "always_review":
            relative = Path("packets") / "feedback" / f"{packet.packet_id}.json"
            packet_path = condition_root / relative
            packet_path.parent.mkdir(parents=True)
            packet_path.write_text(packet.model_dump_json(), encoding="utf-8")
            records = [
                {
                    "status": "awaiting_feedback",
                    "feedback_packet_id": packet.packet_id,
                    "feedback_packet_path": relative.as_posix(),
                }
            ]
            completed = False
        else:
            records = []
            completed = True
        (condition_root / "state.json").write_text(
            json.dumps({"records": records, "completed": completed}),
            encoding="utf-8",
        )

    batch = tmp_path / "blind-batch"
    manifest = MODULE.export_batch(run_root, batch)
    visible = json.dumps(manifest, sort_keys=True)

    assert "always_review" not in visible
    assert "hidden-condition" not in visible
    assert manifest["packets"][0]["packet_id"] == packet.packet_id

    result_path = batch / "results" / "feedback" / f"{packet.packet_id}.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "packet_id": packet.packet_id,
                "judge_id": "feedback",
                "decision": "accept",
                "confidence": 0.9,
                "reason": "The visible claim is supported.",
                "evidence_ids": ["E001"],
            }
        ),
        encoding="utf-8",
    )

    imported = MODULE.import_results(run_root, batch)
    inbox = (
        run_root
        / "always_review"
        / "inbox"
        / "feedback"
        / f"{packet.packet_id}.json"
    )

    assert imported == {"feedback": 1, "evaluation": 0}
    assert inbox.read_bytes() == result_path.read_bytes()
