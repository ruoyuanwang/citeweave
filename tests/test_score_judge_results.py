from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from citeweave.judge_protocol import (
    BlindMap,
    BlindPacket,
    JudgeResult,
    prepare_blind_pair,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_judge_results.py"


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _exchange(tmp_path: Path) -> dict[str, Path | BlindPacket | BlindMap]:
    record = {
        "sample_id": "locked_1",
        "question": "Which answer is better supported?",
        "canonical_evidence": {"E001": "The network contains 10 nodes."},
        "candidates": {
            "graph_rag": "The network contains 10 nodes.",
            "no_rag": "The network contains nine nodes.",
        },
    }
    packet_a, packet_b, mapping = prepare_blind_pair(
        record,
        condition_a="graph_rag",
        condition_b="no_rag",
        rubric_version="graph-frozen-v1",
        seed=42,
    )

    def result(packet: BlindPacket, judge_id: str) -> JudgeResult:
        assignment = mapping.assignments[judge_id]
        full_slot = "A" if assignment.A == "graph_rag" else "B"
        other_slot = "B" if full_slot == "A" else "A"
        return JudgeResult.model_validate(
            {
                "packet_id": packet.packet_id,
                "judge_id": judge_id,
                "candidates": [
                    {
                        "slot": full_slot,
                        "claims": [
                            {
                                "claim": "The network contains 10 nodes.",
                                "verdict": "supported",
                                "evidence_ids": ["E001"],
                            }
                        ],
                        "completeness_score": 5,
                    },
                    {
                        "slot": other_slot,
                        "claims": [
                            {
                                "claim": "The network contains nine nodes.",
                                "verdict": "contradicted",
                                "evidence_ids": ["E001"],
                            }
                        ],
                        "completeness_score": 2,
                    },
                ],
                "preference": full_slot,
                "rationale": "The preferred answer matches the visible evidence.",
            }
        )

    paths: dict[str, Path | BlindPacket | BlindMap] = {
        "judge_a": tmp_path / "judge_a.jsonl",
        "judge_b": tmp_path / "judge_b.jsonl",
        "packets_a": tmp_path / "packets_a.jsonl",
        "packets_b": tmp_path / "packets_b.jsonl",
        "blind_map": tmp_path / "map.json",
        "output": tmp_path / "resolved",
        "packet_a": packet_a,
        "mapping": mapping,
    }
    _jsonl(paths["judge_a"], [result(packet_a, "eval_a")])  # type: ignore[arg-type]
    _jsonl(paths["judge_b"], [result(packet_b, "eval_b")])  # type: ignore[arg-type]
    _jsonl(paths["packets_a"], [packet_a])  # type: ignore[arg-type]
    _jsonl(paths["packets_b"], [packet_b])  # type: ignore[arg-type]
    paths["blind_map"].write_text(  # type: ignore[union-attr]
        json.dumps([mapping.model_dump(mode="json")]),
        encoding="utf-8",
    )
    return paths


def _run(paths: dict[str, Path | BlindPacket | BlindMap]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--judge-a",
            str(paths["judge_a"]),
            "--judge-b",
            str(paths["judge_b"]),
            "--packets-a",
            str(paths["packets_a"]),
            "--packets-b",
            str(paths["packets_b"]),
            "--blind-map",
            str(paths["blind_map"]),
            "--output-dir",
            str(paths["output"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_scores_only_a_hash_bound_dual_judge_exchange(tmp_path: Path):
    paths = _exchange(tmp_path)
    completed = _run(paths)
    assert completed.returncode == 0, completed.stderr
    output = paths["output"]
    assert isinstance(output, Path)
    manifest = json.loads((output / "scoring_manifest.json").read_text())
    assert manifest["judges_may_modify_pipeline_or_source_artifacts"] is False
    assert manifest["packet_count"] == 1
    row = json.loads(
        (output / "resolved_judgments.jsonl").read_text(encoding="utf-8")
    )
    assert row["source"] == "dual_consensus"
    assert row["preference"] == "graph_rag"


def test_rejects_unknown_evidence_citation(tmp_path: Path):
    paths = _exchange(tmp_path)
    judge_a_path = paths["judge_a"]
    assert isinstance(judge_a_path, Path)
    row = json.loads(judge_a_path.read_text(encoding="utf-8"))
    row["candidates"][0]["claims"][0]["evidence_ids"] = ["E999"]
    judge_a_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    completed = _run(paths)
    assert completed.returncode != 0
    assert "not visible" in (completed.stdout + completed.stderr)


def test_rejects_tampered_second_judge_packet(tmp_path: Path):
    paths = _exchange(tmp_path)
    packets_b_path = paths["packets_b"]
    assert isinstance(packets_b_path, Path)
    row = json.loads(packets_b_path.read_text(encoding="utf-8"))
    row["candidate_a"] = "Tampered after packet preparation."
    packets_b_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    completed = _run(paths)
    assert completed.returncode != 0
    assert "content hash mismatch" in (completed.stdout + completed.stderr)
