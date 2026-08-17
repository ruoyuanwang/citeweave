from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from citeweave.adaptive_assignment_conductor import (
    AdaptiveAssignmentError,
    prepare_assignments,
    validate_and_import,
    validate_complete_results,
)
from citeweave.formal_adaptive_review import (
    FormalAdaptiveCase,
    build_evaluation_packet,
)
from citeweave.judge_protocol import prepare_feedback_packet

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "exchange_adaptive_blind_packets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exchange_adaptive_for_conductor_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
EXCHANGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXCHANGE
SPEC.loader.exec_module(EXCHANGE)


def _case(sample_id: str) -> FormalAdaptiveCase:
    return FormalAdaptiveCase(
        sample_id=sample_id,
        dataset_id="dataset",
        topic_role="locked",
        artifact_type="report",
        canonical_evidence=[{"evidence_id": "E001", "statement": "Fact."}],
        anonymous_candidate="Candidate fact.",
        stage="report",
        issue_signature="claim-risk",
        risk_scope_text="Candidate fact.",
        severity="high",
        detector_score=0.9,
    )


def _mixed_batch(tmp_path: Path) -> tuple[Path, Path, object, object]:
    run_root = tmp_path / "run"
    feedback = prepare_feedback_packet(
        {
            "sample_id": "feedback-sample",
            "canonical_evidence": [{"evidence_id": "E001", "statement": "Fact."}],
            "candidates": {"hidden": "Candidate fact."},
        },
        condition="hidden",
        rubric_version="feedback-human-proxy-v3",
        seed=42,
        risk_notice={
            "severity": "high",
            "message": "Review this flagged claim.",
            "flagged_text": "Candidate fact.",
        },
    )
    evaluation = build_evaluation_packet(
        _case("evaluation-sample"),
        final_candidate="Candidate fact.",
        rubric_version="adaptive-evaluation-v1",
        seed=42,
        condition="static_review",
    )
    for condition in ("always_review", "static_review", "adaptive_review"):
        condition_root = run_root / condition
        condition_root.mkdir(parents=True)
        if condition == "always_review":
            packet = feedback
            kind = "feedback"
        elif condition == "static_review":
            packet = evaluation
            kind = "evaluation"
        else:
            (condition_root / "state.json").write_text(
                json.dumps({"records": [], "completed": True}),
                encoding="utf-8",
            )
            continue
        relative = Path("packets") / kind / f"{packet.packet_id}.json"
        packet_path = condition_root / relative
        packet_path.parent.mkdir(parents=True)
        packet_path.write_text(packet.model_dump_json(), encoding="utf-8")
        (condition_root / "state.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "status": f"awaiting_{kind}",
                            f"{kind}_packet_id": packet.packet_id,
                            f"{kind}_packet_path": relative.as_posix(),
                        }
                    ],
                    "completed": False,
                }
            ),
            encoding="utf-8",
        )
    batch_root = tmp_path / "batch"
    EXCHANGE.export_batch(run_root, batch_root)
    return run_root, batch_root, feedback, evaluation


def _write_valid_results(
    batch_root: Path,
    feedback: object,
    evaluation: object,
) -> None:
    feedback_path = (
        batch_root / "results" / "feedback" / f"{feedback.packet_id}.json"
    )
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(
        json.dumps(
            {
                "packet_id": feedback.packet_id,
                "judge_id": "feedback",
                "decision": "accept",
                "confidence": 0.9,
                "reason": "Supported by visible evidence.",
                "evidence_ids": ["E001"],
            }
        ),
        encoding="utf-8",
    )
    evaluation_path = (
        batch_root / "results" / "evaluation" / f"{evaluation.packet_id}.json"
    )
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(
            {
                "packet_id": evaluation.packet_id,
                "judge_id": "evaluation",
                "decision": "pass",
                "confidence": 0.9,
                "reason": "Supported by visible evidence.",
                "evidence_ids": ["E001"],
            }
        ),
        encoding="utf-8",
    )


def test_prepares_two_strictly_isolated_role_assignments(tmp_path: Path) -> None:
    _run, batch, feedback, evaluation = _mixed_batch(tmp_path)
    assignment_root = tmp_path / "assignments"

    result = prepare_assignments(
        batch_root=batch,
        assignment_root=assignment_root,
    )

    assert result["status"] == "assignments_ready"
    feedback_assignment = json.loads(
        (assignment_root / "feedback" / "assignment.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation_assignment = json.loads(
        (assignment_root / "evaluation" / "assignment.json").read_text(
            encoding="utf-8"
        )
    )
    assert feedback_assignment["assignment_role"] == "constrained_llm_human_proxy"
    assert feedback_assignment["items"][0]["packet_id"] == feedback.packet_id
    assert feedback_assignment["may_directly_modify_candidate"] is False
    assert feedback_assignment["may_return_one_scoped_edit"] is True
    assert feedback_assignment["may_directly_update_feedback_memory"] is False
    assert feedback_assignment["validated_result_eligible_for_feedback_memory"] is True
    constraints = " ".join(feedback_assignment["instructions"]).casefold()
    assert "flagged_text" in constraints
    assert "500" in constraints
    assert "do not browse" in constraints
    assert "complete source artifact" in constraints
    assert evaluation_assignment["assignment_role"].endswith("evaluation_judge")
    assert evaluation_assignment["items"][0]["packet_id"] == evaluation.packet_id
    assert evaluation_assignment["may_directly_modify_candidate"] is False
    assert evaluation_assignment["may_return_one_scoped_edit"] is False
    assert evaluation_assignment["may_directly_update_feedback_memory"] is False
    assert (
        evaluation_assignment["validated_result_eligible_for_feedback_memory"]
        is False
    )
    assert "do not propose" in " ".join(
        evaluation_assignment["instructions"]
    ).casefold()
    assert {
        path.name
        for path in (assignment_root / "feedback" / "packets").iterdir()
    } == {f"{feedback.packet_id}.json"}
    assert {
        path.name
        for path in (assignment_root / "evaluation" / "packets").iterdir()
    } == {f"{evaluation.packet_id}.json"}
    serialized_feedback = json.dumps(feedback_assignment)
    serialized_evaluation = json.dumps(evaluation_assignment)
    assert evaluation.packet_id not in serialized_feedback
    assert feedback.packet_id not in serialized_evaluation

    assert prepare_assignments(
        batch_root=batch,
        assignment_root=assignment_root,
    ) == result


def test_requires_exact_result_coverage_and_strict_schema(tmp_path: Path) -> None:
    _run, batch, feedback, evaluation = _mixed_batch(tmp_path)
    assignment_root = tmp_path / "assignments"
    prepare_assignments(batch_root=batch, assignment_root=assignment_root)
    _write_valid_results(batch, feedback, evaluation)
    missing = batch / "results" / "evaluation" / f"{evaluation.packet_id}.json"
    missing.unlink()
    with pytest.raises(AdaptiveAssignmentError, match="coverage differs"):
        validate_complete_results(
            batch_root=batch,
            assignment_root=assignment_root,
        )

    _write_valid_results(batch, feedback, evaluation)
    extra = batch / "results" / "feedback" / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(AdaptiveAssignmentError, match="coverage differs"):
        validate_complete_results(
            batch_root=batch,
            assignment_root=assignment_root,
        )
    extra.unlink()

    evaluation_path = (
        batch / "results" / "evaluation" / f"{evaluation.packet_id}.json"
    )
    malformed = json.loads(evaluation_path.read_text(encoding="utf-8"))
    malformed["suggested_revision"] = {"action": "delete_span"}
    evaluation_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(AdaptiveAssignmentError, match="schema/packet binding"):
        validate_complete_results(
            batch_root=batch,
            assignment_root=assignment_root,
        )


def test_validates_all_results_before_idempotent_existing_import(tmp_path: Path) -> None:
    run, batch, feedback, evaluation = _mixed_batch(tmp_path)
    assignment_root = tmp_path / "assignments"
    _write_valid_results(batch, feedback, evaluation)

    first = validate_and_import(
        run_root=run,
        batch_root=batch,
        assignment_root=assignment_root,
        importer=EXCHANGE.import_results,
    )
    assert first["status"] == "imported"
    assert first["imported"] == {"feedback": 1, "evaluation": 1}
    assert first["idempotent_replay"] is False
    second = validate_and_import(
        run_root=run,
        batch_root=batch,
        assignment_root=assignment_root,
        importer=EXCHANGE.import_results,
    )
    assert second["idempotent_replay"] is True
    for packet, kind, condition in (
        (feedback, "feedback", "always_review"),
        (evaluation, "evaluation", "static_review"),
    ):
        assert (
            run / condition / "inbox" / kind / f"{packet.packet_id}.json"
        ).is_file()


def test_complete_import_preflights_conflicts_before_any_write(tmp_path: Path) -> None:
    run, batch, feedback, evaluation = _mixed_batch(tmp_path)
    _write_valid_results(batch, feedback, evaluation)
    conflicting = (
        run
        / "static_review"
        / "inbox"
        / "evaluation"
        / f"{evaluation.packet_id}.json"
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text('{"conflict": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="Conflicting immutable"):
        EXCHANGE.import_results(run, batch, require_complete=True)

    assert not (
        run
        / "always_review"
        / "inbox"
        / "feedback"
        / f"{feedback.packet_id}.json"
    ).exists()


def test_refuses_tampered_isolated_packet_and_frozen_batch(tmp_path: Path) -> None:
    _run, batch, _feedback, _evaluation = _mixed_batch(tmp_path)
    assignment_root = tmp_path / "assignments"
    prepare_assignments(batch_root=batch, assignment_root=assignment_root)
    isolated = next((assignment_root / "feedback" / "packets").iterdir())
    isolated.write_text("{}", encoding="utf-8")
    with pytest.raises(AdaptiveAssignmentError, match="isolated packet"):
        prepare_assignments(batch_root=batch, assignment_root=assignment_root)

    source = next((batch / "packets" / "feedback").iterdir())
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(AdaptiveAssignmentError, match="file/hash differs"):
        prepare_assignments(batch_root=batch, assignment_root=tmp_path / "fresh")
