from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from citeweave.formal_adaptive_review import (
    AdaptiveEvaluationPacket,
    AdaptiveEvaluationResult,
    FormalAdaptiveConfig,
    FormalAdaptiveReviewRunner,
    assemble_formal_cases,
    evaluation_to_feedback_memory,
    validate_formal_topic_sequence,
)
from citeweave.judge_protocol import (
    FeedbackPacket,
    FeedbackResult,
    scan_condition_leaks,
    validate_feedback_result,
)


def _write_protocol(tmp_path: Path) -> tuple[Path, Path]:
    topic_ids = ["dev-1", "dev-2", *(f"locked-{index}" for index in range(1, 7))]
    references = tmp_path / "references.yml"
    references.write_text(
        yaml.safe_dump(
            {
                "references": [
                    {
                        "id": topic_id,
                        "role": "development" if index < 2 else "locked",
                    }
                    for index, topic_id in enumerate(topic_ids)
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cases = tmp_path / "cases.jsonl"
    rows = []
    for index, topic_id in enumerate(topic_ids):
        rows.append(
            {
                "sample_id": f"{topic_id}:report",
                "dataset_id": topic_id,
                "topic_role": "development" if index < 2 else "locked",
                "artifact_type": "report" if index % 2 == 0 else "graph",
                "canonical_evidence": [
                    {"evidence_id": f"E{index + 1:03d}", "value": index}
                ],
                "anonymous_candidate": f"Candidate statement [{f'E{index + 1:03d}'}].",
                "stage": "formal_output",
                "issue_signature": "candidate_quality_review",
                "severity": "low",
                "detector_score": 0.10,
                "auto_accept_context_ok": True,
            }
        )
    cases.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return references, cases


def _submit_pending(root: Path, *, fail_last_auto: bool = False) -> bool:
    submitted = False
    for condition_dir in [
        root / "always_review",
        root / "static_review",
        root / "adaptive_review",
    ]:
        state = json.loads((condition_dir / "state.json").read_text(encoding="utf-8"))
        if state["completed"] or not state["records"]:
            continue
        record = state["records"][-1]
        if record["status"] == "awaiting_feedback":
            packet = FeedbackPacket.model_validate_json(
                (condition_dir / record["feedback_packet_path"]).read_text(encoding="utf-8")
            )
            path = condition_dir / "inbox" / "feedback" / f"{packet.packet_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "packet_id": packet.packet_id,
                        "judge_id": "feedback",
                        "decision": "accept",
                        "confidence": 0.99,
                        "reason": "The candidate is fully supported.",
                        "evidence_ids": packet.allowed_evidence_ids,
                    }
                ),
                encoding="utf-8",
            )
            submitted = True
        elif record["status"] == "awaiting_evaluation":
            packet = AdaptiveEvaluationPacket.model_validate_json(
                (condition_dir / record["evaluation_packet_path"]).read_text(
                    encoding="utf-8"
                )
            )
            fail = bool(
                fail_last_auto
                and condition_dir.name == "adaptive_review"
                and record["case_index"] == 7
                and record["auto_accepted"]
            )
            path = condition_dir / "inbox" / "evaluation" / f"{packet.packet_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "packet_id": packet.packet_id,
                        "judge_id": "evaluation",
                        "decision": "fail" if fail else "pass",
                        "confidence": 0.98,
                        "reason": "Independent final-quality assessment.",
                        "evidence_ids": packet.allowed_evidence_ids,
                    }
                ),
                encoding="utf-8",
            )
            submitted = True
    return submitted


def _complete(runner: FormalAdaptiveReviewRunner, *, fail_last_auto: bool = False):
    for _ in range(100):
        status = runner.advance_all()
        if status["completed"]:
            return status
        assert _submit_pending(runner.output_root, fail_last_auto=fail_last_auto)
    raise AssertionError("runner did not complete")


def test_formal_three_condition_online_experiment_and_metrics(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=tmp_path / "run",
        config=FormalAdaptiveConfig(
            minimum_confirmations=2,
            audit_rate=0,
            static_detector_threshold=0.5,
        ),
    )

    result = _complete(runner, fail_last_auto=True)
    metrics = result["metrics"]["metrics"]

    assert metrics["always_review"]["review_request_rate"] == 1
    assert metrics["static_review"]["review_request_rate"] == 0
    assert metrics["adaptive_review"]["review_requests"] == 2
    assert metrics["adaptive_review"]["review_request_rate"] == 0.25
    assert metrics["adaptive_review"]["final_quality_pass_rate"] == 7 / 8
    assert metrics["adaptive_review"]["auto_accepted_items"] == 6
    assert metrics["adaptive_review"]["unsafe_auto_accept_rate"] == 1 / 6
    assert metrics["always_review"]["unsafe_auto_accept_rate"] is None

    adaptive = json.loads(
        (tmp_path / "run" / "adaptive_review" / "state.json").read_text(encoding="utf-8")
    )
    assert [row["dataset_id"] for row in adaptive["records"]] == [
        "dev-1",
        "dev-2",
        "locked-1",
        "locked-2",
        "locked-3",
        "locked-4",
        "locked-5",
        "locked-6",
    ]
    assert adaptive["feedback_memory_records"] == 2
    feedback_lines = (
        tmp_path / "run" / "adaptive_review" / "feedback_memory.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["role"] == "feedback" for line in feedback_lines)
    assert "evaluation" not in (
        tmp_path / "run" / "adaptive_review" / "feedback_memory.jsonl"
    ).read_text(encoding="utf-8")


def test_packets_are_anonymous_and_evaluation_cannot_enter_feedback_memory(
    tmp_path: Path,
):
    references, cases = _write_protocol(tmp_path)
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=tmp_path / "run",
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    runner.advance_all()
    packet_files = list((tmp_path / "run").glob("*/packets/feedback/*.json"))
    assert len(packet_files) == 2  # Always and cold Adaptive; Static auto-accepts.
    for path in packet_files:
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert not scan_condition_leaks(
            packet, ["always_review", "static_review", "adaptive_review"]
        )
        assert "condition" not in json.dumps(packet).casefold()
        assert packet["risk_notice"]["message"]
        assert packet["risk_notice"]["flagged_text"] in packet["anonymous_candidate"]
        assert len(packet["risk_notice"]["flagged_text"]) <= 500
        assert packet["permitted_interventions"] == [
            "accept",
            "reject",
            "replace_span",
            "delete_span",
            "append_caveat",
        ]
        constraints = " ".join(packet["human_proxy_constraints"]).casefold()
        assert "only the risk identified" in constraints
        assert "do not use model knowledge" in constraints

    evaluation_path = next(
        (tmp_path / "run" / "static_review" / "packets" / "evaluation").glob("*.json")
    )
    packet = AdaptiveEvaluationPacket.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    result = AdaptiveEvaluationResult(
        packet_id=packet.packet_id,
        decision="pass",
        confidence=1,
        reason="Supported.",
        evidence_ids=packet.allowed_evidence_ids,
    )
    with pytest.raises(TypeError, match="cannot enter feedback memory"):
        evaluation_to_feedback_memory(result, packet)


def test_human_proxy_can_apply_only_one_exact_visible_local_edit():
    result = FeedbackResult(
        packet_id="FP1234567890abcdef1234",
        decision="revise",
        confidence=0.9,
        reason="The flagged number is unsupported.",
        suggested_revision={
            "action": "replace_span",
            "target_text": "12 links",
            "replacement_text": "10 links",
        },
        evidence_ids=[],
    )
    revised = FormalAdaptiveReviewRunner._apply_human_proxy_edit(
        "The network contains 12 links.",
        result,
    )
    assert revised == "The network contains 10 links."

    with pytest.raises(ValueError, match="exactly once"):
        FormalAdaptiveReviewRunner._apply_human_proxy_edit(
            "12 links and another 12 links.",
            result,
        )

    whole_artifact = FeedbackResult(
        packet_id="FP1234567890abcdef1234",
        decision="revise",
        confidence=0.9,
        reason="Attempted whole-artifact replacement.",
        suggested_revision={
            "action": "replace_span",
            "target_text": "The complete artifact.",
            "replacement_text": "A wholly different artifact.",
        },
        evidence_ids=[],
    )
    with pytest.raises(ValueError, match="full artifact"):
        FormalAdaptiveReviewRunner._apply_human_proxy_edit(
            "The complete artifact.",
            whole_artifact,
        )


def test_human_proxy_revision_is_bound_to_the_flagged_risk_scope():
    packet = FeedbackPacket(
        packet_id="FP1234567890abcdef1234",
        sample_id="sample-1",
        rubric_version="feedback-human-proxy-v2",
        canonical_evidence=[{"evidence_id": "E001", "value": 10}],
        anonymous_candidate=(
            "The flagged claim says 12 links. "
            "An unrelated paragraph says 8 authors."
        ),
        allowed_evidence_ids=["E001"],
        risk_notice={
            "message": "Review only the flagged network-size claim.",
            "flagged_text": "The flagged claim says 12 links.",
        },
        content_sha256="a" * 64,
    )
    allowed = FeedbackResult(
        packet_id=packet.packet_id,
        decision="revise",
        confidence=0.9,
        reason="The visible evidence supports 10 links.",
        suggested_revision={
            "action": "replace_span",
            "target_text": "12 links",
            "replacement_text": "10 links",
        },
        evidence_ids=["E001"],
    )
    validate_feedback_result(allowed, packet)

    outside_payload = allowed.model_dump(mode="json")
    outside_payload["suggested_revision"] = {
        "action": "replace_span",
        "target_text": "8 authors",
        "replacement_text": "7 authors",
    }
    outside_scope = FeedbackResult.model_validate(outside_payload)
    with pytest.raises(ValueError, match="outside"):
        validate_feedback_result(outside_scope, packet)


def test_long_candidate_requires_explicit_local_risk_scope(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines()]
    rows[0]["anonymous_candidate"] = "A" * 501
    cases.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=tmp_path / "run",
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    with pytest.raises(RuntimeError, match="explicit risk_scope_text"):
        runner.advance_condition("always_review")


def test_invalid_local_edit_cannot_enter_feedback_memory(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines()]
    rows[0]["anonymous_candidate"] = (
        "The flagged bad number is here. Another bad number is elsewhere."
    )
    rows[0]["risk_scope_text"] = "The flagged bad number is here."
    cases.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    root = tmp_path / "run"
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=root,
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    runner.advance_condition("always_review")
    state = json.loads(
        (root / "always_review" / "state.json").read_text(encoding="utf-8")
    )
    record = state["records"][-1]
    packet = FeedbackPacket.model_validate_json(
        (root / "always_review" / record["feedback_packet_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert packet.anonymous_candidate == rows[0]["risk_scope_text"]
    assert "elsewhere" not in packet.anonymous_candidate
    inbox = root / "always_review" / "inbox" / "feedback" / f"{packet.packet_id}.json"
    inbox.parent.mkdir(parents=True)
    inbox.write_text(
        FeedbackResult(
            packet_id=packet.packet_id,
            decision="revise",
            confidence=0.9,
            reason="Local numeric correction.",
            suggested_revision={
                "action": "replace_span",
                "target_text": "bad number",
                "replacement_text": "corrected number",
            },
            evidence_ids=[],
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly once"):
        runner.advance_condition("always_review")
    assert not (root / "always_review" / "feedback_memory.jsonl").exists()
    assert not (root / "always_review" / "policy_memory.jsonl").exists()


def test_resume_verifies_frozen_input_and_packet_hashes(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    output = tmp_path / "run"
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=output,
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    runner.advance_all()

    resumed = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=output,
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    assert resumed.advance_all()["completed"] is False

    packet_path = next((output / "always_review" / "packets" / "feedback").glob("*.json"))
    packet_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resumed.advance_condition("always_review")


def test_rejects_noncontiguous_or_wrong_formal_topic_sequence(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines()]
    rows[2], rows[3] = rows[3], rows[2]
    cases.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from frozen registry"):
        FormalAdaptiveReviewRunner(
            cases_path=cases,
            reference_registry=references,
            output_root=tmp_path / "run",
        )


def test_development_calibration_is_explicit_and_never_formal(tmp_path: Path):
    references, cases = _write_protocol(tmp_path)
    registry = yaml.safe_load(references.read_text(encoding="utf-8"))
    registry["references"] = registry["references"][:2]
    references.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    rows = [
        json.loads(line)
        for line in cases.read_text(encoding="utf-8").splitlines()
    ][:2]
    cases.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="2 development \\+ 6 locked"):
        FormalAdaptiveReviewRunner(
            cases_path=cases,
            reference_registry=references,
            output_root=tmp_path / "formal-run",
        )

    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=tmp_path / "calibration-run",
        config=FormalAdaptiveConfig(experiment_mode="development_calibration"),
    )
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))

    assert runner.topic_sequence == ["dev-1", "dev-2"]
    assert manifest["formal_results_used"] is False
    assert manifest["config"]["experiment_mode"] == "development_calibration"


def test_assembles_report_and_graph_checkpoint_artifacts(tmp_path: Path):
    references, _ = _write_protocol(tmp_path)
    topics = []
    for index, dataset_id in enumerate(
        ["dev-1", "dev-2", *(f"locked-{item}" for item in range(1, 7))]
    ):
        evidence = tmp_path / f"{dataset_id}-evidence.json"
        evidence.write_text(
            json.dumps([{"evidence_id": f"E{index + 1:03d}", "value": index}]),
            encoding="utf-8",
        )
        if index % 2:
            candidate = tmp_path / f"{dataset_id}.jsonl"
            candidate.write_text(
                json.dumps(
                    {
                        "item_id": f"{dataset_id}:G001",
                        "status": "complete",
                        "response": {
                            "choices": [
                                {"message": {"content": '{"answer": {"nodes": 2}}'}}
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            artifact = {
                "artifact_type": "graph",
                "evidence_path": evidence.name,
                "candidate_path": candidate.name,
                "checkpoint_item_id": f"{dataset_id}:G001",
            }
        else:
            candidate = tmp_path / f"{dataset_id}.md"
            candidate.write_text("# English report\n", encoding="utf-8")
            artifact = {
                "artifact_type": "report",
                "evidence_path": evidence.name,
                "candidate_path": candidate.name,
            }
        topics.append(
            {
                "dataset_id": dataset_id,
                "topic_role": "development" if index < 2 else "locked",
                "artifacts": [artifact],
            }
        )
    spec = tmp_path / "assembly.yml"
    spec.write_text(yaml.safe_dump({"topics": topics}, sort_keys=False), encoding="utf-8")
    output = tmp_path / "formal-cases.jsonl"

    cases = assemble_formal_cases(spec, output)

    assert len(cases) == 8
    assert {case.artifact_type for case in cases} == {"report", "graph"}
    assert validate_formal_topic_sequence(cases, references) == [
        "dev-1",
        "dev-2",
        "locked-1",
        "locked-2",
        "locked-3",
        "locked-4",
        "locked-5",
        "locked-6",
    ]


def test_feedback_checkpoint_recovers_after_memory_append_before_state_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    references, cases = _write_protocol(tmp_path)
    output = tmp_path / "run"
    runner = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=output,
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    runner.advance_all()
    condition_dir = output / "always_review"
    state = json.loads((condition_dir / "state.json").read_text(encoding="utf-8"))
    record = state["records"][-1]
    packet = FeedbackPacket.model_validate_json(
        (condition_dir / record["feedback_packet_path"]).read_text(encoding="utf-8")
    )
    inbox = condition_dir / "inbox" / "feedback" / f"{packet.packet_id}.json"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    inbox.write_text(
        json.dumps(
            {
                "packet_id": packet.packet_id,
                "decision": "accept",
                "confidence": 1,
                "reason": "Supported.",
                "evidence_ids": packet.allowed_evidence_ids,
            }
        ),
        encoding="utf-8",
    )
    original_save = runner._save_state

    def crash_after_memory(condition, value):
        if (condition_dir / "policy_memory.jsonl").exists():
            raise RuntimeError("simulated crash")
        original_save(condition, value)

    monkeypatch.setattr(runner, "_save_state", crash_after_memory)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.advance_condition("always_review")

    recovered = FormalAdaptiveReviewRunner(
        cases_path=cases,
        reference_registry=references,
        output_root=output,
        config=FormalAdaptiveConfig(audit_rate=0),
    )
    status = recovered.advance_condition("always_review")
    assert status["pending"] == "awaiting_evaluation"
    assert len(
        (condition_dir / "feedback_memory.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1
    assert len(
        (condition_dir / "policy_memory.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1
