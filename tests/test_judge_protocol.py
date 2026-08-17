import pytest
from pydantic import ValidationError

from citeweave.judge_protocol import (
    BlindAssignment,
    BlindMap,
    CandidateJudgment,
    ClaimJudgment,
    FeedbackPacket,
    FeedbackResult,
    JudgeResult,
    aggregate_resolved_results,
    build_adjudication_packet,
    collect_reference_ids,
    detect_conflicts,
    prepare_blind_pair,
    prepare_dual_evidence_blind_pair,
    prepare_feedback_packet,
    resolve_packet_results,
    scan_condition_leaks,
    to_feedback_memory_record,
    validate_blind_packet,
    validate_judge_result,
)


def _record():
    return {
        "sample_id": "sample-1",
        "question": "Which explanation is better supported?",
        "canonical_evidence": {"E001": "The network contains 10 nodes."},
        "candidates": {
            "structured_one_shot": "The network contains nine nodes.",
            "citeweave_full": "The network contains 10 nodes.",
        },
    }


def _risk_notice() -> dict[str, str]:
    return {
        "severity": "high",
        "message": "The system flagged a visible grounding risk.",
    }


def _result(packet_id, judge_id, full_slot, *, preference=None, full_score=5):
    other_slot = "B" if full_slot == "A" else "A"
    return JudgeResult(
        packet_id=packet_id,
        judge_id=judge_id,
        candidates=[
            CandidateJudgment(
                slot=full_slot,
                claims=[
                    ClaimJudgment(
                        claim="The network contains 10 nodes.",
                        verdict="supported",
                        evidence_ids=["E001"],
                    )
                ],
                completeness_score=full_score,
            ),
            CandidateJudgment(
                slot=other_slot,
                claims=[
                    ClaimJudgment(
                        claim="The network contains nine nodes.",
                        verdict="contradicted",
                        evidence_ids=["E001"],
                    )
                ],
                completeness_score=2,
            ),
        ],
        preference=preference or full_slot,
        rationale="The supported candidate matches the canonical evidence.",
    )


def test_packet_ids_are_deterministic_and_orders_are_reversed():
    first = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )
    second = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )

    packet_a, packet_b, mapping = first
    assert first == second
    assert packet_a.packet_id == packet_b.packet_id
    assert packet_a.candidate_a == packet_b.candidate_b
    assert packet_a.candidate_b == packet_b.candidate_a
    assert mapping.assignments["eval_a"].A == mapping.assignments["eval_b"].B


def test_human_reference_pair_reverses_each_candidate_with_its_own_evidence():
    record = {
        "sample_id": "topic-1",
        "question": "Which report is the stronger evidence-bounded synthesis?",
        "candidates": {
            "citeweave_full": "System report [E001].",
            "published_human_reference": "Published report.",
        },
        "evidence_by_condition": {
            "citeweave_full": [{"evidence_id": "E001", "statement": "System fact"}],
            "published_human_reference": "Archived source article sections.",
        },
    }

    packet_a, packet_b, mapping = prepare_dual_evidence_blind_pair(
        record,
        condition_a="citeweave_full",
        condition_b="published_human_reference",
        rubric_version="human-gap-v1",
        seed=42,
    )

    assert packet_a.candidate_a == packet_b.candidate_b
    assert (
        packet_a.canonical_evidence["candidate_a_evidence"]
        == packet_b.canonical_evidence["candidate_b_evidence"]
    )
    assert mapping.assignments["eval_a"].A == mapping.assignments["eval_b"].B
    assert not scan_condition_leaks(
        packet_a.model_dump(mode="json"),
        ["citeweave_full", "published_human_reference"],
    )


def test_condition_leak_scanner_finds_hidden_labels():
    assert scan_condition_leaks(
        {"question": "Output from citeweave_full"},
        ["structured_one_shot", "citeweave_full"],
    ) == ["citeweave_full"]
    assert not scan_condition_leaks(
        {"question": "Anonymous output A"},
        ["structured_one_shot", "citeweave_full"],
    )
    assert scan_condition_leaks(
        {"explanation": "This Graph RAG answer uses visible evidence."},
        ["graph_rag", "no_rag"],
    ) == ["graph_rag"]


def test_blind_result_must_bind_to_untampered_packet_and_visible_evidence():
    packet_a, _, mapping = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )
    full_slot = (
        "A" if mapping.assignments["eval_a"].A == "citeweave_full" else "B"
    )
    result = _result(packet_a.packet_id, "eval_a", full_slot)
    validate_blind_packet(packet_a)
    validate_judge_result(result, packet_a, expected_judge_id="eval_a")

    invalid = result.model_copy(deep=True)
    invalid.candidates[0].claims[0].evidence_ids = ["E999"]
    with pytest.raises(ValueError, match="not visible"):
        validate_judge_result(invalid, packet_a, expected_judge_id="eval_a")

    tampered = packet_a.model_copy(update={"candidate_a": "Changed after blinding."})
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_blind_packet(tampered)


def test_human_evidence_ids_are_addressable():
    evidence = [
        {"evidence_id": "H001", "statement": "Published method."},
        {"evidence_id": "H002", "statement": "Published result."},
    ]
    assert collect_reference_ids(evidence) == ["H001", "H002"]


def test_graph_atom_ids_with_visible_labels_and_spaces_are_addressable():
    evidence = {
        "graph_evidence": {
            "atom_ids": [
                "keyword_cooccurrence:node:Climate change:cluster",
                "keyword_cooccurrence:node:Climate change:label",
            ],
            "relationships": [
                {
                    "atom_id": "keyword_cooccurrence:node:Climate change:cluster",
                    "object": 2,
                }
            ],
        }
    }

    assert collect_reference_ids(evidence) == [
        "keyword_cooccurrence:node:Climate change:cluster",
        "keyword_cooccurrence:node:Climate change:label",
    ]


def test_pydantic_result_schema_rejects_duplicate_slots_and_bad_score():
    packet_id = "JP" + "0" * 20
    with pytest.raises(ValidationError):
        JudgeResult(
            packet_id=packet_id,
            judge_id="eval_a",
            candidates=[
                {"slot": "A", "claims": [], "completeness_score": 6},
                {"slot": "A", "claims": [], "completeness_score": 2},
            ],
            preference="A",
            rationale="Invalid duplicate slots.",
        )
    with pytest.raises(ValidationError, match="must cite"):
        ClaimJudgment(
            claim="The network contains 10 nodes.",
            verdict="supported",
            evidence_ids=[],
        )


def test_reversed_judges_decode_without_false_conflict():
    packet_a, _, mapping = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )
    full_slot_a = (
        "A" if mapping.assignments["eval_a"].A == "citeweave_full" else "B"
    )
    full_slot_b = (
        "A" if mapping.assignments["eval_b"].A == "citeweave_full" else "B"
    )
    left = _result(packet_a.packet_id, "eval_a", full_slot_a)
    right = _result(packet_a.packet_id, "eval_b", full_slot_b)

    assert detect_conflicts(left, right, mapping) == []
    resolved = resolve_packet_results(left, right, mapping)
    metrics = aggregate_resolved_results([resolved])
    assert metrics["conditions"]["citeweave_full"]["unsupported_claim_rate"] == 0.0
    assert (
        metrics["conditions"]["structured_one_shot"]["unsupported_claim_rate"]
        == 1.0
    )
    assert metrics["conditions"]["citeweave_full"]["pairwise_preference_score"] == 1.0


def test_opposite_preferences_require_and_accept_blind_adjudication():
    packet_a, _, mapping = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )
    full_slot_a = (
        "A" if mapping.assignments["eval_a"].A == "citeweave_full" else "B"
    )
    full_slot_b = (
        "A" if mapping.assignments["eval_b"].A == "citeweave_full" else "B"
    )
    left = _result(packet_a.packet_id, "eval_a", full_slot_a)
    opposite = "B" if full_slot_b == "A" else "A"
    right = _result(
        packet_a.packet_id,
        "eval_b",
        full_slot_b,
        preference=opposite,
    )
    reasons = detect_conflicts(left, right, mapping)
    assert "opposite_pairwise_preferences" in reasons
    adjudication_packet = build_adjudication_packet(
        packet_a,
        left,
        right,
        mapping,
    )
    assert [
        item["slot"]
        for item in adjudication_packet["judge_b_result_remapped"]["candidates"]
    ] == ["A", "B"]
    assert not scan_condition_leaks(
        adjudication_packet,
        ["structured_one_shot", "citeweave_full"],
    )
    with pytest.raises(ValueError, match="requires adjudication"):
        resolve_packet_results(left, right, mapping)

    adjudicator_slot = (
        "A" if mapping.assignments["adjudicator"].A == "citeweave_full" else "B"
    )
    adjudication = _result(
        packet_a.packet_id,
        "adjudicator",
        adjudicator_slot,
    )
    resolved = resolve_packet_results(
        left,
        right,
        mapping,
        adjudication=adjudication,
    )
    assert resolved["source"] == "adjudication"
    assert resolved["preference"] == "citeweave_full"


def test_adjudication_packet_blinds_condition_specific_conflicts():
    packet_a, _, mapping = prepare_blind_pair(
        _record(),
        condition_a="structured_one_shot",
        condition_b="citeweave_full",
        rubric_version="v1",
        seed=17,
    )
    full_slot_a = "A" if mapping.assignments["eval_a"].A == "citeweave_full" else "B"
    full_slot_b = "A" if mapping.assignments["eval_b"].A == "citeweave_full" else "B"
    left = _result(packet_a.packet_id, "eval_a", full_slot_a, full_score=5)
    right = _result(packet_a.packet_id, "eval_b", full_slot_b, full_score=2)

    packet = build_adjudication_packet(packet_a, left, right, mapping)

    assert packet["conflicts"]
    assert not scan_condition_leaks(
        packet,
        ["structured_one_shot", "citeweave_full"],
    )


def test_blind_map_requires_reverse_orders():
    packet_id = "JP" + "0" * 20
    with pytest.raises(ValidationError):
        BlindMap(
            packet_id=packet_id,
            sample_id="s1",
            assignments={
                "eval_a": BlindAssignment(A="x", B="y"),
                "eval_b": BlindAssignment(A="x", B="y"),
                "adjudicator": BlindAssignment(A="x", B="y"),
            },
        )


def test_feedback_packet_is_deterministic_anonymous_and_memory_ready():
    first = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v1",
        seed=9,
        risk_notice=_risk_notice(),
    )
    second = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v1",
        seed=9,
        risk_notice=_risk_notice(),
    )
    assert first == second
    assert first.allowed_evidence_ids == ["E001"]
    assert not scan_condition_leaks(
        first.model_dump(mode="json"),
        ["citeweave_full"],
    )

    result = FeedbackResult(
        packet_id=first.packet_id,
        decision="accept",
        confidence=0.98,
        reason="The candidate is fully supported by E001.",
        evidence_ids=["E001"],
    )
    record = to_feedback_memory_record(result, first)
    assert record.role == "feedback"
    assert record.decision == "accept"
    assert record.packet_content_sha256 == first.content_sha256


def test_evaluation_result_cannot_enter_feedback_memory():
    packet = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v1",
        seed=9,
        risk_notice=_risk_notice(),
    )
    evaluation = JudgeResult(
        packet_id="JP" + "1" * 20,
        judge_id="eval_a",
        candidates=[
            {"slot": "A", "claims": [], "completeness_score": 4},
            {"slot": "B", "claims": [], "completeness_score": 3},
        ],
        preference="A",
        rationale="A is more complete.",
    )
    with pytest.raises(TypeError, match="Evaluation Judge"):
        to_feedback_memory_record(evaluation, packet)  # type: ignore[arg-type]


def test_feedback_schema_rejects_illegal_label_and_unknown_evidence():
    packet = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v1",
        seed=9,
        risk_notice=_risk_notice(),
    )
    with pytest.raises(ValidationError):
        FeedbackResult(
            packet_id=packet.packet_id,
            decision="approve",
            confidence=0.9,
            reason="Illegal decision label.",
        )
    result = FeedbackResult(
        packet_id=packet.packet_id,
        decision="reject",
        confidence=0.9,
        reason="The candidate cites evidence that is not in the packet.",
        evidence_ids=["E999"],
    )
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        to_feedback_memory_record(result, packet)


def test_revise_feedback_requires_a_suggested_revision():
    packet = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v1",
        seed=9,
        risk_notice=_risk_notice(),
    )
    with pytest.raises(ValidationError, match="suggested_revision"):
        FeedbackResult(
            packet_id=packet.packet_id,
            decision="revise",
            confidence=0.8,
            reason="One statement should be corrected.",
            evidence_ids=["E001"],
        )


def test_human_proxy_revision_is_one_constrained_local_edit():
    packet = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v2",
        seed=9,
        risk_notice={"severity": "high", "message": "Unsupported numeric claim."},
    )
    result = FeedbackResult(
        packet_id=packet.packet_id,
        decision="revise",
        confidence=0.9,
        reason="Replace the single flagged span.",
        suggested_revision={
            "action": "replace_span",
            "target_text": "10 nodes",
            "replacement_text": "10 nodes (as reported)",
        },
        evidence_ids=["E001"],
    )
    assert result.suggested_revision is not None
    assert result.suggested_revision.action == "replace_span"
    memory_record = to_feedback_memory_record(result, packet)
    assert memory_record.suggested_revision == result.suggested_revision
    assert packet.risk_notice is not None
    assert "replace_span" in packet.permitted_interventions
    constraints = " ".join(packet.human_proxy_constraints).casefold()
    assert "only the risk identified" in constraints
    assert "do not use model knowledge" in constraints

    with pytest.raises(ValidationError):
        FeedbackResult(
            packet_id=packet.packet_id,
            decision="revise",
            confidence=0.9,
            reason="An entire report rewrite is outside the interface.",
            suggested_revision={
                "action": "append_caveat",
                "replacement_text": "x" * 501,
            },
            evidence_ids=["E001"],
        )


def test_human_proxy_packet_requires_visible_risk_and_frozen_capabilities():
    with pytest.raises(ValidationError, match="risk_notice"):
        prepare_feedback_packet(
            _record(),
            condition="citeweave_full",
            rubric_version="feedback-v2",
            seed=9,
        )

    packet = prepare_feedback_packet(
        _record(),
        condition="citeweave_full",
        rubric_version="feedback-v2",
        seed=9,
        risk_notice=_risk_notice(),
    )
    value = packet.model_dump(mode="json")
    value["permitted_interventions"].append("rewrite_full_report")
    with pytest.raises(ValidationError, match="frozen contract"):
        FeedbackPacket.model_validate(value)
