from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["supported", "contradicted", "not_in_evidence"]
Slot = Literal["A", "B"]
Preference = Literal["A", "B", "tie"]
JudgeId = Literal["eval_a", "eval_b", "adjudicator"]
FeedbackDecision = Literal["accept", "revise", "reject"]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class BlindPacket(BaseModel):
    packet_id: str = Field(pattern=r"^JP[0-9a-f]{20}$")
    sample_id: str = Field(min_length=1)
    judge_id: Literal["eval_a", "eval_b"]
    rubric_version: str = Field(min_length=1)
    question: str = Field(min_length=1)
    canonical_evidence: Any
    candidate_a: str
    candidate_b: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BlindAssignment(BaseModel):
    A: str = Field(min_length=1)
    B: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_conditions(self) -> BlindAssignment:
        if self.A == self.B:
            raise ValueError("A and B must map to distinct conditions")
        return self


class BlindMap(BaseModel):
    packet_id: str = Field(pattern=r"^JP[0-9a-f]{20}$")
    sample_id: str = Field(min_length=1)
    assignments: dict[JudgeId, BlindAssignment]

    @model_validator(mode="after")
    def reversed_evaluation_orders(self) -> BlindMap:
        required = {"eval_a", "eval_b", "adjudicator"}
        if set(self.assignments) != required:
            raise ValueError(f"assignments must contain exactly {sorted(required)}")
        first = self.assignments["eval_a"]
        second = self.assignments["eval_b"]
        adjudicator = self.assignments["adjudicator"]
        if second.A != first.B or second.B != first.A:
            raise ValueError("eval_b must use the reverse of eval_a")
        if adjudicator != first:
            raise ValueError("adjudicator must use eval_a slot order")
        return self


class ClaimJudgment(BaseModel):
    claim: str = Field(min_length=1)
    verdict: Verdict
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def supported_claim_requires_evidence(self) -> ClaimJudgment:
        if self.verdict == "supported" and not self.evidence_ids:
            raise ValueError("A supported claim must cite at least one canonical evidence ID")
        return self


class CandidateJudgment(BaseModel):
    slot: Slot
    claims: list[ClaimJudgment]
    completeness_score: int = Field(ge=1, le=5)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(pattern=r"^JP[0-9a-f]{20}$")
    judge_id: JudgeId
    candidates: list[CandidateJudgment] = Field(min_length=2, max_length=2)
    preference: Preference
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def exactly_one_judgment_per_slot(self) -> JudgeResult:
        if {item.slot for item in self.candidates} != {"A", "B"}:
            raise ValueError("candidates must contain exactly one A and one B judgment")
        return self


class HumanProxyEdit(BaseModel):
    """One local edit available to a person reviewing the visible artifact.

    The proxy cannot replace an entire report.  It may only operate on an exact
    visible span or append a short caveat, matching the controls exposed by the
    experimental review interface.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["replace_span", "delete_span", "append_caveat"]
    target_text: str | None = Field(default=None, max_length=500)
    replacement_text: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def valid_local_edit(self) -> HumanProxyEdit:
        target = str(self.target_text or "")
        replacement = str(self.replacement_text or "")
        if self.action == "replace_span" and (not target or not replacement):
            raise ValueError("replace_span requires target_text and replacement_text")
        if self.action == "delete_span" and (not target or replacement):
            raise ValueError("delete_span requires only target_text")
        if self.action == "append_caveat" and (target or not replacement):
            raise ValueError("append_caveat requires only replacement_text")
        return self


class FeedbackPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(pattern=r"^FP[0-9a-f]{20}$")
    sample_id: str = Field(min_length=1)
    judge_id: Literal["feedback"] = "feedback"
    rubric_version: str = Field(min_length=1)
    canonical_evidence: Any
    anonymous_candidate: str = Field(min_length=1)
    allowed_evidence_ids: list[str]
    risk_notice: dict[str, Any]
    permitted_interventions: list[str] = Field(
        default_factory=lambda: [
            "accept",
            "reject",
            "replace_span",
            "delete_span",
            "append_caveat",
        ]
    )
    human_proxy_constraints: list[str] = Field(
        default_factory=lambda: [
            "Use only the candidate, risk notice, and evidence visible in this packet.",
            "Do not retrieve new evidence, invoke tools, rerun analysis, or rewrite the full artifact.",
            "A revision must be one local edit of at most 500 characters.",
            "Address only the risk identified in the notice; do not search for or alter unrelated issues.",
            "Do not use model knowledge or model-scale synthesis; if visible evidence is insufficient, reject or append a caveat instead of inventing a correction.",
        ]
    )
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def enforce_visible_human_proxy_contract(self) -> FeedbackPacket:
        if len(self.allowed_evidence_ids) != len(set(self.allowed_evidence_ids)):
            raise ValueError("allowed_evidence_ids must be unique")
        message = self.risk_notice.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("risk_notice must contain a non-empty visible message")
        flagged_text = self.risk_notice.get("flagged_text")
        if (
            not isinstance(flagged_text, str)
            or not flagged_text.strip()
            or len(flagged_text) > 500
        ):
            raise ValueError(
                "risk_notice must identify one non-empty flagged_text span "
                "of at most 500 characters"
            )
        if self.anonymous_candidate.count(flagged_text) != 1:
            raise ValueError(
                "risk_notice.flagged_text must occur exactly once in the "
                "visible candidate"
            )
        expected_interventions = [
            "accept",
            "reject",
            "replace_span",
            "delete_span",
            "append_caveat",
        ]
        if self.permitted_interventions != expected_interventions:
            raise ValueError("permitted_interventions differs from the frozen contract")
        expected_constraints = [
            "Use only the candidate, risk notice, and evidence visible in this packet.",
            "Do not retrieve new evidence, invoke tools, rerun analysis, or rewrite the full artifact.",
            "A revision must be one local edit of at most 500 characters.",
            "Address only the risk identified in the notice; do not search for or alter unrelated issues.",
            "Do not use model knowledge or model-scale synthesis; if visible evidence is insufficient, reject or append a caveat instead of inventing a correction.",
        ]
        if self.human_proxy_constraints != expected_constraints:
            raise ValueError("human_proxy_constraints differs from the frozen contract")
        return self


class FeedbackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(pattern=r"^FP[0-9a-f]{20}$")
    judge_id: Literal["feedback"] = "feedback"
    decision: FeedbackDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    suggested_revision: HumanProxyEdit | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_is_present_when_required(self) -> FeedbackResult:
        if self.decision == "revise" and self.suggested_revision is None:
            raise ValueError("revise decisions require suggested_revision")
        if self.decision != "revise" and self.suggested_revision is not None:
            raise ValueError("only revise decisions may contain suggested_revision")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        return self


class FeedbackMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^FM[0-9a-f]{20}$")
    role: Literal["feedback"] = "feedback"
    packet_id: str = Field(pattern=r"^FP[0-9a-f]{20}$")
    sample_id: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    packet_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: FeedbackDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    suggested_revision: HumanProxyEdit | None = None
    evidence_ids: list[str]


def _packet_material(
    *,
    sample_id: str,
    question: str,
    canonical_evidence: Any,
    candidates: dict[str, str],
    condition_a: str,
    condition_b: str,
    rubric_version: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "question": question,
        "canonical_evidence": canonical_evidence,
        "candidate_hashes": {
            name: _sha256(candidates[name]) for name in sorted((condition_a, condition_b))
        },
        "conditions": sorted((condition_a, condition_b)),
        "rubric_version": rubric_version,
        "seed": seed,
    }


def prepare_blind_pair(
    record: dict[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    rubric_version: str,
    seed: int,
) -> tuple[BlindPacket, BlindPacket, BlindMap]:
    candidates = record.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("record.candidates must be an object")
    missing = [name for name in (condition_a, condition_b) if name not in candidates]
    if missing:
        raise ValueError(f"Missing candidate conditions: {missing}")
    sample_id = str(record.get("sample_id") or "")
    question = str(record.get("question") or "")
    evidence = record.get("canonical_evidence")
    material = _packet_material(
        sample_id=sample_id,
        question=question,
        canonical_evidence=evidence,
        candidates=candidates,
        condition_a=condition_a,
        condition_b=condition_b,
        rubric_version=rubric_version,
        seed=seed,
    )
    packet_id = f"JP{_sha256(material)[:20]}"
    assignment_digest = _sha256({"packet_id": packet_id, "seed": seed})
    if int(assignment_digest[:8], 16) % 2:
        first_a, first_b = condition_b, condition_a
    else:
        first_a, first_b = condition_a, condition_b

    mapping = BlindMap(
        packet_id=packet_id,
        sample_id=sample_id,
        assignments={
            "eval_a": BlindAssignment(A=first_a, B=first_b),
            "eval_b": BlindAssignment(A=first_b, B=first_a),
            "adjudicator": BlindAssignment(A=first_a, B=first_b),
        },
    )

    def build(judge_id: Literal["eval_a", "eval_b"]) -> BlindPacket:
        assignment = mapping.assignments[judge_id]
        visible = {
            "sample_id": sample_id,
            "judge_id": judge_id,
            "rubric_version": rubric_version,
            "question": question,
            "canonical_evidence": evidence,
            "candidate_a": str(candidates[assignment.A]),
            "candidate_b": str(candidates[assignment.B]),
        }
        return BlindPacket(
            packet_id=packet_id,
            content_sha256=_sha256(visible),
            **visible,
        )

    packet_a = build("eval_a")
    packet_b = build("eval_b")
    assert packet_a.candidate_a == packet_b.candidate_b
    assert packet_a.candidate_b == packet_b.candidate_a
    return packet_a, packet_b, mapping


def prepare_dual_evidence_blind_pair(
    record: dict[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    rubric_version: str,
    seed: int,
) -> tuple[BlindPacket, BlindPacket, BlindMap]:
    """Blind two reports that each have their own non-comparable evidence source.

    This is used for system-versus-human-reference quality comparisons. It prevents
    the human paper's corpus statistics from being treated as Gold for an independently
    retrieved system corpus while still allowing each report to be checked against its
    own evidence.
    """
    candidates = record.get("candidates")
    evidence_by_condition = record.get("evidence_by_condition")
    if not isinstance(candidates, dict) or not isinstance(evidence_by_condition, dict):
        raise TypeError("record requires candidates and evidence_by_condition objects")
    missing = [
        name
        for name in (condition_a, condition_b)
        if name not in candidates or name not in evidence_by_condition
    ]
    if missing:
        raise ValueError(f"Missing candidate or evidence conditions: {missing}")
    sample_id = str(record.get("sample_id") or "")
    question = str(record.get("question") or "")
    material = {
        "sample_id": sample_id,
        "question": question,
        "candidate_hashes": {
            name: _sha256(candidates[name]) for name in sorted((condition_a, condition_b))
        },
        "evidence_hashes": {
            name: _sha256(evidence_by_condition[name])
            for name in sorted((condition_a, condition_b))
        },
        "conditions": sorted((condition_a, condition_b)),
        "rubric_version": rubric_version,
        "seed": seed,
        "evidence_policy": "condition_specific_non_comparable",
    }
    packet_id = f"JP{_sha256(material)[:20]}"
    assignment_digest = _sha256({"packet_id": packet_id, "seed": seed})
    if int(assignment_digest[:8], 16) % 2:
        first_a, first_b = condition_b, condition_a
    else:
        first_a, first_b = condition_a, condition_b
    mapping = BlindMap(
        packet_id=packet_id,
        sample_id=sample_id,
        assignments={
            "eval_a": BlindAssignment(A=first_a, B=first_b),
            "eval_b": BlindAssignment(A=first_b, B=first_a),
            "adjudicator": BlindAssignment(A=first_a, B=first_b),
        },
    )

    def build(judge_id: Literal["eval_a", "eval_b"]) -> BlindPacket:
        assignment = mapping.assignments[judge_id]
        paired_evidence = {
            "policy": (
                "Evaluate each anonymous report only against its paired evidence. "
                "Do not compare corpus counts across candidates."
            ),
            "candidate_a_evidence": evidence_by_condition[assignment.A],
            "candidate_b_evidence": evidence_by_condition[assignment.B],
        }
        visible = {
            "sample_id": sample_id,
            "judge_id": judge_id,
            "rubric_version": rubric_version,
            "question": question,
            "canonical_evidence": paired_evidence,
            "candidate_a": str(candidates[assignment.A]),
            "candidate_b": str(candidates[assignment.B]),
        }
        return BlindPacket(
            packet_id=packet_id,
            content_sha256=_sha256(visible),
            **visible,
        )

    packet_a = build("eval_a")
    packet_b = build("eval_b")
    assert packet_a.candidate_a == packet_b.candidate_b
    assert (
        packet_a.canonical_evidence["candidate_a_evidence"]
        == packet_b.canonical_evidence["candidate_b_evidence"]
    )
    return packet_a, packet_b, mapping


def _collect_reference_ids(value: Any) -> list[str]:
    pattern = re.compile(r"^(?:[EH]\d{3,}|[A-Za-z][\w.-]*:[^\s]+)$")
    found: set[str] = set()

    def visit(item: Any, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                if pattern.fullmatch(str(child_key)):
                    found.add(str(child_key))
                visit(child, str(child_key))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, key)
        elif isinstance(item, str):
            identifier_field = key is not None and (
                key.casefold().endswith(("id", "ids"))
                or key.casefold().startswith("evidence")
            )
            if (identifier_field and item.strip()) or (
                key is None and pattern.fullmatch(item)
            ):
                found.add(item)

    visit(value)
    return sorted(found)


def prepare_feedback_packet(
    record: dict[str, Any],
    *,
    condition: str,
    rubric_version: str,
    seed: int,
    risk_notice: dict[str, Any] | None = None,
) -> FeedbackPacket:
    candidates = record.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("record.candidates must be an object")
    if condition not in candidates:
        raise ValueError(f"Missing feedback candidate condition: {condition}")
    sample_id = str(record.get("sample_id") or "")
    evidence = record.get("canonical_evidence")
    candidate = str(candidates[condition])
    visible_risk_notice = dict(risk_notice or {})
    if "flagged_text" not in visible_risk_notice:
        if len(candidate) > 500:
            raise ValueError(
                "A candidate longer than 500 characters requires an explicit "
                "risk_notice.flagged_text"
            )
        visible_risk_notice["flagged_text"] = candidate
    visible = {
        "sample_id": sample_id,
        "judge_id": "feedback",
        "rubric_version": rubric_version,
        "canonical_evidence": evidence,
        "anonymous_candidate": candidate,
        "allowed_evidence_ids": _collect_reference_ids(evidence),
        "risk_notice": visible_risk_notice,
        "permitted_interventions": [
            "accept",
            "reject",
            "replace_span",
            "delete_span",
            "append_caveat",
        ],
        "human_proxy_constraints": [
            "Use only the candidate, risk notice, and evidence visible in this packet.",
            "Do not retrieve new evidence, invoke tools, rerun analysis, or rewrite the full artifact.",
            "A revision must be one local edit of at most 500 characters.",
            "Address only the risk identified in the notice; do not search for or alter unrelated issues.",
            "Do not use model knowledge or model-scale synthesis; if visible evidence is insufficient, reject or append a caveat instead of inventing a correction.",
        ],
    }
    packet_material = {
        **visible,
        "condition_hash": _sha256(condition),
        "seed": seed,
    }
    return FeedbackPacket(
        packet_id=f"FP{_sha256(packet_material)[:20]}",
        content_sha256=_sha256(visible),
        **visible,
    )


def collect_reference_ids(value: Any) -> list[str]:
    """Return the evidence identifiers visible in a Judge packet."""

    return _collect_reference_ids(value)


def validate_blind_packet(packet: BlindPacket) -> None:
    """Verify that a persisted Judge packet still matches its visible payload."""

    visible = {
        "sample_id": packet.sample_id,
        "judge_id": packet.judge_id,
        "rubric_version": packet.rubric_version,
        "question": packet.question,
        "canonical_evidence": packet.canonical_evidence,
        "candidate_a": packet.candidate_a,
        "candidate_b": packet.candidate_b,
    }
    if packet.content_sha256 != _sha256(visible):
        raise ValueError(f"Blind packet content hash mismatch: {packet.packet_id}")


def validate_judge_result(
    result: JudgeResult,
    packet: BlindPacket,
    *,
    expected_judge_id: JudgeId | None = None,
) -> None:
    """Validate identity, packet binding, and addressable evidence citations."""

    validate_blind_packet(packet)
    if result.packet_id != packet.packet_id:
        raise ValueError("Judge result and packet IDs differ")
    if expected_judge_id is not None and result.judge_id != expected_judge_id:
        raise ValueError(
            f"Judge result identity must be {expected_judge_id}, got {result.judge_id}"
        )
    if result.judge_id != packet.judge_id and result.judge_id != "adjudicator":
        raise ValueError("Judge result identity differs from the visible packet")

    evidence = packet.canonical_evidence
    if (
        isinstance(evidence, dict)
        and evidence.get("policy")
        == (
            "Evaluate each anonymous report only against its paired evidence. "
            "Do not compare corpus counts across candidates."
        )
    ):
        allowed_by_slot = {
            "A": set(_collect_reference_ids(evidence.get("candidate_a_evidence"))),
            "B": set(_collect_reference_ids(evidence.get("candidate_b_evidence"))),
        }
    else:
        allowed = set(_collect_reference_ids(evidence))
        allowed_by_slot = {"A": allowed, "B": allowed}

    for candidate in result.candidates:
        cited = {
            evidence_id
            for claim in candidate.claims
            for evidence_id in claim.evidence_ids
        }
        unknown = sorted(cited - allowed_by_slot[candidate.slot])
        if unknown:
            raise ValueError(
                f"Judge result cites evidence IDs not visible for candidate "
                f"{candidate.slot}: {unknown}"
            )


def validate_feedback_result(
    result: FeedbackResult,
    packet: FeedbackPacket,
) -> None:
    if not isinstance(result, FeedbackResult):
        raise TypeError("Only FeedbackResult can enter feedback memory")
    if result.packet_id != packet.packet_id:
        raise ValueError("Feedback result and packet IDs differ")
    unknown = sorted(set(result.evidence_ids) - set(packet.allowed_evidence_ids))
    if unknown:
        raise ValueError(f"Feedback result cites unknown evidence IDs: {unknown}")
    edit = result.suggested_revision
    if edit is not None and edit.action in {"replace_span", "delete_span"}:
        flagged_text = str(packet.risk_notice["flagged_text"])
        target = str(edit.target_text or "")
        if target not in flagged_text:
            raise ValueError(
                "Human Proxy edit target lies outside the risk-notice flagged_text"
            )


def to_feedback_memory_record(
    result: FeedbackResult,
    packet: FeedbackPacket,
) -> FeedbackMemoryRecord:
    if not isinstance(result, FeedbackResult):
        raise TypeError("Evaluation Judge results cannot enter feedback memory")
    validate_feedback_result(result, packet)
    payload = {
        "packet_id": packet.packet_id,
        "sample_id": packet.sample_id,
        "packet_content_sha256": packet.content_sha256,
        "decision": result.decision,
        "confidence": result.confidence,
        "reason": result.reason,
        "suggested_revision": (
            result.suggested_revision.model_dump(mode="json")
            if result.suggested_revision is not None
            else None
        ),
        "evidence_ids": result.evidence_ids,
    }
    return FeedbackMemoryRecord(
        record_id=f"FM{_sha256(payload)[:20]}",
        rubric_version=packet.rubric_version,
        **payload,
    )


def scan_condition_leaks(value: Any, condition_names: list[str]) -> list[str]:
    text = canonical_json(value).casefold()
    normalized_text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()
    leaks = []
    for name in condition_names:
        normalized = str(name).strip().casefold()
        normalized_words = re.sub(
            r"[\W_]+",
            " ",
            normalized,
            flags=re.UNICODE,
        ).strip()
        if normalized and (
            normalized in text
            or (normalized_words and normalized_words in normalized_text)
        ):
            leaks.append(name)
    return sorted(set(leaks))


def _candidate_by_slot(result: JudgeResult) -> dict[Slot, CandidateJudgment]:
    return {candidate.slot: candidate for candidate in result.candidates}


def decode_judge_result(
    result: JudgeResult,
    mapping: BlindMap,
) -> dict[str, Any]:
    if result.packet_id != mapping.packet_id:
        raise ValueError("Judge result and blind map packet IDs differ")
    assignment = mapping.assignments[result.judge_id]
    by_slot = _candidate_by_slot(result)
    decoded = {
        assignment.A: by_slot["A"],
        assignment.B: by_slot["B"],
    }
    preference = (
        "tie"
        if result.preference == "tie"
        else getattr(assignment, result.preference)
    )
    return {"candidates": decoded, "preference": preference}


def _claim_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip(" .;:")


def _unsupported_rate(candidate: CandidateJudgment) -> float:
    if not candidate.claims:
        return 0.0
    unsupported = sum(claim.verdict != "supported" for claim in candidate.claims)
    return unsupported / len(candidate.claims)


def detect_conflicts(
    left: JudgeResult,
    right: JudgeResult,
    mapping: BlindMap,
    *,
    unsupported_rate_tolerance: float = 0.25,
) -> list[str]:
    if left.packet_id != right.packet_id or left.packet_id != mapping.packet_id:
        raise ValueError("Cannot compare results from different packets")
    decoded_left = decode_judge_result(left, mapping)
    decoded_right = decode_judge_result(right, mapping)
    reasons: list[str] = []
    for condition in sorted(decoded_left["candidates"]):
        first = decoded_left["candidates"][condition]
        second = decoded_right["candidates"][condition]
        if abs(first.completeness_score - second.completeness_score) > 1:
            reasons.append(f"{condition}:completeness_gap")
        if abs(_unsupported_rate(first) - _unsupported_rate(second)) > unsupported_rate_tolerance:
            reasons.append(f"{condition}:unsupported_rate_gap")
        first_claims = {_claim_key(item.claim): item.verdict for item in first.claims}
        second_claims = {_claim_key(item.claim): item.verdict for item in second.claims}
        for claim in sorted(set(first_claims) & set(second_claims)):
            if first_claims[claim] != second_claims[claim]:
                reasons.append(f"{condition}:claim_verdict:{_sha256(claim)[:8]}")
    preferences = {decoded_left["preference"], decoded_right["preference"]}
    preferences.discard("tie")
    if len(preferences) > 1:
        reasons.append("opposite_pairwise_preferences")
    return sorted(set(reasons))


def _candidate_summary(candidates: list[CandidateJudgment]) -> dict[str, Any]:
    claims = [claim for candidate in candidates for claim in candidate.claims]
    supported = sum(claim.verdict == "supported" for claim in claims)
    return {
        "supported_claims": supported,
        "unsupported_claims": len(claims) - supported,
        "claim_count": len(claims),
        "mean_completeness": mean(
            candidate.completeness_score for candidate in candidates
        ),
    }


def resolve_packet_results(
    left: JudgeResult,
    right: JudgeResult,
    mapping: BlindMap,
    *,
    adjudication: JudgeResult | None = None,
) -> dict[str, Any]:
    conflicts = detect_conflicts(left, right, mapping)
    if conflicts:
        if adjudication is None:
            raise ValueError(
                f"Packet {mapping.packet_id} requires adjudication: {conflicts}"
            )
        if adjudication.judge_id != "adjudicator":
            raise ValueError("Conflict resolution requires an adjudicator result")
        decoded = decode_judge_result(adjudication, mapping)
        condition_summaries = {
            condition: _candidate_summary([candidate])
            for condition, candidate in decoded["candidates"].items()
        }
        preference = decoded["preference"]
        source = "adjudication"
    else:
        first = decode_judge_result(left, mapping)
        second = decode_judge_result(right, mapping)
        condition_summaries = {
            condition: _candidate_summary(
                [first["candidates"][condition], second["candidates"][condition]]
            )
            for condition in first["candidates"]
        }
        preferences = [first["preference"], second["preference"]]
        non_ties = [item for item in preferences if item != "tie"]
        preference = non_ties[0] if non_ties else "tie"
        source = "dual_consensus"
    return {
        "packet_id": mapping.packet_id,
        "sample_id": mapping.sample_id,
        "source": source,
        "conflicts": conflicts,
        "conditions": condition_summaries,
        "preference": preference,
    }


def aggregate_resolved_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "supported_claims": 0,
            "unsupported_claims": 0,
            "completeness": [],
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    )
    for row in rows:
        conditions = set(row["conditions"])
        if len(conditions) != 2:
            raise ValueError("Each resolved row must compare exactly two conditions")
        for condition, values in row["conditions"].items():
            target = accumulators[condition]
            target["supported_claims"] += int(values["supported_claims"])
            target["unsupported_claims"] += int(values["unsupported_claims"])
            target["completeness"].append(float(values["mean_completeness"]))
            if row["preference"] == "tie":
                target["ties"] += 1
            elif row["preference"] == condition:
                target["wins"] += 1
            else:
                target["losses"] += 1

    metrics: dict[str, Any] = {}
    for condition, values in sorted(accumulators.items()):
        claim_count = values["supported_claims"] + values["unsupported_claims"]
        comparisons = values["wins"] + values["losses"] + values["ties"]
        metrics[condition] = {
            "evidence_grounded_factuality": (
                values["supported_claims"] / claim_count if claim_count else None
            ),
            "unsupported_claim_rate": (
                values["unsupported_claims"] / claim_count if claim_count else None
            ),
            "mean_completeness": (
                mean(values["completeness"]) if values["completeness"] else None
            ),
            "pairwise_preference_score": (
                (values["wins"] + 0.5 * values["ties"]) / comparisons
                if comparisons
                else None
            ),
            "counts": {
                "supported_claims": values["supported_claims"],
                "unsupported_claims": values["unsupported_claims"],
                "wins": values["wins"],
                "losses": values["losses"],
                "ties": values["ties"],
            },
        }
    return {"packets": len(rows), "conditions": metrics}


def build_adjudication_packet(
    packet_a: BlindPacket,
    result_a: JudgeResult,
    result_b: JudgeResult,
    mapping: BlindMap,
) -> dict[str, Any]:
    conflicts = detect_conflicts(result_a, result_b, mapping)
    if not conflicts:
        raise ValueError("Adjudication packet requested without a conflict")
    decoded_b = decode_judge_result(result_b, mapping)
    assignment = mapping.assignments["adjudicator"]
    remapped_b = {}
    for slot in ("A", "B"):
        candidate = decoded_b["candidates"][getattr(assignment, slot)].model_dump(mode="json")
        candidate["slot"] = slot
        remapped_b[slot] = candidate
    preference_b = decoded_b["preference"]
    if preference_b != "tie":
        preference_b = "A" if preference_b == assignment.A else "B"
    blind_conflicts = []
    for reason in conflicts:
        blind_reason = reason
        for slot in ("A", "B"):
            blind_reason = blind_reason.replace(getattr(assignment, slot), f"candidate_{slot}")
        blind_conflicts.append(blind_reason)
    return {
        "packet_id": packet_a.packet_id,
        "judge_id": "adjudicator",
        "rubric_version": packet_a.rubric_version,
        "question": packet_a.question,
        "canonical_evidence": packet_a.canonical_evidence,
        "candidate_a": packet_a.candidate_a,
        "candidate_b": packet_a.candidate_b,
        "conflicts": blind_conflicts,
        "judge_a_result": result_a.model_dump(mode="json"),
        "judge_b_result_remapped": {
            "packet_id": result_b.packet_id,
            "judge_id": "eval_b",
            "candidates": [remapped_b["A"], remapped_b["B"]],
            "preference": preference_b,
            "rationale": result_b.rationale,
        },
    }
