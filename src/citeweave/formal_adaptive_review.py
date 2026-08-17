from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adaptive_review import (
    AdaptiveReviewPolicy,
    FeedbackMemory,
    ReviewAction,
    ReviewIssue,
    make_review_decision,
)
from .io import atomic_write_bytes, sha256_file, write_json
from .judge_protocol import (
    FeedbackMemoryRecord,
    FeedbackPacket,
    FeedbackResult,
    canonical_json,
    collect_reference_ids,
    prepare_feedback_packet,
    scan_condition_leaks,
    to_feedback_memory_record,
)

AdaptiveCondition = Literal["always_review", "static_review", "adaptive_review"]
TopicRole = Literal["development", "locked"]
ArtifactType = Literal["report", "graph"]
QualityDecision = Literal["pass", "fail"]

FORMAL_CONDITIONS: tuple[AdaptiveCondition, ...] = (
    "always_review",
    "static_review",
    "adaptive_review",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_line(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _append_fsynced(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_line(value))
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _reference_ids(value: Any) -> list[str]:
    return collect_reference_ids(value)


class FormalAdaptiveCase(BaseModel):
    """One pre-generated report or graph answer presented to all three policies."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    topic_role: TopicRole
    artifact_type: ArtifactType
    canonical_evidence: Any
    anonymous_candidate: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    issue_signature: str = Field(min_length=1)
    risk_notice_message: str = Field(
        default="Review only the system-flagged span against the visible evidence.",
        min_length=1,
    )
    risk_scope_text: str | None = Field(default=None, min_length=1, max_length=500)
    severity: Literal["low", "medium", "high", "critical"]
    detector_score: float = Field(ge=0.0, le=1.0)
    auto_accept_context_ok: bool = True

    @model_validator(mode="after")
    def exact_visible_risk_scope(self) -> FormalAdaptiveCase:
        if (
            self.risk_scope_text is not None
            and self.anonymous_candidate.count(self.risk_scope_text) != 1
        ):
            raise ValueError(
                "risk_scope_text must occur exactly once in anonymous_candidate"
            )
        return self


class AdaptiveEvaluationPacket(BaseModel):
    """Anonymous final-output packet for a judge that cannot update policy memory."""

    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(pattern=r"^EP[0-9a-f]{20}$")
    sample_id: str = Field(min_length=1)
    judge_id: Literal["evaluation"] = "evaluation"
    rubric_version: str = Field(min_length=1)
    canonical_evidence: Any
    anonymous_candidate: str
    allowed_evidence_ids: list[str]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> AdaptiveEvaluationPacket:
        if len(self.allowed_evidence_ids) != len(set(self.allowed_evidence_ids)):
            raise ValueError("allowed_evidence_ids must be unique")
        return self


class AdaptiveEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(pattern=r"^EP[0-9a-f]{20}$")
    judge_id: Literal["evaluation"] = "evaluation"
    decision: QualityDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> AdaptiveEvaluationResult:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        return self


class FormalAdaptiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = "formal-adaptive-v3-scoped-human-proxy"
    feedback_rubric_version: str = "feedback-human-proxy-v3"
    evaluation_rubric_version: str = "adaptive-evaluation-v1"
    experiment_mode: Literal["formal", "development_calibration"] = "formal"
    seed: int = 42
    minimum_confirmations: int = Field(default=2, ge=1)
    auto_accept_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    audit_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    static_detector_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    static_review_severities: list[Literal["low", "medium", "high", "critical"]] = (
        Field(default_factory=lambda: ["high", "critical"])
    )


def build_evaluation_packet(
    case: FormalAdaptiveCase,
    *,
    final_candidate: str,
    rubric_version: str,
    seed: int,
    condition: AdaptiveCondition,
) -> AdaptiveEvaluationPacket:
    visible = {
        "sample_id": case.sample_id,
        "judge_id": "evaluation",
        "rubric_version": rubric_version,
        "canonical_evidence": case.canonical_evidence,
        "anonymous_candidate": final_candidate,
        "allowed_evidence_ids": _reference_ids(case.canonical_evidence),
    }
    material = {
        **visible,
        "condition_hash": _sha(condition),
        "seed": seed,
    }
    return AdaptiveEvaluationPacket(
        packet_id=f"EP{_sha(material)[:20]}",
        content_sha256=_sha(visible),
        **visible,
    )


def validate_evaluation_result(
    result: AdaptiveEvaluationResult,
    packet: AdaptiveEvaluationPacket,
) -> None:
    if not isinstance(result, AdaptiveEvaluationResult):
        raise TypeError("Only AdaptiveEvaluationResult can complete an Evaluation packet")
    if result.packet_id != packet.packet_id:
        raise ValueError("Evaluation result and packet IDs differ")
    unknown = sorted(set(result.evidence_ids) - set(packet.allowed_evidence_ids))
    if unknown:
        raise ValueError(f"Evaluation result cites unknown evidence IDs: {unknown}")


def evaluation_to_feedback_memory(
    result: AdaptiveEvaluationResult,
    packet: AdaptiveEvaluationPacket,
) -> FeedbackMemoryRecord:
    del result, packet
    raise TypeError("Evaluation Judge results cannot enter feedback memory")


def _issue(case: FormalAdaptiveCase) -> ReviewIssue:
    return ReviewIssue(
        item_id=case.sample_id,
        dataset_id=case.dataset_id,
        stage=case.stage,
        issue_signature=case.issue_signature,
        message="Candidate requires policy-controlled external quality review.",
        severity=case.severity,
        detector_score=case.detector_score,
        payload={
            "artifact_type": case.artifact_type,
            "auto_accept_context_ok": case.auto_accept_context_ok,
        },
    )


def _read_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    rows: list[BaseModel] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    return rows


def load_cases(path: Path) -> list[FormalAdaptiveCase]:
    return [
        FormalAdaptiveCase.model_validate(row.model_dump())
        for row in _read_jsonl(path, FormalAdaptiveCase)
    ]


def _candidate_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("anonymous_candidate"), str):
            return str(value["anonymous_candidate"]).strip()
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(value.get("response"), dict):
            return _candidate_text(value["response"])
    raise ValueError("Candidate artifact has no non-empty report or completion content")


def _load_candidate_artifact(
    path: Path,
    *,
    checkpoint_item_id: str | None = None,
) -> str:
    if checkpoint_item_id is None and path.suffix.casefold() not in {".json", ".jsonl"}:
        return _candidate_text(path.read_text(encoding="utf-8"))
    if path.suffix.casefold() == ".jsonl":
        if not checkpoint_item_id:
            raise ValueError("JSONL candidate artifacts require checkpoint_item_id")
        matched: dict[str, Any] | None = None
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            if str(row.get("item_id")) == checkpoint_item_id:
                if matched and matched.get("status") == "complete":
                    raise ValueError(
                        f"Checkpoint contains a record after completion: {checkpoint_item_id}"
                    )
                matched = row
        if not matched or matched.get("status") != "complete":
            raise ValueError(f"No completed checkpoint item: {checkpoint_item_id}")
        return _candidate_text(matched)
    return _candidate_text(json.loads(path.read_text(encoding="utf-8")))


def assemble_formal_cases(spec_path: Path, output_path: Path) -> list[FormalAdaptiveCase]:
    """Assemble immutable report/graph candidates from formal output artifacts.

    Paths are resolved relative to the YAML specification. Graph JSONL checkpoints
    name an ``item_id``; report Markdown files are read directly.
    """

    specification = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    topics = specification.get("topics") or []
    cases: list[FormalAdaptiveCase] = []
    for topic in topics:
        dataset_id = str(topic["dataset_id"])
        role = str(topic["topic_role"])
        for artifact in topic.get("artifacts") or []:
            evidence_path = (spec_path.parent / artifact["evidence_path"]).resolve()
            candidate_path = (spec_path.parent / artifact["candidate_path"]).resolve()
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            item_id = artifact.get("checkpoint_item_id")
            candidate = _load_candidate_artifact(
                candidate_path,
                checkpoint_item_id=str(item_id) if item_id is not None else None,
            )
            sample_id = str(
                artifact.get("sample_id")
                or f"{dataset_id}:{artifact['artifact_type']}:{item_id or candidate_path.stem}"
            )
            cases.append(
                FormalAdaptiveCase(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    topic_role=role,
                    artifact_type=artifact["artifact_type"],
                    canonical_evidence=evidence,
                    anonymous_candidate=candidate,
                    stage=str(artifact.get("stage") or "formal_output"),
                    issue_signature=str(
                        artifact.get("issue_signature")
                        or f"{artifact['artifact_type']}_candidate_quality_review"
                    ),
                    risk_notice_message=str(
                        artifact.get("risk_notice_message")
                        or "Review only the system-flagged span against the visible evidence."
                    ),
                    risk_scope_text=(
                        str(artifact["risk_scope_text"])
                        if artifact.get("risk_scope_text") is not None
                        else None
                    ),
                    severity=artifact.get("severity", "low"),
                    detector_score=float(artifact.get("detector_score", 0.25)),
                    auto_accept_context_ok=bool(
                        artifact.get("auto_accept_context_ok", True)
                    ),
                )
            )
    payload = b"".join(
        _canonical_line(case.model_dump(mode="json")) for case in cases
    )
    if output_path.exists() and output_path.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite different formal cases: {output_path}")
    if not output_path.exists():
        atomic_write_bytes(output_path, payload)
    return cases


def validate_formal_topic_sequence(
    cases: list[FormalAdaptiveCase],
    human_reference_registry: Path,
    *,
    experiment_mode: Literal["formal", "development_calibration"] = "formal",
) -> list[str]:
    registry = yaml.safe_load(human_reference_registry.read_text(encoding="utf-8"))
    references = registry.get("references") or []
    expected = [
        str(reference["id"])
        for reference in references
        if reference.get("role") in {"development", "locked"}
    ]
    expected_roles = {
        str(reference["id"]): str(reference["role"])
        for reference in references
        if reference.get("role") in {"development", "locked"}
    }
    roles = [expected_roles[item] for item in expected]
    if experiment_mode == "formal":
        if len(expected) != 8:
            raise ValueError(
                "Formal adaptive experiment requires exactly 2 development + 6 locked topics"
            )
        if roles != ["development", "development", *(["locked"] * 6)]:
            raise ValueError(
                "Reference registry order must be 2 development topics then 6 locked topics"
            )
    else:
        if len(expected) != 2 or roles != ["development", "development"]:
            raise ValueError(
                "Development calibration requires exactly two development topics"
            )
    observed: list[str] = []
    for case in cases:
        if not observed or observed[-1] != case.dataset_id:
            if case.dataset_id in observed:
                raise ValueError("Cases for a topic must form one contiguous block")
            observed.append(case.dataset_id)
        if expected_roles.get(case.dataset_id) != case.topic_role:
            raise ValueError(f"Topic role mismatch for {case.dataset_id}")
    if observed != expected:
        raise ValueError(f"Case topic order differs from frozen registry: {observed} != {expected}")
    if len({case.sample_id for case in cases}) != len(cases):
        raise ValueError("Formal adaptive sample_id values must be unique")
    return expected


class FormalAdaptiveReviewRunner:
    """File-mediated, resumable runner; it never invokes an LLM."""

    def __init__(
        self,
        *,
        cases_path: Path,
        reference_registry: Path,
        output_root: Path,
        config: FormalAdaptiveConfig | None = None,
    ):
        self.cases_path = cases_path.resolve()
        self.reference_registry = reference_registry.resolve()
        self.output_root = output_root.resolve()
        self.config = config or FormalAdaptiveConfig()
        self.cases = load_cases(self.cases_path)
        self.topic_sequence = validate_formal_topic_sequence(
            self.cases,
            self.reference_registry,
            experiment_mode=self.config.experiment_mode,
        )
        self.manifest_path = self.output_root / "manifest.json"
        self._initialize_or_verify_manifest()

    def _manifest(self) -> dict[str, Any]:
        return {
            "protocol_version": self.config.protocol_version,
            "cases_path": str(self.cases_path),
            "cases_sha256": sha256_file(self.cases_path),
            "reference_registry": str(self.reference_registry),
            "reference_registry_sha256": sha256_file(self.reference_registry),
            "topic_sequence": self.topic_sequence,
            "cases": len(self.cases),
            "conditions": list(FORMAL_CONDITIONS),
            "formal_results_used": self.config.experiment_mode == "formal",
            "config": self.config.model_dump(mode="json"),
            "config_sha256": _sha(self.config.model_dump(mode="json")),
            "judge_separation": {
                "feedback_updates_online_memory": True,
                "evaluation_updates_online_memory": False,
            },
            "human_proxy_capability": {
                "experiment_only": True,
                "active_only_after_visible_risk_notice": True,
                "visible_packet_only": True,
                "flagged_risk_only": True,
                "candidate_visibility": "flagged_excerpt_only",
                "edit_target_bound_to_flagged_text": True,
                "maximum_flagged_text_characters": 500,
                "may_search_for_additional_issues": False,
                "may_use_model_knowledge_or_model_scale_synthesis": False,
                "permitted_local_edits": [
                    "replace_span",
                    "delete_span",
                    "append_caveat",
                ],
                "maximum_edits_per_intervention": 1,
                "maximum_edit_characters": 500,
                "external_tools_or_retrieval": False,
                "full_artifact_rewrite": False,
            },
        }

    def _initialize_or_verify_manifest(self) -> None:
        expected = self._manifest()
        if self.manifest_path.exists():
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError("Formal adaptive manifest/input/config hash mismatch")
        else:
            write_json(self.manifest_path, expected)
        for condition in FORMAL_CONDITIONS:
            self._initialize_condition(condition)

    def _condition_dir(self, condition: AdaptiveCondition) -> Path:
        return self.output_root / condition

    def _state_path(self, condition: AdaptiveCondition) -> Path:
        return self._condition_dir(condition) / "state.json"

    def _initialize_condition(self, condition: AdaptiveCondition) -> None:
        path = self._state_path(condition)
        if path.exists():
            return
        write_json(
            path,
            {
                "condition": condition,
                "next_case_index": 0,
                "records": [],
                "feedback_memory_records": 0,
                "completed": False,
            },
        )

    def _load_state(self, condition: AdaptiveCondition) -> dict[str, Any]:
        state = json.loads(self._state_path(condition).read_text(encoding="utf-8"))
        if state.get("condition") != condition:
            raise RuntimeError(f"Condition state identity mismatch: {condition}")
        self._verify_record_hashes(condition, state)
        return state

    def _verify_record_hashes(
        self, condition: AdaptiveCondition, state: dict[str, Any]
    ) -> None:
        root = self._condition_dir(condition)
        for record in state.get("records", []):
            for stem in ("feedback_packet", "feedback_result", "evaluation_packet", "evaluation_result"):
                relative = record.get(f"{stem}_path")
                expected = record.get(f"{stem}_sha256")
                if not relative:
                    continue
                path = root / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeError(f"Immutable {stem} hash mismatch: {path}")

    def _save_state(self, condition: AdaptiveCondition, state: dict[str, Any]) -> None:
        write_json(self._state_path(condition), state)

    def _policy_memory(self, condition: AdaptiveCondition) -> FeedbackMemory:
        path = self._condition_dir(condition) / "policy_memory.jsonl"
        return FeedbackMemory.load(path)

    def _action(
        self,
        condition: AdaptiveCondition,
        case: FormalAdaptiveCase,
        memory: FeedbackMemory,
    ) -> ReviewAction:
        issue = _issue(case)
        if condition == "always_review":
            return ReviewAction(
                case.sample_id, "escalate", "Always Review requests every item.", 0, None
            )
        if condition == "static_review":
            requested = (
                case.severity in set(self.config.static_review_severities)
                or case.detector_score >= self.config.static_detector_threshold
                or not case.auto_accept_context_ok
            )
            return ReviewAction(
                case.sample_id,
                "escalate" if requested else "auto_accept",
                "Frozen static rule requested review."
                if requested
                else "Frozen static rule accepted without review.",
                0,
                None,
            )
        return AdaptiveReviewPolicy(
            memory,
            minimum_confirmations=self.config.minimum_confirmations,
            auto_accept_threshold=self.config.auto_accept_threshold,
            audit_rate=self.config.audit_rate,
        ).decide(issue)

    def _write_immutable_packet(
        self, path: Path, packet: BaseModel, *, condition: AdaptiveCondition
    ) -> None:
        value = packet.model_dump(mode="json")
        leaks = scan_condition_leaks(value, list(FORMAL_CONDITIONS))
        if leaks:
            raise RuntimeError(f"Condition-name leakage in anonymous packet: {leaks}")
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"Refusing to overwrite immutable Judge packet: {path}")
            return
        atomic_write_bytes(path, payload)

    def _feedback_packet(
        self, condition: AdaptiveCondition, case: FormalAdaptiveCase
    ) -> FeedbackPacket:
        risk_scope = case.risk_scope_text
        if risk_scope is None:
            if len(case.anonymous_candidate) > 500:
                raise RuntimeError(
                    "Candidates longer than 500 characters require an explicit "
                    "risk_scope_text before Human Proxy review"
                )
            risk_scope = case.anonymous_candidate
        return prepare_feedback_packet(
            {
                "sample_id": case.sample_id,
                "canonical_evidence": case.canonical_evidence,
                # The Human Proxy sees the exact flagged excerpt, not the full
                # report. The validated local edit is later applied to the
                # frozen full candidate by the state machine.
                "candidates": {condition: risk_scope},
            },
            condition=condition,
            rubric_version=self.config.feedback_rubric_version,
            seed=self.config.seed,
            risk_notice={
                "stage": case.stage,
                "issue_signature": case.issue_signature,
                "severity": case.severity,
                "detector_score": case.detector_score,
                "message": case.risk_notice_message,
                "flagged_text": risk_scope,
            },
        )

    @staticmethod
    def _apply_human_proxy_edit(candidate: str, result: FeedbackResult) -> str:
        edit = result.suggested_revision
        if edit is None:
            raise ValueError("A revise decision requires one local human-proxy edit")
        target = edit.target_text or ""
        replacement = edit.replacement_text or ""
        if edit.action in {"replace_span", "delete_span"}:
            occurrences = candidate.count(target)
            if occurrences != 1:
                raise ValueError(
                    "Human-proxy target_text must occur exactly once in the visible candidate"
                )
            if target == candidate:
                raise ValueError(
                    "Human-proxy local edits cannot replace or delete the full artifact"
                )
            return candidate.replace(
                target,
                replacement if edit.action == "replace_span" else "",
                1,
            )
        if edit.action == "append_caveat":
            return f"{candidate.rstrip()}\n\n{replacement.strip()}"
        raise ValueError(f"Unsupported human-proxy edit action: {edit.action}")

    def _start_case(
        self,
        condition: AdaptiveCondition,
        state: dict[str, Any],
        memory: FeedbackMemory,
    ) -> bool:
        index = int(state["next_case_index"])
        if index >= len(self.cases):
            state["completed"] = True
            self._save_state(condition, state)
            return False
        case = self.cases[index]
        action = self._action(condition, case, memory)
        record: dict[str, Any] = {
            "case_index": index,
            "sample_id": case.sample_id,
            "dataset_id": case.dataset_id,
            "topic_role": case.topic_role,
            "artifact_type": case.artifact_type,
            "action": asdict(action),
            "review_requested": action.action in {"escalate", "audit"},
            "auto_accepted": action.action == "auto_accept",
            "feedback_memory_version_before": memory.version,
            "status": "new",
        }
        state["records"].append(record)
        if record["review_requested"]:
            packet = self._feedback_packet(condition, case)
            relative = Path("packets") / "feedback" / f"{packet.packet_id}.json"
            path = self._condition_dir(condition) / relative
            self._write_immutable_packet(path, packet, condition=condition)
            record.update(
                {
                    "feedback_packet_id": packet.packet_id,
                    "feedback_packet_path": relative.as_posix(),
                    "feedback_packet_sha256": sha256_file(path),
                    "status": "awaiting_feedback",
                }
            )
        else:
            self._create_evaluation_packet(condition, case, record, case.anonymous_candidate)
        self._save_state(condition, state)
        return True

    def _inbox(self, condition: AdaptiveCondition, kind: str, packet_id: str) -> Path:
        return self._condition_dir(condition) / "inbox" / kind / f"{packet_id}.json"

    def _archive_result(
        self,
        condition: AdaptiveCondition,
        *,
        kind: str,
        packet_id: str,
        inbox: Path,
    ) -> tuple[str, str]:
        relative = Path("results") / kind / f"{packet_id}.json"
        target = self._condition_dir(condition) / relative
        payload = inbox.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError(f"Conflicting immutable Judge result: {target}")
        if not target.exists():
            atomic_write_bytes(target, payload)
        return relative.as_posix(), sha256_file(target)

    def _consume_feedback(
        self,
        condition: AdaptiveCondition,
        state: dict[str, Any],
        memory: FeedbackMemory,
        record: dict[str, Any],
        case: FormalAdaptiveCase,
    ) -> bool:
        if not record.get("review_requested") or record.get("status") != "awaiting_feedback":
            raise RuntimeError(
                "Human Proxy feedback is allowed only after a policy-requested review"
            )
        packet_path = self._condition_dir(condition) / record["feedback_packet_path"]
        packet = FeedbackPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
        inbox = self._inbox(condition, "feedback", packet.packet_id)
        if not inbox.is_file():
            return False
        result = FeedbackResult.model_validate_json(inbox.read_text(encoding="utf-8"))
        # Validate and materialize the proposed intervention before appending either
        # feedback or policy memory. A malformed/overbroad edit must fail without
        # teaching the adaptive policy from an intervention that was never applied.
        final_candidate = case.anonymous_candidate
        if result.decision == "revise":
            final_candidate = self._apply_human_proxy_edit(final_candidate, result)
        elif result.decision == "reject":
            final_candidate = (
                "ABSTAINED AFTER EXTERNAL FEEDBACK REVIEW. "
                "The submitted candidate was rejected as unsafe or unsupported."
            )
        memory_record = to_feedback_memory_record(result, packet)
        memory_path = self._condition_dir(condition) / "feedback_memory.jsonl"
        existing_ids = {
            item.record_id
            for item in _read_jsonl(memory_path, FeedbackMemoryRecord)
        } if memory_path.exists() else set()
        if memory_record.record_id not in existing_ids:
            _append_fsynced(memory_path, memory_record.model_dump(mode="json"))
        decision_map = {"accept": "accept", "revise": "correct", "reject": "reject"}
        existing_policy = [
            decision
            for decision in memory.decisions
            if decision.item_id == case.sample_id
            and decision.dataset_id == case.dataset_id
        ]
        if len(existing_policy) > 1:
            raise RuntimeError(f"Duplicate policy-memory item: {case.sample_id}")
        if not existing_policy:
            policy_decision = make_review_decision(
                _issue(case),
                reviewer_code="CODEX-HUMAN-PROXY",
                decision=decision_map[result.decision],  # type: ignore[arg-type]
                original={"candidate_sha256": _sha(case.anonymous_candidate)},
                correction=result.suggested_revision,
                reason=result.reason,
                review_seconds=0.0,
                feedback_memory_version=memory.version,
            )
            memory.append(
                self._condition_dir(condition) / "policy_memory.jsonl",
                policy_decision,
            )
        relative, result_sha = self._archive_result(
            condition,
            kind="feedback",
            packet_id=packet.packet_id,
            inbox=inbox,
        )
        record.update(
            {
                "feedback_result_path": relative,
                "feedback_result_sha256": result_sha,
                "feedback_decision": result.decision,
                "feedback_memory_record_id": memory_record.record_id,
                "feedback_memory_version_after": memory.version,
            }
        )
        state["feedback_memory_records"] = memory.version
        self._create_evaluation_packet(condition, case, record, final_candidate)
        self._save_state(condition, state)
        return True

    def _create_evaluation_packet(
        self,
        condition: AdaptiveCondition,
        case: FormalAdaptiveCase,
        record: dict[str, Any],
        final_candidate: str,
    ) -> None:
        packet = build_evaluation_packet(
            case,
            final_candidate=final_candidate,
            rubric_version=self.config.evaluation_rubric_version,
            seed=self.config.seed,
            condition=condition,
        )
        relative = Path("packets") / "evaluation" / f"{packet.packet_id}.json"
        path = self._condition_dir(condition) / relative
        self._write_immutable_packet(path, packet, condition=condition)
        record.update(
            {
                "final_candidate_sha256": _sha(final_candidate),
                "evaluation_packet_id": packet.packet_id,
                "evaluation_packet_path": relative.as_posix(),
                "evaluation_packet_sha256": sha256_file(path),
                "status": "awaiting_evaluation",
            }
        )

    def _consume_evaluation(
        self,
        condition: AdaptiveCondition,
        state: dict[str, Any],
        record: dict[str, Any],
    ) -> bool:
        packet_path = self._condition_dir(condition) / record["evaluation_packet_path"]
        packet = AdaptiveEvaluationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        inbox = self._inbox(condition, "evaluation", packet.packet_id)
        if not inbox.is_file():
            return False
        result = AdaptiveEvaluationResult.model_validate_json(
            inbox.read_text(encoding="utf-8")
        )
        validate_evaluation_result(result, packet)
        relative, result_sha = self._archive_result(
            condition,
            kind="evaluation",
            packet_id=packet.packet_id,
            inbox=inbox,
        )
        record.update(
            {
                "evaluation_result_path": relative,
                "evaluation_result_sha256": result_sha,
                "quality_passed": result.decision == "pass",
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        state["next_case_index"] = int(state["next_case_index"]) + 1
        state["completed"] = state["next_case_index"] == len(self.cases)
        self._save_state(condition, state)
        return True

    def advance_condition(self, condition: AdaptiveCondition) -> dict[str, Any]:
        state = self._load_state(condition)
        memory = self._policy_memory(condition)
        memory_delta = memory.version - int(state["feedback_memory_records"])
        active_status = (
            state["records"][-1].get("status") if state.get("records") else None
        )
        if memory_delta not in {0, 1} or (
            memory_delta == 1 and active_status != "awaiting_feedback"
        ):
            raise RuntimeError("Feedback memory version differs from condition checkpoint")
        progressed = False
        while not state["completed"]:
            index = int(state["next_case_index"])
            active = (
                state["records"][-1]
                if state["records"]
                and state["records"][-1]["case_index"] == index
                and state["records"][-1]["status"] != "complete"
                else None
            )
            if active is None:
                progressed = self._start_case(condition, state, memory) or progressed
                active = state["records"][-1]
            case = self.cases[index]
            if active["status"] == "awaiting_feedback":
                if not self._consume_feedback(condition, state, memory, active, case):
                    break
                progressed = True
            if active["status"] == "awaiting_evaluation":
                if not self._consume_evaluation(condition, state, active):
                    break
                progressed = True
        return {
            "condition": condition,
            "progressed": progressed,
            "completed": state["completed"],
            "completed_cases": state["next_case_index"],
            "total_cases": len(self.cases),
            "pending": (
                state["records"][-1]["status"]
                if state["records"] and not state["completed"]
                else None
            ),
        }

    def advance_all(self) -> dict[str, Any]:
        rows = [self.advance_condition(condition) for condition in FORMAL_CONDITIONS]
        summary = {"conditions": rows, "completed": all(row["completed"] for row in rows)}
        if summary["completed"]:
            summary["metrics"] = self.finalize_metrics()
        return summary

    def finalize_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for condition in FORMAL_CONDITIONS:
            state = self._load_state(condition)
            if not state["completed"] or len(state["records"]) != len(self.cases):
                raise RuntimeError(f"Condition is incomplete: {condition}")
            records = state["records"]
            requests = sum(bool(row["review_requested"]) for row in records)
            passes = sum(bool(row["quality_passed"]) for row in records)
            auto = [row for row in records if row["auto_accepted"]]
            unsafe = sum(not bool(row["quality_passed"]) for row in auto)
            metrics[condition] = {
                "items": len(records),
                "review_requests": requests,
                "review_request_rate": requests / len(records),
                "quality_passes": passes,
                "final_quality_pass_rate": passes / len(records),
                "auto_accepted_items": len(auto),
                "unsafe_auto_accepts": unsafe,
                "unsafe_auto_accept_rate": unsafe / len(auto) if auto else None,
                "unsafe_auto_accept_rate_denominator": "auto_accepted_items",
                "feedback_memory_records": state["feedback_memory_records"],
            }
        result = {
            "protocol_version": self.config.protocol_version,
            "manifest_sha256": sha256_file(self.manifest_path),
            "metrics": metrics,
            "evaluation_feedback_leakage": False,
            "computed_at": datetime.now(UTC).isoformat(),
        }
        write_json(self.output_root / "metrics.json", result)
        return result
