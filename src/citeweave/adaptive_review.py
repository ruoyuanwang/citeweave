from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scipy.stats import beta

from .io import write_json

DecisionValue = Literal["accept", "correct", "reject", "abstain"]


@dataclass(frozen=True)
class ReviewIssue:
    item_id: str
    dataset_id: str
    stage: str
    issue_signature: str
    message: str
    severity: Literal["low", "medium", "high", "critical"]
    detector_score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReviewDecision:
    decision_id: str
    timestamp: str
    reviewer_code: str
    dataset_id: str
    stage: str
    item_id: str
    issue_signature: str
    detector_score: float
    decision: DecisionValue
    original: Any
    correction: Any
    reason: str
    review_seconds: float
    feedback_memory_version: int


@dataclass(frozen=True)
class ReviewAction:
    item_id: str
    action: Literal["escalate", "auto_accept", "audit"]
    reason: str
    prior_examples: int
    estimated_accept_probability: float | None


class FeedbackMemory:
    def __init__(self, decisions: list[ReviewDecision] | None = None):
        self.decisions = decisions or []

    @property
    def version(self) -> int:
        return len(self.decisions)

    @classmethod
    def load(cls, path: Path) -> FeedbackMemory:
        if not path.exists():
            return cls()
        decisions: list[ReviewDecision] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decisions.append(ReviewDecision(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid review record at {path}:{line_number}") from exc
        return cls(decisions)

    def append(self, path: Path, decision: ReviewDecision) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False, default=str))
            handle.write("\n")
        self.decisions.append(decision)

    def examples(self, issue: ReviewIssue, limit: int = 5) -> list[ReviewDecision]:
        matches = [
            decision
            for decision in self.decisions
            if decision.stage == issue.stage
            and decision.issue_signature == issue.issue_signature
            and decision.dataset_id != issue.dataset_id
        ]
        return matches[-limit:]

    def prompt_packet(self, issue: ReviewIssue, limit: int = 5) -> str:
        return json.dumps(
            [asdict(decision) for decision in self.examples(issue, limit=limit)],
            ensure_ascii=False,
            indent=2,
        )


class AdaptiveReviewPolicy:
    def __init__(
        self,
        memory: FeedbackMemory,
        *,
        minimum_confirmations: int = 2,
        auto_accept_threshold: float = 0.95,
        audit_rate: float = 0.10,
    ):
        if minimum_confirmations < 1:
            raise ValueError("minimum_confirmations must be positive")
        if not 0 <= audit_rate <= 1:
            raise ValueError("audit_rate must be between zero and one")
        self.memory = memory
        self.minimum_confirmations = minimum_confirmations
        self.auto_accept_threshold = auto_accept_threshold
        self.audit_rate = audit_rate

    def decide(self, issue: ReviewIssue) -> ReviewAction:
        examples = self.memory.examples(issue, limit=10_000)
        confirmation_datasets = {example.dataset_id for example in examples}
        if issue.severity in {"critical", "high"}:
            return ReviewAction(
                issue.item_id,
                "escalate",
                "High-severity issues always require independent review.",
                len(examples),
                None,
            )
        if issue.payload.get("auto_accept_context_ok") is False:
            return ReviewAction(
                issue.item_id,
                "escalate",
                "The current item does not satisfy the context guard learned with prior accepts.",
                len(examples),
                None,
            )
        if len(confirmation_datasets) < self.minimum_confirmations:
            return ReviewAction(
                issue.item_id,
                "escalate",
                "Insufficient distinct cross-dataset feedback for automatic handling.",
                len(examples),
                None,
            )
        accept_probability = sum(
            decision.decision == "accept" for decision in examples
        ) / len(examples)
        if accept_probability < self.auto_accept_threshold:
            return ReviewAction(
                issue.item_id,
                "escalate",
                "Prior reviewers did not consistently accept this issue class.",
                len(examples),
                accept_probability,
            )
        if self._selected_for_audit(issue):
            return ReviewAction(
                issue.item_id,
                "audit",
                "Stratified audit sample from an otherwise auto-accepted issue class.",
                len(examples),
                accept_probability,
            )
        return ReviewAction(
            issue.item_id,
            "auto_accept",
            "Cross-dataset feedback consistently accepted this issue class.",
            len(examples),
            accept_probability,
        )

    def _selected_for_audit(self, issue: ReviewIssue) -> bool:
        if self.audit_rate <= 0:
            return False
        digest = hashlib.sha256(
            f"{issue.dataset_id}|{issue.stage}|{issue.item_id}|{issue.issue_signature}".encode()
        ).digest()
        value = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        return value < self.audit_rate


def detect_acquisition_issues(dataset_id: str, manifest: dict[str, Any]) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []

    def add(signature: str, message: str, severity: str, score: float) -> None:
        issues.append(
            ReviewIssue(
                item_id=f"{dataset_id}:acquisition:{signature}",
                dataset_id=dataset_id,
                stage="acquisition",
                issue_signature=signature,
                message=message,
                severity=severity,  # type: ignore[arg-type]
                detector_score=score,
                payload=manifest,
            )
        )

    if manifest.get("truncated"):
        add(
            "truncated_registered_sample",
            "The acquisition is capped and must be interpreted as a benchmark sample.",
            "medium",
            1.0,
        )
    if manifest.get("failed_pages", 0):
        add("failed_pages", "One or more acquisition pages failed.", "critical", 1.0)
    if not manifest.get("complete") and not manifest.get("truncated"):
        add(
            "incomplete_uncapped_acquisition",
            "The uncapped acquisition did not satisfy source completeness.",
            "critical",
            1.0,
        )
    if manifest.get("duplicate_records", 0):
        add(
            "source_duplicates_observed",
            "Duplicate source identifiers were observed and deterministically removed.",
            "low",
            min(1.0, float(manifest["duplicate_records"]) / 10),
        )
    return issues


def detect_generation_issues(
    dataset_id: str,
    validation: dict[str, Any],
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    checks = (
        ("invalid_evidence_ids", "invalid_evidence_id", "critical"),
        ("unsupported_numbers", "unsupported_numeric_claim", "critical"),
        ("uncited_numeric_sentences", "uncited_numeric_claim", "high"),
        ("uncited_or_unapproved_dois", "unapproved_doi", "critical"),
        ("incomplete_paragraphs", "incomplete_paragraph", "high"),
    )
    for field, signature, severity in checks:
        values = validation.get(field) or []
        for index, value in enumerate(values):
            issues.append(
                ReviewIssue(
                    item_id=f"{dataset_id}:generation:{signature}:{index}",
                    dataset_id=dataset_id,
                    stage="generation",
                    issue_signature=signature,
                    message=f"{field}: {value}",
                    severity=severity,  # type: ignore[arg-type]
                    detector_score=1.0,
                    payload={"field": field, "value": value},
                )
            )
    return issues


def detect_data_quality_issues(
    dataset_id: str,
    snapshot: dict[str, Any],
    *,
    abstract_target: float = 0.70,
    abstract_hard_floor: float = 0.30,
    relevance_floor: float = 0.90,
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    abstract_coverage = float(snapshot.get("abstract_coverage") or 0.0)
    relevance = float(
        (snapshot.get("topic_relevance") or {}).get("all_terms_rate") or 0.0
    )
    if abstract_coverage < abstract_target:
        severity = "high" if abstract_coverage < abstract_hard_floor else "medium"
        issues.append(
            ReviewIssue(
                item_id=f"{dataset_id}:quality:abstract_coverage_below_target",
                dataset_id=dataset_id,
                stage="data_quality",
                issue_signature="abstract_coverage_below_target",
                message=(
                    f"Abstract coverage is {abstract_coverage:.3f}, below the "
                    f"registered target of {abstract_target:.3f}."
                ),
                severity=severity,
                detector_score=min(
                    1.0,
                    (abstract_target - abstract_coverage) / abstract_target,
                ),
                payload={
                    "abstract_coverage": abstract_coverage,
                    "target": abstract_target,
                    "hard_floor": abstract_hard_floor,
                    "topic_relevance": relevance,
                    "relevance_floor": relevance_floor,
                    "auto_accept_context_ok": relevance >= relevance_floor,
                },
            )
        )

    if relevance < relevance_floor:
        issues.append(
            ReviewIssue(
                item_id=f"{dataset_id}:quality:topic_relevance_below_floor",
                dataset_id=dataset_id,
                stage="data_quality",
                issue_signature="topic_relevance_below_floor",
                message=(
                    f"Core-term relevance is {relevance:.3f}, below the "
                    f"registered floor of {relevance_floor:.3f}."
                ),
                severity="critical",
                detector_score=min(1.0, (relevance_floor - relevance) / relevance_floor),
                payload={"all_terms_rate": relevance, "floor": relevance_floor},
            )
        )
    return issues


def detect_graph_claim_issues(
    dataset_id: str,
    items: list[dict[str, Any]],
) -> list[ReviewIssue]:
    return [
        ReviewIssue(
            item_id=item["item_id"],
            dataset_id=dataset_id,
            stage="graph_interpretation",
            issue_signature="structured_graph_claim_unverified",
            message=(
                "A structured graph claim requires calibration against its fact node, "
                "evidence path, and abstention rule."
            ),
            severity="low",
            detector_score=0.25,
            payload=item,
        )
        for item in items
    ]


def make_review_decision(
    issue: ReviewIssue,
    *,
    reviewer_code: str,
    decision: DecisionValue,
    original: Any,
    correction: Any = None,
    reason: str,
    review_seconds: float,
    feedback_memory_version: int,
) -> ReviewDecision:
    timestamp = datetime.now(UTC).isoformat()
    digest = hashlib.sha256(
        f"{timestamp}|{reviewer_code}|{issue.item_id}|{decision}".encode()
    ).hexdigest()[:16]
    return ReviewDecision(
        decision_id=f"R{digest}",
        timestamp=timestamp,
        reviewer_code=reviewer_code,
        dataset_id=issue.dataset_id,
        stage=issue.stage,
        item_id=issue.item_id,
        issue_signature=issue.issue_signature,
        detector_score=issue.detector_score,
        decision=decision,
        original=original,
        correction=correction,
        reason=reason,
        review_seconds=review_seconds,
        feedback_memory_version=feedback_memory_version,
    )


def summarize_review_program(
    actions: list[ReviewAction],
    audit_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = {
        action: sum(item.action == action for item in actions)
        for action in ("escalate", "audit", "auto_accept")
    }
    interventions = counts["escalate"] + counts["audit"]
    audits = audit_results or []
    audited_errors = sum(not bool(item.get("correct")) for item in audits)
    summary = {
        "items": len(actions),
        "counts": counts,
        "review_requests": interventions,
        "review_request_rate": interventions / len(actions) if actions else 0.0,
        "auto_accept_coverage": counts["auto_accept"] / len(actions) if actions else 0.0,
        "audited_auto_accept_items": len(audits),
        "audited_auto_accept_errors": audited_errors,
        "unsafe_auto_accept_rate": (
            audited_errors / len(audits) if audits else None
        ),
        "audited_auto_accept_precision": (
            1 - audited_errors / len(audits) if audits else None
        ),
    }
    # Compatibility aliases for development records created before Codex judges
    # replaced human annotators. Formal experiment reports use review_request_*.
    summary["human_interventions"] = summary["review_requests"]
    summary["human_intervention_rate"] = summary["review_request_rate"]
    return summary


def one_sided_clopper_pearson_lower(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials and trials >= 1")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1 - confidence, successes, trials - successes + 1))


def save_review_summary(summary: dict[str, Any], path: Path) -> None:
    write_json(path, summary)
