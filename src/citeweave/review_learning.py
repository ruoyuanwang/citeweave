from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

ReviewAction = Literal[
    "accept",
    "rewrite",
    "reject_claim",
    "reject_evidence",
    "reject_operator",
    "reject_plan",
    "abstain",
]


@dataclass(frozen=True)
class ReviewTarget:
    target_id: str
    dataset_id: str
    issue_type: str
    severity: Literal["low", "medium", "high", "critical"]
    features: dict[str, str | int | float | bool]
    estimated_review_seconds: float
    descendants: int = 0


@dataclass(frozen=True)
class StructuredFeedback:
    feedback_id: str
    reviewer_id: str
    dataset_id: str
    target_id: str
    target_type: Literal["claim", "evidence", "operator", "plan"]
    issue_type: str
    action: ReviewAction
    rationale: str
    replacement: str | None = None
    guard: dict[str, Any] = field(default_factory=dict)
    review_seconds: float = 0.0


@dataclass(frozen=True)
class FeedbackLearningSignal:
    """Component-specific supervision compiled from one typed human judgment."""

    signal_id: str
    feedback_id: str
    dataset_id: str
    target_id: str
    component: Literal["retriever", "planner", "generator", "risk_router"]
    reward: float
    supervision: Literal["pointwise", "pairwise_preference", "escalation"]
    preferred_payload: str | None
    issue_type: str


@dataclass(frozen=True)
class PolicyReplayObservation:
    """Held-out paired replay result for a candidate review policy."""

    case_id: str
    dataset_id: str
    severity: Literal["low", "medium", "high", "critical"]
    baseline_correct: bool
    candidate_correct: bool
    baseline_review_seconds: float
    candidate_review_seconds: float
    candidate_route: Literal["human_review", "auto_accept", "auto_correct"]
    unrelated: bool = False


def compile_feedback_learning_signals(
    feedback: list[StructuredFeedback],
) -> list[FeedbackLearningSignal]:
    """Credit typed feedback to the component that can act on it.

    The compiler deliberately avoids blaming retrieval for a rejected claim or
    the generator for rejected evidence. Ambiguous feedback becomes an
    escalation signal instead of weak supervision for every component.
    """
    component_by_target = {
        "evidence": "retriever",
        "operator": "planner",
        "plan": "planner",
        "claim": "generator",
    }
    signals: list[FeedbackLearningSignal] = []
    for item in feedback:
        component = component_by_target[item.target_type]
        if item.action == "accept":
            reward = 1.0
            supervision = "pointwise"
            preferred = None
        elif item.action == "rewrite":
            if item.replacement is None:
                raise ValueError("Rewrite feedback requires replacement text")
            reward = 1.0
            supervision = "pairwise_preference"
            preferred = item.replacement
        elif item.action == "abstain":
            component = "risk_router"
            reward = 1.0
            supervision = "escalation"
            preferred = None
        else:
            expected_target = {
                "reject_claim": "claim",
                "reject_evidence": "evidence",
                "reject_operator": "operator",
                "reject_plan": "plan",
            }[item.action]
            if item.target_type != expected_target:
                component = "risk_router"
                reward = 1.0
                supervision = "escalation"
            else:
                reward = -1.0
                supervision = "pointwise"
            preferred = None
        digest = hashlib.sha256(
            repr(
                [
                    item.feedback_id,
                    item.target_id,
                    component,
                    reward,
                    supervision,
                    preferred,
                ]
            ).encode()
        ).hexdigest()[:16]
        signals.append(
            FeedbackLearningSignal(
                signal_id=f"FS{digest}",
                feedback_id=item.feedback_id,
                dataset_id=item.dataset_id,
                target_id=item.target_id,
                component=component,  # type: ignore[arg-type]
                reward=reward,
                supervision=supervision,  # type: ignore[arg-type]
                preferred_payload=preferred,
                issue_type=item.issue_type,
            )
        )
    return signals


def _wilson_upper(errors: int, total: int, *, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    rate = errors / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return min(1.0, (center + radius) / denominator)


def audit_policy_promotion(
    observations: list[PolicyReplayObservation],
    *,
    minimum_datasets: int = 2,
    quality_noninferiority_margin: float = 0.0,
    maximum_unsafe_autoaccept_upper: float = 0.05,
    maximum_unrelated_regression_rate: float = 0.05,
) -> dict[str, Any]:
    """Decide promotion using paired held-out quality, safety, and labor gates."""
    if not observations:
        return {"decision": "hold", "reason": "no_held_out_replay", "cases": 0}
    datasets = sorted({row.dataset_id for row in observations})
    baseline_accuracy = sum(row.baseline_correct for row in observations) / len(observations)
    candidate_accuracy = sum(row.candidate_correct for row in observations) / len(observations)
    baseline_seconds = sum(row.baseline_review_seconds for row in observations)
    candidate_seconds = sum(row.candidate_review_seconds for row in observations)
    autoaccept = [
        row for row in observations if row.candidate_route == "auto_accept"
    ]
    high_risk_autoaccept = [
        row
        for row in autoaccept
        if row.severity in {"high", "critical"}
    ]
    unsafe_autoaccepts = sum(not row.candidate_correct for row in autoaccept)
    unsafe_upper = _wilson_upper(unsafe_autoaccepts, len(autoaccept))
    critical_automation = sum(
        row.severity == "critical" and row.candidate_route != "human_review"
        for row in observations
    )
    unrelated = [row for row in observations if row.unrelated]
    unrelated_regressions = sum(
        row.baseline_correct and not row.candidate_correct for row in unrelated
    )
    unrelated_rate = unrelated_regressions / len(unrelated) if unrelated else 1.0
    checks = {
        "independent_dataset_gate": len(datasets) >= minimum_datasets,
        "quality_noninferiority_gate": (
            candidate_accuracy - baseline_accuracy >= -quality_noninferiority_margin
        ),
        "review_time_saving_gate": candidate_seconds < baseline_seconds,
        "unsafe_autoaccept_gate": unsafe_upper <= maximum_unsafe_autoaccept_upper,
        "critical_never_automated_gate": critical_automation == 0,
        "unrelated_regression_gate": (
            unrelated_rate <= maximum_unrelated_regression_rate
        ),
    }
    return {
        "decision": "promote" if all(checks.values()) else "hold",
        "cases": len(observations),
        "datasets": datasets,
        "checks": checks,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_difference": candidate_accuracy - baseline_accuracy,
        "baseline_review_seconds": baseline_seconds,
        "candidate_review_seconds": candidate_seconds,
        "review_seconds_saved": baseline_seconds - candidate_seconds,
        "autoaccept_cases": len(autoaccept),
        "high_risk_autoaccept_cases": len(high_risk_autoaccept),
        "unsafe_autoaccepts": unsafe_autoaccepts,
        "unsafe_autoaccept_wilson_upper_95": unsafe_upper,
        "critical_automation_cases": critical_automation,
        "unrelated_cases": len(unrelated),
        "unrelated_regressions": unrelated_regressions,
        "unrelated_regression_rate": unrelated_rate,
    }


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    issue_type: str
    target_type: str
    guard: dict[str, Any]
    action: ReviewAction
    replacement_template: str | None
    supporting_datasets: tuple[str, ...]
    supporting_reviewers: tuple[str, ...]
    support: int
    conflicts: int
    support_gate_passed: bool
    validation_examples: int
    validation_datasets: tuple[str, ...]
    transfer_precision: float | None
    unrelated_regression_rate: float | None
    enabled: bool


@dataclass(frozen=True)
class RuleValidationCase:
    case_id: str
    dataset_id: str
    issue_type: str
    target_type: Literal["claim", "evidence", "operator", "plan"]
    features: dict[str, str | int | float | bool]
    expected_action: ReviewAction
    semantic_neighbor: bool


@dataclass
class DependencyNode:
    node_id: str
    node_type: Literal["source", "evidence", "operator", "claim", "paragraph"]
    content: Any
    status: Literal["valid", "needs_review", "invalid", "rewritten"] = "valid"
    reason: str | None = None


class ReviewDependencyGraph:
    """Claim/evidence dependency graph with deterministic review propagation."""

    def __init__(self) -> None:
        self.nodes: dict[str, DependencyNode] = {}
        self.children: dict[str, set[str]] = defaultdict(set)
        self.parents: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: DependencyNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate dependency node: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_dependency(self, parent_id: str, child_id: str) -> None:
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise KeyError("Both dependency endpoints must exist")
        self.children[parent_id].add(child_id)
        self.parents[child_id].add(parent_id)

    def descendants(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            for child in sorted(self.children[current]):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return seen

    def apply_feedback(self, feedback: StructuredFeedback) -> dict[str, Any]:
        if feedback.target_id not in self.nodes:
            raise KeyError(feedback.target_id)
        target = self.nodes[feedback.target_id]
        changed: list[str] = []
        if feedback.action in {
            "reject_claim",
            "reject_evidence",
            "reject_operator",
            "reject_plan",
            "abstain",
        }:
            target.status = "invalid"
            target.reason = feedback.rationale
            changed.append(target.node_id)
            for node_id in sorted(self.descendants(target.node_id)):
                node = self.nodes[node_id]
                if node.status != "invalid":
                    node.status = "needs_review"
                    node.reason = f"Upstream dependency {target.node_id} was invalidated."
                    changed.append(node_id)
        elif feedback.action == "rewrite":
            if feedback.replacement is None:
                raise ValueError("Rewrite feedback requires replacement text")
            target.content = feedback.replacement
            target.status = "rewritten"
            target.reason = feedback.rationale
            changed.append(target.node_id)
            for node_id in sorted(self.descendants(target.node_id)):
                node = self.nodes[node_id]
                node.status = "needs_review"
                node.reason = f"Upstream dependency {target.node_id} changed."
                changed.append(node_id)
        return {
            "feedback_id": feedback.feedback_id,
            "changed_nodes": changed,
            "review_queue": [
                node_id
                for node_id in changed
                if self.nodes[node_id].status == "needs_review"
            ],
        }


def _canonical_guard(guard: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in guard.items()))


class FeedbackCompiler:
    """Compiles reviewer actions into guarded, executable policy rules.

    A rule is enabled only after it transfers across independent datasets and
    reviewers and has no contradictory action under the same guard.
    """

    def __init__(
        self,
        *,
        minimum_datasets: int = 2,
        minimum_reviewers: int = 2,
        minimum_validation_cases: int = 2,
        minimum_validation_datasets: int = 2,
        transfer_precision_threshold: float = 0.8,
        maximum_unrelated_regression_rate: float = 0.05,
    ):
        self.minimum_datasets = minimum_datasets
        self.minimum_reviewers = minimum_reviewers
        self.minimum_validation_cases = minimum_validation_cases
        self.minimum_validation_datasets = minimum_validation_datasets
        self.transfer_precision_threshold = transfer_precision_threshold
        self.maximum_unrelated_regression_rate = maximum_unrelated_regression_rate

    def compile(
        self,
        feedback: list[StructuredFeedback],
        *,
        validation_cases: list[RuleValidationCase] | None = None,
    ) -> list[CompiledRule]:
        groups: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], list[StructuredFeedback]
        ] = defaultdict(list)
        for item in feedback:
            groups[(item.issue_type, item.target_type, _canonical_guard(item.guard))].append(item)
        rules: list[CompiledRule] = []
        for (issue_type, target_type, _), examples in sorted(groups.items()):
            action_counts: dict[str, int] = defaultdict(int)
            for example in examples:
                action_counts[example.action] += 1
            action = min(action_counts, key=lambda value: (-action_counts[value], value))
            matching = [example for example in examples if example.action == action]
            conflicts = len(examples) - len(matching)
            datasets = tuple(sorted({example.dataset_id for example in matching}))
            reviewers = tuple(sorted({example.reviewer_id for example in matching}))
            replacements = {
                example.replacement for example in matching if example.replacement is not None
            }
            replacement = next(iter(replacements)) if len(replacements) == 1 else None
            guard = dict(matching[0].guard)
            digest = hashlib.sha256(
                repr([issue_type, target_type, _canonical_guard(guard), action]).encode()
            ).hexdigest()[:16]
            support_gate_passed = (
                conflicts == 0
                and len(datasets) >= self.minimum_datasets
                and len(reviewers) >= self.minimum_reviewers
                and (action != "rewrite" or replacement is not None)
            )
            cases = [
                case
                for case in (validation_cases or [])
                if case.issue_type == issue_type and case.target_type == target_type
            ]
            neighbors = [
                case
                for case in cases
                if case.semantic_neighbor
                and all(case.features.get(key) == value for key, value in guard.items())
            ]
            validation_datasets = tuple(sorted({case.dataset_id for case in neighbors}))
            transfer_precision = (
                sum(case.expected_action == action for case in neighbors) / len(neighbors)
                if neighbors
                else None
            )
            unrelated = [case for case in cases if not case.semantic_neighbor]
            unrelated_applications = [
                case
                for case in unrelated
                if all(case.features.get(key) == value for key, value in guard.items())
            ]
            unrelated_regressions = sum(
                case.expected_action != action for case in unrelated_applications
            )
            unrelated_regression_rate = (
                unrelated_regressions / len(unrelated)
                if unrelated
                else None
            )
            validation_gate_passed = (
                len(neighbors) >= self.minimum_validation_cases
                and len(validation_datasets) >= self.minimum_validation_datasets
                and transfer_precision is not None
                and transfer_precision >= self.transfer_precision_threshold
                and unrelated_regression_rate is not None
                and unrelated_regression_rate <= self.maximum_unrelated_regression_rate
            )
            enabled = support_gate_passed and validation_gate_passed
            rules.append(
                CompiledRule(
                    rule_id=f"CR{digest}",
                    issue_type=issue_type,
                    target_type=target_type,
                    guard=guard,
                    action=action,  # type: ignore[arg-type]
                    replacement_template=replacement,
                    supporting_datasets=datasets,
                    supporting_reviewers=reviewers,
                    support=len(matching),
                    conflicts=conflicts,
                    support_gate_passed=support_gate_passed,
                    validation_examples=len(neighbors),
                    validation_datasets=validation_datasets,
                    transfer_precision=transfer_precision,
                    unrelated_regression_rate=unrelated_regression_rate,
                    enabled=enabled,
                )
            )
        return rules

    @staticmethod
    def applicable(rule: CompiledRule, target: ReviewTarget) -> bool:
        if not rule.enabled or rule.issue_type != target.issue_type:
            return False
        return all(target.features.get(key) == value for key, value in rule.guard.items())


class GuardedReviewPolicy:
    """Turns independently validated rules into auditable routing decisions."""

    def route(self, target: ReviewTarget, rules: list[CompiledRule]) -> dict[str, Any]:
        matching = [rule for rule in rules if FeedbackCompiler.applicable(rule, target)]
        provenance = [rule.rule_id for rule in matching]
        if target.severity == "critical":
            return {
                "target_id": target.target_id,
                "route": "human_review",
                "action": None,
                "rule_ids": provenance,
                "reason": "critical_severity",
            }
        actions = {rule.action for rule in matching}
        if not matching:
            return {
                "target_id": target.target_id,
                "route": "human_review",
                "action": None,
                "rule_ids": [],
                "reason": "no_validated_rule",
            }
        if len(actions) != 1:
            return {
                "target_id": target.target_id,
                "route": "human_review",
                "action": None,
                "rule_ids": provenance,
                "reason": "rule_conflict",
            }
        action = next(iter(actions))
        replacements = {
            rule.replacement_template
            for rule in matching
            if rule.replacement_template is not None
        }
        if action == "rewrite" and len(replacements) != 1:
            return {
                "target_id": target.target_id,
                "route": "human_review",
                "action": None,
                "rule_ids": provenance,
                "reason": "rewrite_template_ambiguous",
            }
        return {
            "target_id": target.target_id,
            "route": "auto_accept" if action == "accept" else "auto_correct",
            "action": action,
            "replacement": next(iter(replacements)) if replacements else None,
            "rule_ids": provenance,
            "reason": "validated_guard_match",
        }


class ActiveReviewController:
    """Cost-sensitive active review with posterior risk and feedback novelty."""

    HARM: ClassVar[dict[str, float]] = {
        "low": 1.0,
        "medium": 3.0,
        "high": 8.0,
        "critical": 20.0,
    }

    def __init__(
        self,
        feedback: list[StructuredFeedback],
        *,
        reviewer_second_cost: float = 0.03,
        information_value: float = 1.0,
        escalation_threshold: float = 1.0,
    ) -> None:
        self.feedback = feedback
        self.reviewer_second_cost = reviewer_second_cost
        self.information_value = information_value
        self.escalation_threshold = escalation_threshold

    @staticmethod
    def _signature(target: ReviewTarget) -> tuple[str, tuple[tuple[str, str], ...]]:
        return target.issue_type, _canonical_guard(target.features)

    def posterior(self, target: ReviewTarget) -> dict[str, float | int]:
        exact = [
            item
            for item in self.feedback
            if item.issue_type == target.issue_type
            and all(target.features.get(key) == value for key, value in item.guard.items())
        ]
        errors = sum(item.action != "accept" for item in exact)
        accepts = len(exact) - errors
        alpha = 1.0 + errors
        beta = 1.0 + accepts
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return {
            "examples": len(exact),
            "errors": errors,
            "error_probability": mean,
            "posterior_variance": variance,
        }

    def priority(self, target: ReviewTarget) -> dict[str, Any]:
        posterior = self.posterior(target)
        impact = 1.0 + math_log1p(target.descendants)
        expected_harm = (
            float(posterior["error_probability"]) * self.HARM[target.severity] * impact
        )
        novelty = self.information_value * float(posterior["posterior_variance"]) ** 0.5
        review_cost = target.estimated_review_seconds * self.reviewer_second_cost
        utility = expected_harm + novelty - review_cost
        mandatory = target.severity == "critical"
        return {
            "target_id": target.target_id,
            "review": mandatory or utility >= self.escalation_threshold,
            "priority": utility,
            "expected_harm": expected_harm,
            "information_gain_proxy": novelty,
            "review_cost": review_cost,
            "posterior": posterior,
            "mandatory": mandatory,
        }

    def select(self, targets: list[ReviewTarget], *, budget_seconds: float) -> dict[str, Any]:
        ranked = sorted(
            ((target, self.priority(target)) for target in targets),
            key=lambda pair: (-pair[1]["priority"], pair[0].target_id),
        )
        selected: list[dict[str, Any]] = []
        spent = 0.0
        for target, score in ranked:
            if not score["review"]:
                continue
            if spent + target.estimated_review_seconds > budget_seconds and not score["mandatory"]:
                continue
            selected.append({"target": asdict(target), "decision": score})
            spent += target.estimated_review_seconds
        return {
            "budget_seconds": budget_seconds,
            "selected": selected,
            "selected_count": len(selected),
            "estimated_seconds": spent,
            "coverage": len(selected) / len(targets) if targets else 0.0,
        }


def math_log1p(value: int) -> float:
    # Kept local to make the controller's impact transform explicit and testable.
    import math

    return math.log1p(max(0, value))
