from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

Severity = Literal["low", "medium", "high", "critical"]
OversightRoute = Literal[
    "auto_accept",
    "single_review",
    "dual_review_with_adjudication",
    "blocked_insufficient_reviewers",
]


@dataclass(frozen=True)
class ReviewerObservation:
    observation_id: str
    reviewer_id: str
    dataset_id: str
    domain: str
    issue_type: str
    correct_after_adjudication: bool
    review_seconds: float


@dataclass(frozen=True)
class ReviewerCapability:
    reviewer_id: str
    domain: str
    issue_type: str
    evidence_tier: Literal["exact", "issue", "domain", "global", "prior"]
    observations: int
    independent_datasets: int
    posterior_mean_accuracy: float
    posterior_lower_95: float
    posterior_variance: float
    expected_review_seconds: float


@dataclass(frozen=True)
class OversightCase:
    case_id: str
    dataset_id: str
    domain: str
    issue_type: str
    severity: Severity
    predecision_error_risk: float
    estimated_default_review_seconds: float
    descendants: int = 0
    minimum_reviewers: int = 0


class ReviewerCapabilityModel:
    """Hierarchical Beta capability estimates without using future case labels."""

    def __init__(
        self,
        observations: list[ReviewerObservation],
        *,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        minimum_exact_observations: int = 3,
        minimum_issue_observations: int = 3,
        minimum_domain_observations: int = 5,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if any(row.review_seconds <= 0 for row in observations):
            raise ValueError("Reviewer observations require positive server time")
        ids = [row.observation_id for row in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate reviewer observation ID")
        self.observations = observations
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.minimum_exact_observations = minimum_exact_observations
        self.minimum_issue_observations = minimum_issue_observations
        self.minimum_domain_observations = minimum_domain_observations
        self._groups: dict[
            tuple[str, str, str], list[ReviewerObservation]
        ] = defaultdict(list)
        for row in observations:
            self._groups[(row.reviewer_id, row.domain, row.issue_type)].append(row)
            self._groups[(row.reviewer_id, "*", row.issue_type)].append(row)
            self._groups[(row.reviewer_id, row.domain, "*")].append(row)
            self._groups[(row.reviewer_id, "*", "*")].append(row)

    def reviewers(self) -> tuple[str, ...]:
        return tuple(sorted({row.reviewer_id for row in self.observations}))

    def estimate(
        self, reviewer_id: str, *, domain: str, issue_type: str
    ) -> ReviewerCapability:
        tiers: list[tuple[str, tuple[str, str, str], int]] = [
            (
                "exact",
                (reviewer_id, domain, issue_type),
                self.minimum_exact_observations,
            ),
            (
                "issue",
                (reviewer_id, "*", issue_type),
                self.minimum_issue_observations,
            ),
            (
                "domain",
                (reviewer_id, domain, "*"),
                self.minimum_domain_observations,
            ),
            ("global", (reviewer_id, "*", "*"), 1),
        ]
        selected: list[ReviewerObservation] = []
        evidence_tier = "prior"
        for tier, key, minimum in tiers:
            rows = self._groups.get(key, [])
            if len(rows) >= minimum:
                selected = rows
                evidence_tier = tier
                break
        successes = sum(row.correct_after_adjudication for row in selected)
        alpha = self.prior_alpha + successes
        beta = self.prior_beta + len(selected) - successes
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        lower = max(0.0, mean - 1.959963984540054 * math.sqrt(variance))
        seconds = (
            sum(row.review_seconds for row in selected) / len(selected)
            if selected
            else 60.0
        )
        return ReviewerCapability(
            reviewer_id=reviewer_id,
            domain=domain,
            issue_type=issue_type,
            evidence_tier=evidence_tier,  # type: ignore[arg-type]
            observations=len(selected),
            independent_datasets=len({row.dataset_id for row in selected}),
            posterior_mean_accuracy=mean,
            posterior_lower_95=lower,
            posterior_variance=variance,
            expected_review_seconds=seconds,
        )


class ComplementaryOversightRouter:
    """Jointly decides whether, who, and how many people should review."""

    HARM: ClassVar[dict[Severity, float]] = {
        "low": 1.0,
        "medium": 4.0,
        "high": 12.0,
        "critical": 40.0,
    }

    def __init__(
        self,
        capability_model: ReviewerCapabilityModel,
        *,
        reviewer_second_cost: float = 0.002,
        auto_accept_risk_ceiling: float = 0.05,
        minimum_capability_lower_95: float = 0.5,
    ) -> None:
        self.capability_model = capability_model
        self.reviewer_second_cost = reviewer_second_cost
        self.auto_accept_risk_ceiling = auto_accept_risk_ceiling
        self.minimum_capability_lower_95 = minimum_capability_lower_95

    @staticmethod
    def _decision_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def route(
        self, case: OversightCase, *, reviewer_ids: list[str] | None = None
    ) -> dict[str, Any]:
        if not 0.0 <= case.predecision_error_risk <= 1.0:
            raise ValueError("Predecision error risk must lie in [0, 1]")
        reviewer_ids = sorted(set(reviewer_ids or self.capability_model.reviewers()))
        capabilities = {
            reviewer_id: self.capability_model.estimate(
                reviewer_id, domain=case.domain, issue_type=case.issue_type
            )
            for reviewer_id in reviewer_ids
        }
        eligible = {
            reviewer_id: profile
            for reviewer_id, profile in capabilities.items()
            if profile.posterior_lower_95 >= self.minimum_capability_lower_95
        }
        impact = 1.0 + math.log1p(max(0, case.descendants))
        harm = self.HARM[case.severity] * impact
        candidates: list[dict[str, Any]] = []
        minimum_reviewers = max(
            case.minimum_reviewers, 2 if case.severity == "critical" else 0
        )
        if (
            minimum_reviewers == 0
            and case.predecision_error_risk <= self.auto_accept_risk_ceiling
        ):
            candidates.append(
                {
                    "route": "auto_accept",
                    "reviewer_ids": [],
                    "adjudicator_id": None,
                    "expected_final_accuracy": 1.0 - case.predecision_error_risk,
                    "expected_review_seconds": 0.0,
                    "expected_loss": harm * case.predecision_error_risk,
                }
            )
        if minimum_reviewers <= 1:
            for reviewer_id, profile in eligible.items():
                expected_loss = harm * (1.0 - profile.posterior_mean_accuracy)
                expected_loss += (
                    self.reviewer_second_cost * profile.expected_review_seconds
                )
                candidates.append(
                    {
                        "route": "single_review",
                        "reviewer_ids": [reviewer_id],
                        "adjudicator_id": None,
                        "expected_final_accuracy": profile.posterior_mean_accuracy,
                        "expected_review_seconds": profile.expected_review_seconds,
                        "expected_loss": expected_loss,
                    }
                )
        if len(eligible) >= 3:
            for first, second in itertools.combinations(sorted(eligible), 2):
                for adjudicator in sorted(set(eligible) - {first, second}):
                    p1 = eligible[first].posterior_mean_accuracy
                    p2 = eligible[second].posterior_mean_accuracy
                    pa = eligible[adjudicator].posterior_mean_accuracy
                    disagreement = p1 * (1.0 - p2) + (1.0 - p1) * p2
                    team_accuracy = p1 * p2 + disagreement * pa
                    seconds = (
                        eligible[first].expected_review_seconds
                        + eligible[second].expected_review_seconds
                        + disagreement
                        * eligible[adjudicator].expected_review_seconds
                    )
                    candidates.append(
                        {
                            "route": "dual_review_with_adjudication",
                            "reviewer_ids": [first, second],
                            "adjudicator_id": adjudicator,
                            "expected_disagreement_probability": disagreement,
                            "expected_final_accuracy": team_accuracy,
                            "expected_review_seconds": seconds,
                            "expected_loss": harm * (1.0 - team_accuracy)
                            + self.reviewer_second_cost * seconds,
                        }
                    )
        allowed = [
            candidate
            for candidate in candidates
            if len(candidate["reviewer_ids"]) >= minimum_reviewers
        ]
        if not allowed:
            decision: dict[str, Any] = {
                "case_id": case.case_id,
                "route": "blocked_insufficient_reviewers",
                "reviewer_ids": [],
                "adjudicator_id": None,
                "reason": "no_capability-qualified_team_satisfies_minimum_reviewers",
                "minimum_reviewers": minimum_reviewers,
            }
        else:
            selected = min(
                allowed,
                key=lambda row: (
                    row["expected_loss"],
                    row["expected_review_seconds"],
                    row["route"],
                    row["reviewer_ids"],
                    row.get("adjudicator_id") or "",
                ),
            )
            decision = {
                "case_id": case.case_id,
                **selected,
                "reason": "minimum_expected_harm_plus_review_cost",
                "minimum_reviewers": minimum_reviewers,
            }
        audit_payload = {
            "case": asdict(case),
            "decision": decision,
            "capabilities": {
                reviewer_id: asdict(profile)
                for reviewer_id, profile in sorted(capabilities.items())
            },
            "policy": {
                "reviewer_second_cost": self.reviewer_second_cost,
                "auto_accept_risk_ceiling": self.auto_accept_risk_ceiling,
                "minimum_capability_lower_95": self.minimum_capability_lower_95,
            },
        }
        return {
            **audit_payload,
            "predecision_sha256": self._decision_hash(audit_payload),
        }


def build_adversarial_review_packet(
    *,
    case_id: str,
    dataset_id: str,
    domain: str,
    claim: str,
    supporting_evidence: list[dict[str, Any]],
    challenging_evidence: list[dict[str, Any]],
    operator_trace: list[dict[str, Any]],
    alternative_explanations: list[str],
    forbidden_inferences: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blind the roles of pro/con evidence while retaining an internal audit map."""
    if not supporting_evidence or not challenging_evidence:
        raise ValueError("Adversarial packets require both support and challenge evidence")
    support_first = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0
    first = supporting_evidence if support_first else challenging_evidence
    second = challenging_evidence if support_first else supporting_evidence
    public = {
        "schema_version": 1,
        "packet_type": "adversarial_claim_audit",
        "case_id": case_id,
        "dataset_id": dataset_id,
        "domain": domain,
        "claim": claim,
        "evidence_set_a": first,
        "evidence_set_b": second,
        "operator_trace": operator_trace,
        "alternative_explanations": alternative_explanations,
        "forbidden_inferences": forbidden_inferences,
        "review_instructions": [
            "Audit both evidence sets independently before deciding.",
            "Identify the decisive evidence IDs and any invalid operator step.",
            "Choose supported, qualify, reject, or abstain.",
            "If qualification is needed, provide the smallest defensible rewrite.",
        ],
        "response_schema": {
            "verdict": ["supported", "qualify", "reject", "abstain"],
            "decisive_evidence_ids": "list[string]",
            "invalid_operator_steps": "list[int]",
            "failure_mode": "string",
            "minimal_rewrite": "string|null",
            "rationale": "string",
        },
    }
    public_sha256 = hashlib.sha256(
        json.dumps(
            public, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    internal = {
        "schema_version": 1,
        "case_id": case_id,
        "public_packet_canonical_sha256": public_sha256,
        "role_map": {
            "evidence_set_a": "support" if support_first else "challenge",
            "evidence_set_b": "challenge" if support_first else "support",
        },
        "blinded_fields": ["generation_condition", "gold_answer", "role_map"],
    }
    return public, internal
