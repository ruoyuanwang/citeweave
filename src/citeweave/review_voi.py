from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from .article_claim_review import select_article_review_claims

SEVERITY_WEIGHT = {"medium": 1.0, "high": 2.0, "critical": 4.0}
FIXED_POLICIES = {"uniform_hash", "static_risk", "static_dependency"}


def _synthesis_pairs(writer_input: dict[str, Any]) -> dict[str, frozenset[str]]:
    by_type = {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }
    return {
        "SYN-CONNECTIVITY-REDUNDANCY": frozenset(
            {by_type["multi_hop_connector"], by_type["bridge_counterfactual"]}
        ),
        "SYN-CENTRALITY-RESILIENCE": frozenset(
            {by_type["community_role_contrast"], by_type["hub_removal_resilience"]}
        ),
        "SYN-TEMPORAL-STRUCTURAL": frozenset(
            {by_type["temporal_structural_shift"], by_type["community_role_contrast"]}
        ),
    }


def build_claim_dependency_features(
    article: str,
    *,
    article_id: str,
    dataset_id: str,
    writer_input: dict[str, Any],
    claims: int = 20,
    seed: int = 20260827,
) -> list[dict[str, Any]]:
    phenomena = {
        row["phenomenon_id"]: row for row in writer_input["graph_phenomena"]
    }
    selected = select_article_review_claims(
        article,
        phenomenon_ids=sorted(phenomena),
        claims=claims,
        seed=seed,
    )
    synthesis_pairs = _synthesis_pairs(writer_input)
    rows = []
    for claim in selected:
        dependencies = set(claim["evidence_tokens"])
        for phenomenon_id in claim["phenomenon_ids"]:
            phenomenon = phenomena[phenomenon_id]
            dependencies.update(phenomenon["reference_ids"])
            dependencies.update(phenomenon["graph_evidence_ids"])
        claim_phenomena = set(claim["phenomenon_ids"])
        syntheses = sorted(
            synthesis_id
            for synthesis_id, pair in synthesis_pairs.items()
            if pair.issubset(claim_phenomena)
        )
        dependencies.update(syntheses)
        risk = claim["risk_features"]
        risk_score = (
            4 * int(risk["causal_language"])
            + 2 * int(risk["numeric"])
            + 2 * int(risk["multiple_phenomena"])
            + int(risk["discussion_interpretation"])
        )
        severity = (
            "critical"
            if risk["causal_language"] and risk["multiple_phenomena"]
            else "high"
            if risk_score >= 3
            else "medium"
        )
        packet_id = "VOI-" + hashlib.sha256(
            f"{seed}\x1f{article_id}\x1f{claim['candidate_id']}\x1f{claim['text']}".encode()
        ).hexdigest()[:16].upper()
        rows.append(
            {
                "packet_id": packet_id,
                "routing_scope_id": article_id,
                "dataset_id": dataset_id,
                "candidate_id": claim["candidate_id"],
                "paragraph_id": claim["paragraph_id"],
                "section": claim["section"],
                "claim_sha256": hashlib.sha256(claim["text"].encode()).hexdigest(),
                "phenomenon_ids": claim["phenomenon_ids"],
                "synthesis_ids": syntheses,
                "dependency_ids": sorted(dependencies),
                "risk_features": risk,
                "risk_score": risk_score,
                "severity": severity,
            }
        )
    dependency_claims: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for dependency_id in row["dependency_ids"]:
            dependency_claims[dependency_id].add(row["packet_id"])
    for row in rows:
        row["dependency_fanout"] = max(
            len(dependency_claims[value]) for value in row["dependency_ids"]
        )
        row["reachable_claim_ids"] = sorted(
            {
                packet_id
                for dependency_id in row["dependency_ids"]
                for packet_id in dependency_claims[dependency_id]
            }
        )
    return rows


def _claim_prior(row: dict[str, Any]) -> float:
    risk = row["risk_features"]
    probability = (
        0.06
        + 0.24 * int(risk.get("causal_language", False))
        + 0.12 * int(risk.get("numeric", False))
        + 0.18 * int(risk.get("multiple_phenomena", False))
        + 0.10 * int(risk.get("discussion_interpretation", False))
    )
    return min(0.85, probability)


def replay_sequential_dependency_voi(
    features: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    estimated_review_seconds: dict[str, float],
    budget_seconds: float,
    seed: int = 20260827,
) -> dict[str, Any]:
    if budget_seconds <= 0 or not math.isfinite(budget_seconds):
        raise ValueError("VOI replay requires a positive finite budget")
    feature_index = {row["packet_id"]: row for row in features}
    outcome_index = {row["packet_id"]: row for row in outcomes}
    if len(feature_index) != len(features) or set(feature_index) != set(outcome_index):
        raise ValueError("VOI features and hidden outcomes must be complete and unique")
    scopes = {row.get("routing_scope_id") for row in features}
    if len(scopes) != 1 or None in scopes:
        raise ValueError("VOI replay requires exactly one article routing scope")
    if set(estimated_review_seconds) != set(feature_index) or any(
        not math.isfinite(value) or value <= 0
        for value in estimated_review_seconds.values()
    ):
        raise ValueError("Every packet requires a positive pre-outcome time estimate")
    dependency_claims: dict[str, set[str]] = defaultdict(set)
    for row in features:
        for dependency_id in row["dependency_ids"]:
            dependency_claims[dependency_id].add(row["packet_id"])
    posterior = {dependency_id: [1.0, 4.0] for dependency_id in dependency_claims}
    reviewed: set[str] = set()
    propagated: set[str] = set()
    invalid_dependencies: set[str] = set()
    predicted_seconds = 0.0
    actual_seconds = 0.0
    trace = []

    def score(packet_id: str) -> tuple[float, str]:
        row = feature_index[packet_id]
        unresolved = set(feature_index) - reviewed - propagated
        expected_propagation = 0.0
        for dependency_id in row["dependency_ids"]:
            alpha, beta = posterior[dependency_id]
            new_reach = len(dependency_claims[dependency_id] & unresolved)
            expected_propagation += (alpha / (alpha + beta)) * new_reach
        utility = SEVERITY_WEIGHT[row["severity"]] * (
            _claim_prior(row) + expected_propagation
        )
        ratio = utility / estimated_review_seconds[packet_id]
        tie = hashlib.sha256(f"{seed}\x1f{packet_id}".encode()).hexdigest()
        return ratio, tie

    while True:
        candidates = sorted(set(feature_index) - reviewed - propagated)
        affordable = [
            packet_id
            for packet_id in candidates
            if predicted_seconds + estimated_review_seconds[packet_id] <= budget_seconds
        ]
        if not affordable:
            break
        selected = max(affordable, key=score)
        selection_score = score(selected)[0]
        row = feature_index[selected]
        outcome = outcome_index[selected]
        allowed = set(row["dependency_ids"])
        invalid = set(outcome.get("invalid_dependency_ids") or [])
        if not invalid <= allowed:
            raise ValueError(f"Outcome cites an invisible dependency: {selected}")
        action = outcome.get("action")
        if action not in {"accept", "rewrite", "reject_claim", "abstain"}:
            raise ValueError(f"Invalid review action: {selected}")
        review_seconds = float(outcome.get("review_seconds", 0))
        if not math.isfinite(review_seconds) or review_seconds <= 0:
            raise ValueError(f"Invalid actual review time: {selected}")
        reviewed.add(selected)
        predicted_seconds += estimated_review_seconds[selected]
        actual_seconds += review_seconds
        newly_propagated: set[str] = set()
        for dependency_id in allowed:
            alpha, beta = posterior[dependency_id]
            if dependency_id in invalid:
                posterior[dependency_id] = [alpha + 1, beta]
                invalid_dependencies.add(dependency_id)
                newly_propagated.update(dependency_claims[dependency_id] - reviewed)
            elif outcome.get("evidence_sufficient") is True:
                posterior[dependency_id] = [alpha, beta + 1]
        newly_propagated -= propagated
        propagated.update(newly_propagated)
        trace.append(
            {
                "step": len(trace) + 1,
                "selected_packet_id": selected,
                "score_before_reveal": selection_score,
                "estimated_review_seconds": estimated_review_seconds[selected],
                "actual_review_seconds": review_seconds,
                "action": action,
                "invalid_dependency_ids": sorted(invalid),
                "newly_propagated_claim_ids": sorted(newly_propagated),
            }
        )
    actionable = {
        packet_id
        for packet_id, outcome in outcome_index.items()
        if outcome["action"] in {"rewrite", "reject_claim"}
    }
    captured = actionable & (reviewed | propagated)
    overpropagated = propagated - actionable
    return {
        "schema_version": 1,
        "status": "sequential_dependency_voi_replayed",
        "budget_seconds": budget_seconds,
        "predicted_seconds_scheduled": predicted_seconds,
        "actual_review_seconds": actual_seconds,
        "reviewed_claims": len(reviewed),
        "propagated_claims": len(propagated),
        "invalid_dependencies_found": sorted(invalid_dependencies),
        "actionable_claims": len(actionable),
        "captured_actionable_claims": len(captured),
        "actionable_capture_rate": len(captured) / len(actionable) if actionable else None,
        "unresolved_actionable_claims": sorted(actionable - captured),
        "overpropagated_nonactionable_claims": sorted(overpropagated),
        "trace": trace,
        "interpretation_guard": (
            "This is sequential policy replay using independently collected complete "
            "labels. It does not by itself prove causal labor savings in live deployment."
        ),
    }


def replay_fixed_policy_with_propagation(
    features: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    policy: str,
    estimated_review_seconds: dict[str, float],
    budget_seconds: float,
    seed: int = 20260827,
) -> dict[str, Any]:
    if policy not in FIXED_POLICIES:
        raise ValueError(f"Unsupported fixed review policy: {policy}")
    if budget_seconds <= 0 or not math.isfinite(budget_seconds):
        raise ValueError("Fixed-policy replay requires a positive finite budget")
    feature_index = {row["packet_id"]: row for row in features}
    outcome_index = {row["packet_id"]: row for row in outcomes}
    if len(feature_index) != len(features) or set(feature_index) != set(outcome_index):
        raise ValueError("Fixed-policy features and outcomes must be complete and unique")
    scopes = {row.get("routing_scope_id") for row in features}
    if len(scopes) != 1 or None in scopes:
        raise ValueError("Fixed-policy replay requires exactly one article routing scope")
    if set(estimated_review_seconds) != set(feature_index) or any(
        not math.isfinite(value) or value <= 0
        for value in estimated_review_seconds.values()
    ):
        raise ValueError("Every packet requires a positive pre-outcome time estimate")
    dependency_claims: dict[str, set[str]] = defaultdict(set)
    for row in features:
        for dependency_id in row["dependency_ids"]:
            dependency_claims[dependency_id].add(row["packet_id"])

    def fixed_score(packet_id: str) -> tuple[float, str]:
        row = feature_index[packet_id]
        if policy == "uniform_hash":
            value = 0.0
        elif policy == "static_risk":
            value = SEVERITY_WEIGHT[row["severity"]] * _claim_prior(row)
        else:
            reach = len(set(row.get("reachable_claim_ids") or [packet_id]))
            value = SEVERITY_WEIGHT[row["severity"]] * (
                _claim_prior(row) + 0.2 * reach
            )
        tie = hashlib.sha256(f"{seed}\x1f{policy}\x1f{packet_id}".encode()).hexdigest()
        return value / estimated_review_seconds[packet_id], tie

    order = sorted(feature_index, key=fixed_score, reverse=True)
    reviewed: set[str] = set()
    propagated: set[str] = set()
    invalid_dependencies: set[str] = set()
    predicted_seconds = 0.0
    actual_seconds = 0.0
    trace = []
    for selected in order:
        if selected in propagated:
            continue
        estimated = estimated_review_seconds[selected]
        if predicted_seconds + estimated > budget_seconds:
            continue
        row = feature_index[selected]
        outcome = outcome_index[selected]
        allowed = set(row["dependency_ids"])
        invalid = set(outcome.get("invalid_dependency_ids") or [])
        if not invalid <= allowed:
            raise ValueError(f"Outcome cites an invisible dependency: {selected}")
        action = outcome.get("action")
        if action not in {"accept", "rewrite", "reject_claim", "abstain"}:
            raise ValueError(f"Invalid review action: {selected}")
        review_seconds = float(outcome.get("review_seconds", 0))
        if not math.isfinite(review_seconds) or review_seconds <= 0:
            raise ValueError(f"Invalid actual review time: {selected}")
        reviewed.add(selected)
        predicted_seconds += estimated
        actual_seconds += review_seconds
        newly_propagated: set[str] = set()
        for dependency_id in invalid:
            invalid_dependencies.add(dependency_id)
            newly_propagated.update(dependency_claims[dependency_id] - reviewed)
        newly_propagated -= propagated
        propagated.update(newly_propagated)
        trace.append(
            {
                "step": len(trace) + 1,
                "selected_packet_id": selected,
                "fixed_score": fixed_score(selected)[0],
                "estimated_review_seconds": estimated,
                "actual_review_seconds": review_seconds,
                "action": action,
                "invalid_dependency_ids": sorted(invalid),
                "newly_propagated_claim_ids": sorted(newly_propagated),
            }
        )
    actionable = {
        packet_id
        for packet_id, outcome in outcome_index.items()
        if outcome["action"] in {"rewrite", "reject_claim"}
    }
    captured = actionable & (reviewed | propagated)
    return {
        "schema_version": 1,
        "status": "fixed_policy_with_dependency_propagation_replayed",
        "policy": policy,
        "budget_seconds": budget_seconds,
        "predicted_seconds_scheduled": predicted_seconds,
        "actual_review_seconds": actual_seconds,
        "reviewed_claims": len(reviewed),
        "propagated_claims": len(propagated),
        "invalid_dependencies_found": sorted(invalid_dependencies),
        "actionable_claims": len(actionable),
        "captured_actionable_claims": len(captured),
        "actionable_capture_rate": len(captured) / len(actionable) if actionable else None,
        "unresolved_actionable_claims": sorted(actionable - captured),
        "overpropagated_nonactionable_claims": sorted(propagated - actionable),
        "trace": trace,
    }
