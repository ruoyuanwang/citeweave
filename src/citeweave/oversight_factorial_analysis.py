from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

PACKET_ARMS = ("standard_claim_first", "adversarial_two_sided")
ASSIGNMENT_ARMS = ("qualified_random", "capability_cost_router")


@dataclass(frozen=True)
class OversightFactorialOutcome:
    case_id: str
    dataset_id: str
    packet_arm: Literal["standard_claim_first", "adversarial_two_sided"]
    assignment_arm: Literal["qualified_random", "capability_cost_router"]
    adjudicated_final_correct: bool
    review_seconds: float


def _exact_signflip(values: list[float], *, two_sided: bool) -> dict[str, Any]:
    if not values:
        raise ValueError("Dataset sign-flip requires at least one effect")
    observed = sum(values) / len(values)
    draws = [
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    if two_sided:
        extreme = sum(abs(value) >= abs(observed) - 1e-12 for value in draws)
        key = "p_value_two_sided"
    else:
        extreme = sum(value >= observed - 1e-12 for value in draws)
        key = "p_value_one_sided"
    return {
        "estimate": observed,
        "datasets": len(values),
        key: extreme / len(draws),
        "minimum_attainable_p": 1.0 / len(draws),
    }


def _bootstrap(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    generator = Random(seed)
    draws = [
        sum(generator.choices(values, k=len(values))) / len(values)
        for _ in range(samples)
    ]
    draws.sort()
    return {
        "estimate": sum(values) / len(values),
        "ci_low": draws[int(0.025 * (samples - 1))],
        "ci_high": draws[int(0.975 * (samples - 1))],
        "datasets": len(values),
        "cluster_weighting": "equal_dataset",
    }


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def analyze_oversight_factorial(
    outcomes: list[OversightFactorialOutcome],
    *,
    minimum_datasets: int = 8,
    minimum_cases_per_dataset: int = 12,
    minimum_cases_per_arm: int = 3,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260831,
) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("Complementary-oversight panel is empty")
    if len({row.case_id for row in outcomes}) != len(outcomes):
        raise ValueError("Each held-out case must have exactly one final outcome")
    for row in outcomes:
        if row.packet_arm not in PACKET_ARMS:
            raise ValueError(f"Unknown packet arm: {row.packet_arm}")
        if row.assignment_arm not in ASSIGNMENT_ARMS:
            raise ValueError(f"Unknown assignment arm: {row.assignment_arm}")
        if row.review_seconds <= 0:
            raise ValueError("Review time must be positive and server timed")

    grouped: dict[str, list[OversightFactorialOutcome]] = defaultdict(list)
    for row in outcomes:
        grouped[row.dataset_id].append(row)
    if len(grouped) < minimum_datasets:
        raise ValueError(
            f"Confirmatory analysis requires at least {minimum_datasets} datasets"
        )

    required_arms = {
        (packet, assignment)
        for packet in PACKET_ARMS
        for assignment in ASSIGNMENT_ARMS
    }
    dataset_effects: dict[str, dict[str, float]] = {}
    arm_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_id, rows in sorted(grouped.items()):
        if len(rows) < minimum_cases_per_dataset:
            raise ValueError(
                f"Dataset {dataset_id} has fewer than {minimum_cases_per_dataset} cases"
            )
        by_arm: dict[tuple[str, str], list[OversightFactorialOutcome]] = defaultdict(
            list
        )
        for row in rows:
            by_arm[(row.packet_arm, row.assignment_arm)].append(row)
        if set(by_arm) != required_arms:
            raise ValueError(f"Dataset {dataset_id} does not contain all four arms")
        if any(len(by_arm[arm]) < minimum_cases_per_arm for arm in required_arms):
            raise ValueError(
                f"Dataset {dataset_id} has fewer than {minimum_cases_per_arm} cases "
                "in at least one factorial arm"
            )

        metrics: dict[tuple[str, str], dict[str, float]] = {}
        for arm, arm_rows in by_arm.items():
            correct = sum(row.adjudicated_final_correct for row in arm_rows)
            seconds = sum(row.review_seconds for row in arm_rows)
            metrics[arm] = {
                "cases": float(len(arm_rows)),
                "accuracy": correct / len(arm_rows),
                "mean_review_seconds": seconds / len(arm_rows),
                "correct_outcomes_per_review_minute": correct / (seconds / 60.0),
            }

        standard_accuracy = sum(
            metrics[("standard_claim_first", assignment)]["accuracy"]
            for assignment in ASSIGNMENT_ARMS
        ) / 2.0
        adversarial_accuracy = sum(
            metrics[("adversarial_two_sided", assignment)]["accuracy"]
            for assignment in ASSIGNMENT_ARMS
        ) / 2.0
        random_efficiency = sum(
            metrics[(packet, "qualified_random")][
                "correct_outcomes_per_review_minute"
            ]
            for packet in PACKET_ARMS
        ) / 2.0
        routed_efficiency = sum(
            metrics[(packet, "capability_cost_router")][
                "correct_outcomes_per_review_minute"
            ]
            for packet in PACKET_ARMS
        ) / 2.0
        routed_packet_effect = (
            metrics[("adversarial_two_sided", "capability_cost_router")][
                "accuracy"
            ]
            - metrics[("standard_claim_first", "capability_cost_router")][
                "accuracy"
            ]
        )
        random_packet_effect = (
            metrics[("adversarial_two_sided", "qualified_random")]["accuracy"]
            - metrics[("standard_claim_first", "qualified_random")]["accuracy"]
        )
        dataset_effects[dataset_id] = {
            "O1_packet_accuracy": adversarial_accuracy - standard_accuracy,
            "O2_routing_efficiency": routed_efficiency - random_efficiency,
            "O3_accuracy_interaction": routed_packet_effect - random_packet_effect,
        }
        arm_summaries[dataset_id] = {
            f"{packet}__{assignment}": metrics[(packet, assignment)]
            for packet, assignment in sorted(required_arms)
        }

    o1 = [effect["O1_packet_accuracy"] for effect in dataset_effects.values()]
    o2 = [effect["O2_routing_efficiency"] for effect in dataset_effects.values()]
    o3 = [effect["O3_accuracy_interaction"] for effect in dataset_effects.values()]
    exact = {
        "O1": _exact_signflip(o1, two_sided=False),
        "O2": _exact_signflip(o2, two_sided=False),
        "O3": _exact_signflip(o3, two_sided=True),
    }
    raw_p = {
        "O1": exact["O1"]["p_value_one_sided"],
        "O2": exact["O2"]["p_value_one_sided"],
        "O3": exact["O3"]["p_value_two_sided"],
    }
    adjusted = _holm(raw_p)
    values = {"O1": o1, "O2": o2, "O3": o3}
    contrasts = {}
    for offset, name in enumerate(("O1", "O2", "O3")):
        contrasts[name] = {
            "exact": exact[name],
            "equal_dataset_bootstrap": _bootstrap(
                values[name],
                samples=bootstrap_samples,
                seed=bootstrap_seed + offset,
            ),
            "holm_adjusted_p": adjusted[name],
            "confirmatory_passed": adjusted[name] < 0.05,
        }

    return {
        "schema_version": 1,
        "status": "analyzed",
        "cases": len(outcomes),
        "datasets": len(grouped),
        "minimum_cases_per_dataset": minimum_cases_per_dataset,
        "minimum_cases_per_arm": minimum_cases_per_arm,
        "dataset_weighting": "equal_dataset",
        "dataset_effects": dataset_effects,
        "dataset_arm_summaries": arm_summaries,
        "registered_contrasts": contrasts,
        "interpretation_guard": (
            "Cases and repeated judgments within one scientific dataset increase "
            "within-dataset precision but are not independent topic replications. "
            "Mixed-effects case/reviewer models, if reported, are diagnostic only."
        ),
    }
