from __future__ import annotations

import math
from collections import defaultdict
from random import Random
from typing import Any

from .oversight_factorial_analysis import ASSIGNMENT_ARMS, PACKET_ARMS
from .oversight_factorial_assignment import _allocate_arms


class OversightRandomizationError(ValueError):
    pass


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def _statistics(
    outcomes: dict[str, dict[str, Any]],
    arms: dict[str, tuple[str, str]],
) -> dict[str, float]:
    by_dataset: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case_id, row in outcomes.items():
        by_dataset[str(row["dataset_id"])][arms[case_id]].append(row)

    effects: dict[str, list[float]] = {"B1": [], "B2": [], "B3": []}
    required = {
        (packet, assignment)
        for packet in PACKET_ARMS
        for assignment in ASSIGNMENT_ARMS
    }
    for dataset_id, cells in sorted(by_dataset.items()):
        if set(cells) != required or any(not rows for rows in cells.values()):
            raise OversightRandomizationError(
                f"Randomized allocation is incomplete in dataset {dataset_id}"
            )
        metrics: dict[tuple[str, str], dict[str, float]] = {}
        for arm, rows in cells.items():
            correct = sum(bool(row["adjudicated_final_correct"]) for row in rows)
            minutes = sum(float(row["review_seconds"]) for row in rows) / 60.0
            if minutes <= 0:
                raise OversightRandomizationError("Review time must be positive")
            metrics[arm] = {
                "accuracy": correct / len(rows),
                "correct_per_minute": correct / minutes,
            }
        packet_effects = []
        for assignment in ASSIGNMENT_ARMS:
            packet_effects.append(
                metrics[("adversarial_two_sided", assignment)]["accuracy"]
                - metrics[("standard_claim_first", assignment)]["accuracy"]
            )
        routing_effects = []
        for packet in PACKET_ARMS:
            routing_effects.append(
                metrics[(packet, "capability_cost_router")]["correct_per_minute"]
                - metrics[(packet, "qualified_random")]["correct_per_minute"]
            )
        effects["B1"].append(sum(packet_effects) / len(packet_effects))
        effects["B2"].append(sum(routing_effects) / len(routing_effects))
        effects["B3"].append(packet_effects[1] - packet_effects[0])
    return {name: sum(values) / len(values) for name, values in effects.items()}


def _validate_inputs(
    outcome_records: list[dict[str, Any]],
    packet_records: list[dict[str, Any]],
    assignment_manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, str]]]:
    if len(outcome_records) != 96 or len(packet_records) != 96:
        raise OversightRandomizationError("Exactly 96 frozen held-out cases are required")
    outcomes = {str(row.get("case_id")): row for row in outcome_records}
    packets = {str(row.get("case_id")): row for row in packet_records}
    if len(outcomes) != 96 or len(packets) != 96 or set(outcomes) != set(packets):
        raise OversightRandomizationError("Outcome and packet case identities differ")
    routes = {
        str(row.get("case_id")): row
        for row in assignment_manifest.get("case_routes") or []
    }
    if (
        assignment_manifest.get("status") != "frozen_before_heldout_review"
        or assignment_manifest.get("heldout_outcomes_inspected") is not False
        or len(routes) != 96
        or set(routes) != set(outcomes)
    ):
        raise OversightRandomizationError("Assignment manifest is not the frozen panel")
    required_packet_fields = {"case_id", "dataset_id", "domain", "issue_type", "severity"}
    if any(not required_packet_fields <= set(row) for row in packet_records):
        raise OversightRandomizationError("Packet randomization strata are incomplete")
    required_outcome_fields = {
        "case_id",
        "dataset_id",
        "packet_arm",
        "assignment_arm",
        "adjudicated_final_correct",
        "review_seconds",
    }
    for case_id, row in outcomes.items():
        if not required_outcome_fields <= set(row):
            raise OversightRandomizationError("Outcome record is incomplete")
        if row["dataset_id"] != packets[case_id]["dataset_id"]:
            raise OversightRandomizationError("Outcome dataset differs from frozen packet")
        seconds = row["review_seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0
        ):
            raise OversightRandomizationError("Review time must be positive and finite")
        if not isinstance(row["adjudicated_final_correct"], bool):
            raise OversightRandomizationError("Final correctness must be boolean")

    observed = {
        case_id: (str(route["packet_arm"]), str(route["assignment_arm"]))
        for case_id, route in routes.items()
    }
    outcome_arms = {
        case_id: (str(row["packet_arm"]), str(row["assignment_arm"]))
        for case_id, row in outcomes.items()
    }
    if observed != outcome_arms:
        raise OversightRandomizationError("Outcome arms differ from frozen assignment")
    seed = assignment_manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OversightRandomizationError("Frozen assignment seed is invalid")
    reproduced = _allocate_arms(packet_records, seed=seed)
    if reproduced != observed:
        raise OversightRandomizationError(
            "Frozen arm allocation cannot be reproduced from packet strata and seed"
        )
    return outcomes, observed


def analyze_oversight_randomization(
    outcome_records: list[dict[str, Any]],
    packet_records: list[dict[str, Any]],
    assignment_manifest: dict[str, Any],
    *,
    permutations: int = 50_000,
    permutation_seed: int = 20260907,
) -> dict[str, Any]:
    """Fisher-style randomization inference for the frozen 2x2 benchmark panel.

    This analysis targets the finite population of 96 randomized cases. It does not
    replace the dataset-cluster analysis used for cross-topic transportability.
    """
    if permutations < 999:
        raise OversightRandomizationError("At least 999 random allocations are required")
    outcomes, observed_arms = _validate_inputs(
        outcome_records, packet_records, assignment_manifest
    )
    observed = _statistics(outcomes, observed_arms)
    generator = Random(permutation_seed)
    extreme = {"B1": 0, "B2": 0, "B3": 0}
    for _ in range(permutations):
        allocation_seed = generator.randrange(0, 2**63)
        permuted_arms = _allocate_arms(packet_records, seed=allocation_seed)
        draw = _statistics(outcomes, permuted_arms)
        extreme["B1"] += draw["B1"] >= observed["B1"] - 1e-12
        extreme["B2"] += draw["B2"] >= observed["B2"] - 1e-12
        extreme["B3"] += abs(draw["B3"]) >= abs(observed["B3"]) - 1e-12
    raw = {name: (count + 1) / (permutations + 1) for name, count in extreme.items()}
    adjusted = _holm(raw)
    alternatives = {"B1": "greater", "B2": "greater", "B3": "two_sided"}
    contrasts = {}
    for name in ("B1", "B2", "B3"):
        p_value = raw[name]
        contrasts[name] = {
            "estimate": observed[name],
            "alternative": alternatives[name],
            "randomization_p_value": p_value,
            "monte_carlo_standard_error": math.sqrt(
                p_value * (1.0 - p_value) / (permutations + 1)
            ),
            "holm_adjusted_p_value": adjusted[name],
            "reject_at_0_05": adjusted[name] < 0.05,
        }
    return {
        "schema_version": 1,
        "status": "finite_benchmark_randomization_analysis_complete",
        "cases": len(outcomes),
        "datasets": len({str(row["dataset_id"]) for row in outcome_records}),
        "permutations": permutations,
        "permutation_seed": permutation_seed,
        "minimum_monte_carlo_p": 1.0 / (permutations + 1),
        "registered_contrasts": contrasts,
        "estimands": {
            "B1": "packet-format accuracy effect in the 96 frozen cases",
            "B2": "routing-policy correct-outcomes-per-minute effect in the 96 frozen cases",
            "B3": "packet-by-routing accuracy interaction in the 96 frozen cases",
        },
        "sharp_null": (
            "No case-level effect of the assigned packet or routing policy on final "
            "correctness and total server-accounted review time"
        ),
        "multiplicity": "Holm across B1, B2, and B3",
        "scope_guard": (
            "These randomization tests support causal claims only for the frozen "
            "96-case benchmark population. Cross-topic transportability requires the "
            "separate equal-dataset O1/O2/O3 analysis and cannot be rescued here."
        ),
    }
