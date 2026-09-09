from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .oversight_design_sensitivity import simulate_packet_accuracy_power


class ArticleSensitivityError(ValueError):
    pass


def simulate_standardized_topic_power(
    *,
    topics: int,
    standardized_effects: tuple[float, ...],
    simulations: int = 10_000,
    seed: int = 20260908,
    family_alpha: float = 0.05,
    family_tests: int = 3,
) -> dict[str, Any]:
    """Power of the prospective exact topic sign-flip test under Normal effects."""
    if not 2 <= topics <= 12:
        raise ArticleSensitivityError("Exact simulation supports 2 to 12 topics")
    if simulations < 1_000:
        raise ArticleSensitivityError("At least 1000 simulations are required")
    if any(effect <= 0 for effect in standardized_effects):
        raise ArticleSensitivityError("Standardized effects must be positive")
    if family_tests < 1 or not 0 < family_alpha < 1:
        raise ArticleSensitivityError("Multiplicity parameters are invalid")

    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=topics)), dtype=np.float32
    )
    alpha = family_alpha / family_tests
    rng = np.random.default_rng(seed)
    batch_size = max(20, min(500, 1_000_000 // len(signs)))
    results = []
    for standardized_effect in standardized_effects:
        rejected = 0
        for start in range(0, simulations, batch_size):
            size = min(batch_size, simulations - start)
            effects = rng.normal(
                standardized_effect, 1.0, size=(size, topics)
            ).astype(np.float32)
            observed = effects.mean(axis=1)
            null = effects @ signs.T / topics
            p_values = (null >= observed[:, None] - 1e-7).mean(axis=1)
            rejected += int((p_values <= alpha).sum())
        power = rejected / simulations
        results.append(
            {
                "standardized_topic_effect": standardized_effect,
                "power": power,
                "monte_carlo_standard_error": math.sqrt(
                    power * (1.0 - power) / simulations
                ),
            }
        )
    return {
        "topics": topics,
        "simulations": simulations,
        "seed": seed,
        "conservative_holm_threshold": alpha,
        "results": results,
        "assumptions": [
            "independent Normal topic effects with unit between-topic standard deviation",
            "exact one-sided topic sign-flip test",
            "alpha divided by three as a conservative Holm planning threshold",
            "no article, expert rating, human outcome, or machine effect is read",
        ],
    }


def build_article_sensitivity_report(
    *, simulations: int = 10_000, seed: int = 20260908
) -> dict[str, Any]:
    a1 = simulate_packet_accuracy_power(
        datasets=8,
        cases_per_factor_level_per_dataset=20,
        baseline_accuracy=0.60,
        risk_differences=(0.10, 0.15, 0.20, 0.25, 0.30),
        simulations=simulations,
        seed=seed,
    )
    a1["semantic_role"] = (
        "A1 graph-review minus one-shot strict correct-and-supported claim rate"
    )
    holistic = [
        simulate_standardized_topic_power(
            topics=topics,
            standardized_effects=(0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
            simulations=simulations,
            seed=seed + topics,
        )
        for topics in (8, 12)
    ]
    return {
        "schema_version": 1,
        "status": "prospective_article_design_sensitivity_no_outcomes_read",
        "registered_topics": 8,
        "a1_binary_claim_sensitivity": a1,
        "a2_a3_standardized_topic_sensitivity": holistic,
        "scope_guard": (
            "Multiple claim ratings and evaluators improve within-topic measurement but "
            "do not increase the number of independent scientific topics. Failure to "
            "reject is not evidence of equivalence unless the registered noninferiority "
            "test itself passes."
        ),
    }
