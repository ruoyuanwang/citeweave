from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


class OversightSensitivityError(ValueError):
    pass


def simulate_packet_accuracy_power(
    *,
    datasets: int,
    cases_per_factor_level_per_dataset: int,
    baseline_accuracy: float,
    risk_differences: tuple[float, ...],
    simulations: int = 10_000,
    seed: int = 20260907,
    family_alpha: float = 0.05,
    family_tests: int = 3,
) -> dict[str, Any]:
    """Prospective power for the dataset sign-flip packet-accuracy contrast.

    The simulation is deliberately conservative: it uses alpha/m for the smallest
    Holm p-value and independent Bernoulli case outcomes within each dataset. It is a
    design diagnostic, not an analysis of observed human outcomes.
    """
    if not 2 <= datasets <= 12:
        raise OversightSensitivityError("Exact simulation supports 2 to 12 datasets")
    if cases_per_factor_level_per_dataset < 2:
        raise OversightSensitivityError("At least two cases per factor level are required")
    if simulations < 1_000:
        raise OversightSensitivityError("At least 1000 simulations are required")
    if not 0 < baseline_accuracy < 1:
        raise OversightSensitivityError("Baseline accuracy must be inside (0, 1)")
    if family_tests < 1 or not 0 < family_alpha < 1:
        raise OversightSensitivityError("Multiplicity parameters are invalid")
    if any(effect <= 0 or baseline_accuracy + effect >= 1 for effect in risk_differences):
        raise OversightSensitivityError("Risk differences must be positive and feasible")

    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=datasets)), dtype=np.float32
    )
    alpha = family_alpha / family_tests
    rng = np.random.default_rng(seed)
    batch_size = max(20, min(500, 1_000_000 // len(signs)))
    rows = []
    for risk_difference in risk_differences:
        rejected = 0
        for start in range(0, simulations, batch_size):
            size = min(batch_size, simulations - start)
            treated = rng.binomial(
                cases_per_factor_level_per_dataset,
                baseline_accuracy + risk_difference,
                size=(size, datasets),
            ) / cases_per_factor_level_per_dataset
            control = rng.binomial(
                cases_per_factor_level_per_dataset,
                baseline_accuracy,
                size=(size, datasets),
            ) / cases_per_factor_level_per_dataset
            effects = (treated - control).astype(np.float32)
            observed = effects.mean(axis=1)
            null = effects @ signs.T / datasets
            p_values = (null >= observed[:, None] - 1e-7).mean(axis=1)
            rejected += int((p_values <= alpha).sum())
        power = rejected / simulations
        rows.append(
            {
                "risk_difference": risk_difference,
                "power": power,
                "monte_carlo_standard_error": math.sqrt(
                    power * (1.0 - power) / simulations
                ),
            }
        )
    return {
        "datasets": datasets,
        "cases_per_factor_level_per_dataset": cases_per_factor_level_per_dataset,
        "total_factor_level_cases": datasets * cases_per_factor_level_per_dataset,
        "baseline_accuracy": baseline_accuracy,
        "simulations": simulations,
        "seed": seed,
        "conservative_holm_threshold": alpha,
        "results": rows,
        "assumptions": [
            "independent Bernoulli cases within each dataset",
            "equal cases in packet factor levels after averaging assignment arms",
            "exact one-sided dataset sign-flip test",
            "alpha divided by three as a conservative Holm planning threshold",
            "no human or machine outcome file is read",
        ],
    }


def build_oversight_sensitivity_grid(
    *, simulations: int = 10_000, seed: int = 20260907
) -> dict[str, Any]:
    designs = ((8, 6), (8, 12), (12, 6), (12, 12))
    results = []
    for offset, (datasets, cases) in enumerate(designs):
        results.append(
            simulate_packet_accuracy_power(
                datasets=datasets,
                cases_per_factor_level_per_dataset=cases,
                baseline_accuracy=0.60,
                risk_differences=(0.10, 0.15, 0.20, 0.25, 0.30),
                simulations=simulations,
                seed=seed + offset,
            )
        )
    return {
        "schema_version": 1,
        "status": "prospective_design_sensitivity_no_outcomes_read",
        "current_design": {"datasets": 8, "cases_per_factor_level_per_dataset": 6},
        "designs": results,
        "scope_guard": (
            "Power is conditional on planning assumptions and does not predict the "
            "observed effect. The 96-case randomization analysis and the dataset-level "
            "transportability analysis answer different questions."
        ),
    }
