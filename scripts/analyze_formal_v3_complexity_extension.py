from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import yaml
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm

from citeweave.io import read_json, sha256_file, write_json


def _outcome(record: dict[str, Any]) -> dict[str, float]:
    complete = record.get("status", "complete") == "complete"
    score = record.get("score") or {}
    return {
        "correct": float(complete and bool(score.get("answer_exact"))),
        "evidence_f1": float(score.get("evidence_f1") or 0.0) if complete else 0.0,
    }


def _signflip(values: list[float]) -> dict[str, Any]:
    observed = sum(values) / len(values)
    draws = [
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return {
        "estimate": observed,
        "clusters": len(values),
        "cluster_effects": values,
        "p_value_one_sided": sum(value >= observed - 1e-12 for value in draws)
        / len(draws),
        "minimum_attainable_p": 1.0 / len(draws),
    }


def _cluster_bootstrap(
    dataset_effects: dict[str, float], *, samples: int, seed: int
) -> dict[str, float]:
    datasets = sorted(dataset_effects)
    generator = Random(seed)
    draws = [
        sum(dataset_effects[key] for key in generator.choices(datasets, k=len(datasets)))
        / len(datasets)
        for _ in range(samples)
    ]
    draws.sort()
    return {
        "estimate": sum(dataset_effects.values()) / len(dataset_effects),
        "ci_low": draws[round(0.025 * (samples - 1))],
        "ci_high": draws[round(0.975 * (samples - 1))],
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_values[key]))
        adjusted[key] = running
    return {
        key: {
            "raw_p_value_one_sided": p_values[key],
            "holm_adjusted_p_value": adjusted[key],
            "reject_at_0_05": adjusted[key] < 0.05,
        }
        for key in p_values
    }


def _cluster_tost(values: list[float], *, margin: float) -> dict[str, Any]:
    lower = _signflip([value + margin for value in values])
    upper = _signflip([margin - value for value in values])
    return {
        "margin": margin,
        "lower_test": {
            "null": f"effect <= {-margin}",
            **lower,
        },
        "upper_test": {
            "null": f"effect >= {margin}",
            **upper,
        },
        "equivalent_at_0_05": (
            lower["p_value_one_sided"] < 0.05
            and upper["p_value_one_sided"] < 0.05
        ),
    }


def _fit_focal_mixed_logistic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_names = (
        "intercept",
        "graph_program",
        "complex",
        "scale",
        "graph_program_x_complex",
        "graph_program_x_scale",
        "complex_x_scale",
        "graph_program_x_complex_x_scale",
    )
    scale_code = {"small": -0.5, "medium": 0.0, "large": 0.5}
    design = []
    outcomes = []
    datasets = []
    for row in rows:
        for band in ("simple", "complex"):
            complex_value = float(band == "complex")
            scale_value = scale_code[row["scale"]]
            for condition in ("flat_neural_dense", "graph_program"):
                graph_value = float(condition == "graph_program")
                design.append(
                    [
                        1.0,
                        graph_value,
                        complex_value,
                        scale_value,
                        graph_value * complex_value,
                        graph_value * scale_value,
                        complex_value * scale_value,
                        graph_value * complex_value * scale_value,
                    ]
                )
                outcomes.append(row[f"{band}_{condition}"]["correct"])
                datasets.append(row["dataset_id"])
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    dataset_names = sorted(set(datasets))
    group_indices = [
        np.asarray([index for index, value in enumerate(datasets) if value == dataset])
        for dataset in dataset_names
    ]
    nodes, weights = hermgauss(30)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)

    def objective(parameters: np.ndarray) -> float:
        beta = parameters[:-1]
        sigma = np.exp(parameters[-1])
        total = 0.0
        for indices in group_indices:
            eta = x[indices] @ beta
            random_intercepts = np.sqrt(2.0) * sigma * nodes
            linear = eta[:, None] + random_intercepts[None, :]
            log_likelihood = np.sum(
                y[indices, None] * -np.logaddexp(0.0, -linear)
                + (1.0 - y[indices, None]) * -np.logaddexp(0.0, linear),
                axis=0,
            )
            total += logsumexp(log_weights + log_likelihood)
        return float(-total)

    candidates = []
    for sigma_start in (0.2, 0.5, 1.0):
        initial = np.zeros(len(parameter_names) + 1, dtype=np.float64)
        initial[-1] = np.log(sigma_start)
        fitted = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0)] * len(parameter_names) + [(-5.0, 3.0)],
            options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        candidates.append(fitted)
    result = min(candidates, key=lambda fitted: float(fitted.fun))
    parameters = np.asarray(result.x, dtype=np.float64)
    beta = parameters[:-1]
    try:
        covariance = np.asarray(result.hess_inv.todense(), dtype=np.float64)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance)[:-1], 0.0))
    except (AttributeError, ValueError, np.linalg.LinAlgError):
        covariance = np.full((len(parameters), len(parameters)), np.nan)
        standard_errors = np.full(len(parameter_names), np.nan)
    estimates = {}
    for index, name in enumerate(parameter_names):
        standard_error = float(standard_errors[index])
        z_value = float(beta[index] / standard_error) if standard_error > 0 else None
        estimates[name] = {
            "estimate_log_odds": float(beta[index]),
            "standard_error": standard_error if np.isfinite(standard_error) else None,
            "z_value": z_value,
            "p_value_two_sided": (
                float(2.0 * norm.sf(abs(z_value))) if z_value is not None else None
            ),
        }
    standardized = {}
    for scale, scale_value in scale_code.items():
        probabilities = {}
        for band in ("simple", "complex"):
            complex_value = float(band == "complex")
            for condition in ("flat_neural_dense", "graph_program"):
                graph_value = float(condition == "graph_program")
                vector = np.asarray(
                    [
                        1.0,
                        graph_value,
                        complex_value,
                        scale_value,
                        graph_value * complex_value,
                        graph_value * scale_value,
                        complex_value * scale_value,
                        graph_value * complex_value * scale_value,
                    ]
                )
                probabilities[f"{band}_{condition}"] = float(expit(vector @ beta))
        standardized[scale] = {
            "conditional_probability_at_random_intercept_zero": probabilities,
            "complexity_selectivity_probability_difference": (
                probabilities["complex_graph_program"]
                - probabilities["complex_flat_neural_dense"]
                - probabilities["simple_graph_program"]
                + probabilities["simple_flat_neural_dense"]
            ),
        }
    bounded = bool(
        np.any(np.isclose(np.abs(beta), 20.0, atol=1e-4))
        or np.isclose(parameters[-1], -5.0, atol=1e-4)
        or np.isclose(parameters[-1], 3.0, atol=1e-4)
    )
    return {
        "model": "binomial_logit_random_dataset_intercept",
        "estimation": "30_point_gauss_hermite_marginal_maximum_likelihood",
        "population": "48 registered matched anchors x simple/complex x graph_program/flat_neural_dense",
        "observations": len(y),
        "datasets": dataset_names,
        "scale_coding": scale_code,
        "converged": bool(result.success),
        "boundary_solution": bounded,
        "optimizer_message": str(result.message),
        "negative_log_likelihood": float(result.fun),
        "random_intercept_standard_deviation": float(np.exp(parameters[-1])),
        "fixed_effects": estimates,
        "registered_coefficient_mapping": {
            "C1_at_medium_scale": "graph_program_x_complex",
            "C2_large_minus_small": "graph_program_x_complex_x_scale",
        },
        "standardized_effects": standardized,
        "inference_guard": (
            "With eight dataset clusters, exact cluster tests are confirmatory. Wald "
            "statistics are model diagnostics and boundary/separation must be disclosed."
        ),
    }


def analyze_complexity_factorial(
    tasks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260824,
) -> dict[str, Any]:
    conditions = (
        "flat_hybrid",
        "flat_neural_dense",
        "graph_hierarchical_retrieval_v2",
        "graph_program",
    )
    indexed = {(row["item_id"], row["condition"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate complexity-extension item-condition record")
    expected = {
        (task["item_id"], condition) for task in tasks for condition in conditions
    }
    if set(indexed) != expected:
        raise ValueError(
            f"Incomplete complexity-extension panel: missing={len(expected - set(indexed))}, "
            f"unexpected={len(set(indexed) - expected)}"
        )
    task_index = {task["item_id"]: task for task in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("Duplicate task IDs in complexity-extension benchmarks")

    rows = []
    for simple in (task for task in tasks if task["complexity"] == 1):
        parent_id = simple.get("matched_complex_item_id")
        parent = task_index.get(parent_id)
        if parent is None:
            raise ValueError(f"Matched complex parent missing: {simple['item_id']}")
        if (simple["dataset_id"], simple["scale"]) != (
            parent["dataset_id"],
            parent["scale"],
        ):
            raise ValueError(f"Matched pair invariant mismatch: {simple['item_id']}")
        row = {
            "dataset_id": simple["dataset_id"],
            "scale": simple["scale"],
            "simple_item_id": simple["item_id"],
            "complex_item_id": parent_id,
            "pair_type": f"{simple['task_type']}__{parent['task_type']}",
        }
        for condition in conditions:
            row[f"simple_{condition}"] = _outcome(indexed[(simple["item_id"], condition)])
            row[f"complex_{condition}"] = _outcome(indexed[(parent_id, condition)])
        rows.append(row)
    if len(rows) != 48:
        raise ValueError("Registered extension requires 48 dataset-scale-anchor pairs")

    def advantage(row: dict[str, Any], band: str, graph_condition: str) -> float:
        return (
            row[f"{band}_{graph_condition}"]["correct"]
            - row[f"{band}_flat_neural_dense"]["correct"]
        )

    c1_by_dataset: dict[str, list[float]] = defaultdict(list)
    simple_by_dataset: dict[str, list[float]] = defaultdict(list)
    hierarchy_c1_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset_id"]
        c1_by_dataset[dataset].append(
            advantage(row, "complex", "graph_program")
            - advantage(row, "simple", "graph_program")
        )
        simple_by_dataset[dataset].append(advantage(row, "simple", "graph_program"))
        hierarchy_c1_by_dataset[dataset].append(
            advantage(row, "complex", "graph_hierarchical_retrieval_v2")
            - advantage(row, "simple", "graph_hierarchical_retrieval_v2")
        )
    c1_effects = {
        key: sum(values) / len(values) for key, values in sorted(c1_by_dataset.items())
    }
    simple_effects = {
        key: sum(values) / len(values)
        for key, values in sorted(simple_by_dataset.items())
    }
    hierarchy_effects = {
        key: sum(values) / len(values)
        for key, values in sorted(hierarchy_c1_by_dataset.items())
    }
    c2_effects = {}
    for dataset in sorted(c1_by_dataset):
        large = [
            advantage(row, "complex", "graph_program")
            - advantage(row, "simple", "graph_program")
            for row in rows
            if row["dataset_id"] == dataset and row["scale"] == "large"
        ]
        small = [
            advantage(row, "complex", "graph_program")
            - advantage(row, "simple", "graph_program")
            for row in rows
            if row["dataset_id"] == dataset and row["scale"] == "small"
        ]
        c2_effects[dataset] = sum(large) / len(large) - sum(small) / len(small)

    def summary(effects: dict[str, float], offset: int) -> dict[str, Any]:
        return {
            "dataset_effects": effects,
            "cluster_bootstrap": _cluster_bootstrap(
                effects, samples=bootstrap_samples, seed=seed + offset
            ),
            "exact_cluster_signflip": _signflip(list(effects.values())),
        }

    c1_summary = summary(c1_effects, 1)
    c2_summary = summary(c2_effects, 2)
    holm = _holm_adjust(
        {
            "C1": c1_summary["exact_cluster_signflip"]["p_value_one_sided"],
            "C2": c2_summary["exact_cluster_signflip"]["p_value_one_sided"],
        }
    )
    mixed_model = _fit_focal_mixed_logistic(rows)
    c3_summary = summary(simple_effects, 3)
    c3_tost = _cluster_tost(list(simple_effects.values()), margin=0.05)

    def clustered_condition_effect(
        *,
        left: str,
        right: str,
        include: Any,
    ) -> tuple[dict[str, float], dict[str, int]]:
        values: dict[str, list[float]] = defaultdict(list)
        for task in tasks:
            if not include(task):
                continue
            dataset = task["dataset_id"]
            left_outcome = _outcome(indexed[(task["item_id"], left)])["correct"]
            right_outcome = _outcome(indexed[(task["item_id"], right)])["correct"]
            values[dataset].append(left_outcome - right_outcome)
        effects = {
            dataset: sum(dataset_values) / len(dataset_values)
            for dataset, dataset_values in sorted(values.items())
        }
        counts = {
            dataset: len(dataset_values)
            for dataset, dataset_values in sorted(values.items())
        }
        if len(effects) != 8 or any(count == 0 for count in counts.values()):
            raise ValueError("Cluster validation requires all eight datasets")
        return effects, counts

    e1_effects, e1_counts = clustered_condition_effect(
        left="graph_program",
        right="flat_hybrid",
        include=lambda task: 3 <= int(task["complexity"]) <= 5,
    )
    global_counterfactual_types = {
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    e2_effects, e2_counts = clustered_condition_effect(
        left="graph_hierarchical_retrieval_v2",
        right="flat_hybrid",
        include=lambda task: task["task_type"] in global_counterfactual_types,
    )
    e1_summary = summary(e1_effects, 5)
    e2_summary = summary(e2_effects, 6)
    validation_holm = _holm_adjust(
        {
            "E1": e1_summary["exact_cluster_signflip"]["p_value_one_sided"],
            "E2": e2_summary["exact_cluster_signflip"]["p_value_one_sided"],
        }
    )
    return {
        "schema_version": 1,
        "status": "confirmatory_analysis_complete",
        "datasets": len(c1_effects),
        "matched_pairs": len(rows),
        "registered_contrasts": {
            "C1_complexity_selectivity": {**c1_summary, "multiplicity": holm["C1"]},
            "C2_scale_amplification_of_complexity_selectivity": {
                **c2_summary,
                "multiplicity": holm["C2"],
            },
        },
        "mixed_logistic": mixed_model,
        "secondary": {
            "C3_graph_program_advantage_on_simple_tasks": {
                **c3_summary,
                "cluster_tost": c3_tost,
            },
            "C4_hierarchical_complexity_selectivity": summary(hierarchy_effects, 4),
        },
        "cluster_validity_checks": {
            "purpose": (
                "Eight-dataset confirmatory protection against treating tasks within "
                "a dataset as independent inferential replicates."
            ),
            "E1_graph_program_over_flat_hybrid_complex_3_to_5": {
                **e1_summary,
                "task_counts_by_dataset": e1_counts,
                "multiplicity": validation_holm["E1"],
            },
            "E2_hierarchical_over_flat_hybrid_global_counterfactual": {
                **e2_summary,
                "task_counts_by_dataset": e2_counts,
                "task_types": sorted(global_counterfactual_types),
                "multiplicity": validation_holm["E2"],
            },
        },
        "interpretation_guard": (
            "C1 and C2 require their Holm-adjusted exact cluster tests. C1 without C2 "
            "does not establish scale amplification. Simple-task equivalence requires both "
            "cluster TOST p-values below 0.05. E1/E2 are an eight-dataset validation "
            "family and require their Holm-adjusted exact cluster tests; item-level "
            "McNemar results do not replace them. A boundary GLMM fit must be disclosed "
            "and does not replace exact dataset-cluster inference."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis-amendment",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/"
            "formal_v3_complexity_analysis_amendment_008_cluster_validity.yml"
        ),
    )
    parser.add_argument(
        "--analysis-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/graph_discovery_v2/"
            "formal_v3_complexity_analysis_amendment_008_cluster_validity_freeze.json"
        ),
    )
    args = parser.parse_args()

    protocol_hash = sha256_file(args.protocol)
    if read_json(args.freeze).get("sha256") != protocol_hash:
        raise SystemExit("Complexity-extension protocol differs from freeze artifact")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    analysis_amendment_hash = sha256_file(args.analysis_amendment)
    analysis_amendment = yaml.safe_load(
        args.analysis_amendment.read_text(encoding="utf-8")
    )
    if read_json(args.analysis_amendment_freeze).get(
        "sha256"
    ) != analysis_amendment_hash:
        raise SystemExit("Complexity analysis amendment differs from freeze artifact")
    if analysis_amendment.get("base_protocol_sha256") != protocol_hash:
        raise SystemExit("Complexity analysis amendment protocol hash mismatch")
    implementation = analysis_amendment.get("implementation") or {}
    if implementation.get("sha256") != sha256_file(Path(__file__)):
        raise SystemExit("Complexity analysis implementation hash mismatch")
    expected_conditions = set(protocol["factorial_design"]["conditions"])
    construction = read_json(args.benchmark_root / "construction_manifest.json")
    if construction.get("protocol_sha256") != protocol_hash:
        raise SystemExit("Complexity-extension construction protocol hash mismatch")
    tasks = []
    records = []
    for row in construction["records"]:
        dataset_id = row["dataset_id"]
        benchmark_path = args.benchmark_root / dataset_id / "benchmark.json"
        if sha256_file(benchmark_path) != row["benchmark_sha256"]:
            raise SystemExit(f"Complexity-extension benchmark hash mismatch: {dataset_id}")
        benchmark = read_json(benchmark_path)
        tasks.extend(benchmark["tasks"])
        result_path = args.results_root / dataset_id / "results.json"
        if not result_path.is_file():
            raise SystemExit(f"Missing complexity-extension result: {result_path}")
        result = read_json(result_path)
        if result["manifest"]["benchmark_sha256"] != row["benchmark_sha256"]:
            raise SystemExit(f"Complexity-extension result benchmark mismatch: {dataset_id}")
        if set(result["manifest"]["conditions"]) != expected_conditions:
            raise SystemExit(f"Complexity-extension condition mismatch: {dataset_id}")
        records.extend(result["records"])
    analysis = analyze_complexity_factorial(tasks, records)
    analysis["protocol_sha256"] = protocol_hash
    analysis["analysis_amendment_sha256"] = analysis_amendment_hash
    write_json(args.output, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
