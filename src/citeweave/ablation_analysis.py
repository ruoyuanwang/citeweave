from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = [row for row in rows if not row["abstained"]]
    structured_unsupported = sum(not bool(row["correct"]) for row in answered)
    statement_claims = [row for row in rows if row.get("statement_claim")]
    unsupported_statements = sum(
        not bool(row.get("statement_supported")) for row in statement_claims
    )
    answerable = sum(bool(row.get("answerable")) for row in rows)
    evidence_eligible = [
        row for row in rows if row.get("evidence_valid") is not None
    ]
    return {
        "accuracy": correct / total if total else 0.0,
        "unsupported_claim_rate": (
            unsupported_statements / len(statement_claims) if statement_claims else 0.0
        ),
        "structured_unsupported_answer_rate": (
            structured_unsupported / len(answered) if answered else 0.0
        ),
        "statement_claim_coverage": (
            len(statement_claims) / answerable if answerable else 0.0
        ),
        "format_failure_rate": (
            sum(not bool(row.get("schema_valid")) for row in rows) / total
            if total
            else 0.0
        ),
        "evidence_path_validity": (
            sum(bool(row["evidence_valid"]) for row in evidence_eligible)
            / len(evidence_eligible)
            if evidence_eligible
            else 0.0
        ),
    }


def analyze_paired_ablation(
    graph_result: dict[str, Any],
    no_graph_result: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    graph_by_id = {row["item_id"]: row for row in graph_result["rows"]}
    no_graph_by_id = {row["item_id"]: row for row in no_graph_result["rows"]}
    if set(graph_by_id) != set(no_graph_by_id):
        raise ValueError("Conditions must contain the same item ids")
    item_ids = sorted(graph_by_id)
    graph_rows = [graph_by_id[item_id] for item_id in item_ids]
    no_graph_rows = [no_graph_by_id[item_id] for item_id in item_ids]
    graph_metrics = _condition_metrics(graph_rows)
    no_graph_metrics = _condition_metrics(no_graph_rows)

    graph_only_correct = sum(
        graph["correct"] and not baseline["correct"]
        for graph, baseline in zip(graph_rows, no_graph_rows, strict=True)
    )
    no_graph_only_correct = sum(
        baseline["correct"] and not graph["correct"]
        for graph, baseline in zip(graph_rows, no_graph_rows, strict=True)
    )
    discordant = graph_only_correct + no_graph_only_correct
    mcnemar_p = (
        float(
            binomtest(
                min(graph_only_correct, no_graph_only_correct),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )

    rng = np.random.default_rng(seed)
    accuracy_differences = np.empty(bootstrap_samples)
    ucr_reductions = np.empty(bootstrap_samples)
    stratified_accuracy_differences = np.empty(bootstrap_samples)
    stratified_ucr_reductions = np.empty(bootstrap_samples)
    count = len(item_ids)
    strata: dict[str, list[int]] = {}
    for position, row in enumerate(graph_rows):
        strata.setdefault(str(row.get("dataset_id", "<single-dataset>")), []).append(
            position
        )
    for index in range(bootstrap_samples):
        sample = rng.integers(0, count, size=count)
        sampled_graph = [graph_rows[position] for position in sample]
        sampled_no_graph = [no_graph_rows[position] for position in sample]
        graph_sample_metrics = _condition_metrics(sampled_graph)
        no_graph_sample_metrics = _condition_metrics(sampled_no_graph)
        accuracy_differences[index] = (
            graph_sample_metrics["accuracy"] - no_graph_sample_metrics["accuracy"]
        )
        ucr_reductions[index] = (
            no_graph_sample_metrics["unsupported_claim_rate"]
            - graph_sample_metrics["unsupported_claim_rate"]
        )
        stratified_sample = np.concatenate(
            [
                rng.choice(positions, size=len(positions), replace=True)
                for positions in strata.values()
            ]
        )
        stratified_graph = [graph_rows[position] for position in stratified_sample]
        stratified_no_graph = [
            no_graph_rows[position] for position in stratified_sample
        ]
        stratified_graph_metrics = _condition_metrics(stratified_graph)
        stratified_no_graph_metrics = _condition_metrics(stratified_no_graph)
        stratified_accuracy_differences[index] = (
            stratified_graph_metrics["accuracy"]
            - stratified_no_graph_metrics["accuracy"]
        )
        stratified_ucr_reductions[index] = (
            stratified_no_graph_metrics["unsupported_claim_rate"]
            - stratified_graph_metrics["unsupported_claim_rate"]
        )

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    networks = sorted({row["network"] for row in graph_rows})
    by_network = {}
    for network in networks:
        graph_group = [row for row in graph_rows if row["network"] == network]
        no_graph_group = [row for row in no_graph_rows if row["network"] == network]
        graph_group_metrics = _condition_metrics(graph_group)
        no_graph_group_metrics = _condition_metrics(no_graph_group)
        by_network[network] = {
            "items": len(graph_group),
            "graph_accuracy": graph_group_metrics["accuracy"],
            "no_graph_accuracy": no_graph_group_metrics["accuracy"],
            "accuracy_difference": (
                graph_group_metrics["accuracy"] - no_graph_group_metrics["accuracy"]
            ),
        }

    return {
        "analysis_version": 2,
        "paired_items": count,
        "graph_condition": graph_result["condition"],
        "no_graph_condition": no_graph_result["condition"],
        "graph_metrics": graph_metrics,
        "no_graph_metrics": no_graph_metrics,
        "effects": {
            "accuracy_difference": (
                graph_metrics["accuracy"] - no_graph_metrics["accuracy"]
            ),
            "accuracy_difference_bootstrap_95_ci": interval(accuracy_differences),
            "accuracy_difference_topic_stratified_bootstrap_95_ci": interval(
                stratified_accuracy_differences
            ),
            "unsupported_claim_rate_reduction": (
                no_graph_metrics["unsupported_claim_rate"]
                - graph_metrics["unsupported_claim_rate"]
            ),
            "unsupported_claim_rate_reduction_bootstrap_95_ci": interval(
                ucr_reductions
            ),
            "unsupported_claim_rate_reduction_topic_stratified_bootstrap_95_ci": (
                interval(stratified_ucr_reductions)
            ),
        },
        "mcnemar_exact": {
            "graph_only_correct": graph_only_correct,
            "no_graph_only_correct": no_graph_only_correct,
            "discordant_pairs": discordant,
            "two_sided_p_value": mcnemar_p,
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "strata": sorted(strata),
        },
        "by_network": by_network,
    }
