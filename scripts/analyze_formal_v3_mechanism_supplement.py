from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

from citeweave.io import read_json, sha256_file, write_json

CONDITIONS = ("graph_program", "flat_program", "operator_only")
COMPLEX_TASK_TYPES = {
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
}


def _outcome(record: dict[str, Any], field: str) -> float:
    if record.get("status", "complete") != "complete":
        return 0.0
    score = record.get("score") or {}
    if field == "answer_accuracy":
        return float(bool(score.get("answer_exact")))
    if field == "evidence_f1":
        return float(score.get("evidence_f1") or 0.0)
    raise ValueError(f"Unsupported outcome: {field}")


def _signflip(values: list[float]) -> dict[str, Any]:
    if len(values) != 8:
        raise ValueError("Confirmatory mechanism tests require exactly eight datasets")
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


def _tost(values: list[float], margin: float) -> dict[str, Any]:
    lower = _signflip([value + margin for value in values])
    upper = _signflip([margin - value for value in values])
    return {
        "margin": margin,
        "lower_test": lower,
        "upper_test": upper,
        "equivalent_at_0_05": (
            lower["p_value_one_sided"] < 0.05
            and upper["p_value_one_sided"] < 0.05
        ),
    }


def _bootstrap(values: dict[str, float], *, seed: int, samples: int) -> dict[str, Any]:
    datasets = sorted(values)
    generator = Random(seed)
    draws = [
        sum(values[key] for key in generator.choices(datasets, k=len(datasets)))
        / len(datasets)
        for _ in range(samples)
    ]
    draws.sort()
    return {
        "estimate": sum(values.values()) / len(values),
        "ci_low": draws[round(0.025 * (samples - 1))],
        "ci_high": draws[round(0.975 * (samples - 1))],
        "samples": samples,
        "seed": seed,
    }


def _holm(p_values: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return {
        name: {
            "raw_p_value_one_sided": p_values[name],
            "holm_adjusted_p_value": adjusted[name],
            "reject_at_0_05": adjusted[name] < 0.05,
        }
        for name in p_values
    }


def _dataset_effects(
    *,
    task_index: dict[str, dict[str, Any]],
    records: dict[tuple[str, str], dict[str, Any]],
    left: str,
    right: str,
    outcome: str,
    include,
) -> dict[str, float]:
    paired: dict[str, list[float]] = defaultdict(list)
    for item_id, task in task_index.items():
        if not include(task):
            continue
        paired[str(task["dataset_id"])].append(
            _outcome(records[(item_id, left)], outcome)
            - _outcome(records[(item_id, right)], outcome)
        )
    if set(paired) != {str(task["dataset_id"]) for task in task_index.values()}:
        raise ValueError("Every mechanism estimand must cover all eight datasets")
    return {
        dataset: sum(values) / len(values) for dataset, values in sorted(paired.items())
    }


def analyze_mechanism_supplement(
    *,
    tasks: list[dict[str, Any]],
    result_records: list[dict[str, Any]],
    exposure_records: list[dict[str, Any]],
    bootstrap_samples: int = 10_000,
    seed: int = 20260831,
) -> dict[str, Any]:
    task_index = {str(task["item_id"]): task for task in tasks}
    if len(task_index) != 168:
        raise ValueError("Mechanism supplement requires 168 unique tasks")
    datasets = {str(task["dataset_id"]) for task in tasks}
    if len(datasets) != 8 or any(
        sum(str(task["dataset_id"]) == dataset for task in tasks) != 21
        for dataset in datasets
    ):
        raise ValueError("Mechanism supplement requires eight datasets with 21 tasks each")
    records = {
        (str(record["item_id"]), str(record["condition"])): record
        for record in result_records
    }
    if len(records) != len(result_records):
        raise ValueError("Duplicate final mechanism cells")
    expected = {(item_id, condition) for item_id in task_index for condition in CONDITIONS}
    if set(records) != expected:
        raise ValueError("Mechanism results must contain exactly 504 registered cells")

    exposure = {str(row["item_id"]): row for row in exposure_records}
    complex_ids = {
        item_id
        for item_id, task in task_index.items()
        if task["task_type"] in COMPLEX_TASK_TYPES
    }
    if set(exposure) != complex_ids:
        raise ValueError("Exposure audit must cover exactly the 120 complex tasks")
    derivable_types = {
        task_type
        for task_type in COMPLEX_TASK_TYPES
        if all(
            row["derivability"]["operator_trace"]["fully_derivable"] is True
            for item_id, row in exposure.items()
            if task_index[item_id]["task_type"] == task_type
        )
    }
    if derivable_types != {
        "multi_hop_connector",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }:
        raise ValueError("Frozen fully trace-derivable task-type classification changed")

    m1_effects = _dataset_effects(
        task_index=task_index,
        records=records,
        left="graph_program",
        right="flat_program",
        outcome="answer_accuracy",
        include=lambda _task: True,
    )
    m2_evidence_effects = _dataset_effects(
        task_index=task_index,
        records=records,
        left="graph_program",
        right="operator_only",
        outcome="evidence_f1",
        include=lambda task: task["task_type"] in COMPLEX_TASK_TYPES,
    )
    m2_answer_effects = _dataset_effects(
        task_index=task_index,
        records=records,
        left="graph_program",
        right="operator_only",
        outcome="answer_accuracy",
        include=lambda task: task["task_type"] in COMPLEX_TASK_TYPES,
    )
    m3_bridge = _dataset_effects(
        task_index=task_index,
        records=records,
        left="graph_program",
        right="operator_only",
        outcome="answer_accuracy",
        include=lambda task: task["task_type"] == "bridge_counterfactual",
    )
    m3_derivable = _dataset_effects(
        task_index=task_index,
        records=records,
        left="graph_program",
        right="operator_only",
        outcome="answer_accuracy",
        include=lambda task: task["task_type"] in derivable_types,
    )
    m3_effects = {
        dataset: m3_bridge[dataset] - m3_derivable[dataset]
        for dataset in sorted(datasets)
    }

    m2_evidence_test = _signflip(list(m2_evidence_effects.values()))
    m2_answer_ni = _signflip(
        [value + 0.05 for value in m2_answer_effects.values()]
    )
    m3_test = _signflip(list(m3_effects.values()))
    multiplicity = _holm(
        {
            "M2_evidence_superiority": m2_evidence_test["p_value_one_sided"],
            "M3_trace_incomplete_selectivity": m3_test["p_value_one_sided"],
        }
    )
    return {
        "schema_version": 1,
        "status": "analyzed_registered_mechanism_supplement",
        "independent_unit": "dataset_id",
        "datasets": sorted(datasets),
        "tasks": len(tasks),
        "logical_cells": len(records),
        "terminal_parse_failure_cells": sum(
            record.get("status", "complete") != "complete"
            for record in result_records
        ),
        "derivable_task_types": sorted(derivable_types),
        "M1_same_computation_representation": {
            "estimand": "graph_program_minus_flat_program_answer_accuracy",
            "dataset_effects": m1_effects,
            "equivalence": _tost(list(m1_effects.values()), 0.05),
            "descriptive_interval": _bootstrap(
                m1_effects, seed=seed, samples=bootstrap_samples
            ),
        },
        "M2_raw_provenance_value": {
            "evidence_f1_superiority": {
                **m2_evidence_test,
                "dataset_effects": m2_evidence_effects,
                "descriptive_interval": _bootstrap(
                    m2_evidence_effects, seed=seed + 1, samples=bootstrap_samples
                ),
            },
            "answer_accuracy_noninferiority": {
                **m2_answer_ni,
                "margin": 0.05,
                "unshifted_estimate": sum(m2_answer_effects.values()) / 8,
                "dataset_effects": m2_answer_effects,
                "noninferior_at_0_05": m2_answer_ni["p_value_one_sided"] < 0.05,
            },
        },
        "M3_incomplete_trace_selectivity": {
            **m3_test,
            "outcome": "answer_accuracy",
            "dataset_effects": m3_effects,
            "bridge_effects": m3_bridge,
            "fully_derivable_effects": m3_derivable,
            "descriptive_interval": _bootstrap(
                m3_effects, seed=seed + 2, samples=bootstrap_samples
            ),
        },
        "holm_family": multiplicity,
        "registered_success_gates": {
            "M1": "two-sided equivalence within +/-0.05",
            "M2": "Holm-adjusted evidence superiority and answer noninferiority",
            "M3": "Holm-adjusted positive selectivity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--answer-exposure-audit", type=Path, required=True)
    parser.add_argument("--supplement-protocol", type=Path, required=True)
    parser.add_argument("--supplement-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    protocol_hash = sha256_file(args.supplement_protocol)
    if read_json(args.supplement_freeze).get("sha256") != protocol_hash:
        raise SystemExit("Mechanism-supplement protocol differs from its freeze")
    merged = read_json(args.merged)
    exposure = read_json(args.answer_exposure_audit)
    result = analyze_mechanism_supplement(
        tasks=merged["tasks"],
        result_records=merged["records"],
        exposure_records=exposure["records"],
        bootstrap_samples=args.bootstrap_samples,
    )
    result.update(
        {
            "supplement_protocol_sha256": protocol_hash,
            "merged_path": str(args.merged.resolve()),
            "merged_sha256": sha256_file(args.merged),
            "answer_exposure_audit_path": str(args.answer_exposure_audit.resolve()),
            "answer_exposure_audit_sha256": sha256_file(args.answer_exposure_audit),
        }
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
