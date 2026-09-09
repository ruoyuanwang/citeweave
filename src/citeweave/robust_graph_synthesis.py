from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json

TASK_TYPES = (
    "path_stability_profile",
    "community_role_instability",
    "hub_resilience_decomposition",
    "temporal_window_dependence",
    "cross_operator_robustness_triage",
)
MEASUREMENT_TYPES = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)
TOKEN = re.compile(r"[a-z0-9_.-]+")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def _variant_files(topic_dir: Path) -> list[Path]:
    files = []
    for path in sorted(topic_dir.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload.get("variant"), dict) and isinstance(
            payload.get("measurements"), dict
        ):
            files.append(path)
    return files


def _load_cases(topic_dir: Path) -> list[dict[str, Any]]:
    cases = [read_json(path) for path in _variant_files(topic_dir)]
    if len(cases) != 18:
        raise ValueError(f"Expected 18 robustness cases in {topic_dir}, found {len(cases)}")
    variant_ids = [str(case["variant"]["variant_id"]) for case in cases]
    if len(variant_ids) != len(set(variant_ids)) or variant_ids.count("baseline") != 1:
        raise ValueError(f"Invalid robustness variant coverage in {topic_dir}")
    dataset_ids = {str(case["dataset_id"]) for case in cases}
    if dataset_ids != {topic_dir.name}:
        raise ValueError(f"Dataset identity mismatch in {topic_dir}")
    return sorted(cases, key=lambda row: str(row["variant"]["variant_id"]))


def _slim_measurement(task_type: str, measurement: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "multi_hop_connector": (
            "status",
            "hops",
            "path",
            "total_inverse_weight_distance",
            "cross_community",
            "original_path_still_optimal",
        ),
        "bridge_counterfactual": (
            "status",
            "component_increase",
            "alternate_hops_after_deletion",
            "redundant_connector",
            "cross_community",
        ),
        "community_role_contrast": (
            "status",
            "disconnected_communities",
        ),
        "hub_removal_resilience": (
            "status",
            "frozen_hub_rank",
            "frozen_hub_still_argmax",
            "replacement_hub",
            "reselected_hub",
            "components_after",
            "largest_component_fraction_of_original_nodes",
        ),
        "temporal_structural_shift": (
            "status",
            "emerging_label",
            "established_label",
            "same_community",
            "emerging_growth_ratio",
            "established_weighted_degree",
            "early_years",
            "recent_years",
        ),
    }[task_type]
    result = {field: measurement.get(field) for field in fields}
    if task_type == "community_role_contrast":
        for role in ("dominant", "outward"):
            value = measurement.get(role)
            if isinstance(value, dict):
                result[role] = {
                    key: value.get(key)
                    for key in (
                        "community",
                        "representative",
                        "external_share",
                        "importance",
                    )
                }
    return result


def _evidence_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        variant = case["variant"]
        variant_id = str(variant["variant_id"])
        family = str(variant["family"])
        for task_type in MEASUREMENT_TYPES:
            evidence_id = "RG-" + _hash(
                [case["dataset_id"], variant_id, task_type]
            )[:16]
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "variant_id": variant_id,
                    "family": family,
                    "parameters": {
                        key: variant[key]
                        for key in (
                            "minimum_weight",
                            "drop_fraction",
                            "drop_seed",
                            "community_seed",
                            "resolution",
                            "window_years",
                        )
                    },
                    "measurement_type": task_type,
                    "graph": {
                        key: case["graph"].get(key)
                        for key in (
                            "nodes",
                            "edges",
                            "components",
                            "largest_component_fraction",
                            "communities",
                        )
                    },
                    "measurement": _slim_measurement(
                        task_type, case["measurements"][task_type]
                    ),
                    "comparison_to_baseline": (
                        case.get("comparison", {}).get(task_type, {})
                        if variant_id != "baseline"
                        else {}
                    ),
                }
            )
    return rows


def _perturbations(rows: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["measurement_type"] == task_type and row["variant_id"] != "baseline"
    ]


def _rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    return sum(predicate(row) for row in rows) / len(rows) if rows else 0.0


def _family_rate(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> tuple[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    rates = {family: _round(_rate(values, predicate)) for family, values in grouped.items()}
    least = min(rates, key=lambda family: (rates[family], family))
    return least, dict(sorted(rates.items()))


def _decisive(rows: list[dict[str, Any]], unstable: Callable[[dict[str, Any]], bool]) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            not unstable(row),
            str(row["family"]),
            str(row["variant_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in ordered:
        family = str(row["family"])
        if family not in seen_families:
            selected.append(row)
            seen_families.add(family)
    for row in ordered:
        if len(selected) >= 8:
            break
        if row not in selected:
            selected.append(row)
    return [str(row["evidence_id"]) for row in selected[:8]]


def _path_task(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = _perturbations(rows, "multi_hop_connector")
    stable = lambda row: bool(row["comparison_to_baseline"].get("original_path_still_optimal"))
    least, family_rates = _family_rate(relevant, stable)
    extreme = min(
        relevant,
        key=lambda row: (
            -abs(float(row["comparison_to_baseline"].get("distance_change") or 0.0)),
            str(row["variant_id"]),
        ),
    )
    stability = _round(_rate(relevant, stable))
    answer = {
        "observed_variants": len(relevant),
        "stable_variants": sum(stable(row) for row in relevant),
        "stability_rate": stability,
        "least_stable_family": least,
        "largest_distance_change_variant": extreme["variant_id"],
        "largest_absolute_distance_change": _round(
            abs(float(extreme["comparison_to_baseline"].get("distance_change") or 0.0))
        ),
        "robustness_label": "stable" if stability >= 0.9 else "sensitive",
    }
    return _task(
        dataset_id,
        "path_stability_profile",
        6,
        "Across all registered perturbations, quantify whether the baseline multi-hop connector remains optimal, identify the least stable perturbation family and the variant with the largest distance change, and give a calibrated robustness label.",
        answer,
        relevant,
        [{"operator": "grouped_path_stability", "family_rates": family_rates}, {"operator": "select_extreme_distance_change", **answer}],
        _decisive(relevant, lambda row: not stable(row)),
        "Robustness is conditional on the registered perturbation grid and fixed corpus.",
    )


def _community_task(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = _perturbations(rows, "community_role_contrast")
    comparable = [row for row in relevant if row["comparison_to_baseline"].get("comparable")]
    stable = lambda row: bool(row["comparison_to_baseline"].get("outward_representative_equal"))
    least, family_rates = _family_rate(comparable, stable)
    extreme = min(
        comparable,
        key=lambda row: (
            float(row["comparison_to_baseline"].get("outward_membership_jaccard") or 0.0),
            str(row["variant_id"]),
        ),
    )
    rate = _round(_rate(comparable, stable))
    answer = {
        "registered_variants": len(relevant),
        "comparable_variants": len(comparable),
        "outward_representative_stability_rate": rate,
        "least_stable_family": least,
        "lowest_jaccard_variant": extreme["variant_id"],
        "lowest_outward_membership_jaccard": _round(
            float(extreme["comparison_to_baseline"]["outward_membership_jaccard"])
        ),
        "robustness_label": "stable" if rate >= 0.8 else "sensitive",
    }
    return _task(
        dataset_id,
        "community_role_instability",
        7,
        "Determine how sensitive the outward-facing community role is across the registered graph perturbations. Account for non-comparable variants, locate the weakest family and minimum-membership-overlap case, and avoid treating algorithmic communities as semantic themes.",
        answer,
        relevant,
        [{"operator": "filter_comparable_then_group", "family_rates": family_rates}, {"operator": "select_minimum_outward_jaccard", **answer}],
        _decisive(relevant, lambda row: not stable(row)),
        "Community identities are algorithmic and may change with resolution or sparsification.",
    )


def _hub_task(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = _perturbations(rows, "hub_removal_resilience")
    baseline = next(
        row
        for row in rows
        if row["measurement_type"] == "hub_removal_resilience"
        and row["variant_id"] == "baseline"
    )
    base_pre = float(baseline["graph"]["largest_component_fraction"])
    base_post = float(
        baseline["measurement"]["largest_component_fraction_of_original_nodes"]
    )
    decompositions = []
    for row in relevant:
        pre = float(row["graph"]["largest_component_fraction"])
        post = float(row["measurement"]["largest_component_fraction_of_original_nodes"])
        decompositions.append(
            {
                "variant_id": row["variant_id"],
                "family": row["family"],
                "total_post_deletion_decline": base_post - post,
                "pre_existing_topology_loss": base_pre - pre,
                "additional_marginal_deletion_loss": (pre - post) - (base_pre - base_post),
                "evidence_id": row["evidence_id"],
            }
        )
    extreme = min(
        decompositions,
        key=lambda row: (-row["total_post_deletion_decline"], str(row["variant_id"])),
    )
    topology = float(extreme["pre_existing_topology_loss"])
    marginal = float(extreme["additional_marginal_deletion_loss"])
    driver = (
        "pre_existing_topology_loss"
        if topology > marginal + 1e-9
        else "deletion_specific_loss"
        if marginal > topology + 1e-9
        else "mixed_or_equal"
    )
    answer = {
        "largest_decline_variant": extreme["variant_id"],
        "largest_total_post_deletion_decline": _round(extreme["total_post_deletion_decline"]),
        "pre_existing_topology_loss": _round(topology),
        "additional_marginal_deletion_loss": _round(marginal),
        "dominant_driver": driver,
    }
    return _task(
        dataset_id,
        "hub_resilience_decomposition",
        8,
        "Find the perturbation with the greatest decline in the post-hub-deletion largest component, then decompose that decline into topology already lost before deletion and additional marginal loss caused by the deletion. State which component dominates.",
        answer,
        [baseline, *relevant],
        [{"operator": "baseline_relative_deletion_decomposition", "rows": [{key: _round(value) if isinstance(value, float) else value for key, value in row.items()} for row in decompositions]}, {"operator": "select_largest_total_decline", **answer}],
        [str(extreme["evidence_id"]), str(baseline["evidence_id"])],
        "The decomposition is structural and does not measure real-world causal dependence.",
    )


def _temporal_task(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [
        row
        for row in _perturbations(rows, "temporal_structural_shift")
        if row["family"] == "temporal_window"
    ]
    changed_label = lambda row: not bool(
        row["comparison_to_baseline"].get("emerging_label_equal")
    )
    changed_relation = lambda row: not bool(
        row["comparison_to_baseline"].get("same_community_equal")
    )
    label_changes = sum(changed_label(row) for row in relevant)
    relation_changes = sum(changed_relation(row) for row in relevant)
    max_growth = max(
        abs(float(row["comparison_to_baseline"].get("growth_ratio_change") or 0.0))
        for row in relevant
    )
    label = (
        "label_window_sensitive"
        if label_changes
        else "community_relation_sensitive"
        if relation_changes
        else "stable_on_registered_windows"
    )
    answer = {
        "temporal_window_variants": len(relevant),
        "emerging_label_changes": label_changes,
        "same_community_changes": relation_changes,
        "maximum_absolute_growth_ratio_change": _round(max_growth),
        "window_dependence_label": label,
    }
    return _task(
        dataset_id,
        "temporal_window_dependence",
        7,
        "Across the registered alternative temporal windows, distinguish changes in the selected emerging keyword from changes only in its community relation, quantify the largest growth-ratio change, and report the narrowest supported dependence label.",
        answer,
        relevant,
        [{"operator": "temporal_window_contrast", **answer}],
        _decisive(relevant, lambda row: changed_label(row) or changed_relation(row)),
        "Window sensitivity is evaluated only for the registered candidate set and years.",
    )


def _triage_task(dataset_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "multi_hop_connector": lambda row: bool(row["comparison_to_baseline"].get("original_path_still_optimal")),
        "bridge_counterfactual": lambda row: bool(row["comparison_to_baseline"].get("redundancy_equal")) and bool(row["comparison_to_baseline"].get("alternate_hops_equal")),
        "community_role_contrast": lambda row: bool(row["comparison_to_baseline"].get("comparable")) and bool(row["comparison_to_baseline"].get("outward_representative_equal")),
        "hub_removal_resilience": lambda row: bool(row["comparison_to_baseline"].get("frozen_hub_still_argmax")) and bool(row["comparison_to_baseline"].get("replacement_hub_equal")) and abs(float(row["comparison_to_baseline"].get("largest_component_fraction_change") or 0.0)) <= 0.01,
        "temporal_structural_shift": lambda row: bool(row["comparison_to_baseline"].get("emerging_label_equal")) and bool(row["comparison_to_baseline"].get("established_label_equal")) and bool(row["comparison_to_baseline"].get("same_community_equal")),
    }
    rates = {}
    relevant = []
    for measurement_type, predicate in predicates.items():
        subset = _perturbations(rows, measurement_type)
        relevant.extend(subset)
        rates[measurement_type] = _round(_rate(subset, predicate))
    most = min(rates, key=lambda key: (-rates[key], key))
    least = min(rates, key=lambda key: (rates[key], key))
    spread = _round(rates[most] - rates[least])
    label = "heterogeneous" if spread >= 0.1 else "uniformly_stable" if min(rates.values()) >= 0.9 else "uniformly_uncertain"
    answer = {
        "most_stable_operator": most,
        "least_stable_operator": least,
        "most_stable_rate": rates[most],
        "least_stable_rate": rates[least],
        "stability_spread": spread,
        "joint_robustness_label": label,
    }
    summaries = [
        {"measurement_type": key, "stability_rate": value}
        for key, value in sorted(rates.items())
    ]
    decisive = []
    for task_type in (most, least):
        subset = _perturbations(rows, task_type)
        decisive.extend(_decisive(subset, lambda row, p=predicates[task_type]: not p(row))[:2])
    return _task(
        dataset_id,
        "cross_operator_robustness_triage",
        9,
        "Compare robustness across the five registered graph operators, identify the most and least stable operators and their rate spread, and decide whether the evidence is heterogeneous rather than averaging incompatible structural claims into one conclusion.",
        answer,
        relevant,
        [{"operator": "cross_operator_stability_rates", "summaries": summaries}, {"operator": "rank_and_label_heterogeneity", **answer}],
        list(dict.fromkeys(decisive))[:8],
        "Operator-specific stability rates are not interchangeable measures of scientific validity.",
    )


def _bm25(question: str, rows: list[dict[str, Any]], *, limit: int, tie_key: str) -> list[dict[str, Any]]:
    documents = [_canonical(row) for row in rows]
    query_terms = set(TOKEN.findall(question.casefold()))
    tokenized = [TOKEN.findall(document.casefold()) for document in documents]
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    document_frequency = {term: sum(term in tokens for tokens in tokenized) for term in query_terms}
    scores = []
    for row, tokens in zip(rows, tokenized):
        counts = Counter(token for token in tokens if token in query_terms)
        score = 0.0
        for term, frequency in counts.items():
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(rows) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * len(tokens) / max(1.0, average_length))
            score += inverse * frequency * 2.2 / denominator
        scores.append((score, _hash([tie_key, row["evidence_id"]]), row))
    return [row for _, _, row in sorted(scores, key=lambda item: (-item[0], item[1]))[:limit]]


def _task(
    dataset_id: str,
    task_type: str,
    complexity: int,
    question: str,
    answer: dict[str, Any],
    relevant_rows: list[dict[str, Any]],
    operator_trace: list[dict[str, Any]],
    evidence_ids: list[str],
    required_limitation: str,
) -> dict[str, Any]:
    return {
        "item_id": f"{dataset_id}:robustness:{task_type}",
        "dataset_id": dataset_id,
        "network": "keyword_cooccurrence_perturbation_ensemble",
        "scale": "large",
        "task_type": task_type,
        "complexity": complexity,
        "question": question,
        "answer": answer,
        "answer_alternatives": [],
        "evidence_ids": evidence_ids,
        "operator_trace": operator_trace,
        "interpretation_contract": {
            "allowed": "registered perturbation robustness and structural sensitivity",
            "forbidden": ["causality", "scientific quality", "author intent", "unregistered generalization"],
            "required_limitation": required_limitation,
        },
        "_relevant_rows": relevant_rows,
    }


def _summary_nodes(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = task["_relevant_rows"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["measurement_type"]), str(row["family"]))].append(row)
    return [
        {
            "summary_id": "RS-" + _hash([task["item_id"], measurement, family])[:14],
            "measurement_type": measurement,
            "family": family,
            "variant_count": len(values),
            "evidence_ids": [row["evidence_id"] for row in values],
        }
        for (measurement, family), values in sorted(grouped.items())
    ]


def _attach_contexts(task: dict[str, Any], all_rows: list[dict[str, Any]], record_budget: int) -> dict[str, Any]:
    relevant = list(task.pop("_relevant_rows"))
    summaries = _summary_nodes({**task, "_relevant_rows": relevant})
    required_ids = set(task["evidence_ids"])
    required_rows = [row for row in relevant if row["evidence_id"] in required_ids]
    if len(required_rows) != len(required_ids):
        raise ValueError(f"Required evidence is absent for {task['item_id']}")
    remaining_rows = sorted(
        (row for row in relevant if row["evidence_id"] not in required_ids),
        key=lambda row: _hash([task["item_id"], row["evidence_id"]]),
    )
    program_rows = [*required_rows, *remaining_rows[: max(0, record_budget - len(required_rows))]]
    if len(program_rows) > record_budget:
        raise ValueError(f"Required evidence exceeds record budget for {task['item_id']}")
    hierarchy = {
        "representation": "perturbation_dependency_hierarchy_without_answer",
        "record_budget": record_budget,
        "summary_nodes": summaries,
        "family_edges": [
            {"source": node["family"], "target": node["summary_id"]}
            for node in summaries
        ],
        "evidence_rows": program_rows,
    }
    flat_program = {
        "representation": "flat_program_same_information_ablation",
        "record_budget": record_budget,
        "rows": program_rows,
        "derived_rows": task["operator_trace"],
        "interpretation_contract": task["interpretation_contract"],
    }
    graph_program = {
        **hierarchy,
        "representation": "robustness_graph_program",
        "operator_trace": task["operator_trace"],
        "interpretation_contract": task["interpretation_contract"],
    }
    task["contexts"] = {
        "flat_bm25": {
            "representation": "flat_bm25_budgeted_over_all_variant_operator_rows",
            "record_budget": record_budget,
            "parameters": {"k1": 1.2, "b": 0.75},
            "rows": _bm25(task["question"], all_rows, limit=record_budget, tie_key=task["item_id"]),
        },
        "graph_hierarchical_retrieval_v2": hierarchy,
        "flat_program": flat_program,
        "graph_program": graph_program,
        "operator_only": {
            "representation": "operator_trace_without_raw_perturbation_evidence",
            "operator_trace": task["operator_trace"],
            "interpretation_contract": task["interpretation_contract"],
        },
    }
    task["context_hashes"] = {
        condition: _hash(context) for condition, context in task["contexts"].items()
    }
    return task


def build_topic_benchmark(topic_dir: Path, *, record_budget: int = 20) -> dict[str, Any]:
    cases = _load_cases(topic_dir)
    rows = _evidence_rows(cases)
    dataset_id = topic_dir.name
    tasks = [
        _path_task(dataset_id, rows),
        _community_task(dataset_id, rows),
        _hub_task(dataset_id, rows),
        _temporal_task(dataset_id, rows),
        _triage_task(dataset_id, rows),
    ]
    tasks = [_attach_contexts(task, rows, record_budget) for task in tasks]
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "status": "development_benchmark_constructed_before_model_outcomes",
        "design": {
            "role": "difficulty_calibration_not_confirmatory_evidence",
            "record_budget": record_budget,
            "conditions": [
                "flat_bm25",
                "graph_hierarchical_retrieval_v2",
                "flat_program",
                "graph_program",
                "operator_only",
            ],
            "task_types": list(TASK_TYPES),
            "source_cases": len(cases),
            "source_measurement_rows": len(rows),
        },
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in _variant_files(topic_dir)
        ],
        "tasks": tasks,
    }


def build_benchmark_panel(
    source_root: Path, output_root: Path, *, record_budget: int = 20
) -> dict[str, Any]:
    topic_dirs = sorted(
        path for path in source_root.iterdir() if path.is_dir() and (path / "baseline.json").is_file()
    )
    if len(topic_dirs) != 8:
        raise ValueError(f"Expected eight topic directories, found {len(topic_dirs)}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    rates: list[float] = []
    for topic_dir in topic_dirs:
        benchmark = build_topic_benchmark(topic_dir, record_budget=record_budget)
        destination = output_root / topic_dir.name / "benchmark.json"
        write_json(destination, benchmark)
        for task in benchmark["tasks"]:
            rates.extend(
                float(value)
                for key, value in task["answer"].items()
                if key.endswith("_rate") and isinstance(value, (int, float))
            )
        records.append(
            {
                "dataset_id": topic_dir.name,
                "benchmark": str(destination.resolve()),
                "benchmark_sha256": sha256_file(destination),
                "tasks": len(benchmark["tasks"]),
                "planned_calls": len(benchmark["tasks"])
                * len(benchmark["design"]["conditions"]),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "development_panel_constructed_before_model_outcomes",
        "topics": len(records),
        "tasks": sum(row["tasks"] for row in records),
        "conditions": 5,
        "planned_calls": sum(row["planned_calls"] for row in records),
        "difficulty_diagnostic": {
            "rate_fields": len(rates),
            "interior_rate_fields": sum(0.0 < value < 1.0 for value in rates),
            "minimum_rate": min(rates),
            "maximum_rate": max(rates),
            "distinct_rates": len(set(rates)),
        },
        "records": records,
    }
    write_json(output_root / "construction_manifest.json", manifest)
    return manifest


def answer_field_accuracy(gold: Any, observed: Any) -> tuple[int, int]:
    if isinstance(gold, dict):
        if not isinstance(observed, dict):
            return 0, sum(answer_field_accuracy(value, None)[1] for value in gold.values())
        correct = total = 0
        for key, value in gold.items():
            item_correct, item_total = answer_field_accuracy(value, observed.get(key))
            correct += item_correct
            total += item_total
        return correct, total
    if isinstance(gold, list):
        return (int(gold == observed), 1)
    if isinstance(gold, float) and isinstance(observed, (int, float)):
        return (int(math.isclose(gold, float(observed), abs_tol=1e-4, rel_tol=1e-4)), 1)
    return (int(gold == observed), 1)
