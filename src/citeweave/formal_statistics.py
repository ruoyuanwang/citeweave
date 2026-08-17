from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import numpy as np

REPORT_COMPARISONS = {
    "full_vs_oneshot": ("citeweave_full", "structured_one_shot"),
    "full_vs_human": ("citeweave_full", "published_human_reference"),
    "oneshot_vs_human": ("structured_one_shot", "published_human_reference"),
}
GRAPH_COMPARISONS = {
    "graph_vs_no": ("graph_rag", "no_rag"),
    "graph_vs_flat": ("graph_rag", "flat_structured"),
    "graph_vs_figure": ("graph_rag", "figure_vlm"),
}
BASELINE_ORIGINAL_CONDITION = "baseline_original"
POST_REVIEW_CONDITIONS = ("always_review", "static_review", "adaptive_review")
ADAPTIVE_CONDITIONS = (BASELINE_ORIGINAL_CONDITION, *POST_REVIEW_CONDITIONS)
SUPPLEMENTARY_COMPARISONS = {"oneshot_vs_human", "graph_vs_figure"}


class FormalStatisticsError(ValueError):
    """Raised when formal-result inputs do not satisfy the frozen contract."""


@dataclass(frozen=True)
class ResolvedPanel:
    family: str
    comparison: str
    condition_a: str
    condition_b: str
    rows_by_topic: dict[str, list[dict[str, Any]]]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalStatisticsError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FormalStatisticsError(f"Cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalStatisticsError(
                f"{path}:{line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise FormalStatisticsError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise FormalStatisticsError(f"{path} contains no resolved judgments")
    return rows


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise FormalStatisticsError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FormalStatisticsError(f"{label} must be a non-negative integer")
    return value


def _validate_resolved_rows(
    rows: list[dict[str, Any]],
    *,
    topic: str,
    comparison: str,
    condition_a: str,
    condition_b: str,
) -> list[dict[str, Any]]:
    sample_ids: set[str] = set()
    packet_ids: set[str] = set()
    expected_conditions = {condition_a, condition_b}
    for index, row in enumerate(rows, 1):
        location = f"{comparison}/{topic} row {index}"
        sample_id = row.get("sample_id")
        packet_id = row.get("packet_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise FormalStatisticsError(f"{location}: missing sample_id")
        if sample_id in sample_ids:
            raise FormalStatisticsError(
                f"{comparison}/{topic}: duplicate sample_id {sample_id!r}"
            )
        sample_ids.add(sample_id)
        if not isinstance(packet_id, str) or not packet_id.strip():
            raise FormalStatisticsError(f"{location}: missing packet_id")
        if packet_id in packet_ids:
            raise FormalStatisticsError(
                f"{comparison}/{topic}: duplicate packet_id {packet_id!r}"
            )
        packet_ids.add(packet_id)
        conditions = row.get("conditions")
        if not isinstance(conditions, dict):
            raise FormalStatisticsError(f"{location}: conditions must be an object")
        _require_exact_keys(conditions, expected_conditions, label=f"{location} conditions")
        for condition, result in conditions.items():
            if not isinstance(result, dict):
                raise FormalStatisticsError(
                    f"{location}/{condition}: condition result must be an object"
                )
            supported = _nonnegative_int(
                result.get("supported_claims"),
                label=f"{location}/{condition}.supported_claims",
            )
            unsupported = _nonnegative_int(
                result.get("unsupported_claims"),
                label=f"{location}/{condition}.unsupported_claims",
            )
            if supported + unsupported == 0:
                raise FormalStatisticsError(
                    f"{location}/{condition}: Judge supplied no scorable claims"
                )
            if "claim_count" in result and result["claim_count"] != supported + unsupported:
                raise FormalStatisticsError(
                    f"{location}/{condition}: claim_count contradicts Judge counts"
                )
            completeness = result.get("mean_completeness")
            if (
                isinstance(completeness, bool)
                or not isinstance(completeness, (int, float))
                or not math.isfinite(float(completeness))
                or not 1 <= float(completeness) <= 5
            ):
                raise FormalStatisticsError(
                    f"{location}/{condition}: mean_completeness must be within [1, 5]"
                )
        if row.get("preference") not in {*expected_conditions, "tie"}:
            raise FormalStatisticsError(
                f"{location}: preference is not a decoded Judge condition or tie"
            )
        if row.get("source") not in {"dual_consensus", "adjudication"}:
            raise FormalStatisticsError(
                f"{location}: row is not a resolved dual-Judge result"
            )
    return rows


def _load_panels(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    field: str,
    family: str,
    contract: dict[str, tuple[str, str]],
    topics: list[str],
) -> list[ResolvedPanel]:
    specs = manifest.get(field)
    if not isinstance(specs, list):
        raise FormalStatisticsError(f"{field} must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise FormalStatisticsError(f"Every {field} entry needs a name")
        name = spec["name"]
        if name in by_name:
            raise FormalStatisticsError(f"Duplicate {field} comparison {name!r}")
        by_name[name] = spec
    _require_exact_keys(by_name, set(contract), label=field)

    panels = []
    for name, (condition_a, condition_b) in contract.items():
        spec = by_name[name]
        if (spec.get("condition_a"), spec.get("condition_b")) != (
            condition_a,
            condition_b,
        ):
            raise FormalStatisticsError(
                f"{name}: frozen conditions must be {condition_a} versus {condition_b}"
            )
        files = spec.get("files")
        if not isinstance(files, dict):
            raise FormalStatisticsError(f"{name}.files must map every topic to JSONL")
        _require_exact_keys(files, set(topics), label=f"{name}.files")
        rows_by_topic = {}
        for topic in topics:
            raw_path = Path(str(files[topic]))
            path = (
                raw_path
                if raw_path.is_absolute()
                else (manifest_path.parent / raw_path).resolve()
            )
            rows_by_topic[topic] = _validate_resolved_rows(
                _read_jsonl(path),
                topic=topic,
                comparison=name,
                condition_a=condition_a,
                condition_b=condition_b,
            )
        panels.append(
            ResolvedPanel(
                family=family,
                comparison=name,
                condition_a=condition_a,
                condition_b=condition_b,
                rows_by_topic=rows_by_topic,
            )
        )
    return panels


def _condition_metric(rows: list[dict[str, Any]], condition: str, metric: str) -> float:
    values = [row["conditions"][condition] for row in rows]
    if metric == "ucr":
        unsupported = sum(int(item["unsupported_claims"]) for item in values)
        total = sum(
            int(item["supported_claims"]) + int(item["unsupported_claims"])
            for item in values
        )
        return unsupported / total
    if metric == "completeness":
        return mean(float(item["mean_completeness"]) for item in values)
    raise AssertionError(metric)


def _preference_counts(
    rows_by_topic: dict[str, list[dict[str, Any]]],
    condition_a: str,
) -> tuple[int, int, int]:
    preferences = [
        row["preference"] for rows in rows_by_topic.values() for row in rows
    ]
    wins = sum(item == condition_a for item in preferences)
    ties = sum(item == "tie" for item in preferences)
    return wins, ties, len(preferences) - wins - ties


def _cluster_bootstrap(
    rows_by_topic: dict[str, list[dict[str, Any]]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    samples: int,
    rng: Random,
) -> tuple[float, float]:
    topics = sorted(rows_by_topic)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = [rng.choice(topics) for _ in topics]
        rows = [
            row
            for topic in sampled
            for row in rows_by_topic[topic]
        ]
        estimates[index] = statistic(rows)
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        raise FormalStatisticsError("Every cluster-bootstrap replicate was undefined")
    low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _paired_effect(
    rows: list[dict[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    metric: str,
) -> float:
    if metric == "ucr_reduction":
        return _condition_metric(rows, condition_b, "ucr") - _condition_metric(
            rows, condition_a, "ucr"
        )
    if metric == "completeness_difference":
        return _condition_metric(rows, condition_a, "completeness") - _condition_metric(
            rows, condition_b, "completeness"
        )
    if metric == "preference_score":
        wins = sum(row["preference"] == condition_a for row in rows)
        ties = sum(row["preference"] == "tie" for row in rows)
        return (wins + 0.5 * ties) / len(rows)
    raise AssertionError(metric)


def _sign_flip_pvalue(topic_effects: list[float]) -> float:
    """Exact two-sided paired randomization test over topic clusters."""
    observed = abs(mean(topic_effects))
    if not topic_effects:
        raise FormalStatisticsError("Cannot test an empty topic panel")
    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in product((-1.0, 1.0), repeat=len(topic_effects)):
        permuted = abs(mean(value * sign for value, sign in zip(topic_effects, signs)))
        extreme += permuted + tolerance >= observed
        total += 1
    return extreme / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return adjusted


def _format_ci(interval: list[float] | tuple[float, float]) -> str:
    return f"{interval[0]:.3f} to {interval[1]:.3f}"


def _analyze_panel(
    panel: ResolvedPanel,
    *,
    bootstrap_samples: int,
    rng: Random,
) -> dict[str, Any]:
    all_rows = [row for rows in panel.rows_by_topic.values() for row in rows]
    conditions: dict[str, Any] = {}
    for condition in (panel.condition_a, panel.condition_b):
        condition_metrics = {}
        for metric in ("ucr", "completeness"):
            estimate = _condition_metric(all_rows, condition, metric)
            interval = _cluster_bootstrap(
                panel.rows_by_topic,
                lambda rows, c=condition, m=metric: _condition_metric(rows, c, m),
                samples=bootstrap_samples,
                rng=rng,
            )
            condition_metrics[metric] = {
                "estimate": estimate,
                "cluster_bootstrap_95_ci": list(interval),
            }
        conditions[condition] = condition_metrics
    wins, ties, losses = _preference_counts(panel.rows_by_topic, panel.condition_a)
    preference_rates = {}
    for label, preferred in (
        ("win_rate", panel.condition_a),
        ("tie_rate", "tie"),
        ("loss_rate", panel.condition_b),
    ):
        estimate = sum(row["preference"] == preferred for row in all_rows) / len(all_rows)
        interval = _cluster_bootstrap(
            panel.rows_by_topic,
            lambda rows, p=preferred: (
                sum(row["preference"] == p for row in rows) / len(rows)
            ),
            samples=bootstrap_samples,
            rng=rng,
        )
        preference_rates[label] = {
            "estimate": estimate,
            "cluster_bootstrap_95_ci": list(interval),
        }
    effects = {}
    for metric in ("ucr_reduction", "completeness_difference", "preference_score"):
        estimate = _paired_effect(
            all_rows,
            condition_a=panel.condition_a,
            condition_b=panel.condition_b,
            metric=metric,
        )
        interval = _cluster_bootstrap(
            panel.rows_by_topic,
            lambda rows, m=metric: _paired_effect(
                rows,
                condition_a=panel.condition_a,
                condition_b=panel.condition_b,
                metric=m,
            ),
            samples=bootstrap_samples,
            rng=rng,
        )
        effects[metric] = {
            "estimate": estimate,
            "cluster_bootstrap_95_ci": list(interval),
        }
    return {
        "family": panel.family,
        "comparison": panel.comparison,
        "analysis_role": (
            "supplementary"
            if panel.comparison in SUPPLEMENTARY_COMPARISONS
            else "primary"
        ),
        "condition_a": panel.condition_a,
        "condition_b": panel.condition_b,
        "topics": len(panel.rows_by_topic),
        "samples": len(all_rows),
        "conditions": conditions,
        "pairwise_for_condition_a": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            **preference_rates,
        },
        "effects": effects,
        "topic_ucr_reductions": {
            topic: _paired_effect(
                rows,
                condition_a=panel.condition_a,
                condition_b=panel.condition_b,
                metric="ucr_reduction",
            )
            for topic, rows in sorted(panel.rows_by_topic.items())
        },
        "by_topic": {
            topic: {
                "samples": len(rows),
                "conditions": {
                    condition: {
                        "ucr": _condition_metric(rows, condition, "ucr"),
                        "completeness": _condition_metric(
                            rows, condition, "completeness"
                        ),
                    }
                    for condition in (panel.condition_a, panel.condition_b)
                },
                "pairwise_for_condition_a": dict(
                    zip(
                        ("wins", "ties", "losses"),
                        _preference_counts({topic: rows}, panel.condition_a),
                    )
                ),
            }
            for topic, rows in sorted(panel.rows_by_topic.items())
        },
    }


def _validate_graph_pairing(panels: list[ResolvedPanel], topics: list[str]) -> None:
    indexed = {panel.comparison: panel for panel in panels}
    for topic in topics:
        sample_sets = {
            panel.comparison: {row["sample_id"] for row in panel.rows_by_topic[topic]}
            for panel in panels
        }
        text_ids = sample_sets["graph_vs_no"]
        if sample_sets["graph_vs_flat"] != text_ids:
            raise FormalStatisticsError(
                f"Graph text-panel comparisons are not paired for topic {topic}"
            )
        figure_ids = sample_sets["graph_vs_figure"]
        if not figure_ids.issubset(text_ids):
            raise FormalStatisticsError(
                f"Graph Figure/VLM panel is not a paired subset for topic {topic}"
            )
        if not indexed["graph_vs_figure"].rows_by_topic[topic]:
            raise FormalStatisticsError(f"Graph Figure/VLM panel is empty for topic {topic}")


def _load_adaptive(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    topics: list[str],
    bootstrap_samples: int,
    rng: Random,
) -> dict[str, Any]:
    files = manifest.get("adaptive_results")
    if not isinstance(files, dict):
        raise FormalStatisticsError("adaptive_results must map every topic to JSON")
    _require_exact_keys(files, set(topics), label="adaptive_results")
    counts_by_condition: dict[str, dict[str, dict[str, int]]] = {
        condition: {} for condition in ADAPTIVE_CONDITIONS
    }
    for topic in topics:
        raw_path = Path(str(files[topic]))
        path = raw_path if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("topic_id") != topic:
            raise FormalStatisticsError(f"{path}: topic_id must equal {topic!r}")
        contract = payload.get("comparison_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("topic_role") != "locked"
            or contract.get("formal_results_used") is not True
            or contract.get("post_review_conditions")
            != list(POST_REVIEW_CONDITIONS)
        ):
            raise FormalStatisticsError(
                f"{path}: adaptive comparison contract is not locked formal output"
            )
        conditions = payload.get("conditions")
        if not isinstance(conditions, dict):
            raise FormalStatisticsError(f"{path}: conditions must be an object")
        _require_exact_keys(
            conditions, set(ADAPTIVE_CONDITIONS), label=f"{path} conditions"
        )
        for condition, raw in conditions.items():
            if not isinstance(raw, dict):
                raise FormalStatisticsError(f"{path}/{condition} must be an object")
            keys = {
                "items",
                "review_requests",
                "final_quality_passed",
                "auto_accepts",
                "unsafe_auto_accepts",
            }
            if not keys.issubset(raw):
                raise FormalStatisticsError(
                    f"{path}/{condition}: missing {sorted(keys - set(raw))}"
                )
            values = {
                key: _nonnegative_int(raw[key], label=f"{path}/{condition}.{key}")
                for key in keys
            }
            if values["items"] == 0:
                raise FormalStatisticsError(f"{path}/{condition}: items cannot be zero")
            if values["review_requests"] + values["auto_accepts"] != values["items"]:
                raise FormalStatisticsError(
                    f"{path}/{condition}: review_requests + auto_accepts must equal items"
                )
            if values["final_quality_passed"] > values["items"]:
                raise FormalStatisticsError(
                    f"{path}/{condition}: final_quality_passed exceeds items"
                )
            if values["unsafe_auto_accepts"] > values["auto_accepts"]:
                raise FormalStatisticsError(
                    f"{path}/{condition}: unsafe_auto_accepts exceeds auto_accepts"
                )
            if condition == BASELINE_ORIGINAL_CONDITION and (
                values["review_requests"] != 0
                or values["auto_accepts"] != values["items"]
                or values["unsafe_auto_accepts"]
                != values["items"] - values["final_quality_passed"]
            ):
                raise FormalStatisticsError(
                    f"{path}/{condition}: the untouched baseline must have zero "
                    "reviews, all items unreviewed, and every quality failure counted "
                    "as an unsafe unreviewed output"
                )
            counts_by_condition[condition][topic] = values

    def calculate(rows: list[dict[str, int]], metric: str) -> float:
        if metric == "rrr":
            return sum(row["review_requests"] for row in rows) / sum(
                row["items"] for row in rows
            )
        if metric == "fqpr":
            return sum(row["final_quality_passed"] for row in rows) / sum(
                row["items"] for row in rows
            )
        auto_accepts = sum(row["auto_accepts"] for row in rows)
        return (
            sum(row["unsafe_auto_accepts"] for row in rows) / auto_accepts
            if auto_accepts
            else math.nan
        )

    output: dict[str, Any] = {}
    for condition, topic_counts in counts_by_condition.items():
        rows = list(topic_counts.values())
        metrics = {}
        for metric in ("rrr", "fqpr", "unsafe_auto_accept_rate"):
            estimate = calculate(rows, metric)
            if math.isnan(estimate):
                interval: list[float] | None = None
                estimate_value: float | None = None
            else:
                interval = list(
                    _cluster_bootstrap(
                        {topic: [counts] for topic, counts in topic_counts.items()},
                        lambda sampled, m=metric: calculate(sampled, m),
                        samples=bootstrap_samples,
                        rng=rng,
                    )
                )
                estimate_value = estimate
            metrics[metric] = {
                "estimate": estimate_value,
                "cluster_bootstrap_95_ci": interval,
            }
        output[condition] = {
            "counts": dict(
                Counter(
                    {
                        key: sum(row[key] for row in rows)
                        for key in rows[0]
                    }
                )
            ),
            "metrics": metrics,
        }
    baseline = output[BASELINE_ORIGINAL_CONDITION]["metrics"]["fqpr"]["estimate"]
    if baseline is None:
        raise AssertionError("baseline_original final-quality pass rate is undefined")
    comparisons = {}
    for condition in POST_REVIEW_CONDITIONS:
        post = output[condition]["metrics"]["fqpr"]["estimate"]
        if post is None:
            raise AssertionError(f"{condition} final-quality pass rate is undefined")
        comparisons[condition] = {
            "baseline_original_quality_error_rate": 1 - baseline,
            "post_review_quality_error_rate": 1 - post,
            "absolute_quality_error_rate_reduction": post - baseline,
            "relative_quality_error_rate_reduction": (
                (post - baseline) / (1 - baseline) if baseline < 1 else None
            ),
        }
    return {
        "conditions": output,
        "original_to_post_review": comparisons,
    }


def analyze_formal_experiment(
    manifest_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260806,
) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise FormalStatisticsError("bootstrap_samples must be at least 100")
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise FormalStatisticsError("Statistics manifest must declare version=1")
    topics = manifest.get("topics")
    if (
        not isinstance(topics, list)
        or not topics
        or any(not isinstance(topic, str) or not topic.strip() for topic in topics)
        or len(topics) != len(set(topics))
    ):
        raise FormalStatisticsError("topics must be a non-empty unique string list")
    report_panels = _load_panels(
        manifest,
        manifest_path=manifest_path,
        field="report_comparisons",
        family="report",
        contract=REPORT_COMPARISONS,
        topics=topics,
    )
    graph_panels = _load_panels(
        manifest,
        manifest_path=manifest_path,
        field="graph_comparisons",
        family="graph",
        contract=GRAPH_COMPARISONS,
        topics=topics,
    )
    _validate_graph_pairing(graph_panels, topics)
    rng = Random(seed)
    panel_results = [
        _analyze_panel(panel, bootstrap_samples=bootstrap_samples, rng=rng)
        for panel in [*report_panels, *graph_panels]
    ]
    primary_p = {
        panel.comparison: _sign_flip_pvalue(
            list(result["topic_ucr_reductions"].values())
        )
        for panel, result in zip(graph_panels, panel_results[len(report_panels) :])
    }
    adjusted = holm_adjust(primary_p)
    holm = [
        {
            "comparison": name,
            "outcome": "UCR reduction (comparator minus Graph RAG)",
            "raw_p_value": primary_p[name],
            "holm_adjusted_p_value": adjusted[name],
            "reject_at_0_05": adjusted[name] <= 0.05,
        }
        for name in GRAPH_COMPARISONS
    ]
    adaptive = _load_adaptive(
        manifest,
        manifest_path=manifest_path,
        topics=topics,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    human = next(
        result for result in panel_results if result["comparison"] == "full_vs_human"
    )
    human_quality_gap = {
        "definition": {
            "completeness_difference": (
                "CiteWeave Full minus published human reference"
            ),
            "ucr_reduction": (
                "published human reference UCR minus CiteWeave Full UCR"
            ),
            "pairwise_preference_score": (
                "wins plus half of ties for CiteWeave Full, divided by comparisons"
            ),
        },
        "completeness_difference": human["effects"]["completeness_difference"],
        "ucr_reduction": human["effects"]["ucr_reduction"],
        "pairwise_preference_score": human["effects"]["preference_score"],
    }
    return {
        "version": 1,
        "topics": topics,
        "topic_clusters": len(topics),
        "bootstrap": {
            "method": "topic-cluster bootstrap",
            "samples": bootstrap_samples,
            "confidence_level": 0.95,
            "seed": seed,
        },
        "metric_provenance": (
            "UCR, completeness, and preferences use resolved Judge verdicts only; "
            "report-text exact matching is not used."
        ),
        "panels": panel_results,
        "human_quality_gap": human_quality_gap,
        "graph_primary_holm": holm,
        "adaptive": adaptive,
    }


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in summary["panels"]:
        for condition, metrics in panel["conditions"].items():
            for metric, result in metrics.items():
                rows.append(
                    {
                        "family": panel["family"],
                        "comparison": panel["comparison"],
                        "condition": condition,
                        "metric": metric,
                        "estimate": result["estimate"],
                        "ci_low": result["cluster_bootstrap_95_ci"][0],
                        "ci_high": result["cluster_bootstrap_95_ci"][1],
                    }
                )
        for metric, result in panel["effects"].items():
            rows.append(
                {
                    "family": panel["family"],
                    "comparison": panel["comparison"],
                    "condition": f"{panel['condition_a']} vs {panel['condition_b']}",
                    "metric": metric,
                    "estimate": result["estimate"],
                    "ci_low": result["cluster_bootstrap_95_ci"][0],
                    "ci_high": result["cluster_bootstrap_95_ci"][1],
                }
            )
        pairwise = panel["pairwise_for_condition_a"]
        for metric in ("win_rate", "tie_rate", "loss_rate"):
            result = pairwise[metric]
            rows.append(
                {
                    "family": panel["family"],
                    "comparison": panel["comparison"],
                    "condition": panel["condition_a"],
                    "metric": metric,
                    "estimate": result["estimate"],
                    "ci_low": result["cluster_bootstrap_95_ci"][0],
                    "ci_high": result["cluster_bootstrap_95_ci"][1],
                }
            )
    for condition, result in summary["adaptive"]["conditions"].items():
        for metric, value in result["metrics"].items():
            interval = value["cluster_bootstrap_95_ci"]
            rows.append(
                {
                    "family": "adaptive",
                    "comparison": "review_policy",
                    "condition": condition,
                    "metric": metric,
                    "estimate": value["estimate"],
                    "ci_low": interval[0] if interval else None,
                    "ci_high": interval[1] if interval else None,
                }
            )
    return rows


def render_english_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Formal Experiment Results",
        "",
        (
            f"Results cover {summary['topic_clusters']} topic clusters. All 95% "
            f"confidence intervals use {summary['bootstrap']['samples']:,} "
            "topic-cluster bootstrap replicates."
        ),
        "",
        "## Report and graph comparisons",
        "",
        "| Panel | Target vs comparator | Target UCR | Target completeness | Win / tie / loss | UCR reduction (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for panel in summary["panels"]:
        target = panel["conditions"][panel["condition_a"]]
        pairwise = panel["pairwise_for_condition_a"]
        effect = panel["effects"]["ucr_reduction"]
        lines.append(
            f"| {panel['comparison']} | {panel['condition_a']} vs "
            f"{panel['condition_b']} | {target['ucr']['estimate']:.3f} | "
            f"{target['completeness']['estimate']:.3f} | "
            f"{pairwise['wins']} / {pairwise['ties']} / {pairwise['losses']} | "
            f"{effect['estimate']:.3f} ({_format_ci(effect['cluster_bootstrap_95_ci'])}) |"
        )
    lines.extend(
        [
            "",
            "## Human-reference quality gap",
            "",
            (
                "The gap is defined as CiteWeave Full minus the published human "
                "reference for completeness; positive values favor CiteWeave."
            ),
            "",
            "| Metric | Estimate | Topic-cluster bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for key in ("completeness_difference", "ucr_reduction", "pairwise_preference_score"):
        value = summary["human_quality_gap"][key]
        lines.append(
            f"| {key.replace('_', ' ').title()} | {value['estimate']:.3f} | "
            f"{_format_ci(value['cluster_bootstrap_95_ci'])} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive review",
            "",
            "| Policy | Review request rate | Final quality pass rate | Unsafe auto-accept rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ADAPTIVE_CONDITIONS:
        metrics = summary["adaptive"]["conditions"][condition]["metrics"]

        def display(name: str, condition_metrics: dict[str, Any] = metrics) -> str:
            value = condition_metrics[name]["estimate"]
            return "NA" if value is None else f"{value:.3f}"

        lines.append(
            f"| {condition} | {display('rrr')} | {display('fqpr')} | "
            f"{display('unsafe_auto_accept_rate')} |"
        )
    lines.extend(
        [
            "",
            "### Original-to-post-review quality-error reduction",
            "",
            (
                "The untouched original candidate was independently evaluated once "
                "per case before any Human Proxy intervention."
            ),
            "",
            "| Policy | Original error rate | Final error rate | Absolute reduction | Relative reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in POST_REVIEW_CONDITIONS:
        comparison = summary["adaptive"]["original_to_post_review"][condition]
        relative = comparison["relative_quality_error_rate_reduction"]
        relative_text = "NA" if relative is None else f"{relative:.3f}"
        lines.append(
            f"| {condition} | "
            f"{comparison['baseline_original_quality_error_rate']:.3f} | "
            f"{comparison['post_review_quality_error_rate']:.3f} | "
            f"{comparison['absolute_quality_error_rate_reduction']:.3f} | "
            f"{relative_text} |"
        )
    lines.extend(
        [
            "",
            "## Holm-adjusted preregistered graph comparisons",
            "",
            "| Comparison | Raw p | Holm-adjusted p | Reject at 0.05 |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in summary["graph_primary_holm"]:
        lines.append(
            f"| {row['comparison']} | {row['raw_p_value']:.4f} | "
            f"{row['holm_adjusted_p_value']:.4f} | "
            f"{'Yes' if row['reject_at_0_05'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            (
                "UCR, completeness, and pairwise results were computed from resolved "
                "LLM-as-Judge records. No report-text exact-match proxy was used."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_formal_statistics(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formal_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = _csv_rows(summary)
    with (output_dir / "formal_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "comparison",
                "condition",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "graph_holm.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "outcome",
                "raw_p_value",
                "holm_adjusted_p_value",
                "reject_at_0_05",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["graph_primary_holm"])
    (output_dir / "formal_results.md").write_text(
        render_english_markdown(summary), encoding="utf-8"
    )
