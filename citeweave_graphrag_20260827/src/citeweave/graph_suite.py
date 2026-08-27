from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from .exceptions import ConfigurationError
from .graph_ablation import ABLATION_MODES, _freeze_case, run_graph_ablation
from .graph_explanation import alias_graph, displayed_graph, graph_rag_context
from .io import (
    atomic_write_bytes,
    load_config,
    read_json,
    save_config,
    sha256_file,
    write_json,
    write_jsonl,
)
from .models import (
    AcquisitionPolicy,
    GraphExplanationPolicy,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)
from .workflow import _load_analyses, _load_figures, run_project

SUITE_VERSION = "1.0"
FORMAL_METRICS = (
    "verified_slot_coverage",
    "edge_hallucination_rate",
    "claim_support_rate",
    "path_validity_rate",
    "verified_complex_claims",
    "abstention_rate",
    "prompt_tokens",
)


def load_graph_suite_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    topics = spec.get("topics") or []
    if not isinstance(topics, list) or not topics:
        raise ConfigurationError("Graph suite requires a non-empty topics list.")
    topic_ids = [str(topic.get("id") or "").strip() for topic in topics]
    if any(not topic_id for topic_id in topic_ids):
        raise ConfigurationError("Every graph-suite topic requires an id.")
    if len(set(topic_ids)) != len(topic_ids):
        raise ConfigurationError("Graph-suite topic ids must be unique.")
    for topic in topics:
        if not topic.get("title") or not topic.get("keywords"):
            raise ConfigurationError(
                f"Topic {topic.get('id')!r} requires title and keywords."
            )
    return spec


def _resolved_suite_root(spec_path: Path, spec: dict[str, Any], output: Path | None) -> Path:
    if output is not None:
        return output.resolve()
    configured = Path(str(spec.get("suite_root") or "runs/formal-graph-suite"))
    if configured.is_absolute():
        return configured
    return (spec_path.resolve().parent.parent / configured).resolve()


def _graph_policy(spec: dict[str, Any], *, mode: str) -> GraphExplanationPolicy:
    configured = spec.get("graph_explanation") or {}
    return GraphExplanationPolicy(
        mode=mode,
        networks=["keyword_cooccurrence"],
        model=str(configured.get("model") or "qwen3-vl-plus-2025-12-19"),
        base_url=str(
            configured.get("base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        api_key_env=str(configured.get("api_key_env") or "DASHSCOPE_API_KEY"),
        max_paths=int(configured.get("max_paths") or 8),
        max_hops=int(configured.get("max_hops") or 4),
        temperature=float(configured.get("temperature") or 0.0),
    )


def _project_config(spec: dict[str, Any], topic: dict[str, Any]) -> ProjectConfig:
    max_nodes = int(spec.get("max_nodes") or 50)
    label_budget = int(spec.get("label_budget") or min(40, max_nodes))
    protocol = SearchProtocol(
        title=str(topic["title"]),
        keywords=[str(value) for value in topic["keywords"]],
        year_from=int(spec.get("year_from") or 2020),
        year_to=int(spec.get("year_to") or 2025),
        source=SourceName(str(spec.get("source") or "europe_pmc")),
        query_mode=str(spec.get("query_mode") or "all"),
        max_records=topic.get("max_records") or spec.get("max_records"),
        include_references=bool(spec.get("include_references", False)),
        notes=(
            "Pre-registered unseen topic for CiteWeave multi-topic graph-explanation "
            f"suite {spec.get('name') or 'formal-suite'}."
        ),
    )
    return ProjectConfig(
        project_id=str(topic["id"]),
        protocol=protocol,
        acquisition=AcquisitionPolicy(mode="standard"),
        graph_explanation=_graph_policy(spec, mode="disabled"),
        visualization_max_nodes=max_nodes,
        visualization_label_budget=label_budget,
        random_seed=int(spec.get("random_seed") or 42),
    )


def _project_ready(project: Path) -> bool:
    required = (
        project / "project.yml",
        project / "canonical" / "works.parquet",
        project / "analyses" / "network_keyword_cooccurrence_nodes.parquet",
        project / "analyses" / "network_keyword_cooccurrence_edges.parquet",
        project / "figures" / "network_keyword_cooccurrence.png",
    )
    return all(path.exists() for path in required)


def _work_ids(project: Path) -> set[str]:
    works = pd.read_parquet(project / "canonical" / "works.parquet", columns=["work_id"])
    return {str(value) for value in works["work_id"].dropna().tolist()}


def _corpus_hash(work_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(work_ids)).encode("utf-8")).hexdigest()


def _inspect_project(
    project: Path, topic_id: str, spec: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    paths = ProjectPaths(project)
    analyses = _load_analyses(paths)
    figures = _load_figures(paths)
    network = analyses.networks.get("keyword_cooccurrence")
    figure = next(
        (item for item in figures if item.name == "network_keyword_cooccurrence"), None
    )
    if network is None or figure is None:
        raise ConfigurationError(f"Missing keyword graph artifacts in {project}")
    max_nodes = int(spec.get("max_nodes") or 50)
    graph, lookup = displayed_graph(network, max_nodes)
    id_to_alias, _, _, _ = alias_graph(graph, lookup)
    retrieval = graph_rag_context(
        graph,
        lookup,
        id_to_alias,
        max_paths=int((spec.get("graph_explanation") or {}).get("max_paths") or 8),
        max_hops=int((spec.get("graph_explanation") or {}).get("max_hops") or 4),
    )
    communities = {int(data.get("cluster") or 0) for data in lookup.values()}
    thresholds = spec.get("eligibility") or {}
    checks = {
        "minimum_nodes": graph.number_of_nodes()
        >= int(thresholds.get("minimum_nodes") or 40),
        "minimum_edges": graph.number_of_edges()
        >= int(thresholds.get("minimum_edges") or 40),
        "minimum_communities": len(communities)
        >= int(thresholds.get("minimum_communities") or 2),
        "minimum_cross_edges": len(retrieval["cross_community_edges"])
        >= int(thresholds.get("minimum_cross_edges") or 1),
        "minimum_multi_hop_paths": len(retrieval["multi_hop_paths"])
        >= int(thresholds.get("minimum_multi_hop_paths") or 2),
    }
    work_ids = _work_ids(project)
    case = _freeze_case(network, figure, max_nodes)
    return (
        {
            "topic_id": topic_id,
            "project": str(project),
            "documents": len(work_ids),
            "corpus_sha256": _corpus_hash(work_ids),
            "case_id": case["case_id"],
            "figure_sha256": case["figure_sha256"],
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "communities": len(communities),
            "available_cross_edges": len(retrieval["cross_community_edges"]),
            "available_multi_hop_paths": len(retrieval["multi_hop_paths"]),
            "available_articulation_candidates": len(
                retrieval["articulation_candidates"]
            ),
            "available_hub_candidates": len(retrieval.get("hub_candidates") or []),
            "eligibility_checks": checks,
            "eligible": all(checks.values()),
        },
        work_ids,
    )


def _overlap_rows(
    project_rows: list[dict[str, Any]], work_ids: dict[str, set[str]]
) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(project_rows, 2):
        left_ids = work_ids[left["topic_id"]]
        right_ids = work_ids[right["topic_id"]]
        union = left_ids | right_ids
        intersection = left_ids & right_ids
        rows.append(
            {
                "left_topic": left["topic_id"],
                "right_topic": right["topic_id"],
                "shared_documents": len(intersection),
                "jaccard": round(len(intersection) / len(union), 6) if union else 0.0,
            }
        )
    return rows


def prepare_graph_suite(
    spec_path: Path,
    *,
    output: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build unseen CiteWeave projects without making graph-model API calls."""

    spec_path = spec_path.resolve()
    spec = load_graph_suite_spec(spec_path)
    suite_root = _resolved_suite_root(spec_path, spec, output)
    suite_root.mkdir(parents=True, exist_ok=True)
    topics = list(spec["topics"][:limit] if limit else spec["topics"])
    preparation_records: list[dict[str, Any]] = []
    project_rows: list[dict[str, Any]] = []
    work_ids: dict[str, set[str]] = {}
    enabled_policy = _graph_policy(spec, mode="graph_rag")

    for topic in topics:
        topic_id = str(topic["id"])
        project = suite_root / "projects" / topic_id
        status = "reused"
        preparation_warning = None
        try:
            if not _project_ready(project):
                if project.exists() and any(project.iterdir()):
                    raise ConfigurationError(
                        f"Partial project requires inspection before retry: {project}"
                    )
                config = _project_config(spec, topic)
                try:
                    run_project(
                        project,
                        config,
                        use_llm=False,
                        review_rounds=0,
                        allow_truncated=False,
                    )
                    status = "created"
                except Exception as exc:
                    if not _project_ready(project):
                        raise
                    status = "created_core_artifacts"
                    preparation_warning = (
                        "The full report export failed after all graph artifacts were "
                        f"saved: {type(exc).__name__}: {str(exc)[:500]}"
                    )
            config = load_config(project / "project.yml")
            save_config(
                project / "project.yml",
                config.model_copy(update={"graph_explanation": enabled_policy}),
            )
            inspected, ids = _inspect_project(project, topic_id, spec)
            inspected["preparation_status"] = status
            inspected["preparation_warning"] = preparation_warning
            project_rows.append(inspected)
            work_ids[topic_id] = ids
            preparation_records.append(
                {"topic_id": topic_id, "status": status, "project": str(project)}
            )
        except Exception as exc:  # noqa: BLE001 - retain auditable partial suite state.
            preparation_records.append(
                {
                    "topic_id": topic_id,
                    "status": "error",
                    "project": str(project),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1_000],
                }
            )

    overlaps = _overlap_rows(project_rows, work_ids)
    max_jaccard = max((row["jaccard"] for row in overlaps), default=0.0)
    overlap_limit = float((spec.get("eligibility") or {}).get("max_corpus_jaccard") or 0.2)
    eligible_rows = [row for row in project_rows if row["eligible"]]
    unique_cases = len({row["case_id"] for row in eligible_rows})
    manifest = {
        "suite_version": SUITE_VERSION,
        "prepared_at": datetime.now(UTC),
        "name": spec.get("name") or suite_root.name,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "suite_root": str(suite_root),
        "topics_requested": len(topics),
        "projects_prepared": len(project_rows),
        "eligible_topics": len(eligible_rows),
        "unique_case_ids": unique_cases,
        "maximum_corpus_jaccard": max_jaccard,
        "corpus_overlap_limit": overlap_limit,
        "overlap_check_passed": max_jaccard <= overlap_limit,
        "projects": project_rows,
        "preparation_records": preparation_records,
    }
    write_json(suite_root / "preparation_manifest.json", manifest)
    pd.DataFrame(project_rows).drop(columns=["eligibility_checks"], errors="ignore").to_csv(
        suite_root / "projects.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(overlaps).to_csv(
        suite_root / "corpus_overlap.csv", index=False, encoding="utf-8-sig"
    )
    return {
        "suite_root": str(suite_root),
        "topics_requested": len(topics),
        "projects_prepared": len(project_rows),
        "eligible_topics": len(eligible_rows),
        "unique_case_ids": unique_cases,
        "maximum_corpus_jaccard": max_jaccard,
        "overlap_check_passed": manifest["overlap_check_passed"],
        "manifest": str(suite_root / "preparation_manifest.json"),
    }


def _mean_metric(records: list[dict[str, Any]], metric: str) -> float | None:
    values = []
    for record in records:
        if metric == "prompt_tokens":
            value = (record.get("usage") or {}).get("prompt_tokens")
        else:
            value = (record.get("metrics") or {}).get(metric)
        if value is not None:
            values.append(float(value))
    return round(fmean(values), 6) if values else None


def _bootstrap_ci(values: list[float], seed: int, samples: int = 10_000) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = round(float(values[0]), 6)
        return [value, value]
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [round(float(lower), 6), round(float(upper), 6)]


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    valid = [
        (index, float(row["p_value"]))
        for index, row in enumerate(rows)
        if row["p_value"] is not None
    ]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid):
        adjusted = min(1.0, value * (total - rank))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = round(running, 6)


def aggregate_graph_suite(output: Path, *, seed: int = 42) -> dict[str, Any]:
    output = output.resolve()
    records_path = output / "records.jsonl"
    if not records_path.exists():
        raise ConfigurationError(f"Missing suite records: {records_path}")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [record for record in records if record.get("status") == "complete"]
    topic_mode_rows: list[dict[str, Any]] = []
    topic_ids = sorted({str(record["topic_id"]) for record in records})
    for topic_id in topic_ids:
        for mode in ABLATION_MODES:
            group = [
                record
                for record in successful
                if record["topic_id"] == topic_id and record["mode"] == mode
            ]
            row: dict[str, Any] = {
                "topic_id": topic_id,
                "mode": mode,
                "successful_repeats": len(group),
                "case_id": group[0]["case_id"] if group else None,
            }
            for metric in FORMAL_METRICS:
                row[metric] = _mean_metric(group, metric)
            topic_mode_rows.append(row)
    topic_mode = pd.DataFrame(topic_mode_rows)
    topic_mode.to_csv(output / "topic_mode_summary.csv", index=False, encoding="utf-8-sig")

    aggregate_rows: list[dict[str, Any]] = []
    for mode in ABLATION_MODES:
        for metric in FORMAL_METRICS:
            values = [
                float(row[metric])
                for row in topic_mode_rows
                if row["mode"] == mode and row.get(metric) is not None
            ]
            aggregate_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "graphs": len(values),
                    "mean": round(fmean(values), 6) if values else None,
                    "median": round(median(values), 6) if values else None,
                    "std": round(stdev(values), 6) if len(values) > 1 else 0.0,
                    "bootstrap_ci_95": _bootstrap_ci(values, seed),
                }
            )

    paired_rows: list[dict[str, Any]] = []
    paired_stats: list[dict[str, Any]] = []
    lookup = {(row["topic_id"], row["mode"]): row for row in topic_mode_rows}
    for reference in ("vlm", "flat_kg"):
        for metric in FORMAL_METRICS:
            differences = []
            for topic_id in topic_ids:
                graph_rag = lookup.get((topic_id, "graph_rag"), {}).get(metric)
                baseline = lookup.get((topic_id, reference), {}).get(metric)
                if graph_rag is None or baseline is None:
                    continue
                difference = round(float(graph_rag) - float(baseline), 6)
                differences.append(difference)
                paired_rows.append(
                    {
                        "topic_id": topic_id,
                        "reference": reference,
                        "metric": metric,
                        "graph_rag": graph_rag,
                        "baseline": baseline,
                        "difference": difference,
                    }
                )
            p_value = None
            if len(differences) >= 2 and any(value != 0 for value in differences):
                try:
                    p_value = round(
                        float(wilcoxon(differences, alternative="two-sided").pvalue), 6
                    )
                except ValueError:
                    p_value = None
            paired_stats.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "graphs": len(differences),
                    "mean_difference": (
                        round(fmean(differences), 6) if differences else None
                    ),
                    "bootstrap_ci_95": _bootstrap_ci(differences, seed + 1),
                    "p_value": p_value,
                    "p_value_holm": None,
                }
            )
    _holm_adjust(paired_stats)
    pd.DataFrame(aggregate_rows).to_csv(
        output / "aggregate_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired_rows).to_csv(
        output / "paired_differences.csv", index=False, encoding="utf-8-sig"
    )
    write_json(output / "statistics.json", {"groups": aggregate_rows, "paired": paired_stats})

    def group_value(mode: str, metric: str) -> str:
        row = next(
            item
            for item in aggregate_rows
            if item["mode"] == mode and item["metric"] == metric
        )
        return "—" if row["mean"] is None else f"{float(row['mean']) * 100:.1f}%"

    lines = [
        "# CiteWeave 多主题图解释正式消融",
        "",
        f"- 独立主题图：{len(topic_ids)}",
        f"- 成功调用：{len(successful)}/{len(records)}",
        "- 统计单位：主题图；每张图内先平均重复调用，再进行图间配对比较。",
        "",
        "| 方法 | 有效槽位覆盖 | 边幻觉率 | 声明支持率 | 路径有效率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ABLATION_MODES:
        lines.append(
            f"| {mode} | {group_value(mode, 'verified_slot_coverage')} | "
            f"{group_value(mode, 'edge_hallucination_rate')} | "
            f"{group_value(mode, 'claim_support_rate')} | "
            f"{group_value(mode, 'path_validity_rate')} |"
        )
    lines.extend(
        [
            "",
            (
                "配对差值、图级bootstrap 95%置信区间与Wilcoxon检验见 "
                "`statistics.json`；未经图级聚合的逐次输出不作为独立样本。"
            ),
            "",
        ]
    )
    atomic_write_bytes(output / "summary.md", "\n".join(lines).encode("utf-8"))
    return {
        "output": str(output),
        "independent_topics": len(topic_ids),
        "successful_runs": len(successful),
        "total_runs": len(records),
        "summary": str(output / "summary.md"),
        "topic_mode_summary": str(output / "topic_mode_summary.csv"),
        "statistics": str(output / "statistics.json"),
    }


def run_graph_suite(
    spec_path: Path,
    *,
    suite_root: Path | None = None,
    output: Path | None = None,
    repeats: int | None = None,
    allow_underpowered: bool = False,
) -> dict[str, Any]:
    """Run a frozen three-way ablation across independent CiteWeave topics."""

    spec_path = spec_path.resolve()
    spec = load_graph_suite_spec(spec_path)
    root = _resolved_suite_root(spec_path, spec, suite_root)
    preparation_path = root / "preparation_manifest.json"
    if not preparation_path.exists():
        raise ConfigurationError(
            f"Prepare the suite before model calls: missing {preparation_path}"
        )
    preparation = read_json(preparation_path)
    if preparation.get("spec_sha256") != sha256_file(spec_path):
        raise ConfigurationError("Suite specification changed after preparation.")
    if not preparation.get("overlap_check_passed"):
        raise ConfigurationError("Corpus overlap gate failed; inspect corpus_overlap.csv.")
    projects = [row for row in preparation["projects"] if row.get("eligible")]
    minimum_topics = int(spec.get("minimum_topics") or 10)
    if len(projects) < minimum_topics and not allow_underpowered:
        raise ConfigurationError(
            f"Only {len(projects)} eligible topics; formal minimum is {minimum_topics}."
        )
    if len({row["case_id"] for row in projects}) != len(projects):
        raise ConfigurationError("Duplicate graph case ids detected across topics.")
    api_key_env = _graph_policy(spec, mode="graph_rag").api_key_env
    if not os.getenv(api_key_env):
        raise ConfigurationError(
            f"{api_key_env} must be set before the formal model-call stage."
        )

    active_repeats = int(repeats or spec.get("repeats") or 3)
    if active_repeats < 1:
        raise ConfigurationError("repeats must be at least 1")
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = root / "experiments" / f"formal-graph-suite-{timestamp}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError(f"Suite output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "suite_version": SUITE_VERSION,
        "started_at": datetime.now(UTC),
        "name": spec.get("name") or root.name,
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "preparation_manifest_sha256": sha256_file(preparation_path),
        "suite_code_sha256": sha256_file(Path(__file__)),
        "statistical_unit": "independent_topic_graph",
        "repeats_nested_within_graph": active_repeats,
        "minimum_topics": minimum_topics,
        "topics": [row["topic_id"] for row in projects],
        "expected_runs": len(projects) * len(ABLATION_MODES) * active_repeats,
    }
    write_json(output / "run_manifest.json", manifest)
    atomic_write_bytes(output / "frozen_suite.yml", spec_path.read_bytes())
    records: list[dict[str, Any]] = []
    project_results: list[dict[str, Any]] = []
    for project_row in projects:
        topic_id = str(project_row["topic_id"])
        project = Path(project_row["project"])
        try:
            inspected, _ = _inspect_project(project, topic_id, spec)
            for field in ("case_id", "corpus_sha256", "figure_sha256"):
                if inspected[field] != project_row[field]:
                    raise ConfigurationError(
                        f"Frozen project changed for {topic_id}: {field} mismatch."
                    )
            paths = ProjectPaths(project)
            analyses = _load_analyses(paths)
            figures = _load_figures(paths)
            config = load_config(project / "project.yml")
            project_output = output / "projects" / topic_id
            result = run_graph_ablation(
                analyses,
                figures,
                config.graph_explanation,
                max_nodes=config.visualization_max_nodes,
                repeats=active_repeats,
                output=project_output,
            )
            project_results.append({"topic_id": topic_id, "status": "complete", **result})
            for line in (project_output / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                record = json.loads(line)
                record.update(
                    {
                        "topic_id": topic_id,
                        "project_id": config.project_id,
                        "corpus_sha256": project_row["corpus_sha256"],
                    }
                )
                records.append(record)
        except Exception as exc:  # noqa: BLE001 - continue other frozen topics.
            project_results.append(
                {
                    "topic_id": topic_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1_000],
                }
            )
        write_jsonl(output / "records.jsonl", records)
        write_json(output / "project_results.json", project_results)
        manifest["topics_finished"] = len(project_results)
        manifest["runs_persisted"] = len(records)
        write_json(output / "run_manifest.json", manifest)
    aggregate = aggregate_graph_suite(output, seed=int(spec.get("random_seed") or 42))
    manifest["finished_at"] = datetime.now(UTC)
    manifest["actual_runs"] = len(records)
    manifest["successful_runs"] = sum(
        record.get("status") == "complete" for record in records
    )
    manifest["failed_projects"] = sum(
        result["status"] != "complete" for result in project_results
    )
    manifest["formal_completion_gate"] = bool(
        len(projects) >= minimum_topics
        and manifest["successful_runs"] == manifest["expected_runs"]
        and manifest["failed_projects"] == 0
    )
    write_json(output / "run_manifest.json", manifest)
    return {**aggregate, "formal_completion_gate": manifest["formal_completion_gate"]}
