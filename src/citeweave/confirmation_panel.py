from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .formal_request import build_messages, canonical_sha256, task_payload_sha256
from .graph_discovery import _load_graph, build_discovery_benchmark
from .graph_robustness import (
    baseline_checks,
    compare_variants,
    measure_variant,
    perturb_graph,
    registered_variants,
)
from .io import read_json, sha256_file, write_json, write_parquet
from .robust_graph_synthesis import (
    MEASUREMENT_TYPES,
    TASK_TYPES,
    build_topic_benchmark,
)
from .token_budget import CommandTokenizer, verify_api_usage_probe_artifact

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = (
    "src/citeweave/confirmation_panel.py",
    "src/citeweave/robust_graph_synthesis.py",
    "src/citeweave/graph_robustness.py",
    "src/citeweave/graph_discovery.py",
    "src/citeweave/formal_request.py",
    "src/citeweave/token_budget.py",
    "src/citeweave/io.py",
    "scripts/build_robust_graph_confirmation_panel.py",
    "scripts/run_robust_graph_confirmation_panel.py",
    "scripts/run_graph_discovery_experiment.py",
)


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _materialize_keyword_trends(workspace: Path, output_path: Path) -> dict[str, Any]:
    canonical = workspace / "canonical"
    visual = canonical / "visualization"
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    try:
        frame = connection.execute(
            f"""
            WITH top_keywords AS (
              SELECT keyword, occurrences AS global_documents
              FROM read_parquet('{_sql_path(visual / 'keyword_occurrences.parquet')}')
              WHERE keyword IS NOT NULL
              ORDER BY occurrences DESC, keyword
              LIMIT 15
            )
            SELECT works.year, keywords.keyword,
                   count(DISTINCT keywords.work_id) AS documents,
                   max(top_keywords.global_documents) AS global_documents
            FROM read_parquet('{_sql_path(canonical / 'keywords.parquet')}') keywords
            JOIN top_keywords USING (keyword)
            JOIN read_parquet('{_sql_path(canonical / 'works.parquet')}') works
              USING (work_id)
            WHERE works.year IS NOT NULL
            GROUP BY works.year, keywords.keyword
            ORDER BY global_documents DESC, keyword, year
            """
        ).df()
    finally:
        connection.close()
    write_parquet(output_path, frame)
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "rows": len(frame),
        "keywords": int(frame["keyword"].nunique()),
        "years": int(frame["year"].nunique()),
    }


def _verify_tokenizer(
    tokenizer_manifest_path: Path,
) -> tuple[dict[str, Any], CommandTokenizer]:
    manifest = read_json(tokenizer_manifest_path)
    if manifest.get("passed") is not True or manifest.get("model") != "deepseek-v4-pro":
        raise ValueError("Exact DeepSeek tokenizer manifest is not accepted")
    root = tokenizer_manifest_path.parent
    for artifact in manifest.get("artifacts") or []:
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Tokenizer artifact drift: {path}")
    tokenizer = CommandTokenizer(manifest, manifest_dir=root)
    if not tokenizer.verify(manifest.get("verification_probes") or [])["passed"]:
        raise ValueError("Exact tokenizer failed frozen raw-text probes")
    _, message_verification, reasons = verify_api_usage_probe_artifact(
        manifest, manifest_dir=root, tokenizer=tokenizer
    )
    if reasons or not message_verification["passed"]:
        raise ValueError("Exact tokenizer failed frozen provider-usage probes")
    return manifest, tokenizer


def _validate_selection(
    *,
    protocol_path: Path,
    protocol_freeze_path: Path,
    selection_path: Path,
    collection_manifest_path: Path,
    candidate_audit_path: Path,
    initialization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    protocol_sha = sha256_file(protocol_path)
    freeze = read_json(protocol_freeze_path)
    if freeze.get("sha256") != protocol_sha:
        raise ValueError("Confirmation protocol differs from its freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    selection = read_json(selection_path)
    if selection.get("status") != "selected_topics_ready_for_post_selection_freeze":
        raise ValueError("Human query-relevance selection is not complete")
    if selection.get("protocol_sha256") != protocol_sha:
        raise ValueError("Query selection belongs to another protocol")
    integrity = selection.get("integrity") or {}
    if (
        integrity.get("provider_responses_at_selection") != 0
        or integrity.get("perturbation_outcomes_at_selection") != 0
    ):
        raise ValueError("Selection was not completed before downstream outcomes")
    if selection.get("collection_manifest_sha256") != sha256_file(
        collection_manifest_path
    ):
        raise ValueError("Query selection is not bound to this review collection")
    collection = read_json(collection_manifest_path)
    if collection.get("protocol_sha256") != protocol_sha:
        raise ValueError("Review collection belongs to another protocol")
    if collection.get("candidate_audit_sha256") != sha256_file(candidate_audit_path):
        raise ValueError("Review collection is not bound to this candidate audit")
    audit = read_json(candidate_audit_path)
    initialization = read_json(initialization_path)
    if audit.get("initialization_sha256") != sha256_file(initialization_path):
        raise ValueError("Candidate audit is not bound to this initialization")
    if initialization.get("protocol_sha256") != protocol_sha:
        raise ValueError("Workspace initialization belongs to another protocol")
    target = int(protocol["selection"]["target_topics"])
    summaries = sorted(
        selection.get("candidate_summaries") or [], key=lambda row: row["priority"]
    )
    expected = [row["dataset_id"] for row in summaries if row.get("eligible")][:target]
    selected = selection.get("selected_topics")
    if (
        not isinstance(selected, list)
        or len(selected) != target
        or len(set(selected)) != target
        or selected != expected
    ):
        raise ValueError("Selected topics violate the frozen first-eight-eligible rule")
    return protocol, audit, initialization, selected


def _verify_workspace_sources(
    dataset_id: str,
    workspace: Path,
    *,
    audit_record: dict[str, Any],
    initialization_record: dict[str, Any],
) -> dict[str, str]:
    project = workspace / "project.yml"
    if (
        not project.is_file()
        or sha256_file(project) != initialization_record["project_sha256"]
    ):
        raise ValueError(f"Workspace project identity drift: {dataset_id}")
    hashes = {}
    for raw_path, expected in audit_record["source_hashes"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Selected canonical source drift: {dataset_id}: {path.name}")
        hashes[str(path.resolve())] = expected
    required = (
        workspace / "canonical/keywords.parquet",
        workspace / "canonical/works.parquet",
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"Selected workspace lacks prerequisite: {path}")
        hashes[str(path.resolve())] = sha256_file(path)
    return hashes


def _build_robustness_cases(
    *,
    dataset_id: str,
    workspace: Path,
    baseline_benchmark: dict[str, Any],
    trends_path: Path,
    output_dir: Path,
    selection_sha256: str,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    import pandas as pd

    tasks = baseline_benchmark["tasks"]
    if (
        len(tasks) != 5
        or {task["task_type"] for task in tasks} != set(MEASUREMENT_TYPES)
        or any(task["network"] != "keyword_cooccurrence" for task in tasks)
        or any(task["scale"] != "large" for task in tasks)
    ):
        raise ValueError("Baseline benchmark must contain five Large keyword phenomena")
    graph, _ = _load_graph(workspace, "keyword_cooccurrence", "large")
    trends = pd.read_parquet(trends_path)
    variants = registered_variants()
    baseline, base_partition = measure_variant(graph, tasks, trends, variants[0])
    checks = baseline_checks(baseline["measurements"], tasks)
    if not all(row["passed"] for row in checks):
        raise ValueError(f"Baseline arithmetic mismatch: {dataset_id}")
    results = []
    for spec in variants:
        started = time.perf_counter()
        if spec["family"] == "baseline":
            row, communities = baseline, base_partition
        else:
            changed = perturb_graph(graph, spec)
            row, communities = measure_variant(
                changed,
                tasks,
                trends,
                spec,
                precomputed_partition=(
                    base_partition if spec["family"] == "temporal_window" else None
                ),
            )
        payload = {
            **row,
            "dataset_id": dataset_id,
            "selection_sha256": selection_sha256,
            "protocol_sha256": protocol_sha256,
            "run_identity": canonical_sha256(
                {
                    "selection_sha256": selection_sha256,
                    "protocol_sha256": protocol_sha256,
                    "dataset_id": dataset_id,
                    "variant": spec,
                }
            ),
            "baseline_checks": checks,
            "comparison": compare_variants(
                baseline, row, base_partition, communities
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload["payload_sha256"] = canonical_sha256(payload)
        path = output_dir / f"{spec['variant_id']}.json"
        write_json(path, payload)
        results.append(
            {
                "variant_id": spec["variant_id"],
                "family": spec["family"],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    if len(results) != 18:
        raise AssertionError("Registered robustness grid must contain 18 cases")
    return results


def _freeze_request_identities(
    construction: dict[str, Any],
    *,
    conditions: tuple[str, ...],
    model: str,
    token_budget: int,
    tokenizer: CommandTokenizer,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    rows = []
    by_condition: dict[str, list[int]] = {condition: [] for condition in conditions}
    for record in construction["records"]:
        benchmark = read_json(Path(record["benchmark"]))
        for task in benchmark["tasks"]:
            for condition in conditions:
                messages, context_audit = build_messages(
                    task,
                    condition,
                    tokenizer=tokenizer,
                    token_budget=token_budget,
                )
                if condition != "no_reference" and (
                    context_audit["full_context_tokens"]
                    != context_audit["budgeted_context_tokens"]
                ):
                    raise ValueError(
                        f"Frozen context would require truncation: {task['item_id']}:{condition}"
                    )
                context_tokens = int(context_audit["budgeted_context_tokens"])
                if context_tokens > token_budget:
                    raise ValueError("A request exceeds the registered context token budget")
                request = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 2200,
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }
                prompt_tokens = tokenizer.count_messages(messages)
                by_condition[condition].append(prompt_tokens)
                rows.append(
                    {
                        "dataset_id": record["dataset_id"],
                        "item_id": task["item_id"],
                        "condition": condition,
                        "task_payload_sha256": task_payload_sha256(task),
                        "context_sha256": context_audit["context_sha256"],
                        "request_messages_sha256": canonical_sha256(messages),
                        "request_payload_sha256": canonical_sha256(request),
                        "context_tokens": context_tokens,
                        "prompt_tokens": prompt_tokens,
                    }
                )
    expected = int(construction["planned_calls"])
    identities = {(row["item_id"], row["condition"]) for row in rows}
    if len(rows) != expected or len(identities) != expected:
        raise ValueError("Request identity panel is incomplete or duplicated")
    summary = {
        condition: {
            "minimum_prompt_tokens": min(values),
            "maximum_prompt_tokens": max(values),
            "mean_prompt_tokens": sum(values) / len(values),
        }
        for condition, values in by_condition.items()
    }
    return rows, summary


def build_confirmation_panel(
    protocol_path: Path,
    protocol_freeze_path: Path,
    selection_path: Path,
    collection_manifest_path: Path,
    candidate_audit_path: Path,
    initialization_path: Path,
    tokenizer_manifest_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Build and freeze the held-out panel only after human query selection."""
    protocol, audit, initialization, selected = _validate_selection(
        protocol_path=protocol_path,
        protocol_freeze_path=protocol_freeze_path,
        selection_path=selection_path,
        collection_manifest_path=collection_manifest_path,
        candidate_audit_path=candidate_audit_path,
        initialization_path=initialization_path,
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite confirmation panel: {output_root}")
    tokenizer_manifest, tokenizer = _verify_tokenizer(tokenizer_manifest_path)
    conditions = tuple(protocol["benchmark"]["conditions"])
    expected_conditions = (
        "flat_bm25",
        "graph_hierarchical_retrieval_v2",
        "flat_program",
        "graph_program",
        "operator_only",
    )
    if conditions != expected_conditions:
        raise ValueError("Confirmation conditions differ from the registered panel")

    audit_records = {row["dataset_id"]: row for row in audit["records"]}
    initialized = {
        row["dataset_id"]: row for row in initialization["records"]
    }
    protocol_sha = sha256_file(protocol_path)
    selection_sha = sha256_file(selection_path)
    output_root.mkdir(parents=True, exist_ok=False)
    construction_records = []
    source_hashes: dict[str, str] = {}
    case_records = []
    rates: list[float] = []
    for dataset_id in selected:
        init_record = initialized.get(dataset_id)
        audit_record = audit_records.get(dataset_id)
        if init_record is None or audit_record is None:
            raise ValueError("Selected topic is absent from frozen candidate records")
        workspace = Path(init_record["workspace"])
        source_hashes.update(
            _verify_workspace_sources(
                dataset_id,
                workspace,
                audit_record=audit_record,
                initialization_record=init_record,
            )
        )
        prerequisite = _materialize_keyword_trends(
            workspace,
            output_root / "prerequisites" / dataset_id / "keyword_trends.parquet",
        )
        baseline = build_discovery_benchmark(
            workspace,
            output_root / "baseline_tasks" / dataset_id,
            networks=("keyword_cooccurrence",),
            scales=("large",),
            task_types=MEASUREMENT_TYPES,
            record_budget=int(protocol["benchmark"]["record_budget"]),
        )
        baseline_path = output_root / "baseline_tasks" / dataset_id / "benchmark.json"
        topic_cases = _build_robustness_cases(
            dataset_id=dataset_id,
            workspace=workspace,
            baseline_benchmark=baseline,
            trends_path=Path(prerequisite["path"]),
            output_dir=output_root / "robustness_cases" / dataset_id,
            selection_sha256=selection_sha,
            protocol_sha256=protocol_sha,
        )
        case_records.extend(topic_cases)
        benchmark = build_topic_benchmark(
            output_root / "robustness_cases" / dataset_id,
            record_budget=int(protocol["benchmark"]["record_budget"]),
        )
        if [task["task_type"] for task in benchmark["tasks"]] != list(TASK_TYPES):
            raise ValueError("Final robust benchmark task order or coverage differs")
        final_path = output_root / "benchmarks" / dataset_id / "benchmark.json"
        write_json(final_path, benchmark)
        for task in benchmark["tasks"]:
            rates.extend(
                float(value)
                for key, value in task["answer"].items()
                if key.endswith("_rate") and isinstance(value, (int, float))
            )
        construction_records.append(
            {
                "dataset_id": dataset_id,
                "workspace": str(workspace.resolve()),
                "baseline_benchmark": str(baseline_path.resolve()),
                "baseline_benchmark_sha256": sha256_file(baseline_path),
                "temporal_prerequisite": prerequisite,
                "robustness_cases": topic_cases,
                "benchmark": str(final_path.resolve()),
                "benchmark_sha256": sha256_file(final_path),
                "tasks": len(benchmark["tasks"]),
                "planned_calls": len(benchmark["tasks"]) * len(conditions),
            }
        )
    construction = {
        "schema_version": 1,
        "status": "confirmatory_panel_constructed_before_provider_outcomes",
        "confirmatory": True,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "topics": len(construction_records),
        "tasks": sum(row["tasks"] for row in construction_records),
        "conditions": len(conditions),
        "planned_calls": sum(row["planned_calls"] for row in construction_records),
        "robustness_cases": len(case_records),
        "difficulty_diagnostic": {
            "rate_fields": len(rates),
            "interior_rate_fields": sum(0.0 < value < 1.0 for value in rates),
            "minimum_rate": min(rates),
            "maximum_rate": max(rates),
            "distinct_rates": len(set(rates)),
        },
        "records": construction_records,
    }
    if (
        construction["topics"] != 8
        or construction["tasks"] != 40
        or construction["planned_calls"] != 200
        or construction["robustness_cases"] != 144
    ):
        raise AssertionError("Confirmatory panel size differs from the protocol")
    construction_path = output_root / "construction_manifest.json"
    write_json(construction_path, construction)
    request_rows, token_summary = _freeze_request_identities(
        construction,
        conditions=conditions,
        model=tokenizer_manifest["model"],
        token_budget=int(protocol["benchmark"]["context_token_budget"]),
        tokenizer=tokenizer,
    )
    request_manifest = {
        "schema_version": 1,
        "status": "all_confirmation_requests_hashed_before_provider_execution",
        "confirmatory": True,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "construction_manifest_sha256": sha256_file(construction_path),
        "tokenizer_manifest": str(tokenizer_manifest_path.resolve()),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
        "model": tokenizer_manifest["model"],
        "cells": len(request_rows),
        "token_summary": token_summary,
        "records": request_rows,
    }
    request_path = output_root / "request_identity_manifest.json"
    write_json(request_path, request_manifest)
    freeze_payload = {
        "schema_version": 1,
        "status": "ready_for_independent_confirmation_readiness_audit",
        "confirmatory": True,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "protocol_freeze_sha256": sha256_file(protocol_freeze_path),
        "selection": str(selection_path.resolve()),
        "selection_sha256": selection_sha,
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "candidate_audit_sha256": sha256_file(candidate_audit_path),
        "initialization_sha256": sha256_file(initialization_path),
        "construction_manifest": str(construction_path.resolve()),
        "construction_manifest_sha256": sha256_file(construction_path),
        "request_identity_manifest": str(request_path.resolve()),
        "request_identity_manifest_sha256": sha256_file(request_path),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
        "implementation_sha256": {
            relative: sha256_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "source_sha256": dict(sorted(source_hashes.items())),
        "baseline_benchmark_sha256": {
            row["dataset_id"]: row["baseline_benchmark_sha256"]
            for row in construction_records
        },
        "robustness_case_sha256": {
            f"{Path(row['path']).parent.name}/{row['variant_id']}": row["sha256"]
            for row in case_records
        },
        "benchmark_sha256": {
            row["dataset_id"]: row["benchmark_sha256"]
            for row in construction_records
        },
        "counts": {
            "topics": 8,
            "robustness_cases": 144,
            "tasks": 40,
            "request_identities": 200,
            "provider_responses": 0,
        },
    }
    freeze_path = output_root / "post_selection_input_freeze.json"
    write_json(freeze_path, freeze_payload)
    return freeze_payload
