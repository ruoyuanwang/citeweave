"""Construct time-allocation witnesses for static-context temporal questions.

This is an information-sufficiency diagnostic, not an accuracy experiment. All
work-keyword incidences stay fixed when years are permuted, so the all-period
graph stays fixed while temporal ground truth can change.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from citeweave.formal_request import build_messages, canonical_sha256
from citeweave.graph_discovery import _load_graph
from citeweave.graph_robustness import (
    baseline_checks,
    load_verified_trends,
    measure_temporal,
    partition,
    registered_variants,
)
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
CONDITIONS = (
    "flat_hybrid",
    "flat_tfidf",
    "flat_bm25",
    "flat_lsa",
    "graph_query_retrieval",
    "graph_hierarchical_retrieval_v2",
)


def swap_windows(trends):
    years = sorted(int(y) for y in trends.year.dropna().unique())
    if len(years) < 6:
        raise ValueError("At least six non-overlapping observed years required")
    early, recent = years[:3], years[-3:]
    mapping = dict(zip(early, recent, strict=True)) | dict(zip(recent, early, strict=True))
    changed = trends.copy()
    changed["year"] = changed.year.map(lambda year: mapping.get(int(year), int(year)))
    return changed, mapping


def witness(graph, communities, trends, task):
    original = measure_temporal(graph, communities, trends, 3)
    if not baseline_checks({"temporal_structural_shift": original}, [task])[0]["passed"]:
        raise ValueError("Temporal baseline differs from benchmark")
    permuted, mapping = swap_windows(trends)
    alternative = measure_temporal(graph, communities, permuted, 3)
    if alternative["status"] != "measured":
        raise ValueError("Counterexample must remain measurable")
    new_answer = {
        key: round(alternative[key], 6) if isinstance(alternative[key], float) else alternative[key]
        for key in task["answer"]
    }
    fields = [key for key, value in task["answer"].items() if value != new_answer[key]]
    altered_task = copy.deepcopy(task)
    altered_task["answer"] = new_answer
    # These six conditions never serialize operator_trace. The hidden answer
    # values are likewise absent: only answer-field names enter the request.
    messages = []
    for condition in CONDITIONS:
        first, _ = build_messages(task, condition)
        second, _ = build_messages(altered_task, condition)
        messages.append(
            {
                "condition": condition,
                "identical_public_messages": first == second,
                "message_sha256": canonical_sha256(first),
                "alternative_message_sha256": canonical_sha256(second),
            }
        )
    return {
        "item_id": task["item_id"],
        "scale": task["scale"],
        "year_permutation": mapping,
        "original_answer": task["answer"],
        "alternative_answer": new_answer,
        "changed_answer_fields": fields,
        "conditions": messages,
        "temporal_answer_not_identifiable_from_static_context": bool(fields)
        and all(row["identical_public_messages"] for row in messages),
        "all_period_keyword_document_totals_preserved": trends.groupby("keyword")
        .documents.sum()
        .equals(permuted.groupby("keyword").documents.sum()),
        "witness_scope": "Global publication-year reassignment; work-keyword incidence and all-period graph unchanged.",
        "message_check_scope": "Before token truncation; identical inputs remain identical under the same deterministic truncator.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    construction_path = (
        BASE / "formal_v3_complexity_extension_benchmarks/construction_manifest.json"
    )
    construction = read_json(construction_path)
    records, sources = [], {str(construction_path): sha256_file(construction_path)}
    for row in construction["records"]:
        topic = row["dataset_id"]
        workspace = (
            ROOT
            / "experiments/formal_v3_identity_corrected_workspaces"
            / row["source_panel"]
            / topic
        )
        old_root = (
            "formal_v3_workspaces"
            if row["source_panel"] == "primary"
            else "formal_v3_replication_workspaces"
        )
        trends_path = ROOT / "experiments" / old_root / topic / "analyses/keyword_trends.parquet"
        trends = load_verified_trends(workspace, trends_path)
        benchmark_path = construction_path.parent / topic / "benchmark.json"
        if sha256_file(benchmark_path) != row["benchmark_sha256"]:
            raise ValueError("Frozen extension benchmark drift")
        sources[str(benchmark_path)] = sha256_file(benchmark_path)
        sources[str(trends_path)] = sha256_file(trends_path)
        for relative in (
            "canonical/keywords.parquet",
            "canonical/works.parquet",
            "canonical/visualization/keyword_occurrences.parquet",
            "canonical/visualization/keyword_cooccurrence_edges.parquet",
        ):
            sources[str(workspace / relative)] = sha256_file(workspace / relative)
        for task in read_json(benchmark_path)["tasks"]:
            if task["task_type"] != "temporal_structural_shift":
                continue
            graph, _ = _load_graph(workspace, "keyword_cooccurrence", task["scale"])
            communities = partition(graph, registered_variants()[0])
            records.append({"dataset_id": topic, **witness(graph, communities, trends, task)})
        print(json.dumps({"topic": topic, "checked": len(records)}), flush=True)
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/citeweave/graph_robustness.py",
        ROOT / "src/citeweave/graph_discovery.py",
        ROOT / "src/citeweave/formal_request.py",
    ):
        sources[str(path)] = sha256_file(path)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "information_sufficiency_diagnostic",
            "tasks": len(records),
            "witnesses": sum(
                r["temporal_answer_not_identifiable_from_static_context"] for r in records
            ),
            "checked_static_conditions": list(CONDITIONS),
            "source_sha256": sources,
            "records": records,
            "limitations": [
                "No provider calls and no changes to formal artifacts.",
                "Neural sidecars are still building and are not counted among the six audited conditions.",
                "Does not invalidate non-temporal tasks; temporal method gaps confound access to time information with reasoning.",
                "Equal-token budgets do not repair missing information. A shared time-evidence control is needed for temporal attribution.",
            ],
        },
    )


if __name__ == "__main__":
    main()
