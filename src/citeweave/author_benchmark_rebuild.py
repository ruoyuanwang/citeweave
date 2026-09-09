"""Prospective author-only benchmark replacement with unchanged-layer proofs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .author_graph_sensitivity import SparseAuthorGraph, task_arithmetic
from .canonical_identity_audit import _contains_identity
from .graph_discovery import build_discovery_benchmark
from .harvest_acceptance import verify_bulk_harvest
from .io import read_json, sha256_file, write_json
from .operator_verification import verify_benchmark_operator_replay
from .processing_acceptance import verify_large_processing

UNCHANGED_GRAPH_FILES = (
    "keyword_occurrences.parquet",
    "keyword_cooccurrence_edges.parquet",
    "institution_productivity.parquet",
    "institution_collaboration_edges.parquet",
)


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def replace_author_tasks(
    original: dict[str, Any], component: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_author = {t["item_id"]: t for t in original["tasks"] if t["network"] == "coauthorship"}
    new_author = {t["item_id"]: t for t in component["tasks"]}
    if len(old_author) != 5 or len(component["tasks"]) != 5 or set(old_author) != set(new_author):
        raise ValueError(
            "All five registered author tasks must be replaced without dropping or adding IDs"
        )
    if any(t["network"] != "coauthorship" or t["scale"] != "large" for t in new_author.values()):
        raise ValueError("Author replacement escaped registered network/scale")
    if _contains_identity(
        component,
        {"openalex-author:None", "openalex-author:", "openalex-author:null", "openalex-author:nan"},
    ):
        raise ValueError("Rebuilt author task still contains a placeholder identity")
    rebuilt = copy.deepcopy(original)
    rebuilt["tasks"] = [copy.deepcopy(new_author.get(t["item_id"], t)) for t in original["tasks"]]
    scales = {(r["network"], r["name"]): r for r in component["scales"]}
    if set(scales) != {("coauthorship", "large")}:
        raise ValueError("Expected exactly one author graph scale")
    rebuilt["scales"] = [
        copy.deepcopy(scales.get((r["network"], r["name"]), r)) for r in original["scales"]
    ]
    unchanged = [t for t in original["tasks"] if t["network"] != "coauthorship"]
    after = [t for t in rebuilt["tasks"] if t["network"] != "coauthorship"]
    if unchanged != after or len(unchanged) != 20 or len(rebuilt["tasks"]) != 25:
        raise ValueError("Unchanged task content/count invariant failed")
    return rebuilt, {
        "unchanged_tasks": len(unchanged),
        "rebuilt_author_tasks": len(new_author),
        "unchanged_task_content_sha256": object_hash(unchanged),
        "author_task_hash_mapping": [
            {
                "item_id": key,
                "old_sha256": object_hash(old_author[key]),
                "new_sha256": object_hash(new_author[key]),
            }
            for key in sorted(old_author)
        ],
    }


def rebuild_author_benchmark(
    *,
    original_workspace: Path,
    corrected_workspace: Path,
    original_benchmark: Path,
    output: Path,
    author_task_types: tuple[str, ...],
    amendment_hash: str,
) -> dict[str, Any]:
    if (output / "benchmark.json").exists():
        raise ValueError(f"Refusing to overwrite rebuilt benchmark: {output}")
    verification_path = corrected_workspace / "identity_correction_verification.json"
    verification = read_json(verification_path)
    if verification.get("status") != "reprocessed_not_promoted" or any(
        r["row_multiset_differences"] for r in verification["unaffected_tables"].values()
    ):
        raise ValueError("Corrected workspace lacks passed non-author invariants")
    lineage = verification["lineage"]
    if (
        lineage["amendment_sha256"] != amendment_hash
        or Path(lineage["source_workspace"]).resolve() != original_workspace.resolve()
    ):
        raise ValueError("Author correction lineage mismatch")
    original = read_json(original_benchmark)
    source_hashes = {
        str(original_benchmark.resolve()): sha256_file(original_benchmark),
        str(verification_path.resolve()): sha256_file(verification_path),
    }
    for name, row in verification["unaffected_tables"].items():
        for root, expected in (
            (original_workspace, row["old_sha256"]),
            (corrected_workspace, row["new_sha256"]),
        ):
            path = root / "canonical" / name
            if sha256_file(path) != expected:
                raise ValueError(f"Canonical input drift: {path}")
            source_hashes[str(path.resolve())] = expected
    graph_checks = {}
    for name in UNCHANGED_GRAPH_FILES:
        old = original_workspace / "canonical" / "visualization" / name
        new = corrected_workspace / "canonical" / "visualization" / name
        old_frame, new_frame = pd.read_parquet(old), pd.read_parquet(new)
        if not old_frame.equals(new_frame):
            raise ValueError(f"Unchanged graph table content/order differs: {name}")
        graph_checks[name] = {
            "ordered_rows_equal": True,
            "rows": len(old_frame),
            "old_sha256": sha256_file(old),
            "new_sha256": sha256_file(new),
        }
        source_hashes[str(old.resolve())] = sha256_file(old)
        source_hashes[str(new.resolve())] = sha256_file(new)
    temporal = original_workspace / "analyses" / "keyword_trends.parquet"
    target_temporal = corrected_workspace / "analyses" / temporal.name
    if target_temporal.exists() and sha256_file(target_temporal) != sha256_file(temporal):
        raise ValueError("Temporal prerequisite differs")
    if not target_temporal.exists():
        target_temporal.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporal, target_temporal)
    source_hashes[str(temporal.resolve())] = sha256_file(temporal)
    source_hashes[str(target_temporal.resolve())] = sha256_file(target_temporal)
    harvest = verify_bulk_harvest(corrected_workspace)
    processing = verify_large_processing(corrected_workspace)
    if not harvest["passed"] or not processing["passed"]:
        raise ValueError("Fresh harvest/processing acceptance failed")
    component = build_discovery_benchmark(
        corrected_workspace,
        output / "components" / "corrected_author_large",
        networks=("coauthorship",),
        scales=("large",),
        task_types=author_task_types,
        record_budget=original["design"]["record_budget"],
    )
    rebuilt, comparisons = replace_author_tasks(original, component)
    replay = verify_benchmark_operator_replay(component, workspace=corrected_workspace)
    if replay["invalid_tasks"]:
        raise ValueError("Rebuilt author operator replay failed")
    graph_path = corrected_workspace / "canonical" / "visualization" / "coauthor_edges.parquet"
    graph = SparseAuthorGraph(pd.read_parquet(graph_path))
    arithmetic = []
    for task in component["tasks"]:
        if task["task_type"] not in {
            "multi_hop_connector",
            "bridge_counterfactual",
            "hub_removal_resilience",
        }:
            continue
        observed = task_arithmetic(graph, task)
        passed = observed is not None and all(
            a == task["answer"][key]
            if a is None or task["answer"][key] is None
            else math.isclose(float(a), float(task["answer"][key]), abs_tol=1e-6)
            for key, a in (observed or {}).items()
        )
        arithmetic.append({"item_id": task["item_id"], "passed": passed, "computed": observed})
    if len(arithmetic) != 3 or not all(r["passed"] for r in arithmetic):
        raise ValueError("Independent sparse author arithmetic failed")
    for name in (
        "authors.parquet",
        "authorships.parquet",
        "visualization/author_productivity.parquet",
        "visualization/coauthor_edges.parquet",
    ):
        path = corrected_workspace / "canonical" / name
        source_hashes[str(path.resolve())] = sha256_file(path)
    rebuilt["formal_v3"]["author_identity_correction_sha256"] = amendment_hash
    rebuilt["formal_v3"]["original_benchmark_sha256"] = sha256_file(original_benchmark)
    gate = read_json(original_benchmark.parent / "data_gate_audit.json")
    gate["harvest"] = harvest
    gate["processing"] = processing
    gate["checks"]["author_identity_correction_verified"] = True
    gate["author_identity_correction_sha256"] = amendment_hash
    write_json(output / "data_gate_audit.json", gate)
    write_json(output / "benchmark.json", rebuilt)
    receipt = {
        "schema_version": 1,
        "status": "rebuilt_not_promoted",
        "dataset_id": original["dataset_id"],
        "author_identity_correction_sha256": amendment_hash,
        "source_sha256": source_hashes,
        "unchanged_graph_tables": graph_checks,
        "task_comparisons": comparisons,
        "operator_replay": replay,
        "independent_arithmetic": arithmetic,
        "benchmark_sha256": sha256_file(output / "benchmark.json"),
        "code_sha256": sha256_file(Path(__file__)),
        "limitations": [
            "Sparse arithmetic checks do not independently validate community or highest-betweenness selection.",
            "No generation results; corrected artifacts await neural, token and promotion audits.",
        ],
    }
    write_json(output / "author_identity_rebuild_receipt.json", receipt)
    return receipt
