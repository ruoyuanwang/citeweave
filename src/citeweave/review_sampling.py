from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json


def _stable_hash(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def _balanced_take(
    entries: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        buckets[tuple(entry["stratum"])].append(entry)
    for stratum, bucket in buckets.items():
        bucket.sort(key=lambda item: _stable_hash(seed, stage, stratum, item["packet_id"]))
    strata = sorted(buckets, key=lambda value: _stable_hash(seed, stage, value))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _packet_metadata(
    record: dict[str, Any], packet_root: Path, layer: str, packet_id: str
) -> dict[str, Any]:
    task_type = record.get("task_type")
    complexity = record.get("complexity")
    if task_type is None or complexity is None:
        packet = read_json(packet_root / "packets" / layer / f"{packet_id}.json")
        task_type = packet["task_type"]
        complexity = packet["complexity"]
    outcome = (
        bool(record["deterministic_answer_exact"]) if layer == "factual" else None
    )
    stratum = [record["condition"], task_type, int(complexity)]
    if layer == "factual":
        stratum.append(outcome)
    return {
        "packet_id": packet_id,
        "layer": layer,
        "condition": record["condition"],
        "task_type": task_type,
        "complexity": int(complexity),
        "deterministic_outcome": outcome,
        "stratum": stratum,
    }


def _allocate_layer(
    entries: list[dict[str, Any]],
    reviewers: tuple[str, str],
    *,
    per_reviewer: int,
    overlap_fraction: float,
    seed: int,
    layer: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be between zero and one")
    population = len(entries)
    effective = min(max(0, per_reviewer), population)
    minimum_overlap = max(0, 2 * effective - population)
    desired_overlap = round(effective * overlap_fraction)
    overlap = min(effective, max(minimum_overlap, desired_overlap))
    unique_per_reviewer = effective - overlap

    common = _balanced_take(entries, overlap, seed=seed, stage=f"{layer}:common")
    common_ids = {entry["packet_id"] for entry in common}
    remaining = [entry for entry in entries if entry["packet_id"] not in common_ids]
    left_unique = _balanced_take(
        remaining,
        unique_per_reviewer,
        seed=seed,
        stage=f"{layer}:{reviewers[0]}",
    )
    left_ids = {entry["packet_id"] for entry in left_unique}
    remaining = [entry for entry in remaining if entry["packet_id"] not in left_ids]
    right_unique = _balanced_take(
        remaining,
        unique_per_reviewer,
        seed=seed,
        stage=f"{layer}:{reviewers[1]}",
    )
    selected = {entry["packet_id"]: entry for entry in [*common, *left_unique, *right_unique]}
    assignments = {
        reviewers[0]: [entry["packet_id"] for entry in [*common, *left_unique]],
        reviewers[1]: [entry["packet_id"] for entry in [*common, *right_unique]],
    }
    for reviewer, packet_ids in assignments.items():
        packet_ids.sort(key=lambda packet_id: _stable_hash(seed, reviewer, layer, packet_id))

    population_by_stratum: dict[tuple[Any, ...], int] = defaultdict(int)
    sampled_by_stratum: dict[tuple[Any, ...], int] = defaultdict(int)
    for entry in entries:
        population_by_stratum[tuple(entry["stratum"])] += 1
    for entry in selected.values():
        sampled_by_stratum[tuple(entry["stratum"])] += 1
    strata = []
    for stratum in sorted(population_by_stratum, key=str):
        population_count = population_by_stratum[stratum]
        sampled_count = sampled_by_stratum.get(stratum, 0)
        strata.append(
            {
                "stratum": list(stratum),
                "population": population_count,
                "sampled_unique": sampled_count,
                "inverse_probability_weight": (
                    population_count / sampled_count if sampled_count else None
                ),
            }
        )
    audit = {
        "layer": layer,
        "population": population,
        "requested_per_reviewer": per_reviewer,
        "effective_per_reviewer": effective,
        "common_double_review": overlap,
        "unique_per_reviewer": unique_per_reviewer,
        "unique_packets_sampled": len(selected),
        "realized_overlap_fraction": overlap / effective if effective else 0.0,
        "strata": strata,
    }
    return assignments, audit


def build_sampled_review_manifest(
    manifest: dict[str, Any],
    *,
    packet_root: Path,
    factual_per_reviewer: int,
    semantic_per_reviewer: int,
    overlap_fraction: float = 0.5,
    seed: int = 20260820,
) -> dict[str, Any]:
    reviewers = tuple(manifest["reviewers"])
    if len(reviewers) != 2:
        raise ValueError("The stratified protocol requires exactly two reviewers")
    layer_entries: dict[str, list[dict[str, Any]]] = {"factual": [], "semantic": []}
    for record in manifest["records"]:
        factual_id = record.get("factual_packet_id")
        if factual_id:
            layer_entries["factual"].append(
                _packet_metadata(record, packet_root, "factual", factual_id)
            )
        semantic_id = record.get("semantic_packet_id")
        if semantic_id:
            layer_entries["semantic"].append(
                _packet_metadata(record, packet_root, "semantic", semantic_id)
            )

    factual_assignments, factual_audit = _allocate_layer(
        layer_entries["factual"],
        reviewers,
        per_reviewer=factual_per_reviewer,
        overlap_fraction=overlap_fraction,
        seed=seed,
        layer="factual",
    )
    semantic_assignments, semantic_audit = _allocate_layer(
        layer_entries["semantic"],
        reviewers,
        per_reviewer=semantic_per_reviewer,
        overlap_fraction=overlap_fraction,
        seed=seed,
        layer="semantic",
    )
    assignments = {
        reviewer: {
            "factual": factual_assignments[reviewer],
            "semantic": semantic_assignments[reviewer],
        }
        for reviewer in reviewers
    }
    selected_ids = {
        packet_id
        for assignment in assignments.values()
        for layer in ("factual", "semantic")
        for packet_id in assignment[layer]
    }
    selected_records = [
        record
        for record in manifest["records"]
        if record.get("factual_packet_id") in selected_ids
        or record.get("semantic_packet_id") in selected_ids
    ]
    return {
        **manifest,
        "schema_version": 2,
        "factual_packets": factual_audit["unique_packets_sampled"],
        "semantic_packets": semantic_audit["unique_packets_sampled"],
        "double_review": factual_audit["unique_per_reviewer"] == 0
        and semantic_audit["unique_per_reviewer"] == 0,
        "partial_overlap_review": True,
        "records": selected_records,
        "assignments": assignments,
        "sampling": {
            "method": "deterministic_balanced_stratified_partial_overlap",
            "seed": seed,
            "requested_overlap_fraction": overlap_fraction,
            "offline_stratification_fields": [
                "condition",
                "task_type",
                "complexity",
                "deterministic_answer_exact_for_factual_only",
            ],
            "online_policy_leakage_prohibited": True,
            "note": (
                "Deterministic outcome is used only to balance the hidden evaluation sample; "
                "it must never be exposed to reviewers or used as an online policy feature."
            ),
            "layers": {"factual": factual_audit, "semantic": semantic_audit},
        },
    }
