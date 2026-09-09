from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def build_source_relevance_packets(
    *, manifest_path: Path, output_root: Path
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "same_evidence_packs_ready":
        raise ValueError("Writer-pack manifest is not ready")
    public_root = output_root / "public"
    internal_root = output_root / "internal"
    records = []
    dataset_counts: Counter[str] = Counter()
    for manifest_row in sorted(manifest["records"], key=lambda row: row["dataset_id"]):
        pack_path = Path(manifest_row["pack"])
        if sha256_file(pack_path) != manifest_row["pack_sha256"]:
            raise ValueError(f"Writer pack hash mismatch: {pack_path}")
        pack = read_json(pack_path)
        if (
            pack.get("passed") is not True
            or pack.get("source_selection_version")
            != "topic-task-bm25-v4-title-deduplicated"
        ):
            raise ValueError(f"Writer pack is not the audited v4 version: {pack_path}")
        sources = {row["reference_id"]: row for row in pack["representative_sources"]}
        for phenomenon in pack["graph_phenomena"]:
            for reference_id in phenomenon["reference_ids"]:
                source = sources[reference_id]
                identity = f"{phenomenon['phenomenon_id']}\x1f{reference_id}"
                packet_id = f"SR-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
                public = {
                    "schema_version": 1,
                    "packet_type": "phenomenon_source_relevance_warmup",
                    "packet_id": packet_id,
                    "dataset_id": pack["dataset_id"],
                    "domain": pack["dataset_id"],
                    "issue_type": "evidence_relevance",
                    "confirmatory_exclusion": True,
                    "phenomenon": {
                        "phenomenon_id": phenomenon["phenomenon_id"],
                        "task_type": phenomenon["task_type"],
                        "question": phenomenon["question"],
                        "verified_structural_answer": phenomenon["verified_answer"],
                        "allowed_interpretations": phenomenon[
                            "interpretation_contract"
                        ]["allowed"],
                        "required_limitation": phenomenon[
                            "interpretation_contract"
                        ]["required_limitation"],
                    },
                    "source": {
                        "reference_id": reference_id,
                        "title": source["title"],
                        "year": source["year"],
                        "doi": source.get("doi"),
                        "abstract_excerpt": source["abstract_excerpt"],
                    },
                    "review_instructions": [
                        "Judge topical relevance separately from support for the graph interpretation.",
                        "Copy the smallest decisive span exactly from the visible title or abstract.",
                        "Mark generic-anchor or broad-query failures and propose query terms when useful.",
                        "Use cannot_assess when the supplied excerpt is insufficient.",
                    ],
                    "response_schema": {
                        "topic_relevance": [
                            "direct",
                            "contextual",
                            "irrelevant",
                            "cannot_assess",
                        ],
                        "evidence_role": [
                            "supports_interpretation",
                            "context_only",
                            "challenges",
                            "irrelevant",
                            "cannot_assess",
                        ],
                        "decisive_span": "exact visible string|null",
                        "failure_type": [
                            "none",
                            "generic_graph_anchor",
                            "query_too_broad",
                            "off_topic",
                            "insufficient_detail",
                            "conflicting_evidence",
                            "other",
                        ],
                        "suggested_query_terms": "list[string]",
                        "rationale": "string",
                    },
                }
                public_path = public_root / f"{packet_id}.json"
                write_json(public_path, public)
                internal = {
                    "schema_version": 1,
                    "packet_id": packet_id,
                    "public_packet_canonical_sha256": _canonical_hash(public),
                    "writer_pack": str(pack_path),
                    "writer_pack_sha256": manifest_row["pack_sha256"],
                    "selection_metadata": {
                        field: source.get(field)
                        for field in (
                            "selection_basis",
                            "retrieval_query",
                            "retrieval_rank",
                            "bm25_rank",
                            "matched_keyword",
                            "topic_terms",
                            "topic_term_matches",
                        )
                    },
                    "hidden_fields": [
                        "selection_basis",
                        "retrieval_query",
                        "retrieval_rank",
                        "bm25_rank",
                        "matched_keyword",
                        "topic_term_matches",
                        "other_reviewer_judgments",
                    ],
                    "gold_label_present": False,
                }
                internal_path = internal_root / f"{packet_id}.json"
                write_json(internal_path, internal)
                dataset_counts[str(pack["dataset_id"])] += 1
                records.append(
                    {
                        "packet_id": packet_id,
                        "dataset_id": pack["dataset_id"],
                        "phenomenon_id": phenomenon["phenomenon_id"],
                        "reference_id": reference_id,
                        "public_path": str(public_path.relative_to(output_root)),
                        "public_sha256": sha256_file(public_path),
                        "internal_path": str(internal_path.relative_to(output_root)),
                        "internal_sha256": sha256_file(internal_path),
                    }
                )
    result = {
        "schema_version": 1,
        "status": "warmup_packets_built_before_human_judgments",
        "human_judgments_inspected": False,
        "confirmatory_exclusion": True,
        "writer_pack_manifest": str(manifest_path),
        "writer_pack_manifest_sha256": sha256_file(manifest_path),
        "datasets": len(dataset_counts),
        "packets": len(records),
        "dataset_packet_counts": dict(sorted(dataset_counts.items())),
        "records": records,
    }
    write_json(output_root / "manifest.json", result)
    return result


def audit_source_relevance_packets(packet_root: Path) -> dict[str, Any]:
    manifest_path = packet_root / "manifest.json"
    manifest = read_json(manifest_path)
    reasons = []
    packet_ids = [str(row.get("packet_id") or "") for row in manifest.get("records", [])]
    counts: Counter[str] = Counter()
    for row in manifest.get("records", []):
        packet_id = str(row["packet_id"])
        counts[str(row["dataset_id"])] += 1
        public_path = packet_root / row["public_path"]
        internal_path = packet_root / row["internal_path"]
        if not public_path.is_file() or sha256_file(public_path) != row["public_sha256"]:
            reasons.append(f"{packet_id}:public_hash_mismatch")
            continue
        if not internal_path.is_file() or sha256_file(internal_path) != row["internal_sha256"]:
            reasons.append(f"{packet_id}:internal_hash_mismatch")
            continue
        public = read_json(public_path)
        internal = read_json(internal_path)
        serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
        if any(
            field in serialized
            for field in (
                "selection_basis",
                "retrieval_query",
                "retrieval_rank",
                "bm25_rank",
                "matched_keyword",
                "topic_term_matches",
            )
        ):
            reasons.append(f"{packet_id}:selection_metadata_leakage")
        if internal.get("public_packet_canonical_sha256") != _canonical_hash(public):
            reasons.append(f"{packet_id}:canonical_hash_mismatch")
        if internal.get("gold_label_present") is not False:
            reasons.append(f"{packet_id}:gold_label_present")
    checks = {
        "pre_judgment_status": manifest.get("status")
        == "warmup_packets_built_before_human_judgments",
        "human_judgments_absent": manifest.get("human_judgments_inspected") is False,
        "confirmatory_exclusion": manifest.get("confirmatory_exclusion") is True,
        "eight_datasets": len(counts) == 8,
        "fifteen_packets_per_dataset": set(counts.values()) == {15},
        "one_hundred_twenty_packets": len(packet_ids) == 120,
        "unique_nonempty_packet_ids": all(packet_ids)
        and len(packet_ids) == len(set(packet_ids)),
        "all_artifacts_valid": not reasons,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reasons": reasons,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_packet_counts": dict(sorted(counts.items())),
    }
