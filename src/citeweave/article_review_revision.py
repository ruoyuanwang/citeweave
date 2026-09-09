from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from .article_claim_review import EVIDENCE_TOKEN, _article_paragraphs
from .io import read_json, sha256_file, write_json
from .review_learning import DependencyNode, ReviewDependencyGraph, StructuredFeedback
from .review_ui import ReviewStore

JUDGMENT_FIELDS = (
    "factual_supported",
    "interpretation_calibrated",
    "alternative_adequate",
    "evidence_sufficient",
)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _return_hashes(root: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted((root / "returns").glob("*.json"))}


def _decision_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    action = result.get("action")
    return (
        *(result.get(field) for field in JUDGMENT_FIELDS),
        action,
        tuple(sorted(result.get("decisive_evidence_ids", []))),
        tuple(sorted(result.get("invalid_dependency_ids", []))),
        str(result.get("replacement") or "").strip() if action == "rewrite" else "",
        tuple(sorted((str(key), repr(value)) for key, value in result.get("guard", {}).items())),
    )


def _returned_results(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = read_json(root / "internal_manifest.json")
    assignments = manifest["assignments"]
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((root / "returns").glob("*.json")):
        payload = read_json(path)
        reviewer = str(payload["reviewer_code"])
        for row in payload.get("results", []):
            if row.get("packet_id") not in assignments.get(reviewer, {}).get("article", []):
                raise ValueError("Unassigned article review return")
            key = (reviewer, str(row["packet_id"]))
            if key in results:
                raise ValueError(f"Duplicate frozen review result: {key}")
            if row.get("reviewer_code") != reviewer:
                raise ValueError(f"Reviewer identity mismatch in {path}")
            packet = read_json(root / "packets" / "article" / f"{row['packet_id']}.json")
            metadata = {
                "packet_id",
                "reviewer_code",
                "review_seconds",
                "server_elapsed_seconds",
                "timing_method",
                "submitted_at_unix",
                "session_id",
            }
            ReviewStore._validate_answers(
                "article", {k: v for k, v in row.items() if k not in metadata}, packet
            )
            seconds = row.get("review_seconds")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
                or row.get("timing_method") != "visibility_heartbeat_server_accounted"
            ):
                raise ValueError("Article review requires positive server-accounted time")
            results[key] = row
    return results


def _packet_index(root: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(root / "internal_manifest.json")
    rows = manifest["internal_packets"]
    indexed = {str(row["packet_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate article packet identity")
    records = {row["packet_id"]: row for row in manifest.get("packet_records", [])}
    for packet_id, row in indexed.items():
        primary = row["primary_reviewers"]
        if len(primary) != 2 or len(set(primary)) != 2 or row["adjudicator"] in primary:
            raise ValueError("Article review requires two distinct primaries and a third person")
        packet_path = root / "packets" / "article" / f"{packet_id}.json"
        if packet_id not in records or records[packet_id]["sha256"] != sha256_file(packet_path):
            raise ValueError(f"Article packet hash mismatch: {packet_id}")
    return indexed


def prepare_article_adjudication(
    primary_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Create third-person packets only after both primary returns are frozen."""
    manifest = read_json(primary_root / "internal_manifest.json")
    packet_index = _packet_index(primary_root)
    returned = _returned_results(primary_root)
    missing_pairs = [
        {
            "packet_id": packet_id,
            "missing_reviewers": [
                r for r in row["primary_reviewers"] if (r, packet_id) not in returned
            ],
        }
        for packet_id, row in sorted(packet_index.items())
        if any((r, packet_id) not in returned for r in row["primary_reviewers"])
    ]
    if missing_pairs:
        return {
            "status": "awaiting_primary_reviews",
            "missing_primary_pairs": missing_pairs,
            "adjudication_required": None,
        }
    if (output_root / "internal_manifest.json").exists():
        previous = read_json(output_root / "internal_manifest.json")
        if previous.get("primary_return_hashes") != _return_hashes(primary_root) or previous.get(
            "primary_manifest_sha256"
        ) != sha256_file(primary_root / "internal_manifest.json"):
            raise ValueError("Frozen adjudication inputs changed")
        return read_json(output_root / "manifest.json")
    missing = []
    conflicts = []
    assignments: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"article": []})
    internal_packets = []
    packet_records = []
    for packet_id, internal in sorted(packet_index.items()):
        reviewers = list(internal["primary_reviewers"])
        rows = [returned.get((reviewer, packet_id)) for reviewer in reviewers]
        if any(row is None for row in rows):
            missing.append(
                {
                    "packet_id": packet_id,
                    "missing_reviewers": [
                        reviewer
                        for reviewer, row in zip(reviewers, rows, strict=True)
                        if row is None
                    ],
                }
            )
            continue
        left, right = rows
        if _decision_signature(left) == _decision_signature(right):
            continue
        adjudicator = str(internal["adjudicator"])
        source_path = primary_root / "packets" / "article" / f"{packet_id}.json"
        packet = read_json(source_path)
        packet["adjudication"] = True
        packet["adjudication_instruction"] = (
            "Resolve this claim independently. Primary reviewer identities and judgments "
            "are withheld; use only the visible claim, graph result, and source excerpts."
        )
        packet_path = output_root / "packets" / "article" / f"{packet_id}.json"
        write_json(packet_path, packet)
        packet_records.append({"packet_id": packet_id, "sha256": sha256_file(packet_path)})
        assignments[adjudicator]["article"].append(packet_id)
        conflicts.append(packet_id)
        internal_packets.append(
            {
                **internal,
                "primary_result_sha256": [_canonical_hash(row) for row in rows],
            }
        )
    status = "awaiting_primary_reviews" if missing else "adjudication_packets_ready"
    result = {
        "schema_version": 1,
        "status": status,
        "primary_manifest_sha256": sha256_file(primary_root / "internal_manifest.json"),
        "primary_packets": len(packet_index),
        "primary_return_hashes": _return_hashes(primary_root),
        "complete_primary_pairs": len(packet_index) - len(missing),
        "missing_primary_pairs": missing,
        "adjudication_required": len(conflicts),
        "conflict_packet_ids": conflicts,
        "assignment_rule": "Only exact primary-decision disagreements receive one frozen independent third reviewer.",
    }
    write_json(output_root / "manifest.json", result)
    write_json(
        output_root / "internal_manifest.json",
        {
            **result,
            "assignments": dict(assignments),
            "internal_packets": internal_packets,
            "packet_records": packet_records,
            "source_topic_adjudicators": manifest["topic_adjudicators"],
        },
    )
    return result


def resolve_article_claim_reviews(
    primary_root: Path,
    *,
    adjudication_root: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    packet_index = _packet_index(primary_root)
    primary = _returned_results(primary_root)
    adjudicated = _returned_results(adjudication_root) if adjudication_root is not None else {}
    if adjudication_root is not None:
        adjudication_manifest = read_json(adjudication_root / "internal_manifest.json")
        _packet_index(adjudication_root)
        if adjudication_manifest.get("primary_return_hashes") != _return_hashes(
            primary_root
        ) or adjudication_manifest.get("primary_manifest_sha256") != sha256_file(
            primary_root / "internal_manifest.json"
        ):
            raise ValueError("Primary reviews changed after adjudication freeze")
    resolved = []
    conflicts = 0
    expected_adjudications = set()
    for packet_id, internal in sorted(packet_index.items()):
        reviewers = list(internal["primary_reviewers"])
        rows = [primary.get((reviewer, packet_id)) for reviewer in reviewers]
        if any(row is None for row in rows):
            raise ValueError(f"Incomplete primary pair: {packet_id}")
        left, right = rows
        disagreement = _decision_signature(left) != _decision_signature(right)
        adjudicator = str(internal["adjudicator"])
        adjudication = adjudicated.get((adjudicator, packet_id))
        if disagreement:
            conflicts += 1
            expected_adjudications.add((adjudicator, packet_id))
            if adjudication is None:
                raise ValueError(f"Disputed article claim lacks adjudication: {packet_id}")
            decision = adjudication
            source = "adjudicated"
        else:
            if adjudication is not None:
                raise ValueError(f"Undisputed article claim must not be adjudicated: {packet_id}")
            decision = left
            source = "dual_consensus"
        resolved.append(
            {
                "packet_id": packet_id,
                "dataset_id": internal["dataset_id"],
                "paragraph_id": internal["paragraph_id"],
                "candidate_id": internal["candidate_id"],
                "source": source,
                "resolver_code": decision["reviewer_code"],
                **{field: decision[field] for field in JUDGMENT_FIELDS},
                "action": decision["action"],
                "decisive_evidence_ids": decision["decisive_evidence_ids"],
                "invalid_dependency_ids": decision["invalid_dependency_ids"],
                "replacement": decision.get("replacement"),
                "guard": decision.get("guard", {}),
                "rationale": decision["rationale"],
                "review_seconds": decision["review_seconds"],
                "primary_review_seconds": sum(row["review_seconds"] for row in rows),
                "adjudication_seconds": adjudication["review_seconds"] if disagreement else 0,
                "total_review_seconds": sum(row["review_seconds"] for row in rows)
                + (adjudication["review_seconds"] if disagreement else 0),
            }
        )
    if set(adjudicated) != expected_adjudications:
        raise ValueError("Unexpected adjudication outside the frozen disagreements")
    result = {
        "schema_version": 1,
        "status": "article_claim_reviews_resolved",
        "claims": len(resolved),
        "dual_consensus": len(resolved) - conflicts,
        "adjudicated": conflicts,
        "primary_manifest_sha256": sha256_file(primary_root / "internal_manifest.json"),
        "primary_return_hashes": _return_hashes(primary_root),
        "adjudication_root": str(adjudication_root.resolve()) if adjudication_root else None,
        "adjudication_return_hashes": _return_hashes(adjudication_root)
        if adjudication_root
        else {},
        "resolved": resolved,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def compile_article_revision_worklist(
    primary_root: Path,
    resolved_path: Path,
    machine_plan_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Propagate resolved evidence/claim judgments into paragraph-only edits."""
    resolved = read_json(resolved_path)
    adjudication_root = resolved.get("adjudication_root")
    verified = resolve_article_claim_reviews(
        primary_root,
        adjudication_root=Path(adjudication_root) if adjudication_root else None,
    )
    if resolved != verified:
        raise ValueError("Resolved reviews differ from validated frozen returns")
    packet_index = _packet_index(primary_root)
    manifest = read_json(primary_root / "internal_manifest.json")
    if manifest.get("machine_plan_sha256") != sha256_file(machine_plan_path):
        raise ValueError("Machine plan differs from article review packet freeze")
    plan = read_json(machine_plan_path)
    article_paths = {
        row["dataset_id"]: Path(row["output_dir"]) / "draft.md"
        for row in plan["cells"]
        if row["condition"] == "citeweave_graph_review"
    }
    graph = ReviewDependencyGraph()
    paragraph_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    packet_payloads = {
        packet_id: read_json(primary_root / "packets" / "article" / f"{packet_id}.json")
        for packet_id in packet_index
    }
    article_inventory = []
    evidence_dependencies: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add_evidence(dataset_id: str, evidence_id: str) -> str:
        node_id = f"{dataset_id}:evidence:{evidence_id}"
        if node_id not in graph.nodes:
            graph.add_node(DependencyNode(node_id, "evidence", evidence_id))
        return node_id

    for packet_id, packet in packet_payloads.items():
        dataset_id = packet_index[packet_id]["dataset_id"]
        for phenomenon in packet.get("phenomena", []):
            phenomenon_id = phenomenon["phenomenon_id"]
            parents = set(phenomenon.get("graph_evidence_ids", [])) | set(
                phenomenon.get("reference_ids", [])
            )
            evidence_dependencies[(dataset_id, phenomenon_id)].update(parents)
            for parent in parents:
                graph.add_dependency(
                    add_evidence(dataset_id, parent), add_evidence(dataset_id, phenomenon_id)
                )
    for dataset_id, article_path in article_paths.items():
        draft_hash = sha256_file(article_path)
        topic_packets = [row for row in packet_index.values() if row["dataset_id"] == dataset_id]
        if any(row.get("draft_sha256") != draft_hash for row in topic_packets):
            raise ValueError(f"Draft changed after claim review: {dataset_id}")
        article = article_path.read_bytes().decode("utf-8")
        article_inventory.append(
            {
                "dataset_id": dataset_id,
                "draft_path": str(article_path.resolve()),
                "draft_sha256": draft_hash,
            }
        )
        for paragraph in _article_paragraphs(article):
            paragraph_lookup[(dataset_id, paragraph["paragraph_id"])] = paragraph
            graph.add_node(
                DependencyNode(
                    f"{dataset_id}:paragraph:{paragraph['paragraph_id']}",
                    "paragraph",
                    paragraph["text"],
                )
            )
            for token in set(EVIDENCE_TOKEN.findall(paragraph["text"])):
                graph.add_dependency(
                    add_evidence(dataset_id, token),
                    f"{dataset_id}:paragraph:{paragraph['paragraph_id']}",
                )
    for packet_id, internal in packet_index.items():
        dataset_id = internal["dataset_id"]
        claim_node = f"{dataset_id}:claim:{packet_id}"
        paragraph_node = f"{dataset_id}:paragraph:{internal['paragraph_id']}"
        graph.add_node(DependencyNode(claim_node, "claim", packet_payloads[packet_id]["claim"]))
        graph.add_dependency(claim_node, paragraph_node)
        for evidence_id in internal["evidence_tokens"]:
            evidence_node = f"{dataset_id}:evidence:{evidence_id}"
            if evidence_node not in graph.nodes:
                graph.add_node(DependencyNode(evidence_node, "evidence", evidence_id))
            graph.add_dependency(evidence_node, claim_node)
    propagation = []
    directives: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unresolved_abstentions = []
    paragraph_node_lookup = {
        f"{dataset}:paragraph:{paragraph_id}": (dataset, paragraph_id)
        for dataset, paragraph_id in paragraph_lookup
    }

    def register_propagation(event: dict[str, Any], verdict: dict[str, Any]) -> None:
        propagation.append(event)
        for node_id in event["changed_nodes"]:
            key = paragraph_node_lookup.get(node_id)
            if key is not None and not any(
                item["packet_id"] == verdict["packet_id"] for item in directives[key]
            ):
                directives[key].append(verdict)

    for row in resolved["resolved"]:
        internal = packet_index[row["packet_id"]]
        dataset_id = row["dataset_id"]
        for evidence_id in row["invalid_dependency_ids"]:
            feedback = StructuredFeedback(
                feedback_id=f"{row['packet_id']}:evidence:{evidence_id}",
                reviewer_id=row["resolver_code"],
                dataset_id=dataset_id,
                target_id=f"{dataset_id}:evidence:{evidence_id}",
                target_type="evidence",
                issue_type="article_dependency_invalid",
                action="reject_evidence",
                rationale=row["rationale"],
                guard=row["guard"],
                review_seconds=row["review_seconds"],
            )
            register_propagation(graph.apply_feedback(feedback), row)
        if row["action"] in {"rewrite", "reject_claim"}:
            feedback = StructuredFeedback(
                feedback_id=f"{row['packet_id']}:claim",
                reviewer_id=row["resolver_code"],
                dataset_id=dataset_id,
                target_id=f"{dataset_id}:claim:{row['packet_id']}",
                target_type="claim",
                issue_type="article_claim_correction",
                action=row["action"],
                rationale=row["rationale"],
                replacement=row.get("replacement"),
                guard=row["guard"],
                review_seconds=row["review_seconds"],
            )
            register_propagation(graph.apply_feedback(feedback), row)
        if row["action"] == "abstain":
            unresolved_abstentions.append(row["packet_id"])
        defect = (
            row["action"] != "accept"
            or bool(row["invalid_dependency_ids"])
            or not all(row[field] for field in JUDGMENT_FIELDS)
        )
        direct_key = (dataset_id, internal["paragraph_id"])
        if defect and not any(
            item["packet_id"] == row["packet_id"] for item in directives[direct_key]
        ):
            directives[direct_key].append(row)
    affected = []
    for (dataset_id, paragraph_id), rows in sorted(directives.items()):
        paragraph = paragraph_lookup[(dataset_id, paragraph_id)]
        paragraph_tokens = set(EVIDENCE_TOKEN.findall(paragraph["text"]))
        allowed = sorted(
            paragraph_tokens
            | {
                evidence_id
                for token in paragraph_tokens
                for evidence_id in evidence_dependencies[(dataset_id, token)]
            }
        )
        affected.append(
            {
                "dataset_id": dataset_id,
                "draft_path": str(article_paths[dataset_id].resolve()),
                "draft_sha256": sha256_file(article_paths[dataset_id]),
                "paragraph_id": paragraph_id,
                "paragraph_sha256": paragraph["paragraph_sha256"],
                "section": paragraph["section"],
                "start_char": paragraph["start_char"],
                "end_char": paragraph["end_char"],
                "original_text": paragraph["text"],
                "allowed_evidence_ids": allowed,
                "directives": rows,
                "propagated_from_other_paragraph": any(
                    item["paragraph_id"] != paragraph_id for item in rows
                ),
            }
        )
    status = (
        "blocked_unresolved_abstentions"
        if unresolved_abstentions
        else "controlled_paragraph_revision_ready"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "resolved_reviews_sha256": sha256_file(resolved_path),
        "affected_paragraphs": len(affected),
        "unaffected_paragraphs_locked": True,
        "unresolved_abstentions": unresolved_abstentions,
        "propagation_events": propagation,
        "articles": article_inventory,
        "propagation_scope": "Explicit PH/REF citations plus graph-evidence and literature-context dependencies; uncited semantic dependencies require separate expert audit.",
        "worklist": affected,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def apply_controlled_paragraph_revisions(
    worklist_path: Path,
    replacements_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Apply only hash-bound paragraph replacements; all other bytes stay locked."""
    worklist = read_json(worklist_path)
    if worklist["status"] != "controlled_paragraph_revision_ready":
        raise ValueError("Revision worklist is not ready")
    replacements_payload = read_json(replacements_path)
    replacements = {
        (row["dataset_id"], row["paragraph_id"]): row
        for row in replacements_payload["replacements"]
    }
    if len(replacements) != len(replacements_payload["replacements"]):
        raise ValueError("Duplicate paragraph replacement")
    expected = {(row["dataset_id"], row["paragraph_id"]) for row in worklist["worklist"]}
    if set(replacements) != expected:
        raise ValueError("Replacements must match the complete affected-paragraph set")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worklist["worklist"]:
        by_dataset[row["dataset_id"]].append(row)
    records = []
    prepared_outputs = []
    for inventory in sorted(worklist["articles"], key=lambda row: row["dataset_id"]):
        dataset_id = inventory["dataset_id"]
        rows = by_dataset[dataset_id]
        draft_path = Path(inventory["draft_path"])
        if sha256_file(draft_path) != inventory["draft_sha256"]:
            raise ValueError(f"Original draft hash changed: {dataset_id}")
        original_bytes = draft_path.read_bytes()
        original_article = original_bytes.decode("utf-8")
        article = original_article
        ordered = sorted(rows, key=lambda item: item["start_char"])
        if any(left["end_char"] > right["start_char"] for left, right in pairwise(ordered)):
            raise ValueError("Overlapping paragraph revision spans")
        changed_original_bytes = 0
        for row in sorted(rows, key=lambda item: item["start_char"], reverse=True):
            replacement = replacements[(dataset_id, row["paragraph_id"])]
            if replacement.get("original_paragraph_sha256") != row["paragraph_sha256"]:
                raise ValueError(f"Paragraph hash mismatch: {row['paragraph_id']}")
            text = str(replacement.get("replacement_text") or "").strip()
            if not text:
                raise ValueError("Controlled paragraph replacement cannot be empty")
            unknown = sorted(set(EVIDENCE_TOKEN.findall(text)) - set(row["allowed_evidence_ids"]))
            if unknown:
                raise ValueError(f"Replacement introduces unknown evidence IDs: {unknown}")
            if not 0 <= row["start_char"] < row["end_char"] <= len(original_article):
                raise ValueError("Paragraph span outside original draft")
            original = original_article[row["start_char"] : row["end_char"]]
            if hashlib.sha256(original.encode()).hexdigest() != row["paragraph_sha256"]:
                raise ValueError(f"Original paragraph bytes changed: {row['paragraph_id']}")
            changed_original_bytes += len(original.encode("utf-8"))
            article = article[: row["start_char"]] + text + article[row["end_char"] :]
        output_path = output_root / dataset_id / "draft.reviewed.md"
        if output_path.resolve() == draft_path.resolve():
            raise ValueError("Revision output cannot overwrite the original draft")
        output_bytes = article.encode("utf-8")
        if output_path.exists() and output_path.read_bytes() != output_bytes:
            raise ValueError("A different reviewed draft is already frozen")
        prepared_outputs.append((output_path, output_bytes))
        records.append(
            {
                "dataset_id": dataset_id,
                "original_sha256": sha256_file(draft_path),
                "reviewed_path": str(output_path.resolve()),
                "reviewed_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "replaced_paragraphs": len(rows),
                "original_bytes": len(original_bytes),
                "unchanged_original_bytes": len(original_bytes) - changed_original_bytes,
                "publication_status": "candidate_requires_independent_expert_verification",
            }
        )
    result = {
        "schema_version": 1,
        "status": "controlled_revisions_applied",
        "worklist_sha256": sha256_file(worklist_path),
        "replacement_manifest_sha256": sha256_file(replacements_path),
        "unaffected_text_policy": "byte-identical by reconstructing only frozen paragraph spans",
        "articles": records,
    }
    manifest_path = output_root / "revision_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != result:
        raise ValueError("A different revision manifest is already frozen")
    for output_path, output_bytes in prepared_outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_bytes)
    write_json(output_root / "revision_manifest.json", result)
    return result
