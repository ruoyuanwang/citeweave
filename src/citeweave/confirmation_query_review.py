from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .io import read_json, sha256_file, write_json


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_contract(
    reviewer_id: str,
    candidate_code: str,
    item_code: str,
    *,
    role: str,
) -> dict[str, str]:
    identity = {
        "reviewer_id": reviewer_id,
        "candidate_code": candidate_code,
        "item_code": item_code,
        "role": role,
    }
    task_id = "QRT-" + _canonical_sha(identity)[:16].upper()
    definition = {"task_id": task_id, **identity}
    return {**definition, "task_definition_sha256": _canonical_sha(definition)}


def _reviewer_catalog(roster: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reviewers = roster.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers:
        raise ValueError("A nonempty real-reviewer roster is required")
    catalog: dict[str, dict[str, Any]] = {}
    for row in reviewers:
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("Every reviewer requires a nonempty reviewer_id")
        if reviewer_id in catalog:
            raise ValueError("Duplicate reviewer identity is prohibited")
        if row.get("real_human_attestation") is not True:
            raise ValueError("Every reviewer must attest that they are a real human")
        if row.get("synthetic_or_proxy_reviewer") is not False:
            raise ValueError("Synthetic or proxy query reviewers are prohibited")
        domains = row.get("qualified_domains")
        if not isinstance(domains, list) or not domains or any(
            not isinstance(value, str) or not value.strip() for value in domains
        ):
            raise ValueError("Every reviewer requires at least one qualified domain")
        conflicts = row.get("conflicts_with_candidate_codes", [])
        if not isinstance(conflicts, list) or any(
            not isinstance(value, str) for value in conflicts
        ):
            raise ValueError("Reviewer conflicts must be a list of candidate codes")
        catalog[reviewer_id] = row
    return catalog


def _choose_reviewers(
    *,
    candidate_code: str,
    domain: str,
    catalog: dict[str, dict[str, Any]],
    loads: Counter[str],
    seed: int,
) -> tuple[list[str], str]:
    eligible = [
        reviewer_id
        for reviewer_id, row in catalog.items()
        if domain in row["qualified_domains"]
        and candidate_code not in row.get("conflicts_with_candidate_codes", [])
    ]
    if len(eligible) < 3:
        raise ValueError(
            f"Candidate {candidate_code} requires three conflict-free, "
            f"domain-qualified real reviewers"
        )
    eligible.sort(
        key=lambda reviewer_id: (
            loads[reviewer_id],
            _canonical_sha(
                {
                    "seed": seed,
                    "candidate_code": candidate_code,
                    "reviewer_id": reviewer_id,
                }
            ),
        )
    )
    chosen = eligible[:3]
    for reviewer_id in chosen:
        loads[reviewer_id] += 1
    return chosen[:2], chosen[2]


def build_query_reviewer_roster_template(
    protocol_path: Path,
    candidate_audit_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Create a non-overwriting roster contract without fabricating reviewers."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite reviewer roster: {output_path}")
    protocol_sha = sha256_file(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    audit = read_json(candidate_audit_path)
    if audit.get("protocol_sha256") != protocol_sha:
        raise ValueError("Candidate audit is not bound to this protocol")
    topics = {row["id"]: row for row in protocol["candidate_topics"]}
    requirements = []
    for row in audit.get("records", []):
        if row.get("automatic_data_gate_passed") is not True:
            continue
        topic = topics[str(row["dataset_id"])]
        requirements.append(
            {
                "candidate_code": row["candidate_code"],
                "qualified_domain": topic["domain"],
                "minimum_conflict_free_reviewers": 3,
                "roles": "two_primary_plus_one_independent_adjudicator",
            }
        )
    result = {
        "schema_version": 1,
        "status": "awaiting_real_query_reviewers",
        "protocol_sha256": protocol_sha,
        "candidate_audit_sha256": sha256_file(candidate_audit_path),
        "reviewer_contract": {
            "real_humans_only": True,
            "synthetic_or_proxy_reviewers_prohibited": True,
            "domain_qualification_must_be_documented": True,
            "conflicts_declared_before_assignment": True,
            "minimum_three_eligible_reviewers_per_candidate": True,
            "primary_answers_hidden_from_adjudicator": True,
        },
        "candidate_requirements": requirements,
        "reviewer_schema_example": {
            "reviewer_id": "replace_with_pseudonymous_id",
            "real_human_attestation": True,
            "synthetic_or_proxy_reviewer": False,
            "qualified_domains": ["replace_with_registered_domain"],
            "qualification_note": "degree, publications, or documented experience",
            "conflicts_with_candidate_codes": [],
        },
        "reviewers": [],
    }
    write_json(output_path, result)
    return result


def build_query_review_collection(
    protocol_path: Path,
    protocol_freeze_path: Path,
    candidate_audit_path: Path,
    roster_path: Path,
    *,
    output_dir: Path,
    seed: int = 20260908,
) -> dict[str, Any]:
    """Freeze blind double-review assignments for query-relevance screening."""
    protocol_sha = sha256_file(protocol_path)
    freeze = read_json(protocol_freeze_path)
    if freeze.get("sha256") != protocol_sha:
        raise ValueError("Confirmation protocol differs from its freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    audit = read_json(candidate_audit_path)
    if audit.get("protocol_sha256") != protocol_sha:
        raise ValueError("Candidate audit is not bound to this protocol")
    if audit.get("status") != "awaiting_blind_query_relevance_review":
        raise ValueError("Candidate audit is not ready for blind query review")
    roster = read_json(roster_path)
    catalog = _reviewer_catalog(roster)
    topics = {row["id"]: row for row in protocol["candidate_topics"]}

    reviewable = [
        row
        for row in audit.get("records", [])
        if row.get("automatic_data_gate_passed") is True
    ]
    if not reviewable:
        raise ValueError("No candidate passed the automatic data gate")
    output_dir.mkdir(parents=True, exist_ok=False)
    loads: Counter[str] = Counter()
    packet_groups: dict[str, list[dict[str, Any]]] = {}
    private_assignments = []
    primary_registry: dict[str, list[dict[str, str]]] = {}

    for record in sorted(reviewable, key=lambda row: int(row["priority"])):
        dataset_id = str(record["dataset_id"])
        topic = topics.get(dataset_id)
        if topic is None:
            raise ValueError("Candidate audit contains a topic outside the protocol")
        packet_path = Path(record["query_review_packet"])
        if (
            not packet_path.is_file()
            or sha256_file(packet_path) != record["query_review_packet_sha256"]
        ):
            raise ValueError(f"Query packet changed after candidate audit: {dataset_id}")
        packet = read_json(packet_path)
        candidate_code = str(record["candidate_code"])
        if packet.get("candidate_code") != candidate_code:
            raise ValueError("Candidate code differs between audit and blind packet")
        primary_reviewers, adjudicator = _choose_reviewers(
            candidate_code=candidate_code,
            domain=str(topic["domain"]),
            catalog=catalog,
            loads=loads,
            seed=seed,
        )
        private_assignments.append(
            {
                "dataset_id": dataset_id,
                "priority": int(record["priority"]),
                "domain": topic["domain"],
                "candidate_code": candidate_code,
                "query_review_packet": str(packet_path.resolve()),
                "query_review_packet_sha256": record["query_review_packet_sha256"],
                "primary_reviewers": primary_reviewers,
                "adjudicator": adjudicator,
            }
        )
        for reviewer_id in primary_reviewers:
            tasks = [
                _task_contract(
                    reviewer_id,
                    candidate_code,
                    str(item["item_code"]),
                    role="primary",
                )
                for item in packet["items"]
            ]
            primary_registry.setdefault(reviewer_id, []).extend(tasks)
            packet_groups.setdefault(reviewer_id, []).append(
                {
                    "candidate_code": candidate_code,
                    "concepts": packet["concepts"],
                    "instructions": packet["instructions"],
                    "items": [
                        {
                            **item,
                            **tasks[index],
                        }
                        for index, item in enumerate(packet["items"])
                    ],
                }
            )

    packet_records = {}
    for reviewer_id, candidates in sorted(packet_groups.items()):
        packet = {
            "schema_version": 1,
            "status": "blind_primary_query_review_packet",
            "reviewer_id": reviewer_id,
            "hidden_fields": [
                "dataset_id",
                "candidate_priority",
                "graph_statistics",
                "perturbations",
                "task_gold",
                "model_outcomes",
                "other_reviewer_identity",
                "other_reviewer_answers",
            ],
            "decision_rule": (
                "Relevant only when both displayed concepts are scientifically central "
                "to the title or abstract. Use cannot_assess only when the visible text "
                "is insufficient to make that decision."
            ),
            "candidates": candidates,
        }
        serialized = json.dumps(packet, ensure_ascii=False)
        for assignment in private_assignments:
            if assignment["dataset_id"] in serialized:
                raise AssertionError("Dataset identity leaked into a blind reviewer packet")
        path = output_dir / "primary_packets" / f"{reviewer_id}.json"
        write_json(path, packet)
        packet_records[reviewer_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "candidates": len(candidates),
            "tasks": sum(len(row["items"]) for row in candidates),
        }
        template = {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "blind_attestation": True,
            "independent_completion_attestation": True,
            "submitted_at": None,
            "results": [
                {
                    **task,
                    "relevant": None,
                    "cannot_assess": None,
                    "rationale_code": None,
                    "review_seconds": None,
                }
                for task in primary_registry[reviewer_id]
            ],
        }
        write_json(output_dir / "return_templates" / f"{reviewer_id}.json", template)

    private_manifest = {
        "schema_version": 1,
        "status": "blind_query_review_assignments_frozen",
        "seed": seed,
        "assignments": private_assignments,
    }
    private_path = output_dir / "private_assignment_manifest.json"
    write_json(private_path, private_manifest)
    manifest = {
        "schema_version": 1,
        "status": "blind_primary_query_review_ready",
        "confirmatory": True,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "protocol_freeze_sha256": sha256_file(protocol_freeze_path),
        "candidate_audit": str(candidate_audit_path.resolve()),
        "candidate_audit_sha256": sha256_file(candidate_audit_path),
        "reviewer_roster": str(roster_path.resolve()),
        "reviewer_roster_sha256": sha256_file(roster_path),
        "seed": seed,
        "reviewed_candidates": len(private_assignments),
        "primary_reviews_per_item": 2,
        "independent_adjudicator_frozen_per_candidate": True,
        "primary_packets": packet_records,
        "primary_task_registry": primary_registry,
        "private_assignment_manifest": str(private_path.resolve()),
        "private_assignment_manifest_sha256": sha256_file(private_path),
        "submission_contract": {
            "allowed_rationale_codes": [
                "both_central",
                "one_or_both_peripheral",
                "insufficient_visible_information",
            ],
            "complete_return_required": True,
            "immutable_submission": True,
            "positive_review_seconds_required": True,
            "blind_and_independent_attestations_required": True,
        },
    }
    write_json(output_dir / "collection_manifest.json", manifest)
    return manifest


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every return requires a submitted_at timestamp")
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("submitted_at is not a valid ISO-8601 timestamp") from error


def _validate_result(
    result: dict[str, Any],
    contract: dict[str, str],
    *,
    allowed_rationales: set[str],
) -> dict[str, Any]:
    for field, expected in contract.items():
        if result.get(field) != expected:
            raise ValueError("Reviewer changed a server-owned task field")
    relevant = result.get("relevant")
    cannot_assess = result.get("cannot_assess")
    if cannot_assess is True:
        if relevant is not None:
            raise ValueError("Cannot-assess result must not include a relevance label")
        expected_rationale = "insufficient_visible_information"
    else:
        if cannot_assess is not False or not isinstance(relevant, bool):
            raise ValueError("Each assessable task requires one boolean relevance label")
        expected_rationale = (
            "both_central" if relevant else "one_or_both_peripheral"
        )
    if result.get("rationale_code") != expected_rationale:
        raise ValueError("Rationale code is inconsistent with the relevance decision")
    if result.get("rationale_code") not in allowed_rationales:
        raise ValueError("Unknown rationale code")
    seconds = result.get("review_seconds")
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or seconds <= 0
    ):
        raise ValueError("Every task requires positive review_seconds")
    return {
        **contract,
        "relevant": relevant,
        "cannot_assess": cannot_assess,
        "rationale_code": result["rationale_code"],
        "review_seconds": float(seconds),
    }


def validate_query_review_primary_returns(
    collection_manifest_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Validate complete independent primary returns and enumerate disagreements."""
    collection = read_json(collection_manifest_path)
    if collection.get("status") != "blind_primary_query_review_ready":
        raise ValueError("Query-review collection is not ready")
    private_path = Path(collection["private_assignment_manifest"])
    if sha256_file(private_path) != collection["private_assignment_manifest_sha256"]:
        raise ValueError("Private query-review assignments changed after freeze")
    private = read_json(private_path)
    allowed = set(collection["submission_contract"]["allowed_rationale_codes"])
    return_hashes = {}
    rows = []
    task_ids: set[str] = set()
    for reviewer_id, contracts in collection["primary_task_registry"].items():
        path = collection_manifest_path.parent / "returns" / "primary" / f"{reviewer_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing primary reviewer return: {reviewer_id}")
        payload = read_json(path)
        if payload.get("reviewer_id") != reviewer_id:
            raise ValueError("Primary return reviewer identity mismatch")
        if (
            payload.get("blind_attestation") is not True
            or payload.get("independent_completion_attestation") is not True
        ):
            raise ValueError("Primary reviewer did not attest blind independent completion")
        _validate_timestamp(payload.get("submitted_at"))
        contract_by_id = {row["task_id"]: row for row in contracts}
        results = payload.get("results")
        if not isinstance(results, list) or {
            row.get("task_id") for row in results
        } != set(contract_by_id):
            raise ValueError("Primary return set is incomplete, duplicated, or foreign")
        if len(results) != len(contract_by_id):
            raise ValueError("Primary return contains duplicate tasks")
        for raw in results:
            task_id = str(raw["task_id"])
            if task_id in task_ids:
                raise ValueError("A primary task was returned more than once")
            task_ids.add(task_id)
            rows.append(
                _validate_result(raw, contract_by_id[task_id], allowed_rationales=allowed)
            )
        return_hashes[reviewer_id] = sha256_file(path)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["candidate_code"], row["item_code"]), []).append(row)
    resolved = []
    disagreements = []
    assignments = {
        row["candidate_code"]: row for row in private["assignments"]
    }
    for (candidate_code, item_code), decisions in sorted(grouped.items()):
        if len(decisions) != 2 or len({row["reviewer_id"] for row in decisions}) != 2:
            raise ValueError("Every item requires two unique primary reviewers")
        expected_reviewers = set(assignments[candidate_code]["primary_reviewers"])
        if {row["reviewer_id"] for row in decisions} != expected_reviewers:
            raise ValueError("Primary reviewer assignment mismatch")
        labels = [
            None if row["cannot_assess"] else bool(row["relevant"])
            for row in decisions
        ]
        base = {"candidate_code": candidate_code, "item_code": item_code}
        if labels[0] is not None and labels[0] == labels[1]:
            resolved.append({**base, "relevant": labels[0], "resolution": "primary_agreement"})
        else:
            disagreements.append(
                {
                    **base,
                    "adjudicator": assignments[candidate_code]["adjudicator"],
                    "excluded_primary_reviewers": sorted(expected_reviewers),
                }
            )
    result = {
        "schema_version": 1,
        "status": (
            "awaiting_blind_query_adjudication"
            if disagreements
            else "query_relevance_ready_without_disagreement"
        ),
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "protocol_sha256": collection["protocol_sha256"],
        "primary_return_sha256": return_hashes,
        "primary_integrity": {
            "reviewers": len(collection["primary_task_registry"]),
            "decisions": len(rows),
            "item_pairs": len(grouped),
            "agreements": len(resolved),
            "disagreements": len(disagreements),
            "all_tasks_returned_once": True,
            "all_server_owned_fields_match": True,
            "blind_independent_attestations": True,
        },
        "resolved_items": resolved,
        "adjudication_worklist": disagreements,
    }
    write_json(output_path, result)
    return result


def build_query_review_adjudication(
    primary_validation_path: Path,
    collection_manifest_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Route only disagreements to a frozen third reviewer without primary answers."""
    primary = read_json(primary_validation_path)
    collection = read_json(collection_manifest_path)
    if primary.get("status") != "awaiting_blind_query_adjudication":
        raise ValueError("Primary validation has no query disagreements")
    if primary.get("collection_manifest_sha256") != sha256_file(
        collection_manifest_path
    ):
        raise ValueError("Primary validation is not bound to this collection")
    private_path = Path(collection["private_assignment_manifest"])
    if sha256_file(private_path) != collection["private_assignment_manifest_sha256"]:
        raise ValueError("Private query-review assignments changed after freeze")

    item_catalog = {}
    for record in collection["primary_packets"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("A primary blind packet changed after assignment")
        packet = read_json(path)
        for candidate in packet["candidates"]:
            for item in candidate["items"]:
                key = (candidate["candidate_code"], item["item_code"])
                public_item = {
                    field: item.get(field)
                    for field in ("item_code", "title", "abstract", "year")
                }
                previous = item_catalog.setdefault(
                    key,
                    {
                        "candidate_code": candidate["candidate_code"],
                        "concepts": candidate["concepts"],
                        "instructions": candidate["instructions"],
                        "item": public_item,
                    },
                )
                if previous["item"] != public_item:
                    raise ValueError("Item content differs across primary packets")

    output_dir.mkdir(parents=True, exist_ok=False)
    grouped: dict[str, list[dict[str, Any]]] = {}
    registry: dict[str, list[dict[str, str]]] = {}
    for row in primary["adjudication_worklist"]:
        key = (row["candidate_code"], row["item_code"])
        public = item_catalog.get(key)
        if public is None:
            raise ValueError("Adjudication item is absent from frozen primary packets")
        reviewer_id = str(row["adjudicator"])
        if reviewer_id in row["excluded_primary_reviewers"]:
            raise ValueError("A primary reviewer cannot adjudicate their own item")
        contract = _task_contract(
            reviewer_id,
            row["candidate_code"],
            row["item_code"],
            role="adjudication",
        )
        registry.setdefault(reviewer_id, []).append(contract)
        grouped.setdefault(reviewer_id, []).append({**public, **contract})

    packet_records = {}
    for reviewer_id, tasks in sorted(grouped.items()):
        packet = {
            "schema_version": 1,
            "status": "blind_query_adjudication_packet",
            "reviewer_id": reviewer_id,
            "hidden_fields": [
                "dataset_id",
                "candidate_priority",
                "graph_statistics",
                "primary_reviewer_identity",
                "primary_answers",
                "model_outcomes",
            ],
            "tasks": tasks,
        }
        serialized = json.dumps(packet, ensure_ascii=False)
        if "excluded_primary_reviewers" in serialized:
            raise AssertionError("Primary reviewer identity leaked to adjudication")
        path = output_dir / "adjudication_packets" / f"{reviewer_id}.json"
        write_json(path, packet)
        packet_records[reviewer_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "tasks": len(tasks),
        }
        template = {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "blind_attestation": True,
            "independent_completion_attestation": True,
            "submitted_at": None,
            "results": [
                {
                    **contract,
                    "relevant": None,
                    "cannot_assess": False,
                    "rationale_code": None,
                    "review_seconds": None,
                }
                for contract in registry[reviewer_id]
            ],
        }
        write_json(output_dir / "return_templates" / f"{reviewer_id}.json", template)

    manifest = {
        "schema_version": 1,
        "status": "blind_query_adjudication_ready",
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "primary_validation": str(primary_validation_path.resolve()),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_packets": packet_records,
        "adjudication_task_registry": registry,
        "submission_contract": collection["submission_contract"],
        "primary_answers_hidden": True,
        "primary_reviewer_identities_hidden": True,
    }
    write_json(output_dir / "adjudication_manifest.json", manifest)
    return manifest


def finalize_query_relevance_selection(
    primary_validation_path: Path,
    collection_manifest_path: Path,
    *,
    output_path: Path,
    adjudication_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve relevance rates and apply the frozen first-eight-eligible rule."""
    primary = read_json(primary_validation_path)
    collection = read_json(collection_manifest_path)
    if primary.get("collection_manifest_sha256") != sha256_file(
        collection_manifest_path
    ):
        raise ValueError("Primary validation is not bound to this collection")
    resolved = {
        (row["candidate_code"], row["item_code"]): row
        for row in primary["resolved_items"]
    }
    adjudication_hashes = {}
    if primary["adjudication_worklist"]:
        if adjudication_manifest_path is None:
            raise ValueError("Disagreements require blind third-person adjudication")
        adjudication = read_json(adjudication_manifest_path)
        if adjudication.get("primary_validation_sha256") != sha256_file(
            primary_validation_path
        ):
            raise ValueError("Adjudication is not bound to the primary validation")
        allowed = set(
            adjudication["submission_contract"]["allowed_rationale_codes"]
        )
        expected = {
            row["task_id"]: row
            for contracts in adjudication["adjudication_task_registry"].values()
            for row in contracts
        }
        observed: set[str] = set()
        for reviewer_id, contracts in adjudication[
            "adjudication_task_registry"
        ].items():
            path = (
                adjudication_manifest_path.parent
                / "returns"
                / "adjudication"
                / f"{reviewer_id}.json"
            )
            if not path.is_file():
                raise ValueError(f"Missing adjudicator return: {reviewer_id}")
            payload = read_json(path)
            if payload.get("reviewer_id") != reviewer_id:
                raise ValueError("Adjudicator return identity mismatch")
            if (
                payload.get("blind_attestation") is not True
                or payload.get("independent_completion_attestation") is not True
            ):
                raise ValueError("Adjudicator did not attest blind independent completion")
            _validate_timestamp(payload.get("submitted_at"))
            contract_by_id = {row["task_id"]: row for row in contracts}
            results = payload.get("results")
            if not isinstance(results, list) or {
                row.get("task_id") for row in results
            } != set(contract_by_id):
                raise ValueError("Adjudicator return set is incomplete, duplicated, or foreign")
            if len(results) != len(contract_by_id):
                raise ValueError("Adjudicator return contains duplicate tasks")
            for raw in results:
                task_id = str(raw["task_id"])
                decision = _validate_result(
                    raw, contract_by_id[task_id], allowed_rationales=allowed
                )
                if decision["cannot_assess"]:
                    raise ValueError("A third reviewer must resolve each disputed item")
                key = (decision["candidate_code"], decision["item_code"])
                if key in resolved:
                    raise ValueError("An agreed primary item received post-hoc adjudication")
                resolved[key] = {
                    "candidate_code": key[0],
                    "item_code": key[1],
                    "relevant": bool(decision["relevant"]),
                    "resolution": "independent_blind_adjudication",
                }
                observed.add(task_id)
            adjudication_hashes[reviewer_id] = sha256_file(path)
        if observed != set(expected):
            raise ValueError("Not every frozen disagreement received one adjudication")
    elif adjudication_manifest_path is not None:
        raise ValueError("Agreed items must not receive post-hoc adjudication")

    private_path = Path(collection["private_assignment_manifest"])
    if sha256_file(private_path) != collection["private_assignment_manifest_sha256"]:
        raise ValueError("Private query-review assignments changed after freeze")
    private = read_json(private_path)
    summaries = []
    for assignment in sorted(private["assignments"], key=lambda row: row["priority"]):
        candidate_rows = [
            row
            for key, row in resolved.items()
            if key[0] == assignment["candidate_code"]
        ]
        packet = read_json(Path(assignment["query_review_packet"]))
        if len(candidate_rows) != len(packet["items"]):
            raise ValueError("Candidate does not have exactly one resolved label per item")
        relevant = sum(row["relevant"] is True for row in candidate_rows)
        rate = relevant / len(candidate_rows)
        summaries.append(
            {
                "dataset_id": assignment["dataset_id"],
                "priority": assignment["priority"],
                "candidate_code": assignment["candidate_code"],
                "items": len(candidate_rows),
                "relevant": relevant,
                "resolved_relevance_rate": rate,
                "eligible": rate >= 0.80,
                "primary_agreements": sum(
                    row["resolution"] == "primary_agreement" for row in candidate_rows
                ),
                "blind_adjudications": sum(
                    row["resolution"] == "independent_blind_adjudication"
                    for row in candidate_rows
                ),
            }
        )
    target = 8
    protocol = yaml.safe_load(Path(collection["protocol"]).read_text(encoding="utf-8"))
    target = int(protocol["selection"]["target_topics"])
    eligible = [row for row in summaries if row["eligible"]]
    audit = read_json(Path(collection["candidate_audit"]))
    unreviewed_registered = len(audit["records"]) - len(summaries)
    if len(eligible) >= target:
        status = "selected_topics_ready_for_post_selection_freeze"
        selected = [row["dataset_id"] for row in eligible[:target]]
    elif unreviewed_registered > 0:
        status = "awaiting_registered_reserve_candidate_processing"
        selected = []
    else:
        status = "stopped_insufficient_eligible_topics"
        selected = []
    result = {
        "schema_version": 1,
        "status": status,
        "confirmatory": True,
        "protocol_sha256": collection["protocol_sha256"],
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_manifest_sha256": (
            sha256_file(adjudication_manifest_path)
            if adjudication_manifest_path is not None
            else None
        ),
        "adjudicator_return_sha256": adjudication_hashes,
        "threshold": 0.80,
        "selection_rule": "first_eight_eligible_candidates_by_frozen_priority",
        "candidate_summaries": summaries,
        "selected_topics": selected,
        "integrity": {
            "two_independent_primary_decisions_per_item": True,
            "disagreements_only_received_third_review": True,
            "adjudicators_were_not_primary_reviewers": True,
            "reviewers_blind_to_priority_graph_and_model_outcomes": True,
            "provider_responses_at_selection": 0,
            "perturbation_outcomes_at_selection": 0,
        },
    }
    write_json(output_path, result)
    return result
