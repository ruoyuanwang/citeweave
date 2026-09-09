from __future__ import annotations

import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Any

import yaml

from .complementary_oversight import ReviewerCapabilityModel, ReviewerObservation
from .io import read_json, sha256_file, write_json
from .review_ui import ReviewStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SERVER_FIELDS = {
    "packet_id",
    "reviewer_code",
    "review_seconds",
    "server_elapsed_seconds",
    "timing_method",
    "submitted_at_unix",
}
_RESOLUTION_FIELDS = ("topic_relevance", "evidence_role")


class SourceReviewOutcomeError(ValueError):
    pass


def validate_source_outcome_protocol(
    *,
    protocol_path: Path,
    protocol_freeze_path: Path,
    execution_amendment_path: Path,
    execution_amendment_freeze_path: Path,
    outcome_amendment_path: Path,
    outcome_amendment_freeze_path: Path,
) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_freeze = read_json(protocol_freeze_path)
    if (
        protocol_freeze.get("protocol_sha256") != protocol_hash
        or protocol_freeze.get("protocol_id") != protocol.get("protocol_id")
    ):
        raise RuntimeError("Source outcome base protocol identity/hash mismatch")
    execution_hash = sha256_file(execution_amendment_path)
    execution = yaml.safe_load(
        execution_amendment_path.read_text(encoding="utf-8")
    )
    execution_freeze = read_json(execution_amendment_freeze_path)
    if (
        execution_freeze.get("sha256") != execution_hash
        or execution.get("base_protocol_sha256") != protocol_hash
    ):
        raise RuntimeError("Source outcome execution-amendment chain mismatch")
    outcome_hash = sha256_file(outcome_amendment_path)
    outcome = yaml.safe_load(outcome_amendment_path.read_text(encoding="utf-8"))
    outcome_freeze = read_json(outcome_amendment_freeze_path)
    if (
        outcome_freeze.get("sha256") != outcome_hash
        or outcome_freeze.get("amendment_id") != outcome.get("amendment_id")
        or outcome.get("base_protocol_sha256") != protocol_hash
        or outcome.get("previous_amendment_sha256") != execution_hash
    ):
        raise RuntimeError("Source outcome amendment identity/hash mismatch")
    for label, artifact in (outcome.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Source outcome implementation mismatch: {label}")
    return outcome


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise SourceReviewOutcomeError(f"Unsafe or empty {label}: {value!r}")
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceReviewOutcomeError(f"{label} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise SourceReviewOutcomeError(f"{label} must be a finite positive number")
    return converted


def _packet_index(assignment_root: Path) -> dict[str, dict[str, Any]]:
    assignment = read_json(assignment_root / "assignment_manifest.json")
    if (
        assignment.get("status") != "source_warmup_panel_ready_before_returns"
        or assignment.get("human_returns_present") is not False
        or assignment.get("confirmatory_exclusion") is not True
        or assignment.get("packets") != 120
        or assignment.get("decisions") != 180
        or assignment.get("double_reviewed_packets") != 60
    ):
        raise SourceReviewOutcomeError("Source assignment manifest is not a formal warmup panel")
    rows = assignment.get("records") or []
    packet_ids = [_safe_id(row.get("packet_id"), label="packet ID") for row in rows]
    if len(rows) != 120 or len(packet_ids) != len(set(packet_ids)):
        raise SourceReviewOutcomeError("Source assignment requires 120 unique packet records")
    return {row["packet_id"]: row for row in rows}


def build_source_adjudicator_commitment(
    assignment_root: Path,
    reviewer_roster_path: Path,
    *,
    output_path: Path,
    seed: int = 20260907,
) -> dict[str, Any]:
    assignment_root = assignment_root.resolve()
    assignment_path = assignment_root / "assignment_manifest.json"
    assignment = read_json(assignment_path)
    packet_rows = _packet_index(assignment_root)
    roster = read_json(reviewer_roster_path)
    if assignment.get("reviewer_roster_sha256") != sha256_file(reviewer_roster_path):
        raise SourceReviewOutcomeError("Reviewer roster changed after primary assignment")
    if output_path.exists():
        raise SourceReviewOutcomeError("Refusing to overwrite adjudicator commitment")
    returns_root = assignment_root / "returns"
    if returns_root.exists() and any(returns_root.glob("*.json")):
        raise SourceReviewOutcomeError("Adjudicators must be committed before primary returns")

    reviewers = roster.get("reviewers") or []
    reviewer_ids = [_safe_id(row.get("reviewer_id"), label="reviewer ID") for row in reviewers]
    if len(reviewer_ids) != 6 or len(reviewer_ids) != len(set(reviewer_ids)):
        raise SourceReviewOutcomeError("Adjudicator commitment requires six unique reviewers")
    eligible = {
        row["reviewer_id"]: set(map(str, row.get("eligible_domains") or []))
        - set(map(str, row.get("conflicted_domains") or []))
        for row in reviewers
    }
    potential_load: Counter[str] = Counter()
    potential_domain_load: dict[str, Counter[str]] = defaultdict(Counter)
    commitments = []
    for packet_id, row in sorted(packet_rows.items()):
        primaries = list(map(str, row.get("reviewer_ids") or []))
        if len(primaries) != 2:
            continue
        domain = str(row["dataset_id"])
        candidates = [
            reviewer_id
            for reviewer_id in reviewer_ids
            if reviewer_id not in primaries and domain in eligible[reviewer_id]
        ]
        if not candidates:
            raise SourceReviewOutcomeError(f"No independent adjudicator for {packet_id}")
        rng = Random(f"{seed}:{packet_id}")
        tie_breaks = {reviewer_id: rng.random() for reviewer_id in candidates}
        adjudicator = min(
            candidates,
            key=lambda reviewer_id: (
                potential_load[reviewer_id],
                potential_domain_load[reviewer_id][domain],
                tie_breaks[reviewer_id],
                reviewer_id,
            ),
        )
        potential_load[adjudicator] += 1
        potential_domain_load[adjudicator][domain] += 1
        packet_path = assignment_root / "packets" / "source" / f"{packet_id}.json"
        if not packet_path.is_file():
            raise SourceReviewOutcomeError(f"Committed packet file is missing: {packet_id}")
        commitments.append(
            {
                "packet_id": packet_id,
                "dataset_id": domain,
                "primary_reviewers": primaries,
                "adjudicator_id": adjudicator,
                "packet_sha256": sha256_file(packet_path),
            }
        )
    if len(commitments) != 60:
        raise SourceReviewOutcomeError("Exactly 60 double-reviewed cases need adjudicators")
    result = {
        "schema_version": 1,
        "status": "adjudicators_frozen_before_primary_returns",
        "primary_returns_present_at_commitment": False,
        "seed": seed,
        "assignment_manifest_sha256": sha256_file(assignment_path),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
        "double_reviewed_cases": 60,
        "potential_adjudications_per_reviewer": dict(sorted(potential_load.items())),
        "records": commitments,
    }
    write_json(output_path, result)
    return result


def _validate_server_result(
    result: dict[str, Any],
    *,
    reviewer_id: str,
    packet_id: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    if result.get("reviewer_code") != reviewer_id:
        raise SourceReviewOutcomeError(f"Reviewer identity mismatch for {packet_id}")
    if result.get("packet_id") != packet_id:
        raise SourceReviewOutcomeError(f"Packet identity mismatch for {packet_id}")
    review_seconds = _finite_positive(
        result.get("review_seconds"), label=f"review_seconds for {packet_id}"
    )
    elapsed_seconds = _finite_positive(
        result.get("server_elapsed_seconds"),
        label=f"server_elapsed_seconds for {packet_id}",
    )
    if review_seconds > elapsed_seconds + 0.002:
        raise SourceReviewOutcomeError(
            f"Active review time exceeds server elapsed time for {packet_id}"
        )
    _finite_positive(
        result.get("submitted_at_unix"), label=f"submitted_at_unix for {packet_id}"
    )
    if result.get("timing_method") != "visibility_heartbeat_server_accounted":
        raise SourceReviewOutcomeError(f"Untrusted timing method for {packet_id}")
    answers = {key: value for key, value in result.items() if key not in _SERVER_FIELDS}
    try:
        ReviewStore._validate_answers("source", answers, packet)
    except (TypeError, ValueError) as error:
        raise SourceReviewOutcomeError(
            f"Invalid source-review answers for {packet_id}: {error}"
        ) from error
    return dict(result)


def validate_source_primary_returns(
    assignment_root: Path,
    commitment_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    assignment_root = assignment_root.resolve()
    packet_rows = _packet_index(assignment_root)
    internal_path = assignment_root / "internal_manifest.json"
    internal = read_json(internal_path)
    commitment = read_json(commitment_path)
    if (
        commitment.get("status") != "adjudicators_frozen_before_primary_returns"
        or commitment.get("assignment_manifest_sha256")
        != sha256_file(assignment_root / "assignment_manifest.json")
    ):
        raise SourceReviewOutcomeError("Invalid or mismatched adjudicator commitment")
    committed_packets = {row["packet_id"]: row for row in commitment["records"]}
    reviewers = list(map(str, internal.get("reviewers") or []))
    if len(reviewers) != 6 or len(reviewers) != len(set(reviewers)):
        raise SourceReviewOutcomeError("Primary source review requires six reviewers")

    validated = []
    return_hashes = []
    for reviewer_id in sorted(reviewers):
        _safe_id(reviewer_id, label="reviewer ID")
        assigned = list(internal["assignments"][reviewer_id].get("source") or [])
        if len(assigned) != 30 or len(assigned) != len(set(assigned)):
            raise SourceReviewOutcomeError(
                f"Reviewer {reviewer_id} must have 30 unique source assignments"
            )
        return_path = assignment_root / "returns" / f"{reviewer_id}.json"
        if not return_path.is_file():
            raise SourceReviewOutcomeError(f"Missing primary return for {reviewer_id}")
        payload = read_json(return_path)
        if payload.get("reviewer_code") != reviewer_id:
            raise SourceReviewOutcomeError(f"Return identity mismatch for {reviewer_id}")
        results = payload.get("results")
        if not isinstance(results, list):
            raise SourceReviewOutcomeError(f"Return results must be a list for {reviewer_id}")
        by_packet = {str(row.get("packet_id")): row for row in results}
        if len(results) != len(by_packet) or set(by_packet) != set(assigned):
            raise SourceReviewOutcomeError(
                f"Return assignments are incomplete, duplicated, or unexpected for {reviewer_id}"
            )
        for packet_id in assigned:
            _safe_id(packet_id, label="packet ID")
            if packet_id not in packet_rows:
                raise SourceReviewOutcomeError(f"Unknown assigned packet {packet_id}")
            packet_path = assignment_root / "packets" / "source" / f"{packet_id}.json"
            packet = read_json(packet_path)
            if packet_id in committed_packets and sha256_file(packet_path) != committed_packets[
                packet_id
            ]["packet_sha256"]:
                raise SourceReviewOutcomeError(f"Packet changed after commitment: {packet_id}")
            validated.append(
                _validate_server_result(
                    by_packet[packet_id],
                    reviewer_id=reviewer_id,
                    packet_id=packet_id,
                    packet=packet,
                )
            )
        return_hashes.append(
            {
                "reviewer_id": reviewer_id,
                "path": str(return_path),
                "sha256": sha256_file(return_path),
            }
        )
    if len(validated) != 180:
        raise SourceReviewOutcomeError("Primary source review requires 180 validated decisions")
    result = {
        "schema_version": 1,
        "status": "primary_source_returns_validated",
        "assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "internal_manifest_sha256": sha256_file(internal_path),
        "adjudicator_commitment_sha256": sha256_file(commitment_path),
        "reviewers": reviewers,
        "decisions": len(validated),
        "return_files": return_hashes,
        "validated_results": validated,
    }
    write_json(output_path, result)
    return result


def _resolution_key(result: dict[str, Any]) -> tuple[str, str]:
    return tuple(str(result[field]) for field in _RESOLUTION_FIELDS)  # type: ignore[return-value]


def prepare_source_adjudication_panel(
    assignment_root: Path,
    commitment_path: Path,
    primary_validation_path: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    assignment_root = assignment_root.resolve()
    commitment = read_json(commitment_path)
    primary = read_json(primary_validation_path)
    if primary.get("status") != "primary_source_returns_validated":
        raise SourceReviewOutcomeError("Primary returns are not validated")
    if primary.get("adjudicator_commitment_sha256") != sha256_file(commitment_path):
        raise SourceReviewOutcomeError("Primary validation is not bound to commitment")
    if output_root.exists() and any(output_root.iterdir()):
        raise SourceReviewOutcomeError("Refusing to overwrite adjudication panel")
    indexed_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary["validated_results"]:
        indexed_results[row["packet_id"]].append(row)
    assignments: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"source": []})
    records = []
    for commitment_row in commitment["records"]:
        packet_id = commitment_row["packet_id"]
        rows = indexed_results[packet_id]
        if len(rows) != 2:
            raise SourceReviewOutcomeError(
                f"Double-reviewed packet lacks two validated returns: {packet_id}"
            )
        if _resolution_key(rows[0]) == _resolution_key(rows[1]):
            continue
        adjudicator_id = commitment_row["adjudicator_id"]
        assignments[adjudicator_id]["source"].append(packet_id)
        source = assignment_root / "packets" / "source" / f"{packet_id}.json"
        target = output_root / "packets" / "source" / f"{packet_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "packet_id": packet_id,
                "dataset_id": commitment_row["dataset_id"],
                "adjudicator_id": adjudicator_id,
                "packet_sha256": sha256_file(target),
            }
        )
    frozen_assignments = {
        reviewer: {"source": sorted(layers["source"])}
        for reviewer, layers in sorted(assignments.items())
    }
    internal = {
        "schema_version": 1,
        "status": "source_adjudication_assignments_frozen",
        "reviewers": sorted(frozen_assignments),
        "assignments": frozen_assignments,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "internal_manifest.json", internal)
    manifest = {
        "schema_version": 1,
        "status": "source_adjudication_panel_ready",
        "primary_judgments_hidden_from_adjudicators": True,
        "adjudicator_commitment_sha256": sha256_file(commitment_path),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudications_required": len(records),
        "records": records,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _validate_adjudication_returns(
    adjudication_root: Path,
    adjudication_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    internal = read_json(adjudication_root / "internal_manifest.json")
    expected = {
        packet_id: reviewer_id
        for reviewer_id, layers in internal.get("assignments", {}).items()
        for packet_id in layers.get("source", [])
    }
    if len(expected) != adjudication_manifest.get("adjudications_required"):
        raise SourceReviewOutcomeError("Adjudication manifest assignment mismatch")
    validated = {}
    for reviewer_id, layers in internal.get("assignments", {}).items():
        return_path = adjudication_root / "returns" / f"{reviewer_id}.json"
        if not return_path.is_file():
            raise SourceReviewOutcomeError(f"Missing adjudication return for {reviewer_id}")
        payload = read_json(return_path)
        results = payload.get("results")
        if payload.get("reviewer_code") != reviewer_id or not isinstance(results, list):
            raise SourceReviewOutcomeError(f"Invalid adjudication return for {reviewer_id}")
        by_packet = {str(row.get("packet_id")): row for row in results}
        assigned = set(layers.get("source", []))
        if len(results) != len(by_packet) or set(by_packet) != assigned:
            raise SourceReviewOutcomeError(
                f"Adjudication return assignment mismatch for {reviewer_id}"
            )
        for packet_id in sorted(assigned):
            packet_path = adjudication_root / "packets" / "source" / f"{packet_id}.json"
            packet = read_json(packet_path)
            validated[packet_id] = _validate_server_result(
                by_packet[packet_id],
                reviewer_id=reviewer_id,
                packet_id=packet_id,
                packet=packet,
            )
    return validated


def finalize_source_warmup(
    assignment_root: Path,
    commitment_path: Path,
    primary_validation_path: Path,
    adjudication_root: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise SourceReviewOutcomeError("Refusing to overwrite finalized source warmup")
    assignment_root = assignment_root.resolve()
    packet_rows = _packet_index(assignment_root)
    primary = read_json(primary_validation_path)
    adjudication_manifest_path = adjudication_root / "manifest.json"
    adjudication_manifest = read_json(adjudication_manifest_path)
    if (
        adjudication_manifest.get("adjudicator_commitment_sha256")
        != sha256_file(commitment_path)
        or adjudication_manifest.get("primary_validation_sha256")
        != sha256_file(primary_validation_path)
    ):
        raise SourceReviewOutcomeError("Adjudication panel is not bound to primary inputs")
    adjudications = _validate_adjudication_returns(
        adjudication_root, adjudication_manifest
    )
    primary_by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary["validated_results"]:
        primary_by_packet[row["packet_id"]].append(row)
    resolved_labels = []
    observations = []
    for packet_id, assignment_row in sorted(packet_rows.items()):
        rows = primary_by_packet[packet_id]
        if len(rows) not in {1, 2}:
            raise SourceReviewOutcomeError(f"Unexpected primary count for {packet_id}")
        if len(rows) == 1:
            resolved = rows[0]
            resolution_source = "single_review"
            confidence = "single_review_training_only"
        elif _resolution_key(rows[0]) == _resolution_key(rows[1]):
            resolved = rows[0]
            resolution_source = "double_review_consensus"
            confidence = "independent_double_review"
        else:
            if packet_id not in adjudications:
                raise SourceReviewOutcomeError(f"Missing required adjudication for {packet_id}")
            resolved = adjudications[packet_id]
            resolution_source = "precommitted_third_adjudication"
            confidence = "independent_third_resolution"
        packet = read_json(
            assignment_root / "packets" / "source" / f"{packet_id}.json"
        )
        resolved_labels.append(
            {
                "packet_id": packet_id,
                "dataset_id": assignment_row["dataset_id"],
                "phenomenon_id": packet["phenomenon"]["phenomenon_id"],
                "reference_id": packet["source"]["reference_id"],
                "topic_relevance": resolved["topic_relevance"],
                "evidence_role": resolved["evidence_role"],
                "decisive_spans": sorted(
                    {
                        str(row.get("decisive_span"))
                        for row in rows
                        if row.get("decisive_span")
                    }
                    | (
                        {str(resolved["decisive_span"])}
                        if resolved.get("decisive_span")
                        else set()
                    )
                ),
                "suggested_query_terms": sorted(
                    {
                        str(term)
                        for row in [*rows, resolved]
                        for term in row.get("suggested_query_terms", [])
                    }
                ),
                "resolution_source": resolution_source,
                "confidence_tier": confidence,
                "eligible_for_capability_calibration": len(rows) == 2,
                "eligible_for_retriever_training": True,
                "eligible_for_retriever_promotion": len(rows) == 2,
            }
        )
        if len(rows) == 2:
            resolution_key = _resolution_key(resolved)
            for row in rows:
                observations.append(
                    ReviewerObservation(
                        observation_id=f"SRC-{packet_id}-{row['reviewer_code']}",
                        reviewer_id=row["reviewer_code"],
                        dataset_id=assignment_row["dataset_id"],
                        domain=assignment_row["dataset_id"],
                        issue_type="evidence_relevance",
                        correct_after_adjudication=_resolution_key(row)
                        == resolution_key,
                        review_seconds=float(row["review_seconds"]),
                    )
                )
    if len(resolved_labels) != 120 or len(observations) != 120:
        raise SourceReviewOutcomeError(
            "Final source warmup requires 120 labels and 120 calibration observations"
        )
    reviewers = sorted({row.reviewer_id for row in observations})
    if reviewers != sorted(primary["reviewers"]):
        raise SourceReviewOutcomeError("Capability observations do not cover frozen reviewers")
    by_reviewer: dict[str, list[ReviewerObservation]] = defaultdict(list)
    for row in observations:
        by_reviewer[row.reviewer_id].append(row)
    if any(len(rows) < 12 for rows in by_reviewer.values()):
        raise SourceReviewOutcomeError("Every reviewer needs at least 12 resolved warmup cases")
    if any(len({row.domain for row in rows}) < 2 for rows in by_reviewer.values()):
        raise SourceReviewOutcomeError("Every reviewer needs two resolved warmup domains")

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved_source_labels.json"
    observations_path = output_root / "reviewer_observations.json"
    registry_path = output_root / "reviewer_registry.json"
    write_json(
        resolved_path,
        {
            "schema_version": 1,
            "status": "source_labels_resolved",
            "assignment_manifest_sha256": sha256_file(
                assignment_root / "assignment_manifest.json"
            ),
            "primary_validation_sha256": sha256_file(primary_validation_path),
            "adjudication_manifest_sha256": sha256_file(
                adjudication_manifest_path
            ),
            "labels": resolved_labels,
        },
    )
    write_json(
        observations_path,
        {
            "schema_version": 1,
            "status": "warmup_capability_observations_resolved",
            "observations": [asdict(row) for row in observations],
        },
    )
    registry = {
        "schema_version": 1,
        "status": "reviewer_registry_frozen_after_warmup",
        "reviewers": [
            {
                "reviewer_id": reviewer_id,
                "warmup_cases_completed": len(by_reviewer[reviewer_id]),
                "warmup_domains": sorted(
                    {row.domain for row in by_reviewer[reviewer_id]}
                ),
                "warmup_outcomes_adjudicated": True,
                "warmup_excluded_from_confirmatory": True,
            }
            for reviewer_id in reviewers
        ],
    }
    write_json(registry_path, registry)
    model = ReviewerCapabilityModel(observations)
    domains = sorted({row.domain for row in observations})
    profiles = [
        asdict(
            model.estimate(
                reviewer_id,
                domain=domain,
                issue_type="evidence_relevance",
            )
        )
        for reviewer_id in reviewers
        for domain in domains
    ]
    capability_path = output_root / "capability_freeze.json"
    adjudication_return_files = [
        {
            "reviewer_id": reviewer_id,
            "path": str(adjudication_root / "returns" / f"{reviewer_id}.json"),
            "sha256": sha256_file(
                adjudication_root / "returns" / f"{reviewer_id}.json"
            ),
        }
        for reviewer_id in sorted(
            read_json(adjudication_root / "internal_manifest.json").get(
                "reviewers", []
            )
        )
    ]
    capability = {
        "schema_version": 1,
        "status": "frozen_before_heldout_assignment",
        "heldout_outcomes_inspected": False,
        "predecision_sha256": sha256_file(observations_path),
        "reviewer_ids": reviewers,
        "prior_alpha": 2.0,
        "prior_beta": 2.0,
        "minimum_exact_observations": 3,
        "minimum_domain_observations": 5,
        "observations_sha256": sha256_file(observations_path),
        "resolved_source_labels_sha256": sha256_file(resolved_path),
        "reviewer_registry_sha256": sha256_file(registry_path),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_manifest_sha256": sha256_file(adjudication_manifest_path),
        "adjudication_return_files": adjudication_return_files,
        "profiles": profiles,
    }
    write_json(capability_path, capability)
    result = {
        "schema_version": 1,
        "status": "source_warmup_finalized_before_heldout_assignment",
        "assignment_manifest_sha256": sha256_file(
            assignment_root / "assignment_manifest.json"
        ),
        "adjudicator_commitment_sha256": sha256_file(commitment_path),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_manifest_sha256": sha256_file(adjudication_manifest_path),
        "adjudication_return_files": adjudication_return_files,
        "resolved_labels": len(resolved_labels),
        "double_resolved_labels": sum(
            row["eligible_for_capability_calibration"] for row in resolved_labels
        ),
        "single_review_training_labels": sum(
            not row["eligible_for_capability_calibration"] for row in resolved_labels
        ),
        "capability_observations": len(observations),
        "reviewers": len(reviewers),
        "resolved_source_labels_sha256": sha256_file(resolved_path),
        "reviewer_observations_sha256": sha256_file(observations_path),
        "reviewer_registry_sha256": sha256_file(registry_path),
        "capability_freeze_sha256": sha256_file(capability_path),
    }
    write_json(output_root / "finalization_receipt.json", result)
    return result
