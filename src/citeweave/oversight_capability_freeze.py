from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .complementary_oversight import ReviewerCapabilityModel, ReviewerObservation
from .io import read_json, sha256_file, write_json
from .oversight_calibration import CALIBRATION_ISSUES

REQUIRED_ISSUES = ("evidence_relevance", *CALIBRATION_ISSUES)


class OversightCapabilityFreezeError(ValueError):
    pass


def _observations(path: Path, *, expected_status: str) -> list[ReviewerObservation]:
    payload = read_json(path)
    if payload.get("status") != expected_status:
        raise OversightCapabilityFreezeError(f"Warmup observations are not finalized: {path}")
    if payload.get("confirmatory_exclusion") is not True and expected_status.startswith(
        "multidimensional"
    ):
        raise OversightCapabilityFreezeError("Calibration observations lack confirmatory exclusion")
    rows = payload.get("observations") or []
    result = []
    for row in rows:
        try:
            result.append(ReviewerObservation(**row))
        except (TypeError, ValueError) as exc:
            raise OversightCapabilityFreezeError("Invalid warmup observation schema") from exc
    return result


def freeze_integrated_reviewer_capabilities(
    *,
    source_observations_path: Path,
    calibration_observations_path: Path,
    reviewer_roster_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Fuse independent warmups once, before any held-out assignment or outcome."""
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise OversightCapabilityFreezeError("Refusing to overwrite capability freeze")
    source = _observations(
        source_observations_path,
        expected_status="warmup_capability_observations_resolved",
    )
    calibration = _observations(
        calibration_observations_path,
        expected_status="multidimensional_calibration_observations_resolved",
    )
    if len(source) != 120 or {row.issue_type for row in source} != {"evidence_relevance"}:
        raise OversightCapabilityFreezeError(
            "Source warmup must contain 120 evidence-relevance observations"
        )
    if len(calibration) != 144 or {row.issue_type for row in calibration} != set(
        CALIBRATION_ISSUES
    ):
        raise OversightCapabilityFreezeError(
            "Calibration warmup must contain 144 observations across all four issues"
        )
    combined = [*source, *calibration]
    observation_ids = [row.observation_id for row in combined]
    if len(observation_ids) != len(set(observation_ids)):
        raise OversightCapabilityFreezeError("Warmup observation IDs overlap")

    roster = read_json(reviewer_roster_path)
    roster_rows = roster.get("reviewers") or []
    reviewer_ids = [str(row.get("reviewer_id") or "") for row in roster_rows]
    if len(reviewer_ids) != 6 or any(not value for value in reviewer_ids):
        raise OversightCapabilityFreezeError("Integrated capability freeze requires six reviewers")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise OversightCapabilityFreezeError("Reviewer identities are duplicated")
    observed_reviewers = {row.reviewer_id for row in combined}
    if observed_reviewers != set(reviewer_ids):
        raise OversightCapabilityFreezeError("Warmup reviewer identities do not match roster")
    roster_index = {str(row["reviewer_id"]): row for row in roster_rows}
    for reviewer_id, row in roster_index.items():
        if row.get("conflicts_declared") is not True:
            raise OversightCapabilityFreezeError(
                f"Reviewer {reviewer_id} lacks an explicit conflict declaration"
            )
        eligible = set(map(str, row.get("eligible_domains") or []))
        conflicted = set(map(str, row.get("conflicted_domains") or []))
        if not eligible or eligible & conflicted:
            raise OversightCapabilityFreezeError(
                f"Reviewer {reviewer_id} has invalid domain eligibility"
            )

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    domains: dict[str, set[str]] = defaultdict(set)
    for row in combined:
        counts[row.reviewer_id][row.issue_type] += 1
        domains[row.reviewer_id].add(row.domain)
    for reviewer_id in reviewer_ids:
        if len(domains[reviewer_id]) < 2:
            raise OversightCapabilityFreezeError(
                f"Reviewer {reviewer_id} has fewer than two warmup domains"
            )
        for issue in REQUIRED_ISSUES:
            if counts[reviewer_id][issue] < 3:
                raise OversightCapabilityFreezeError(
                    f"Reviewer {reviewer_id} lacks three observations for {issue}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / "combined_reviewer_observations.json"
    registry_path = output_root / "reviewer_registry.json"
    write_json(
        combined_path,
        {
            "schema_version": 1,
            "status": "integrated_warmup_observations_frozen_before_heldout",
            "heldout_outcomes_inspected": False,
            "confirmatory_exclusion": True,
            "source_observations_sha256": sha256_file(source_observations_path),
            "calibration_observations_sha256": sha256_file(
                calibration_observations_path
            ),
            "observations": [asdict(row) for row in combined],
        },
    )
    registry_rows = []
    for reviewer_id in sorted(reviewer_ids):
        roster_row = roster_index[reviewer_id]
        registry_rows.append(
            {
                "reviewer_id": reviewer_id,
                "eligible_domains": sorted(set(roster_row["eligible_domains"])),
                "conflicted_domains": sorted(
                    set(roster_row.get("conflicted_domains") or [])
                ),
                "conflicts_declared": True,
                "warmup_cases_completed": sum(counts[reviewer_id].values()),
                "warmup_domains": sorted(domains[reviewer_id]),
                "warmup_issue_types": dict(sorted(counts[reviewer_id].items())),
                "warmup_outcomes_adjudicated": True,
                "warmup_objective_injected_gold_scoring": True,
                "warmup_excluded_from_confirmatory": True,
            }
        )
    write_json(
        registry_path,
        {
            "schema_version": 1,
            "status": "integrated_reviewer_registry_frozen_before_heldout",
            "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
            "combined_observations_sha256": sha256_file(combined_path),
            "reviewers": registry_rows,
        },
    )

    model = ReviewerCapabilityModel(
        combined,
        prior_alpha=2.0,
        prior_beta=2.0,
        minimum_exact_observations=3,
        minimum_issue_observations=3,
        minimum_domain_observations=5,
    )
    all_domains = sorted({row.domain for row in combined})
    profiles = []
    for reviewer_id in sorted(reviewer_ids):
        for domain in all_domains:
            for issue in REQUIRED_ISSUES:
                profile = asdict(
                    model.estimate(reviewer_id, domain=domain, issue_type=issue)
                )
                profile["personalized_evidence"] = profile["evidence_tier"] in {
                    "exact",
                    "issue",
                    "domain",
                }
                profiles.append(profile)
    capability_path = output_root / "capability_freeze.json"
    capability = {
        "schema_version": 2,
        "status": "frozen_before_heldout_assignment",
        "heldout_outcomes_inspected": False,
        "predecision_sha256": sha256_file(combined_path),
        "reviewer_ids": sorted(reviewer_ids),
        "required_issue_types": list(REQUIRED_ISSUES),
        "prior_alpha": 2.0,
        "prior_beta": 2.0,
        "minimum_exact_observations": 3,
        "minimum_issue_observations": 3,
        "minimum_domain_observations": 5,
        "fallback_order": ["exact", "issue", "domain", "global", "prior"],
        "minimum_personalized_evidence_tiers": ["exact", "issue", "domain"],
        "source_observations_sha256": sha256_file(source_observations_path),
        "calibration_observations_sha256": sha256_file(
            calibration_observations_path
        ),
        "combined_observations_sha256": sha256_file(combined_path),
        "reviewer_registry_sha256": sha256_file(registry_path),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
        "profiles": profiles,
    }
    write_json(capability_path, capability)
    result = {
        "schema_version": 1,
        "status": "integrated_reviewer_capability_frozen_before_heldout",
        "heldout_outcomes_inspected": False,
        "reviewers": len(reviewer_ids),
        "observations": len(combined),
        "source_observations": len(source),
        "multidimensional_observations": len(calibration),
        "issue_types": list(REQUIRED_ISSUES),
        "combined_observations_sha256": sha256_file(combined_path),
        "reviewer_registry_sha256": sha256_file(registry_path),
        "capability_freeze_sha256": sha256_file(capability_path),
    }
    write_json(output_root / "freeze_receipt.json", result)
    return result
