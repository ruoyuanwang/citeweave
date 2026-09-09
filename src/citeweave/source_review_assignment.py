from __future__ import annotations

import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .io import read_json, sha256_file, write_json
from .source_relevance_packets import audit_source_relevance_packets


def _order(seed: int, *parts: str) -> str:
    return hashlib.sha256("\x1f".join((str(seed), *parts)).encode()).hexdigest()


def validate_source_warmup_protocol(
    *,
    protocol_path: Path,
    protocol_freeze_path: Path,
    packet_root: Path,
    amendment_path: Path,
    amendment_freeze_path: Path,
) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_freeze = read_json(protocol_freeze_path)
    if protocol_freeze.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Source warmup protocol differs from its freeze")
    if protocol_freeze.get("protocol_id") != protocol.get("protocol_id"):
        raise RuntimeError("Source warmup protocol identity mismatch")
    if protocol.get("human_judgments_inspected") is not False:
        raise RuntimeError("Source warmup protocol must remain pre-judgment")
    if protocol.get("cases", {}).get("confirmatory_exclusion") is not True:
        raise RuntimeError("Source warmup cases must remain confirmatory-excluded")

    packet_manifest_path = packet_root / "manifest.json"
    packet_manifest = read_json(packet_manifest_path)
    expected_cases = int(protocol_freeze.get("expected_cases", 0))
    if (
        packet_manifest.get("status")
        != "warmup_packets_built_before_human_judgments"
        or packet_manifest.get("confirmatory_exclusion") is not True
        or int(packet_manifest.get("packets", 0)) != expected_cases
    ):
        raise RuntimeError("Source warmup packet manifest violates frozen protocol")

    amendment_hash = sha256_file(amendment_path)
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    amendment_freeze = read_json(amendment_freeze_path)
    if (
        amendment_freeze.get("sha256") != amendment_hash
        or amendment_freeze.get("amendment_id") != amendment.get("amendment_id")
        or amendment.get("base_protocol_sha256") != protocol_hash
        or amendment.get("packet_manifest_sha256")
        != sha256_file(packet_manifest_path)
    ):
        raise RuntimeError("Source warmup execution amendment identity/hash mismatch")
    for label, artifact in (amendment.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Source warmup implementation mismatch: {label}")
    return protocol


def assess_source_warmup_panel_readiness(
    *, packet_root: Path, reviewer_roster_path: Path
) -> dict[str, Any]:
    packet_audit = audit_source_relevance_packets(packet_root)
    roster = read_json(reviewer_roster_path)
    reviewers = roster.get("reviewers") or []
    reviewer_ids = [str(row.get("reviewer_id") or "") for row in reviewers]
    domains = set(packet_audit.get("dataset_packet_counts") or {})
    eligible = {
        str(row.get("reviewer_id") or ""): (
            set(map(str, row.get("eligible_domains") or []))
            - set(map(str, row.get("conflicted_domains") or []))
        )
        & domains
        for row in reviewers
    }
    eligible_by_domain = {
        domain: sum(domain in values for values in eligible.values())
        for domain in domains
    }
    checks = {
        "packet_audit_passed": packet_audit.get("status") == "passed",
        "exactly_six_reviewers": len(reviewers) == 6,
        "unique_nonempty_reviewer_ids": bool(reviewer_ids)
        and all(reviewer_ids)
        and len(reviewer_ids) == len(set(reviewer_ids)),
        "each_reviewer_has_two_domains": bool(reviewers)
        and all(len(values) >= 2 for values in eligible.values()),
        "each_domain_supports_two_plus_adjudicator": bool(domains)
        and all(value >= 3 for value in eligible_by_domain.values()),
        "conflicts_declared": bool(reviewers)
        and all("conflicted_domains" in row for row in reviewers),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "status": "ready" if not blockers else "blocked",
        "checks": checks,
        "blocking_reasons": blockers,
        "reviewers": len(reviewers),
        "eligible_reviewers_by_domain": dict(sorted(eligible_by_domain.items())),
        "packet_manifest_sha256": packet_audit.get("manifest_sha256"),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
    }


def build_source_warmup_assignment(
    *,
    packet_root: Path,
    reviewer_roster_path: Path,
    output_root: Path,
    seed: int = 20260826,
) -> dict[str, Any]:
    packet_manifest_path = packet_root / "manifest.json"
    packet_manifest = read_json(packet_manifest_path)
    if (
        packet_manifest.get("status")
        != "warmup_packets_built_before_human_judgments"
        or packet_manifest.get("confirmatory_exclusion") is not True
    ):
        raise ValueError("Source warmup packets are not frozen and excluded")
    roster = read_json(reviewer_roster_path)
    reviewers = roster.get("reviewers") or []
    reviewer_ids = [str(row.get("reviewer_id") or "") for row in reviewers]
    if (
        len(reviewers) != 6
        or any(not value for value in reviewer_ids)
        or len(reviewer_ids) != len(set(reviewer_ids))
    ):
        raise ValueError("Source warmup requires exactly six unique reviewers")
    domains = set(packet_manifest["dataset_packet_counts"])
    eligible = {
        str(row["reviewer_id"]): (
            set(map(str, row.get("eligible_domains") or []))
            - set(map(str, row.get("conflicted_domains") or []))
        )
        & domains
        for row in reviewers
    }
    if any(len(values) < 2 for values in eligible.values()):
        raise ValueError("Every reviewer requires at least two eligible domains")
    if any(
        sum(domain in values for values in eligible.values()) < 3
        for domain in domains
    ):
        raise ValueError(
            "Every domain requires two independent reviewers plus an adjudicator"
        )
    records = packet_manifest["records"]
    if len(records) != 120:
        raise ValueError("Source warmup requires exactly 120 packets")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_domain[str(row["dataset_id"])].append(row)
    if set(map(len, by_domain.values())) != {15}:
        raise ValueError("Source warmup requires 15 packets per domain")
    if any(
        not any(domain in eligible[reviewer_id] for reviewer_id in reviewer_ids)
        for domain in domains
    ):
        raise ValueError("At least one domain has no eligible reviewer")

    first_load: Counter[str] = Counter()
    total_load: Counter[str] = Counter()
    reviewer_domains: dict[str, Counter[str]] = defaultdict(Counter)
    assignments: dict[str, list[str]] = defaultdict(list)
    case_reviewers: dict[str, list[str]] = defaultdict(list)
    ordered_records = sorted(
        records,
        key=lambda row: (
            sum(row["dataset_id"] in eligible[value] for value in reviewer_ids),
            _order(seed, "first", str(row["packet_id"])),
        ),
    )
    for row in ordered_records:
        packet_id = str(row["packet_id"])
        domain = str(row["dataset_id"])
        candidates = [value for value in reviewer_ids if domain in eligible[value]]
        selected = min(
            candidates,
            key=lambda reviewer_id: (
                first_load[reviewer_id],
                reviewer_domains[reviewer_id][domain],
                total_load[reviewer_id],
                _order(seed, "first-reviewer", packet_id, reviewer_id),
            ),
        )
        assignments[selected].append(packet_id)
        case_reviewers[packet_id].append(selected)
        first_load[selected] += 1
        total_load[selected] += 1
        reviewer_domains[selected][domain] += 1

    double_cases = []
    extra_domains = set(
        sorted(domains, key=lambda value: _order(seed, "extra-domain", value))[:4]
    )
    for domain, rows in sorted(by_domain.items()):
        ordered = sorted(
            rows, key=lambda row: _order(seed, "double", str(row["packet_id"]))
        )
        double_cases.extend(ordered[: 8 if domain in extra_domains else 7])
    if len(double_cases) != 60:
        raise AssertionError("Double-review selection must contain 60 cases")

    second_load: Counter[str] = Counter()
    for row in sorted(
        double_cases, key=lambda value: _order(seed, "second", value["packet_id"])
    ):
        packet_id = str(row["packet_id"])
        domain = str(row["dataset_id"])
        first = case_reviewers[packet_id][0]
        candidates = [
            value
            for value in reviewer_ids
            if value != first and domain in eligible[value]
        ]
        if not candidates:
            raise ValueError(f"No independent second reviewer for {packet_id}")
        selected = min(
            candidates,
            key=lambda reviewer_id: (
                second_load[reviewer_id],
                total_load[reviewer_id],
                reviewer_domains[reviewer_id][domain],
                _order(seed, "second-reviewer", packet_id, reviewer_id),
            ),
        )
        assignments[selected].append(packet_id)
        case_reviewers[packet_id].append(selected)
        second_load[selected] += 1
        total_load[selected] += 1
        reviewer_domains[selected][domain] += 1

    if set(total_load.values()) != {30}:
        raise ValueError(f"Unable to balance 30 decisions per reviewer: {total_load}")
    if any(len(values) < 2 for values in reviewer_domains.values()):
        raise ValueError("Assignment failed the two-domain reviewer minimum")
    if sum(len(values) == 2 for values in case_reviewers.values()) != 60:
        raise AssertionError("Exactly 60 cases must receive independent double review")

    packet_output = output_root / "packets" / "source"
    for row in records:
        source = packet_root / row["public_path"]
        target = packet_output / f"{row['packet_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    internal_manifest = {
        "schema_version": 1,
        "status": "source_warmup_assignments_frozen_before_returns",
        "reviewers": sorted(reviewer_ids),
        "assignments": {
            reviewer_id: {
                "source": sorted(
                    assignments[reviewer_id],
                    key=lambda value: _order(seed, "delivery", reviewer_id, value),
                )
            }
            for reviewer_id in sorted(reviewer_ids)
        },
    }
    write_json(output_root / "internal_manifest.json", internal_manifest)
    assignment_rows = [
        {
            "packet_id": packet_id,
            "dataset_id": next(
                str(row["dataset_id"])
                for row in records
                if row["packet_id"] == packet_id
            ),
            "reviewer_ids": reviewers_for_case,
            "independent_first_passes": len(reviewers_for_case),
        }
        for packet_id, reviewers_for_case in sorted(case_reviewers.items())
    ]
    result = {
        "schema_version": 1,
        "status": "source_warmup_panel_ready_before_returns",
        "human_returns_present": False,
        "confirmatory_exclusion": True,
        "seed": seed,
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "reviewer_roster_sha256": sha256_file(reviewer_roster_path),
        "reviewers": 6,
        "packets": 120,
        "decisions": sum(total_load.values()),
        "double_reviewed_packets": 60,
        "decisions_per_reviewer": dict(sorted(total_load.items())),
        "domains_per_reviewer": {
            reviewer_id: sorted(reviewer_domains[reviewer_id])
            for reviewer_id in sorted(reviewer_ids)
        },
        "records": assignment_rows,
    }
    write_json(output_root / "assignment_manifest.json", result)
    return result
