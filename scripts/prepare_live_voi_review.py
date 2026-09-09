from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file, write_json
from citeweave.live_voi_review import LiveSequentialVoiRouter


def _severity(risk: dict[str, Any]) -> str:
    score = (
        4 * int(bool(risk.get("causal_language")))
        + 2 * int(bool(risk.get("numeric")))
        + 2 * int(bool(risk.get("multiple_phenomena")))
        + int(bool(risk.get("discussion_interpretation")))
    )
    if risk.get("causal_language") and risk.get("multiple_phenomena"):
        return "critical"
    return "high" if score >= 3 else "medium"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze development-only live VOI routing from blinded article packets and "
            "pre-outcome time/budget estimates."
        )
    )
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument(
        "--time-estimates",
        type=Path,
        required=True,
        help="JSON object with packet_seconds keyed by every article packet ID.",
    )
    parser.add_argument(
        "--reviewer-budgets",
        type=Path,
        required=True,
        help="JSON object with reviewer_budget_seconds keyed by article reviewer code.",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    packet_root = args.packet_root.resolve()
    config_path = packet_root / "live_voi_config.json"
    freeze_path = packet_root / "live_voi_config_freeze.json"
    if config_path.exists() or freeze_path.exists():
        raise SystemExit("Refusing to overwrite live VOI config or freeze")
    returns = list((packet_root / "returns").glob("*.json"))
    if any((read_json(path).get("results") or []) for path in returns):
        raise SystemExit("Live VOI routing must be frozen before any human return")

    manifest_path = packet_root / "internal_manifest.json"
    manifest = read_json(manifest_path)
    article_packet_ids = sorted(
        {
            packet_id
            for layers in manifest["assignments"].values()
            for packet_id in layers.get("article", [])
        }
    )
    estimates_payload = read_json(args.time_estimates.resolve())
    budget_payload = read_json(args.reviewer_budgets.resolve())
    estimates = estimates_payload.get("packet_seconds") or {}
    budgets = budget_payload.get("reviewer_budget_seconds") or {}
    if set(estimates) != set(article_packet_ids):
        raise SystemExit("Time estimates must cover every and only assigned article packet")
    expected_reviewers = {
        reviewer
        for reviewer, layers in manifest["assignments"].items()
        if layers.get("article")
    }
    if set(budgets) != expected_reviewers:
        raise SystemExit("Reviewer budgets must cover every and only article reviewer")

    features = []
    for packet_id in article_packet_ids:
        path = packet_root / "packets" / "article" / f"{packet_id}.json"
        packet = read_json(path)
        dependencies = sorted(set(packet.get("allowed_decisive_evidence_ids") or []))
        if not dependencies:
            raise SystemExit(f"Article packet has no visible dependency IDs: {packet_id}")
        risk = packet.get("risk_features") or {}
        features.append(
            {
                "packet_id": packet_id,
                "routing_scope_id": packet.get("topic_code") or packet_id,
                "dependency_ids": dependencies,
                "severity": _severity(risk),
                "risk_features": risk,
                "packet_sha256": sha256_file(path),
            }
        )
    config = {
        "schema_version": 1,
        "status": "frozen_live_sequential_voi_development",
        "development_only": True,
        "formal_outcome_collection": False,
        "human_outcomes_before_freeze": 0,
        "policy": "sequential_dependency_value_of_information",
        "seed": args.seed,
        "posterior_prior": {"alpha_invalid": 1.0, "beta_sufficient": 4.0},
        "selection_time_source": "pre_outcome_frozen_estimates_only",
        "actual_time_use": "posthoc_reporting_only",
        "reviewer_outcome_visibility": "reviewer_private_no_cross_reviewer_update",
        "source_artifacts": {
            "internal_manifest_sha256": sha256_file(manifest_path),
            "time_estimates_sha256": sha256_file(args.time_estimates.resolve()),
            "reviewer_budgets_sha256": sha256_file(args.reviewer_budgets.resolve()),
        },
        "features": features,
        "estimated_review_seconds": estimates,
        "reviewer_budget_seconds": budgets,
    }
    write_json(config_path, config)
    write_json(
        freeze_path,
        {
            "schema_version": 1,
            "status": "live_voi_config_frozen_before_human_outcomes",
            "sha256": sha256_file(config_path),
            "human_outcomes_at_freeze": 0,
        },
    )
    LiveSequentialVoiRouter(config, manifest)
    print(config_path)
    print(freeze_path)


if __name__ == "__main__":
    main()
