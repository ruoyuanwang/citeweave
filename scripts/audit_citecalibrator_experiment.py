from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.citecalibrator_benchmark import canonical_sha256
from citeweave.io import read_json, sha256_file, write_json

EXPECTED_METRICS = {
    "major_critical_defect_recall",
    "accept_false_intervention_rate",
    "action_macro_f1",
    "mean_latency_seconds",
}
EXPECTED_LLM_JUDGE_METRICS = {
    "inter_judge_action_kappa",
    "consensus_major_critical_defect_rate",
    "rule_major_critical_defect_recall",
    "rule_accept_false_intervention_rate",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_cases(path: Path, expected_count: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    _require(len(rows) == expected_count, f"Unexpected case count: {path}")
    ids = [str(row["case_id"]) for row in rows]
    _require(len(ids) == len(set(ids)), f"Duplicate case IDs: {path}")
    for row in rows:
        _require("condition" not in row, f"Reviewer-visible condition leaked: {row['case_id']}")
        _require(
            row.get("eligibility", {}).get("eligible_for_natural_prevalence") is False,
            f"Development case incorrectly permits prevalence use: {row['case_id']}",
        )
        material = {key: value for key, value in row.items() if key != "review"}
        expected_hash = row.get("lineage", {}).get("case_material_sha256")
        material.get("lineage", {}).pop("case_material_sha256", None)
        _require(
            canonical_sha256(material) == expected_hash,
            f"Case material hash mismatch: {row['case_id']}",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    protocol_path = root / "protocol.yml"
    freeze_path = root / "protocol_freeze.json"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    _require(
        hashlib.sha256(protocol_path.read_bytes()).hexdigest() == freeze["sha256"],
        "Protocol freeze hash mismatch",
    )
    _require(
        set(protocol["primary_metrics"]) == EXPECTED_METRICS
        and len(protocol["primary_metrics"]) == 4,
        "Primary metric set is not the frozen four-metric set",
    )

    pilot_root = root / "pilot_benchmark"
    pilot_manifest = read_json(pilot_root / "manifest.json")
    _require(pilot_manifest["natural_prevalence_claims_prohibited"] is True, "Pilot role leak")
    _require(
        sha256_file(pilot_root / "cases.jsonl") == pilot_manifest["cases_sha256"],
        "Pilot case file hash mismatch",
    )
    pilot_cases = _verify_cases(pilot_root / "cases.jsonl", 114)
    qualified_cases = _read_jsonl(pilot_root / "qualified_article_cases.jsonl")
    failed_article_cases = _read_jsonl(pilot_root / "failed_article_cases.jsonl")
    _require(len(qualified_cases) == 63, "Qualified pilot subset count mismatch")
    _require(len(failed_article_cases) == 51, "Failed-article pilot subset count mismatch")
    _require(
        {row["case_id"] for row in qualified_cases}
        | {row["case_id"] for row in failed_article_cases}
        == {row["case_id"] for row in pilot_cases},
        "Pilot subsets do not partition the full case set",
    )
    _require(
        not (
            {row["case_id"] for row in qualified_cases}
            & {row["case_id"] for row in failed_article_cases}
        ),
        "Pilot subsets overlap",
    )
    _require(len(pilot_manifest["review_packets"]) == 114, "Pilot packet count mismatch")
    primary_template_path = pilot_root / "primary_review_returns_template.jsonl"
    primary_template = _read_jsonl(primary_template_path)
    _require(len(primary_template) == 228, "Primary review template count mismatch")
    template_slots: dict[str, set[str]] = {}
    for row in primary_template:
        template_slots.setdefault(str(row["case_id"]), set()).add(str(row["review_slot"]))
    _require(
        set(template_slots) == {row["case_id"] for row in pilot_cases}
        and all(slots == {"primary_a", "primary_b"} for slots in template_slots.values()),
        "Every pilot case must have exactly two primary review slots",
    )
    _require(
        all(
            row["provider_calls"] == 9
            and set(row["preexisting_section_repairs"]) == {"Results", "Discussion"}
            for row in pilot_manifest["source_artifacts"]
        ),
        "Preexisting generation/repair calls were not preserved in pilot lineage",
    )
    for packet in pilot_manifest["review_packets"]:
        packet_path = Path(packet["path"])
        _require(packet_path.is_file(), f"Missing review packet: {packet_path}")
        _require(sha256_file(packet_path) == packet["sha256"], "Review packet hash mismatch")

    challenge_root = root / "controlled_challenge"
    challenge_manifest = read_json(challenge_root / "manifest.json")
    _require(
        challenge_manifest["training_need_claims_prohibited"] is True,
        "Challenge role permits an invalid training-need claim",
    )
    _require(
        sha256_file(challenge_root / "cases.jsonl") == challenge_manifest["cases_sha256"],
        "Challenge case file hash mismatch",
    )
    _require(
        sha256_file(challenge_root / "gold.jsonl") == challenge_manifest["gold_sha256"],
        "Challenge gold file hash mismatch",
    )
    challenge_cases = _verify_cases(challenge_root / "cases.jsonl", 40)
    challenge_gold = _read_jsonl(challenge_root / "gold.jsonl")
    _require(
        {row["case_id"] for row in challenge_cases}
        == {row["case_id"] for row in challenge_gold},
        "Challenge cases and gold differ",
    )

    baseline_root = root / "baselines" / "rules"
    pilot_predictions = _read_jsonl(baseline_root / "pilot" / "predictions.jsonl")
    qualified_predictions = _read_jsonl(
        baseline_root / "pilot_qualified" / "predictions.jsonl"
    )
    failed_article_predictions = _read_jsonl(
        baseline_root / "pilot_failed_article" / "predictions.jsonl"
    )
    challenge_predictions = _read_jsonl(
        baseline_root / "controlled_challenge" / "predictions.jsonl"
    )
    _require(
        {row["case_id"] for row in pilot_predictions}
        == {row["case_id"] for row in pilot_cases},
        "Pilot rule predictions are incomplete",
    )
    _require(
        {row["case_id"] for row in qualified_predictions}
        == {row["case_id"] for row in qualified_cases},
        "Qualified pilot rule predictions are incomplete",
    )
    _require(
        {row["case_id"] for row in failed_article_predictions}
        == {row["case_id"] for row in failed_article_cases},
        "Failed-article pilot rule predictions are incomplete",
    )
    _require(
        {row["case_id"] for row in challenge_predictions}
        == {row["case_id"] for row in challenge_cases},
        "Challenge rule predictions are incomplete",
    )
    analysis = read_json(challenge_root / "rule_baseline_analysis.json")
    _require(
        analysis["natural_prevalence_interpretation_permitted"] is False,
        "Challenge analysis incorrectly permits prevalence interpretation",
    )
    metrics = analysis["conditions"]["deterministic_rules_v1"]["metrics"]
    _require(set(metrics) == EXPECTED_METRICS, "Analysis metric set has drifted")

    readiness = read_json(root / "human_review_readiness.json")
    _require(readiness["real_reviewers_registered"] == 0, "Readiness record is stale")

    judge_root = root / "llm_judging_v1"
    judge_protocol_path = judge_root / "judge_protocol.yml"
    judge_freeze = read_json(judge_root / "judge_protocol_freeze.json")
    _require(
        sha256_file(judge_protocol_path) == judge_freeze["sha256"],
        "LLM judge protocol freeze hash mismatch",
    )
    judge_protocol = yaml.safe_load(judge_protocol_path.read_text(encoding="utf-8"))
    _require(
        set(judge_protocol["primary_metrics"]) == EXPECTED_LLM_JUDGE_METRICS
        and len(judge_protocol["primary_metrics"]) == 4,
        "LLM judge primary metric set has drifted",
    )
    assignment_manifest = read_json(judge_root / "assignment_manifest.json")
    dispatch_manifest = read_json(judge_root / "dispatch_manifest.json")
    _require(
        assignment_manifest["context_inheritance"] is False
        and assignment_manifest["cross_judge_visibility"] is False,
        "Judge assignment isolation is not explicit",
    )
    _require(
        dispatch_manifest["fork_context"] is False
        and dispatch_manifest["shared_parent_history"] is False
        and dispatch_manifest["cross_judge_input_or_output_access"] is False,
        "Judge dispatch isolation is not explicit",
    )
    judge_outputs: dict[str, list[dict[str, Any]]] = {}
    for judge_id in ("judge_a", "judge_b"):
        assignment = read_json(judge_root / judge_id / "assignment.json")
        _require(
            assignment["inherit_parent_context"] is False
            and assignment["forbidden_cross_judge_access"] is True,
            f"Isolation flags invalid for {judge_id}",
        )
        input_root = judge_root / judge_id / "input"
        _require(
            sha256_file(input_root / "cases.jsonl") == sha256_file(pilot_root / "cases.jsonl"),
            f"Frozen case input differs for {judge_id}",
        )
        _require(
            sha256_file(input_root / "judge_protocol.yml") == judge_freeze["sha256"],
            f"Frozen protocol input differs for {judge_id}",
        )
        reviews = _read_jsonl(judge_root / judge_id / "output" / "reviews.jsonl")
        _require(len(reviews) == 114, f"Unexpected review count for {judge_id}")
        _require(
            [row["case_id"] for row in reviews] == [row["case_id"] for row in pilot_cases],
            f"Review IDs/order differ for {judge_id}",
        )
        _require(
            all(row.get("reviewer_id") == judge_id for row in reviews),
            f"Reviewer identity mismatch for {judge_id}",
        )
        judge_outputs[judge_id] = reviews

    aggregate_root = judge_root / "aggregate"
    judge_analysis = read_json(aggregate_root / "analysis.json")
    consensus_reviews = _read_jsonl(aggregate_root / "consensus_reviews.jsonl")
    disagreements = _read_jsonl(aggregate_root / "disagreements.jsonl")
    _require(
        set(judge_analysis["primary_metrics"]) == EXPECTED_LLM_JUDGE_METRICS
        and len(judge_analysis["primary_metrics"]) == 4,
        "Aggregated LLM judge metric set has drifted",
    )
    _require(
        len(consensus_reviews) + len(disagreements) == len(pilot_cases)
        and {row["case_id"] for row in consensus_reviews}
        | {row["case_id"] for row in disagreements}
        == {row["case_id"] for row in pilot_cases},
        "Consensus and disagreement outputs do not partition the frozen cases",
    )
    _require(
        judge_analysis["input_hashes"]["judge_a"]
        == sha256_file(judge_root / "judge_a" / "output" / "reviews.jsonl")
        and judge_analysis["input_hashes"]["judge_b"]
        == sha256_file(judge_root / "judge_b" / "output" / "reviews.jsonl"),
        "Aggregated judge output hashes are stale",
    )
    _require(
        judge_analysis["provisional_training_need_supported"] is True
        and judge_analysis["formal_training_need_proven"] is False,
        "Training-need conclusion does not preserve the proxy-gold boundary",
    )
    result = {
        "schema_version": 1,
        "status": "benchmark_rule_and_two_llm_judge_experiment_verified",
        "root": str(root),
        "protocol_sha256": freeze["sha256"],
        "checks": {
            "protocol_frozen": True,
            "primary_metric_count": 4,
            "pilot_cases": len(pilot_cases),
            "qualified_article_cases": len(qualified_cases),
            "failed_article_cases": len(failed_article_cases),
            "pilot_packets": len(pilot_manifest["review_packets"]),
            "pilot_primary_review_slots": len(primary_template),
            "preexisting_provider_calls_per_article": 9,
            "preexisting_repaired_sections": ["Results", "Discussion"],
            "controlled_challenge_cases": len(challenge_cases),
            "rule_pilot_predictions": len(pilot_predictions),
            "rule_qualified_pilot_predictions": len(qualified_predictions),
            "rule_failed_article_predictions": len(failed_article_predictions),
            "rule_challenge_predictions": len(challenge_predictions),
            "reviewer_condition_blinding": True,
            "lineage_hashes_verified": True,
            "isolated_llm_judges": len(judge_outputs),
            "llm_reviews_per_judge": 114,
            "llm_consensus_cases": len(consensus_reviews),
            "llm_unresolved_cases": len(disagreements),
            "llm_primary_metric_count": 4,
            "llm_judge_protocol_frozen": True,
            "cross_judge_visibility": False,
        },
        "training_need_conclusion": {
            "ready_for_training_experiment": True,
            "formal_training_need_proven": False,
            "reason": (
                "Two isolated LLM judges pass the preregistered reliability, problem-size, "
                "and rule-gap gates on development cases. Human gold and held-out topics "
                "remain necessary for a formal training-over-prompting claim."
            ),
            "external_llm_api_calls_used": False,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
