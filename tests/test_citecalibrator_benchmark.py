from __future__ import annotations

import json

import pytest

from citeweave.citecalibrator_benchmark import (
    ACTIONS,
    build_controlled_challenge,
    cohen_kappa,
    deterministic_rule_judge,
    load_perturbation_profile,
    resolve_human_reviews,
    score_predictions,
    summarize_gold,
    validate_review,
)


def _case(claim: str) -> dict:
    return {
        "case_id": "C1",
        "atomic_claim": claim,
        "allowed_evidence_ids": ["PH-one", "REF-one", "N1"],
        "phenomena": [
            {
                "phenomenon_id": "PH-one",
                "verified_answer": {"hops": 2},
                "operator_trace": [{"operator": "shortest_path", "hops": 2}],
            }
        ],
        "sources": [{"reference_id": "REF-one", "year": 2024}],
        "perturbation_profiles": {},
    }


def test_rule_judge_accepts_bounded_supported_claim() -> None:
    result = deterministic_rule_judge(
        _case("The observed path contains 2 hops in this corpus. PH-one REF-one")
    )
    assert result["action"] == "accept"
    assert result["risk_types"] == []


def test_rule_judge_flags_causality_and_unknown_number() -> None:
    result = deterministic_rule_judge(
        _case("The path drives the entire field through 99 hops. PH-one REF-one")
    )
    assert result["action"] == "reject"
    assert set(result["risk_types"]) == {
        "causal_overclaim",
        "numeric_error",
        "scope_overclaim",
    }


def test_rule_judge_accepts_grouped_and_rounded_supported_numbers() -> None:
    case = _case("The observed value is 92,988.0 and the ratio is 382.142857. PH-one REF-one")
    case["phenomena"][0]["operator_trace"] = [
        {"importance": 92988.0, "ratio": 382.14285714285717}
    ]
    result = deterministic_rule_judge(case)
    assert result["action"] == "accept"
    assert "numeric_error" not in result["risk_types"]


def test_rule_judge_flags_missing_evidence_and_unstable_robustness() -> None:
    case = _case("The result is stable. PH-one")
    case["perturbation_profiles"] = {
        "PH-one": {
            "variants": [
                {"comparison_to_baseline": {"comparable": False, "score_change": 0.1}}
            ]
        }
    }
    result = deterministic_rule_judge(case)
    assert result["action"] == "qualify"
    assert set(result["risk_types"]) == {
        "missing_evidence",
        "parameter_sensitivity_omitted",
    }


def test_load_perturbation_profile_compacts_measurements(tmp_path) -> None:
    robustness_root = tmp_path / "robustness"
    topic_dir = robustness_root / "topic-one"
    topic_dir.mkdir(parents=True)
    payload = {
        "variant": {"variant_id": "V1", "family": "threshold", "level": 0.5},
        "measurements": {
            "shortest_path": {
                "answer": 2,
                "members": ["A", "B"],
                "nested": {"score": 0.9, "path": ["A", "B"]},
            }
        },
        "comparison": {"shortest_path": {"answer_equal": True}},
    }
    (topic_dir / "variant.json").write_text(json.dumps(payload), encoding="utf-8")

    profile = load_perturbation_profile(
        robustness_root, dataset_id="topic-one", task_type="shortest_path"
    )
    assert profile["available"] is True
    assert profile["variants"][0]["measurement"] == {
        "answer": 2,
        "nested": {"score": 0.9},
    }
    assert profile["variants"][0]["parameters"] == {"level": 0.5}
    assert len(profile["variants"][0]["artifact_sha256"]) == 64

    missing = load_perturbation_profile(
        robustness_root, dataset_id="missing", task_type="shortest_path"
    )
    assert missing == {
        "available": False,
        "reason": "topic_robustness_missing",
        "variants": [],
    }


def test_controlled_challenge_materializes_four_interventions(tmp_path) -> None:
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_case = {
        **_case("The observed path contains 2 hops. PH-one REF-one"),
        "topic_id": "topic-one",
        "article_id": "article-one",
        "lineage": {"case_material_sha256": "source-sha"},
    }
    pilot_path.write_text(json.dumps(pilot_case) + "\n", encoding="utf-8")

    manifest = build_controlled_challenge(
        pilot_cases_path=pilot_path, output_dir=tmp_path / "challenge"
    )
    assert manifest["cases"] == 4
    assert manifest["phenomena"] == 1
    assert manifest["action_counts"] == {"accept": 1, "qualify": 2, "reject": 1}

    case_rows = [
        json.loads(line)
        for line in (tmp_path / "challenge" / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    gold_rows = [
        json.loads(line)
        for line in (tmp_path / "challenge" / "gold.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(case_rows) == len(gold_rows) == 4
    assert {row["benchmark_split"] for row in case_rows} == {
        "controlled_challenge_development_only"
    }
    assert all(row["lineage"]["source_case_id"] == "C1" for row in case_rows)


def test_controlled_challenge_rejects_pilot_without_phenomena(tmp_path) -> None:
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(json.dumps({"case_id": "C1", "phenomena": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no graph phenomena"):
        build_controlled_challenge(
            pilot_cases_path=pilot_path, output_dir=tmp_path / "challenge"
        )


def test_core_metrics_are_small_and_decision_relevant() -> None:
    gold = [
        {"case_id": "A", "action": "accept", "severity": "minor"},
        {"case_id": "B", "action": "qualify", "severity": "major"},
        {"case_id": "C", "action": "reject", "severity": "critical"},
        {"case_id": "D", "action": "abstain", "severity": "major"},
    ]
    predictions = [
        {"case_id": "A", "action": "accept", "latency_seconds": 1},
        {"case_id": "B", "action": "qualify", "latency_seconds": 1},
        {"case_id": "C", "action": "accept", "latency_seconds": 1},
        {"case_id": "D", "action": "abstain", "latency_seconds": 1},
    ]
    result = score_predictions(gold, predictions)
    assert result["metrics"]["major_critical_defect_recall"] == 0.5
    assert result["metrics"]["accept_false_intervention_rate"] == 0.0
    assert result["metrics"]["mean_latency_seconds"] == 1.0
    assert set(result["metrics"]) == {
        "major_critical_defect_recall",
        "accept_false_intervention_rate",
        "action_macro_f1",
        "mean_latency_seconds",
    }


def test_gold_summary_keeps_only_two_prevalence_outcomes() -> None:
    rows = [
        {
            "case_id": "A",
            "article_id": "P1",
            "action": "accept",
            "severity": "minor",
            "risk_types": [],
        },
        {
            "case_id": "B",
            "article_id": "P2",
            "action": "qualify",
            "severity": "major",
            "risk_types": ["scope_overclaim"],
            "minimal_revision": "A bounded claim.",
        },
    ]
    result = summarize_gold(rows)
    assert result["major_critical_defect_rate"] == 0.5
    assert result["articles_with_major_critical_defect_rate"] == 0.5


def _human_review(case_id: str, reviewer_id: str, action: str) -> dict:
    return {
        "case_id": case_id,
        "reviewer_id": reviewer_id,
        "factual_supported": action == "accept",
        "interpretation_calibrated": action == "accept",
        "alternative_adequate": action == "accept",
        "evidence_sufficient": True,
        "action": action,
        "risk_types": [] if action == "accept" else ["scope_overclaim"],
        "severity": "minor" if action == "accept" else "major",
        "minimal_revision": None if action == "accept" else "Bound the claim.",
    }


def test_human_resolution_requires_blind_adjudication_for_disagreement() -> None:
    cases = [{"case_id": "A", "topic_id": "T1", "article_id": "P1"}]
    primary = [
        _human_review("A", "R1", "accept"),
        _human_review("A", "R2", "qualify"),
    ]
    pending = resolve_human_reviews(cases=cases, primary_reviews=primary)
    assert pending["status"] == "awaiting_human_reviews"
    assert len(pending["adjudication_packets"]) == 1
    assert pending["adjudication_packets"][0]["topic_id"] == "T1"
    assert "review" in pending["adjudication_packets"][0]

    resolved = resolve_human_reviews(
        cases=cases,
        primary_reviews=primary,
        adjudications=[_human_review("A", "R3", "qualify")],
    )
    assert resolved["status"] == "resolved"
    assert resolved["gold"][0]["resolution_source"] == "adjudicated"


def test_human_resolution_reports_missing_reviews_and_accepts_consensus() -> None:
    cases = [
        {"case_id": "A", "topic_id": "T1", "article_id": "P1"},
        {"case_id": "B", "topic_id": "T1", "article_id": "P2"},
    ]
    reviews = [
        _human_review("A", "R1", "accept"),
        _human_review("A", "R2", "accept"),
    ]
    result = resolve_human_reviews(cases=cases, primary_reviews=reviews)
    assert result["status"] == "awaiting_human_reviews"
    assert result["missing_primary_case_ids"] == ["B"]
    assert result["gold"][0]["resolution_source"] == "primary_consensus"


def test_human_resolution_rejects_invalid_review_routing() -> None:
    cases = [{"case_id": "A", "topic_id": "T1", "article_id": "P1"}]
    with pytest.raises(ValueError, match="unknown case"):
        resolve_human_reviews(
            cases=cases,
            primary_reviews=[_human_review("unknown", "R1", "accept")],
        )

    primary = [
        _human_review("A", "R1", "accept"),
        _human_review("A", "R2", "qualify"),
    ]
    with pytest.raises(ValueError, match="Adjudicator must differ"):
        resolve_human_reviews(
            cases=cases,
            primary_reviews=primary,
            adjudications=[_human_review("A", "R1", "accept")],
        )

    consensus = [
        _human_review("A", "R1", "accept"),
        _human_review("A", "R2", "accept"),
    ]
    with pytest.raises(ValueError, match="must not receive adjudication"):
        resolve_human_reviews(
            cases=cases,
            primary_reviews=consensus,
            adjudications=[_human_review("A", "R3", "accept")],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"action": "invalid"}, "Invalid review action"),
        ({"severity": "invalid"}, "Invalid review severity"),
        ({"risk_types": ["unknown"]}, "Unknown risk types"),
        ({"risk_types": ["scope_overclaim"]}, "cannot carry"),
        (
            {
                "action": "qualify",
                "severity": "major",
                "risk_types": ["scope_overclaim"],
            },
            "require a minimal revision",
        ),
    ],
)
def test_validate_review_rejects_invalid_decisions(updates, message) -> None:
    review = {"action": "accept", "severity": "minor", "risk_types": []}
    review.update(updates)
    with pytest.raises(ValueError, match=message):
        validate_review(review)


def test_cohen_kappa_handles_perfect_and_chance_level_agreement() -> None:
    assert cohen_kappa(
        ["accept", "qualify", "reject"],
        ["accept", "qualify", "reject"],
        labels=ACTIONS,
    ) == 1.0
    assert cohen_kappa(
        ["accept", "accept", "reject", "reject"],
        ["accept", "reject", "accept", "reject"],
        labels=ACTIONS,
    ) == 0.0
