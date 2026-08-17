from citeweave.adaptive_review import (
    AdaptiveReviewPolicy,
    FeedbackMemory,
    ReviewDecision,
    ReviewIssue,
    detect_acquisition_issues,
    detect_data_quality_issues,
    detect_graph_claim_issues,
    one_sided_clopper_pearson_lower,
    summarize_review_program,
)


def _issue(dataset_id="d3", severity="low"):
    return ReviewIssue(
        item_id=f"{dataset_id}:item",
        dataset_id=dataset_id,
        stage="acquisition",
        issue_signature="truncated_registered_sample",
        message="The acquisition is capped.",
        severity=severity,
        detector_score=1.0,
        payload={},
    )


def _decision(dataset_id, value="accept"):
    return ReviewDecision(
        decision_id=f"R-{dataset_id}",
        timestamp="2026-08-06T00:00:00+00:00",
        reviewer_code="H01",
        dataset_id=dataset_id,
        stage="acquisition",
        item_id=f"{dataset_id}:item",
        issue_signature="truncated_registered_sample",
        detector_score=1.0,
        decision=value,
        original={},
        correction=None,
        reason="The capped sample is registered and correctly disclosed.",
        review_seconds=5.0,
        feedback_memory_version=0,
    )


def test_policy_reduces_review_after_cross_dataset_confirmations():
    memory = FeedbackMemory([_decision("d1"), _decision("d2")])
    policy = AdaptiveReviewPolicy(memory, minimum_confirmations=2, audit_rate=0)

    action = policy.decide(_issue())

    assert action.action == "auto_accept"
    assert action.prior_examples == 2
    assert action.estimated_accept_probability == 1.0


def test_policy_does_not_auto_accept_high_severity_or_mixed_feedback():
    high_policy = AdaptiveReviewPolicy(
        FeedbackMemory([_decision("d1"), _decision("d2")]),
        audit_rate=0,
    )
    assert high_policy.decide(_issue(severity="high")).action == "escalate"

    mixed_policy = AdaptiveReviewPolicy(
        FeedbackMemory([_decision("d1"), _decision("d2", "reject")]),
        audit_rate=0,
    )
    assert mixed_policy.decide(_issue()).action == "escalate"


def test_repeated_examples_from_one_dataset_are_one_confirmation():
    repeated = [_decision("d1"), _decision("d1")]
    policy = AdaptiveReviewPolicy(FeedbackMemory(repeated), audit_rate=0)

    assert policy.decide(_issue("d3")).action == "escalate"


def test_context_guard_prevents_transfer_even_with_confirmations():
    issue = _issue("d3")
    issue = ReviewIssue(
        **{
            **issue.__dict__,
            "payload": {"auto_accept_context_ok": False},
        }
    )
    policy = AdaptiveReviewPolicy(
        FeedbackMemory([_decision("d1"), _decision("d2")]),
        audit_rate=0,
    )

    assert policy.decide(issue).action == "escalate"


def test_acquisition_detector_distinguishes_registered_truncation_from_failure():
    issues = detect_acquisition_issues(
        "d",
        {
            "truncated": True,
            "complete": False,
            "failed_pages": 0,
            "duplicate_records": 2,
        },
    )
    signatures = {issue.issue_signature for issue in issues}

    assert "truncated_registered_sample" in signatures
    assert "source_duplicates_observed" in signatures
    assert "incomplete_uncapped_acquisition" not in signatures


def test_review_summary_reports_intervention_rate():
    memory = FeedbackMemory([_decision("d1"), _decision("d2")])
    warm = AdaptiveReviewPolicy(memory, audit_rate=0).decide(_issue())
    cold = AdaptiveReviewPolicy(FeedbackMemory(), audit_rate=0).decide(_issue("cold"))

    summary = summarize_review_program([warm, cold])

    assert summary["human_interventions"] == 1
    assert summary["human_intervention_rate"] == 0.5
    assert summary["auto_accept_coverage"] == 0.5


def test_data_quality_detector_uses_hard_floor_and_relevance_gate():
    issues = detect_data_quality_issues(
        "d1",
        {
            "abstract_coverage": 0.25,
            "topic_relevance": {"all_terms_rate": 0.80},
        },
    )

    assert [issue.issue_signature for issue in issues] == [
        "abstract_coverage_below_target",
        "topic_relevance_below_floor",
    ]
    assert [issue.severity for issue in issues] == ["high", "critical"]


def test_graph_claim_detector_creates_atomic_review_units():
    issues = detect_graph_claim_issues(
        "d1",
        [{"item_id": "d1:G001", "answerable": True}],
    )

    assert len(issues) == 1
    assert issues[0].stage == "graph_interpretation"
    assert issues[0].payload["item_id"] == "d1:G001"


def test_one_sided_precision_bound_reaches_target_with_92_successes():
    lower = one_sided_clopper_pearson_lower(92, 92)

    assert lower > 0.95
