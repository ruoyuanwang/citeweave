from __future__ import annotations

from src.citeweave.review_learning import (
    ActiveReviewController,
    DependencyNode,
    FeedbackCompiler,
    GuardedReviewPolicy,
    PolicyReplayObservation,
    ReviewDependencyGraph,
    ReviewTarget,
    RuleValidationCase,
    StructuredFeedback,
    audit_policy_promotion,
    compile_feedback_learning_signals,
)


def _feedback(index: int, dataset: str, reviewer: str, action: str = "rewrite") -> StructuredFeedback:
    return StructuredFeedback(
        feedback_id=f"f{index}",
        reviewer_id=reviewer,
        dataset_id=dataset,
        target_id=f"claim{index}",
        target_type="claim",
        issue_type="causal_overreach",
        action=action,  # type: ignore[arg-type]
        rationale="Centrality does not establish causality.",
        replacement="Describe this as a structural association.",
        guard={"network_metric": "centrality"},
        review_seconds=10,
    )


def test_feedback_compiler_requires_cross_dataset_and_reviewer_support() -> None:
    compiler = FeedbackCompiler(minimum_datasets=2, minimum_reviewers=2)
    one = compiler.compile([_feedback(1, "d1", "r1")])[0]
    assert one.enabled is False
    transferred = compiler.compile(
        [_feedback(1, "d1", "r1"), _feedback(2, "d2", "r2")]
    )[0]
    assert transferred.support_gate_passed is True
    assert transferred.enabled is False
    assert transferred.supporting_datasets == ("d1", "d2")


def test_feedback_rule_requires_holdout_transfer_and_regression_gates() -> None:
    compiler = FeedbackCompiler(minimum_datasets=2, minimum_reviewers=2)
    cases = [
        RuleValidationCase(
            "v1",
            "holdout-1",
            "causal_overreach",
            "claim",
            {"network_metric": "centrality"},
            "rewrite",
            True,
        ),
        RuleValidationCase(
            "v2",
            "holdout-2",
            "causal_overreach",
            "claim",
            {"network_metric": "centrality"},
            "rewrite",
            True,
        ),
        RuleValidationCase(
            "u1",
            "holdout-3",
            "causal_overreach",
            "claim",
            {"network_metric": "citation_count"},
            "accept",
            False,
        ),
    ]
    rule = compiler.compile(
        [_feedback(1, "d1", "r1"), _feedback(2, "d2", "r2")],
        validation_cases=cases,
    )[0]
    assert rule.enabled is True
    assert rule.transfer_precision == 1.0
    assert rule.unrelated_regression_rate == 0.0
    decision = GuardedReviewPolicy().route(
        ReviewTarget(
            "claim-new",
            "test",
            "causal_overreach",
            "high",
            {"network_metric": "centrality"},
            20,
        ),
        [rule],
    )
    assert decision["route"] == "auto_correct"
    assert decision["rule_ids"] == [rule.rule_id]


def test_critical_target_always_routes_to_human() -> None:
    compiler = FeedbackCompiler(
        minimum_datasets=1,
        minimum_reviewers=1,
        minimum_validation_cases=1,
        minimum_validation_datasets=1,
    )
    rule = compiler.compile(
        [_feedback(1, "d1", "r1")],
        validation_cases=[
            RuleValidationCase(
                "v1",
                "holdout",
                "causal_overreach",
                "claim",
                {"network_metric": "centrality"},
                "rewrite",
                True,
            ),
            RuleValidationCase(
                "u1",
                "holdout",
                "causal_overreach",
                "claim",
                {"network_metric": "citation_count"},
                "accept",
                False,
            ),
        ],
    )[0]
    decision = GuardedReviewPolicy().route(
        ReviewTarget(
            "critical",
            "test",
            "causal_overreach",
            "critical",
            {"network_metric": "centrality"},
            20,
        ),
        [rule],
    )
    assert decision["route"] == "human_review"
    assert decision["reason"] == "critical_severity"


def test_invalid_evidence_propagates_to_claim_and_paragraph() -> None:
    graph = ReviewDependencyGraph()
    graph.add_node(DependencyNode("e1", "evidence", {"value": 1}))
    graph.add_node(DependencyNode("c1", "claim", "claim"))
    graph.add_node(DependencyNode("p1", "paragraph", "paragraph"))
    graph.add_dependency("e1", "c1")
    graph.add_dependency("c1", "p1")
    feedback = StructuredFeedback(
        feedback_id="f1",
        reviewer_id="r1",
        dataset_id="d1",
        target_id="e1",
        target_type="evidence",
        issue_type="wrong_value",
        action="reject_evidence",
        rationale="The source does not support the value.",
    )
    result = graph.apply_feedback(feedback)
    assert result["review_queue"] == ["c1", "p1"]
    assert graph.nodes["e1"].status == "invalid"
    assert graph.nodes["p1"].status == "needs_review"


def test_active_review_prioritizes_critical_and_high_impact_items() -> None:
    controller = ActiveReviewController([], reviewer_second_cost=0.01)
    targets = [
        ReviewTarget("low", "d", "style", "low", {}, 20, descendants=0),
        ReviewTarget("critical", "d", "grounding", "critical", {}, 20, descendants=5),
    ]
    selection = controller.select(targets, budget_seconds=20)
    assert selection["selected_count"] == 1
    assert selection["selected"][0]["target"]["target_id"] == "critical"


def test_typed_feedback_compiles_component_specific_credit() -> None:
    feedback = [
        StructuredFeedback(
            "f-evidence",
            "r1",
            "d1",
            "e1",
            "evidence",
            "irrelevant_source",
            "reject_evidence",
            "Not relevant.",
        ),
        StructuredFeedback(
            "f-claim",
            "r1",
            "d1",
            "c1",
            "claim",
            "causal_overreach",
            "rewrite",
            "Needs calibration.",
            replacement="Structurally associated in this corpus.",
        ),
        StructuredFeedback(
            "f-ambiguous",
            "r1",
            "d1",
            "c2",
            "claim",
            "unknown_source",
            "reject_evidence",
            "The reviewer did not identify an evidence target.",
        ),
    ]
    signals = compile_feedback_learning_signals(feedback)
    assert [(row.component, row.reward, row.supervision) for row in signals] == [
        ("retriever", -1.0, "pointwise"),
        ("generator", 1.0, "pairwise_preference"),
        ("risk_router", 1.0, "escalation"),
    ]
    assert signals[1].preferred_payload == "Structurally associated in this corpus."


def test_policy_promotion_requires_quality_safety_and_labor_gates() -> None:
    clean = [
        PolicyReplayObservation(
            case_id=f"case-{index}",
            dataset_id=f"d{index % 2}",
            severity="high",
            baseline_correct=True,
            candidate_correct=True,
            baseline_review_seconds=10,
            candidate_review_seconds=1,
            candidate_route="auto_accept",
            unrelated=True,
        )
        for index in range(100)
    ]
    passed = audit_policy_promotion(clean)
    assert passed["decision"] == "promote"
    assert passed["unsafe_autoaccept_wilson_upper_95"] < 0.05
    unsafe = [*clean]
    unsafe[0] = PolicyReplayObservation(
        case_id="case-0",
        dataset_id="d0",
        severity="high",
        baseline_correct=True,
        candidate_correct=False,
        baseline_review_seconds=10,
        candidate_review_seconds=1,
        candidate_route="auto_accept",
        unrelated=True,
    )
    held = audit_policy_promotion(unsafe)
    assert held["decision"] == "hold"
    assert held["checks"]["quality_noninferiority_gate"] is False
    assert held["checks"]["unsafe_autoaccept_gate"] is False
