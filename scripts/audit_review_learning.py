from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from citeweave.io import write_json
from citeweave.review_learning import (
    FeedbackCompiler,
    StructuredFeedback,
    compile_feedback_learning_signals,
)


def _convert(record: dict) -> StructuredFeedback:
    decision = str(record.get("decision"))
    action = {
        "accept": "accept",
        "correct": "rewrite",
        "reject": "reject_claim",
        "abstain": "abstain",
    }.get(decision, "reject_claim")
    correction = record.get("correction")
    return StructuredFeedback(
        feedback_id=str(record.get("decision_id")),
        reviewer_id=str(record.get("reviewer_code")),
        dataset_id=str(record.get("dataset_id")),
        target_id=str(record.get("item_id")),
        target_type="claim",
        issue_type=str(record.get("issue_signature")),
        action=action,  # type: ignore[arg-type]
        rationale=str(record.get("reason") or ""),
        replacement=str(correction) if correction else None,
        guard={"stage": str(record.get("stage"))},
        review_seconds=float(record.get("review_seconds") or 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.memory.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    feedback = [_convert(record) for record in records]
    learning_signals = compile_feedback_learning_signals(feedback)
    rules = FeedbackCompiler(minimum_datasets=2, minimum_reviewers=2).compile(feedback)
    reviewers = sorted({item.reviewer_id for item in feedback})
    datasets = sorted({item.dataset_id for item in feedback})
    findings = []
    if len(reviewers) < 2:
        findings.append(
            "No learned rule can be independently validated because all decisions come from one reviewer identity."
        )
    if sum(item.review_seconds for item in feedback) <= 0:
        findings.append(
            "Review-time savings are not measurable because all recorded review durations are zero."
        )
    if not any(rule.enabled for rule in rules):
        findings.append(
            "The existing memory yields zero publishable executable rules under cross-dataset and cross-reviewer guards."
        )
    audit = {
        "schema_version": 1,
        "source": str(args.memory.resolve()),
        "records": len(feedback),
        "datasets": datasets,
        "reviewers": reviewers,
        "reviewer_count": len(reviewers),
        "total_review_seconds": sum(item.review_seconds for item in feedback),
        "actions": dict(Counter(item.action for item in feedback)),
        "issue_types": dict(Counter(item.issue_type for item in feedback)),
        "learning_signal_components": dict(
            Counter(item.component for item in learning_signals)
        ),
        "learning_signal_supervision": dict(
            Counter(item.supervision for item in learning_signals)
        ),
        "compiled_rule_candidates": len(rules),
        "enabled_rules": sum(rule.enabled for rule in rules),
        "rules": [rule.__dict__ for rule in rules],
        "findings": findings,
        "required_upgrade": {
            "annotation_unit": "claim/evidence/operator/plan rather than whole output",
            "minimum_reviewers": 2,
            "adjudication": True,
            "propagation_metrics": [
                "correction_lag",
                "post_feedback_accuracy_on_semantic_neighbors",
                "regression_rate_on_unrelated_items",
                "review_seconds",
            ],
            "policy_metrics": [
                "risk_coverage",
                "unsafe_auto_accept_rate",
                "review utility per minute",
                "rule precision under transfer",
            ],
        },
    }
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
