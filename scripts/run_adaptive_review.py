from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from citeweave.adaptive_review import (
    AdaptiveReviewPolicy,
    FeedbackMemory,
    detect_acquisition_issues,
    detect_data_quality_issues,
    detect_generation_issues,
    detect_graph_claim_issues,
    make_review_decision,
    summarize_review_program,
)
from citeweave.io import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the registered adaptive-review program.")
    parser.add_argument(
        "--program",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "reviews" / "program.yml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "reviews" / "results",
    )
    args = parser.parse_args()
    program = yaml.safe_load(args.program.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "review_log.jsonl"
    if log_path.exists():
        raise SystemExit(f"Refusing to overwrite append-only review log: {log_path}")

    memory = FeedbackMemory()
    policy_config = program["policy"]
    policy = AdaptiveReviewPolicy(memory, **policy_config)
    thresholds = program["quality_thresholds"]
    round_results: list[dict[str, Any]] = []

    for round_spec in program["rounds"]:
        run_path = REPOSITORY_ROOT / round_spec["run_record"]
        record = _load(run_path)
        dataset_id = round_spec["dataset_id"]
        workspace = (
            REPOSITORY_ROOT / round_spec["workspace"]
            if round_spec.get("workspace")
            else REPOSITORY_ROOT / "experiments" / "workspaces" / dataset_id
        )
        graph_items = _load(workspace / "evidence" / "graph_qa_benchmark.json")
        issues = [
            *detect_acquisition_issues(
                dataset_id,
                record["snapshot"]["acquisition_manifest"],
            ),
            *detect_data_quality_issues(
                dataset_id,
                record["snapshot"],
                abstract_target=thresholds["abstract_target"],
                abstract_hard_floor=thresholds["abstract_hard_floor"],
                relevance_floor=thresholds["topic_relevance_floor"],
            ),
            *detect_generation_issues(
                dataset_id,
                record["result"]["generation_validation"],
            ),
            *detect_graph_claim_issues(dataset_id, graph_items),
        ]
        actions = []
        decisions = []
        intervention_counts: dict[str, int] = {}
        provisional_actions = [policy.decide(issue) for issue in issues]
        for issue, action in zip(issues, provisional_actions, strict=True):
            if action.action in {"escalate", "audit"}:
                intervention_counts[issue.issue_signature] = (
                    intervention_counts.get(issue.issue_signature, 0) + 1
                )
        for issue in issues:
            action = policy.decide(issue)
            actions.append(action)
            if action.action not in {"escalate", "audit"}:
                continue
            decision_spec = (
                program.get("human_decisions", {})
                .get(round_spec["round_id"], {})
                .get(issue.issue_signature)
            )
            if decision_spec is None:
                raise SystemExit(
                    f"Missing human decision for {round_spec['round_id']}:"
                    f"{issue.issue_signature}"
                )
            review_seconds = (
                float(decision_spec["review_seconds"])
                if "review_seconds" in decision_spec
                else float(decision_spec["review_seconds_total"])
                / intervention_counts[issue.issue_signature]
            )
            decision = make_review_decision(
                issue,
                reviewer_code=program["reviewer_code"],
                decision=decision_spec["decision"],
                original=issue.payload,
                correction=decision_spec.get("correction"),
                reason=decision_spec["reason"],
                review_seconds=review_seconds,
                feedback_memory_version=memory.version,
            )
            memory.append(log_path, decision)
            decisions.append(decision)

        summary = summarize_review_program(actions)
        round_results.append(
            {
                "round_id": round_spec["round_id"],
                "dataset_id": dataset_id,
                "run_record": round_spec["run_record"],
                "issues": [asdict(issue) for issue in issues],
                "actions": [asdict(action) for action in actions],
                "decisions": [asdict(decision) for decision in decisions],
                "summary": summary,
                "feedback_memory_version_after_round": memory.version,
            }
        )

    result = {
        "program_version": program["version"],
        "policy": policy_config,
        "quality_thresholds": thresholds,
        "rounds": round_results,
        "total_review_seconds": sum(
            decision.review_seconds for decision in memory.decisions
        ),
        "feedback_records": memory.version,
    }
    write_json(args.output_dir / "adaptive_review_results.json", result)
    lines = [
        "# Adaptive Human-Review Experiment",
        "",
        f"- Feedback records: {memory.version}",
        f"- Measured online Judge time: {result['total_review_seconds']:.1f} seconds",
        "",
        "| Round | Issues | Review requests | RRR | Auto-accept coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in round_results:
        summary = row["summary"]
        lines.append(
            f"| {row['round_id']} | {summary['items']} | "
            f"{summary['review_requests']} | "
            f"{summary['review_request_rate']:.3f} | "
            f"{summary['auto_accept_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Online intervention counts exclude retrospective outcome annotation.",
            "The append-only JSONL log stores the human decisions and memory versions.",
        ]
    )
    (args.output_dir / "adaptive_review_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
