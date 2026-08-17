from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.adaptive_review import one_sided_clopper_pearson_lower
from citeweave.io import write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _graph_outcome(item: dict[str, Any], workspace: Path) -> tuple[bool, dict[str, Any]]:
    kg = _load(workspace / "evidence" / "bibliometric_kg.json")
    nodes = {node["id"]: node for node in kg["nodes"]}
    edges = {edge.get("edge_id") for edge in kg["edges"] if edge.get("edge_id")}
    evidence_ok = all(node in nodes for node in item["gold_evidence_nodes"]) and all(
        edge in edges for edge in item["gold_evidence_edges"]
    )
    if item["answerable"]:
        fact = nodes.get(f"fact:{item['item_id'].split(':')[-1]}", {})
        content_ok = (
            fact.get("value") == item["gold_answer"]
            and fact.get("label") == item["gold_statement"]
        )
    else:
        target = item["evidence_operation"]["target_label"].casefold()
        content_ok = (
            target not in {str(node.get("label", "")).casefold() for node in kg["nodes"]}
            and item["evidence_operation"]["type"] == "node_absence_check"
        )
    return evidence_ok and content_ok, {
        "evidence_ok": evidence_ok,
        "content_or_abstention_ok": content_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrospectively annotate adaptive-review auto-accept outcomes."
    )
    parser.add_argument(
        "--program",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "reviews" / "program.yml",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "experiments"
            / "reviews"
            / "results"
            / "adaptive_review_results.json"
        ),
    )
    args = parser.parse_args()
    program = yaml.safe_load(args.program.read_text(encoding="utf-8"))
    result = _load(args.results)
    round_specs = {item["round_id"]: item for item in program["rounds"]}
    outcomes: list[dict[str, Any]] = []

    for round_result in result["rounds"]:
        spec = round_specs[round_result["round_id"]]
        workspace = (
            REPOSITORY_ROOT / spec["workspace"]
            if spec.get("workspace")
            else REPOSITORY_ROOT
            / "experiments"
            / "workspaces"
            / spec["dataset_id"]
        )
        issues = {item["item_id"]: item for item in round_result["issues"]}
        for action in round_result["actions"]:
            if action["action"] != "auto_accept":
                continue
            issue = issues[action["item_id"]]
            signature = issue["issue_signature"]
            if signature == "structured_graph_claim_unverified":
                correct, detail = _graph_outcome(issue["payload"], workspace)
            elif signature == "abstract_coverage_below_target":
                correct = issue["payload"].get("auto_accept_context_ok") is True
                detail = {
                    "context_guard": issue["payload"].get("auto_accept_context_ok"),
                    "topic_relevance": issue["payload"].get("topic_relevance"),
                }
            elif signature == "truncated_registered_sample":
                manifest = issue["payload"]
                correct = bool(
                    manifest.get("truncated")
                    and manifest.get("failed_pages") == 0
                    and manifest.get("raw_sha256")
                )
                detail = {
                    "truncated": manifest.get("truncated"),
                    "failed_pages": manifest.get("failed_pages"),
                    "raw_hashes": len(manifest.get("raw_sha256") or []),
                }
            else:
                correct = False
                detail = {"error": "No registered retrospective outcome rule."}
            outcomes.append(
                {
                    "round_id": round_result["round_id"],
                    "item_id": action["item_id"],
                    "issue_signature": signature,
                    "correct": correct,
                    "detail": detail,
                }
            )

    successes = sum(item["correct"] for item in outcomes)
    trials = len(outcomes)
    lower = one_sided_clopper_pearson_lower(successes, trials)
    summary = {
        "evaluation_type": "retrospective_outcome_annotation",
        "feedback_leakage": False,
        "auto_accepted_items": trials,
        "correct_auto_accepts": successes,
        "auto_accept_precision": successes / trials,
        "one_sided_95_percent_clopper_pearson_lower": lower,
        "registered_quality_target": 0.95,
        "quality_target_passed": lower >= 0.95,
        "outcomes": outcomes,
    }
    output = args.results.parent / "adaptive_review_outcomes.json"
    write_json(output, summary)
    lines = [
        "# Adaptive Review Outcome Evaluation",
        "",
        f"- Auto-accepted items: {trials}",
        f"- Correct auto-accepts: {successes}",
        f"- Precision: {successes / trials:.3f}",
        f"- One-sided 95% Clopper-Pearson lower bound: {lower:.3f}",
        f"- Registered 0.95 quality target: {'PASS' if lower >= 0.95 else 'FAIL'}",
        "",
        "Outcome annotations were not added to feedback memory.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
