from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from citeweave.citecalibrator_benchmark import ACTIONS, SEVERITIES, cohen_kappa
from citeweave.io import sha256_file, write_json, write_jsonl

CORE_FIELDS = (
    "factual_supported",
    "interpretation_calibrated",
    "alternative_adequate",
    "evidence_sufficient",
    "action",
    "severity",
)
OUTPUT_FIELDS = {
    "case_id",
    "reviewer_id",
    "factual_supported",
    "interpretation_calibrated",
    "alternative_adequate",
    "evidence_sufficient",
    "action",
    "risk_types",
    "severity",
    "unsupported_spans",
    "decisive_evidence_ids",
    "invalid_dependency_ids",
    "minimal_revision",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_reviews(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    expected_order: list[str],
    judge_id: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != OUTPUT_FIELDS:
            raise ValueError(f"Output schema mismatch for {judge_id}")
        if row.get("action") not in ACTIONS:
            raise ValueError(f"Invalid review action: {row.get('action')}")
        if row.get("severity") not in SEVERITIES:
            raise ValueError(f"Invalid review severity: {row.get('severity')}")
        risk_types = row.get("risk_types")
        if not isinstance(risk_types, list) or any(
            not isinstance(value, str) or not value.strip() for value in risk_types
        ):
            raise TypeError(f"{judge_id} has invalid risk_types: {row.get('case_id')}")
        if row["action"] == "accept" and risk_types:
            raise ValueError("Accept judgments cannot carry defect risk types")
        if row["action"] == "qualify" and not row.get("minimal_revision"):
            raise ValueError("Qualify judgments require a minimal revision")
        case_id = str(row.get("case_id"))
        if case_id not in cases or case_id in index:
            raise ValueError(f"Unknown or duplicate case for {judge_id}: {case_id}")
        if row.get("reviewer_id") != judge_id:
            raise ValueError(f"Reviewer identity mismatch for {case_id}")
        for field in CORE_FIELDS[:4]:
            if not isinstance(row.get(field), bool):
                raise TypeError(f"{judge_id} has non-boolean {field}: {case_id}")
        invalid_ids = set(row.get("decisive_evidence_ids") or []) - set(
            cases[case_id].get("allowed_evidence_ids") or []
        )
        if invalid_ids:
            raise ValueError(f"Unavailable evidence IDs for {case_id}: {sorted(invalid_ids)}")
        claim = str(cases[case_id]["atomic_claim"])
        if any(span not in claim for span in row.get("unsupported_spans") or []):
            raise ValueError(f"Unsupported span is not an exact claim substring: {case_id}")
        if row["action"] == "accept" and row["severity"] != "minor":
            raise ValueError(f"Accepted claim must use minor severity: {case_id}")
        if row["action"] in {"qualify", "reject"} and not row["risk_types"]:
            raise ValueError(f"Intervention lacks a risk type: {case_id}")
        if row["action"] == "abstain" and row["evidence_sufficient"] is not False:
            raise ValueError(f"Abstention must mark evidence insufficient: {case_id}")
        index[case_id] = row
    if set(index) != set(cases):
        raise ValueError(f"{judge_id} did not review exactly the frozen case set")
    if [str(row["case_id"]) for row in rows] != expected_order:
        raise ValueError(f"{judge_id} did not preserve frozen case order")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--judge-a", type=Path, required=True)
    parser.add_argument("--judge-b", type=Path, required=True)
    parser.add_argument("--rule-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(args.cases) != protocol["input"]["source_cases_sha256"]:
        raise ValueError("Cases differ from the frozen judge protocol")
    case_rows = _read_jsonl(args.cases)
    cases = {str(row["case_id"]): row for row in case_rows}
    expected_order = [str(row["case_id"]) for row in case_rows]
    judge_a = _index_reviews(
        _read_jsonl(args.judge_a), cases, expected_order, "judge_a"
    )
    judge_b = _index_reviews(
        _read_jsonl(args.judge_b), cases, expected_order, "judge_b"
    )
    rules = {str(row["case_id"]): row for row in _read_jsonl(args.rule_predictions)}
    if set(rules) != set(cases):
        raise ValueError("Rule predictions do not cover the frozen case set")

    left_actions = [judge_a[key]["action"] for key in sorted(cases)]
    right_actions = [judge_b[key]["action"] for key in sorted(cases)]
    action_agreement = sum(
        left == right for left, right in zip(left_actions, right_actions, strict=True)
    ) / len(cases)
    consensus = []
    disagreements = []
    for case_id in sorted(cases):
        left = judge_a[case_id]
        right = judge_b[case_id]
        differing_fields = [field for field in CORE_FIELDS if left[field] != right[field]]
        if differing_fields:
            disagreements.append(
                {
                    "case_id": case_id,
                    "topic_id": cases[case_id]["topic_id"],
                    "differing_core_fields": differing_fields,
                    "judge_a": left,
                    "judge_b": right,
                }
            )
            continue
        consensus.append(
            {
                "case_id": case_id,
                "topic_id": cases[case_id]["topic_id"],
                "article_id": cases[case_id]["article_id"],
                **{field: left[field] for field in CORE_FIELDS},
                "risk_types": sorted(
                    set(left.get("risk_types") or []) | set(right.get("risk_types") or [])
                ),
                "resolution_source": "two_llm_core_consensus",
            }
        )

    major = [
        row
        for row in consensus
        if row["action"] in {"qualify", "reject"}
        and row["severity"] in {"major", "critical"}
    ]
    accepted = [row for row in consensus if row["action"] == "accept"]
    rule_major_recall = (
        sum(rules[row["case_id"]]["action"] != "accept" for row in major) / len(major)
        if major
        else None
    )
    rule_false_intervention = (
        sum(rules[row["case_id"]]["action"] != "accept" for row in accepted)
        / len(accepted)
        if accepted
        else None
    )
    metrics = {
        "inter_judge_action_kappa": cohen_kappa(
            left_actions, right_actions, labels=ACTIONS
        ),
        "consensus_major_critical_defect_rate": (
            len(major) / len(consensus) if consensus else None
        ),
        "rule_major_critical_defect_recall": rule_major_recall,
        "rule_accept_false_intervention_rate": rule_false_intervention,
    }
    reliability_passed = (
        metrics["inter_judge_action_kappa"] >= 0.70
        and len(consensus) / len(cases) >= 0.80
    )
    problem_passed = (
        metrics["consensus_major_critical_defect_rate"] is not None
        and metrics["consensus_major_critical_defect_rate"] >= 0.08
    )
    rule_gap_passed = (
        rule_major_recall is not None
        and rule_false_intervention is not None
        and (rule_major_recall < 0.80 or rule_false_intervention > 0.10)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "consensus_reviews.jsonl", consensus)
    write_jsonl(args.output_dir / "disagreements.jsonl", disagreements)
    result = {
        "schema_version": 1,
        "status": "two_llm_judges_aggregated",
        "scientific_role": "development_proxy_gold_not_human_gold",
        "cases": len(cases),
        "action_raw_agreement": action_agreement,
        "consensus_cases": len(consensus),
        "consensus_coverage": len(consensus) / len(cases),
        "unresolved_cases": len(disagreements),
        "primary_metrics": metrics,
        "provisional_gates": {
            "reliability_passed": reliability_passed,
            "problem_size_passed": problem_passed,
            "rule_gap_passed": rule_gap_passed,
        },
        "provisional_training_need_supported": (
            reliability_passed and problem_passed and rule_gap_passed
        ),
        "formal_training_need_proven": False,
        "limitation": (
            "Same-family LLM consensus on two inspected topics is development evidence, "
            "not formal human gold or a natural-prevalence estimate."
        ),
        "input_hashes": {
            "cases": sha256_file(args.cases),
            "protocol": sha256_file(args.protocol),
            "judge_a": sha256_file(args.judge_a),
            "judge_b": sha256_file(args.judge_b),
            "rules": sha256_file(args.rule_predictions),
        },
    }
    write_json(args.output_dir / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
