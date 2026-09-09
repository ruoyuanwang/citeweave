from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from citeweave.io import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    records = []
    finish_reasons: Counter[str] = Counter()
    for row in plan["articles"]:
        output_dir = Path(row["output_dir"])
        article_record = read_json(output_dir / "execution_record.json")
        assessment = read_json(output_dir / "draft_assessment.json")
        oversight = read_json(output_dir / "oversight_readiness.json")
        calls = []
        for record_path in sorted((output_dir / "calls").glob("*/execution_record.json")):
            call = read_json(record_path)
            finish_reasons[str(call.get("finish_reason"))] += 1
            calls.append(
                {
                    "section": call["section"],
                    "pass_type": call["pass_type"],
                    "finish_reason": call.get("finish_reason"),
                    "word_count": call["section_word_count"],
                    "prompt_tokens": int(
                        (call.get("usage") or {}).get("prompt_tokens") or 0
                    ),
                    "completion_tokens": int(
                        (call.get("usage") or {}).get("completion_tokens") or 0
                    ),
                }
            )
        retained_nonstop = [
            call["section"]
            for call in calls
            if call["finish_reason"] != "stop"
            and (
                call["pass_type"] == "repair"
                or call["section"] not in {"Results", "Discussion"}
            )
        ]
        records.append(
            {
                "article_id": row["article_id"],
                "dataset_id": row["dataset_id"],
                "condition": row["condition"],
                "provider_calls": len(calls),
                "nonstop_calls": [
                    f"{call['section']}:{call['pass_type']}"
                    for call in calls
                    if call["finish_reason"] != "stop"
                ],
                "retained_nonstop_sections": retained_nonstop,
                "word_count": article_record["word_count"],
                "machine_gate_passed": article_record["machine_gate_passed"],
                "machine_gate_failures": sorted(
                    key
                    for key, value in assessment["quality_gates"].items()
                    if not value
                ),
                "unexpected_evidence_tokens": assessment[
                    "unexpected_evidence_tokens"
                ],
                "oversight_ready": article_record["oversight_ready"],
                "reviewable_claim_candidates": oversight["candidate_claims"],
                "reviewable_claims_by_section": oversight[
                    "candidate_claims_by_section"
                ],
                "reviewable_claims_by_phenomenon": oversight[
                    "candidate_claims_by_phenomenon"
                ],
                "multiple_phenomenon_candidates": oversight[
                    "multiple_phenomenon_candidates"
                ],
                "discussion_interpretation_candidates": oversight[
                    "discussion_interpretation_candidates"
                ],
                "prompt_tokens": article_record["usage"]["prompt_tokens"],
                "completion_tokens": article_record["usage"]["completion_tokens"],
                "fully_passed": article_record["fully_passed"],
                "draft_sha256": article_record["draft_sha256"],
            }
        )
    by_pair = {}
    for dataset_id in sorted({row["dataset_id"] for row in records}):
        pair = {row["condition"]: row for row in records if row["dataset_id"] == dataset_id}
        flat = pair["flat_article_compiler"]
        graph = pair["graph_dependency_compiler"]
        by_pair[dataset_id] = {
            "graph_minus_flat_reviewable_claim_candidates": (
                graph["reviewable_claim_candidates"]
                - flat["reviewable_claim_candidates"]
            ),
            "graph_minus_flat_word_count": graph["word_count"] - flat["word_count"],
            "graph_minus_flat_prompt_tokens": (
                graph["prompt_tokens"] - flat["prompt_tokens"]
            ),
            "graph_minus_flat_completion_tokens": (
                graph["completion_tokens"] - flat["completion_tokens"]
            ),
            "flat_machine_gate_passed": flat["machine_gate_passed"],
            "graph_machine_gate_passed": graph["machine_gate_passed"],
            "flat_oversight_ready": flat["oversight_ready"],
            "graph_oversight_ready": graph["oversight_ready"],
        }
    claim_deltas = [
        row["graph_minus_flat_reviewable_claim_candidates"] for row in by_pair.values()
    ]
    report = {
        "schema_version": 1,
        "status": "matched_article_compiler_development_analyzed",
        "scientific_role": "development_only_not_confirmatory",
        "plan_sha256": sha256_file(args.plan),
        "articles": len(records),
        "provider_calls": sum(row["provider_calls"] for row in records),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "machine_gate_passed": sum(row["machine_gate_passed"] for row in records),
        "oversight_ready": sum(row["oversight_ready"] for row in records),
        "fully_passed": sum(row["fully_passed"] for row in records),
        "pilot_qualified_for_confirmatory_freeze": all(
            row["fully_passed"] for row in records
        ),
        "mean_graph_minus_flat_reviewable_claim_candidates": sum(claim_deltas)
        / len(claim_deltas),
        "paired_diagnostics": by_pair,
        "records": records,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
