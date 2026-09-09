from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_expert_packets import (
    FIGURE_ACCESS_MODE,
    assess_article_packet_readiness,
    build_article_intake_template,
    build_evaluator_roster_template,
)
from citeweave.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-pack-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intake_path = args.output_dir / "article_intake.json"
    roster_path = args.output_dir / "evaluator_roster.json"
    intake = build_article_intake_template(
        args.writer_pack_manifest, output_path=intake_path
    )
    topics = sorted({row["topic_id"] for row in intake["articles"]})
    build_evaluator_roster_template(topics, output_path=roster_path)
    write_json(
        args.output_dir / "claim_inventory_contract.json",
        {
            "schema_version": 1,
            "article_sha256": "<sha256 of submitted article>",
            "abstractor_id": "<accountable evaluator ID>",
            "condition_blinded_to_abstractor": True,
            "sampling_requirement": {
                "selected_claims": 20,
                "unique_claims_per_stratum": 4,
                "strata": [
                    "results",
                    "discussion",
                    "graph_derived",
                    "numerical",
                    "causal_risk",
                ],
            },
            "claims": [
                {
                    "candidate_id": "<stable candidate ID>",
                    "text": "<exact article substring>",
                    "start_char": "<zero-based inclusive character offset>",
                    "end_char": "<zero-based exclusive character offset>",
                    "section": "<article section>",
                    "eligible_strata": ["<one or more registered strata>"],
                    "evidence_tokens": ["<PH-* or REF-* token present in article>"],
                }
            ],
        },
    )
    write_json(
        args.output_dir / "production_record_contract.json",
        {
            "shared_required": {
                "condition": "<registered condition>",
                "article_sha256": "<article hash>",
                "writer_pack_sha256": "<frozen pack hash>",
                "figure_sha256": "<frozen rendered figure hash>",
                "writer_input_sha256": "<audited text-only writer-input hash>",
                "figure_access_mode": FIGURE_ACCESS_MODE,
                "writer_rendered_figure_access": False,
                "draft_sha256": "<writer draft hash before figure attachment>",
                "posthoc_figure_insertion_receipt_sha256": (
                    "<receipt binding draft, final article, and unchanged figure>"
                ),
                "producer_id": "<accountable producer ID>",
                "started_at": "<ISO-8601 timestamp>",
                "completed_at": "<ISO-8601 timestamp>",
                "no_cross_condition_draft_access": True,
            },
            "human_same_evidence": {
                "domain_qualified": True,
                "system_builder": False,
                "machine_drafts_visible": False,
                "writer_input_delivery_log_sha256": "<server delivery log hash>",
            },
            "one_shot_llm": {
                "generation_requests": 1,
                "iterative_revision": False,
                "graph_operator_access": False,
                "input_modality": "text_only_structured_graph",
                "text_request_sha256": "<request hash binding the writer input>",
            },
            "citeweave_graph_review": {
                "graph_program_used": True,
                "validated_review_pipeline_used": True,
                "review_event_log_sha256": "<event-log hash>",
                "input_modality": "text_only_structured_graph",
                "text_request_sha256": "<request hash binding the writer input>",
            },
        },
    )
    readiness = assess_article_packet_readiness(intake_path, roster_path)
    write_json(args.output_dir / "readiness.json", readiness)
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
