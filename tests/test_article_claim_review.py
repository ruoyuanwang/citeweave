from __future__ import annotations

from pathlib import Path

from citeweave.article_claim_review import (
    _claim_candidates,
    assess_article_claim_review_readiness,
    build_article_claim_review_packets,
    select_article_review_claims,
)
from citeweave.io import read_json, sha256_file, write_json


def test_numeric_risk_ignores_digits_in_evidence_identifiers() -> None:
    claims = _claim_candidates(
        "## Results\r\nThe network connects PH-123 with REF-456. The degree is 7 for PH-123.\r\n"
    )
    assert len(claims) == 2
    assert claims[0]["risk_features"]["numeric"] is False
    assert claims[1]["risk_features"]["numeric"] is True


def _phenomenon(index: int) -> dict[str, object]:
    return {
        "phenomenon_id": f"PH-{index:016d}",
        "task_type": "task",
        "question": "question",
        "verified_answer": {"value": index},
        "operator_trace": [{"operator": "operator"}],
        "graph_evidence_ids": [f"graph:node:{index}"],
        "reference_ids": [f"REF-{index:016d}", f"REF-{index + 5:016d}"],
        "interpretation_contract": {
            "allowed": "structure",
            "forbidden": "causality",
            "required_limitation": "corpus",
        },
    }


def _article() -> str:
    sections = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    ]
    blocks = []
    for section in sections:
        sentences = []
        for index in range(5):
            for repeat in range(4):
                sentences.append(
                    f"Structural claim {section} {index} {repeat} reports {repeat + 1}.0 for PH-{index:016d} with REF-{index:016d}."
                )
        blocks.append(f"## {section}\n" + " ".join(sentences))
    return "\n\n".join(blocks) + "\n"


def test_selects_twenty_unique_claims_covering_every_phenomenon() -> None:
    phenomena = [f"PH-{index:016d}" for index in range(5)]
    selected = select_article_review_claims(_article(), phenomenon_ids=phenomena)
    assert len(selected) == 20
    assert len({row["candidate_id"] for row in selected}) == 20
    assert all(
        sum(phenomenon in row["phenomenon_ids"] for row in selected) >= 2
        for phenomenon in phenomena
    )


def test_builds_double_review_packets(tmp_path: Path) -> None:
    topic = "topic"
    writer_input_path = tmp_path / "writer_input.json"
    sources = [
        {"reference_id": f"REF-{index:016d}", "abstract_excerpt": "evidence"} for index in range(10)
    ]
    write_json(
        writer_input_path,
        {
            "graph_phenomena": [_phenomenon(index) for index in range(5)],
            "representative_sources": sources,
        },
    )
    output_dir = tmp_path / "generation" / topic / "citeweave_graph_review"
    output_dir.mkdir(parents=True)
    draft_path = output_dir / "draft.md"
    draft_path.write_text(_article(), encoding="utf-8")
    write_json(
        output_dir / "execution_record.json",
        {
            "status": "draft_ready_for_human_review",
            "draft_sha256": sha256_file(draft_path),
        },
    )
    plan_path = tmp_path / "plan.json"
    write_json(
        plan_path,
        {
            "cells": [
                {
                    "dataset_id": topic,
                    "condition": "citeweave_graph_review",
                    "output_dir": str(output_dir.resolve()),
                    "writer_input": str(writer_input_path.resolve()),
                    "writer_input_sha256": sha256_file(writer_input_path),
                }
            ]
            * 8
        },
    )
    # Eight distinct topics are required in production; create a realistic plan here.
    plan = read_json(plan_path)
    plan["cells"] = []
    roster_topics = []
    for index in range(8):
        dataset = f"topic_{index}"
        roster_topics.append(dataset)
        dataset_output = tmp_path / "generation" / dataset / "citeweave_graph_review"
        dataset_output.mkdir(parents=True)
        dataset_draft = dataset_output / "draft.md"
        dataset_draft.write_text(_article(), encoding="utf-8")
        write_json(
            dataset_output / "execution_record.json",
            {
                "status": "draft_ready_for_human_review",
                "draft_sha256": sha256_file(dataset_draft),
            },
        )
        plan["cells"].append(
            {
                "dataset_id": dataset,
                "condition": "citeweave_graph_review",
                "output_dir": str(dataset_output.resolve()),
                "writer_input": str(writer_input_path.resolve()),
                "writer_input_sha256": sha256_file(writer_input_path),
            }
        )
    write_json(plan_path, plan)
    roster_path = tmp_path / "roster.json"
    write_json(
        roster_path,
        {
            "reviewers": [
                {
                    "reviewer_id": f"reviewer_{index}",
                    "eligible_topics": roster_topics,
                    "conflicted_topics": [],
                    "domain_qualified": True,
                }
                for index in range(3)
            ]
        },
    )
    manifest = build_article_claim_review_packets(
        plan_path,
        roster_path,
        output_root=tmp_path / "review",
    )
    assert manifest["packets"] == 160
    assert manifest["primary_reviews_required"] == 320
    internal = read_json(tmp_path / "review" / "internal_manifest.json")
    assert all(len(row["article"]) > 0 for row in internal["assignments"].values())


def test_readiness_reports_missing_drafts_and_reviewer_coverage(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    cells = []
    for index in range(8):
        cells.append(
            {
                "dataset_id": f"topic_{index}",
                "condition": "citeweave_graph_review",
                "output_dir": str((tmp_path / f"topic_{index}").resolve()),
            }
        )
    write_json(plan_path, {"cells": cells})
    roster_path = tmp_path / "roster.json"
    write_json(roster_path, {"reviewers": []})
    result = assess_article_claim_review_readiness(plan_path, roster_path)
    assert result["status"] == "blocked"
    assert len(result["missing_drafts"]) == 8
    assert result["real_reviewers_registered"] == 0
    assert all(not row["ready"] for row in result["reviewer_coverage"].values())
