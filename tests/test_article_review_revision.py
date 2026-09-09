from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.article_claim_review import _article_paragraphs
from citeweave.article_review_experiment import (
    analyze_article_review_routing,
    analyze_revision_evaluation,
    build_article_review_routing_plan,
    build_revision_evaluation_packets,
    exact_topic_sign_flip,
)
from citeweave.article_review_revision import (
    apply_controlled_paragraph_revisions,
    compile_article_revision_worklist,
    prepare_article_adjudication,
    resolve_article_claim_reviews,
)
from citeweave.io import read_json, sha256_file, write_json


def _result(reviewer: str, *, accept: bool) -> dict[str, object]:
    return {
        "packet_id": "AR1",
        "reviewer_code": reviewer,
        "factual_supported": accept,
        "interpretation_calibrated": accept,
        "alternative_adequate": True,
        "evidence_sufficient": accept,
        "action": "accept" if accept else "rewrite",
        "decisive_evidence_ids": ["PH-1", "REF-1"],
        "invalid_dependency_ids": [] if accept else ["REF-1"],
        "replacement": None if accept else "Calibrated claim PH-1 with REF-1.",
        "guard": {"scope": "corpus"},
        "rationale": "Supported." if accept else "The wording needs calibration.",
        "review_seconds": 12.0,
        "timing_method": "visibility_heartbeat_server_accounted",
    }


def _review_fixture(
    tmp_path: Path, *, shared: bool = False, with_returns: bool = True
) -> tuple[Path, Path, Path]:
    primary = tmp_path / "primary"
    (primary / "packets" / "article").mkdir(parents=True)
    draft_dir = tmp_path / "generation" / "topic" / "citeweave_graph_review"
    draft_dir.mkdir(parents=True)
    article = (
        "## Introduction\nBackground stays byte identical.\n\n"
        "## Results\nOriginal claim PH-1 with REF-1.\n\n"
        f"## Discussion\nSeparate observation {'PH-1' if shared else 'PH-2'}.\n"
    )
    draft_path = draft_dir / "draft.md"
    draft_path.write_text(article, encoding="utf-8")
    paragraph = _article_paragraphs(draft_path.read_bytes().decode("utf-8"))[0]
    write_json(
        primary / "packets" / "article" / "AR1.json",
        {
            "packet_id": "AR1",
            "claim": "Original claim PH-1 with REF-1.",
            "allowed_decisive_evidence_ids": ["PH-1", "REF-1"],
            "phenomena": [
                {"phenomenon_id": "PH-1", "graph_evidence_ids": [], "reference_ids": ["REF-1"]}
            ],
        },
    )
    write_json(
        primary / "internal_manifest.json",
        {
            "topic_adjudicators": {"topic": "R3"},
            "assignments": {"R1": {"article": ["AR1"]}, "R2": {"article": ["AR1"]}},
            "internal_packets": [
                {
                    "packet_id": "AR1",
                    "dataset_id": "topic",
                    "candidate_id": "PAR-0000:S00",
                    "paragraph_id": paragraph["paragraph_id"],
                    "primary_reviewers": ["R1", "R2"],
                    "adjudicator": "R3",
                    "draft_sha256": sha256_file(draft_path),
                    "paragraph_sha256": paragraph["paragraph_sha256"],
                    "evidence_tokens": ["PH-1", "REF-1"],
                }
            ],
        },
    )
    for reviewer, accept in (("R1", True), ("R2", False)):
        if with_returns:
            write_json(
                primary / "returns" / f"{reviewer}.json",
                {"reviewer_code": reviewer, "results": [_result(reviewer, accept=accept)]},
            )
    plan_path = tmp_path / "plan.json"
    write_json(
        plan_path,
        {
            "cells": [
                {
                    "dataset_id": "topic",
                    "condition": "citeweave_graph_review",
                    "output_dir": str(draft_dir.resolve()),
                }
            ]
        },
    )
    manifest = read_json(primary / "internal_manifest.json")
    manifest["machine_plan_sha256"] = sha256_file(plan_path)
    manifest["packet_records"] = [
        {"packet_id": "AR1", "sha256": sha256_file(primary / "packets" / "article" / "AR1.json")}
    ]
    write_json(primary / "internal_manifest.json", manifest)
    return primary, plan_path, draft_path


def test_disagreements_only_are_adjudicated_then_propagated(tmp_path: Path) -> None:
    primary, plan_path, draft_path = _review_fixture(tmp_path)
    adjudication = tmp_path / "adjudication"
    prepared = prepare_article_adjudication(primary, output_root=adjudication)
    assert prepared["adjudication_required"] == 1
    assert read_json(adjudication / "internal_manifest.json")["assignments"] == {
        "R3": {"article": ["AR1"]}
    }
    with pytest.raises(ValueError, match="lacks adjudication"):
        resolve_article_claim_reviews(primary, adjudication_root=adjudication)
    write_json(
        adjudication / "returns" / "R3.json",
        {"reviewer_code": "R3", "results": [_result("R3", accept=False)]},
    )
    resolved_path = tmp_path / "resolved.json"
    resolved = resolve_article_claim_reviews(
        primary, adjudication_root=adjudication, output_path=resolved_path
    )
    assert resolved["adjudicated"] == 1
    worklist_path = tmp_path / "worklist.json"
    worklist = compile_article_revision_worklist(
        primary, resolved_path, plan_path, output_path=worklist_path
    )
    assert worklist["affected_paragraphs"] == 1
    changed_nodes = {
        node for event in worklist["propagation_events"] for node in event["changed_nodes"]
    }
    assert "topic:paragraph:PAR-0000" in changed_nodes
    replacement_path = tmp_path / "replacements.json"
    paragraph = worklist["worklist"][0]
    write_json(
        replacement_path,
        {
            "replacements": [
                {
                    "dataset_id": "topic",
                    "paragraph_id": "PAR-0000",
                    "original_paragraph_sha256": paragraph["paragraph_sha256"],
                    "replacement_text": "Calibrated claim PH-1 with REF-1.",
                }
            ]
        },
    )
    output_root = tmp_path / "reviewed"
    manifest = apply_controlled_paragraph_revisions(
        worklist_path, replacement_path, output_root=output_root
    )
    reviewed = Path(manifest["articles"][0]["reviewed_path"]).read_text(encoding="utf-8")
    assert "Background stays byte identical." in reviewed
    assert "Calibrated claim PH-1 with REF-1." in reviewed
    assert sha256_file(draft_path) == manifest["articles"][0]["original_sha256"]
    assert (
        Path(manifest["articles"][0]["reviewed_path"])
        .read_bytes()
        .startswith(draft_path.read_bytes().split(b"Original claim")[0])
    )
    assert (
        Path(manifest["articles"][0]["reviewed_path"])
        .read_bytes()
        .endswith(draft_path.read_bytes().split(b"REF-1.")[1])
    )


def test_controlled_revision_rejects_unregistered_evidence(tmp_path: Path) -> None:
    primary, plan_path, _ = _review_fixture(tmp_path)
    adjudication = tmp_path / "adjudication"
    prepare_article_adjudication(primary, output_root=adjudication)
    write_json(
        adjudication / "returns" / "R3.json",
        {"reviewer_code": "R3", "results": [_result("R3", accept=False)]},
    )
    resolved_path = tmp_path / "resolved.json"
    resolve_article_claim_reviews(
        primary, adjudication_root=adjudication, output_path=resolved_path
    )
    worklist_path = tmp_path / "worklist.json"
    worklist = compile_article_revision_worklist(
        primary, resolved_path, plan_path, output_path=worklist_path
    )
    paragraph = worklist["worklist"][0]
    replacement_path = tmp_path / "replacements.json"
    write_json(
        replacement_path,
        {
            "replacements": [
                {
                    "dataset_id": "topic",
                    "paragraph_id": "PAR-0000",
                    "original_paragraph_sha256": paragraph["paragraph_sha256"],
                    "replacement_text": "Unsupported PH-999 appears here.",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        apply_controlled_paragraph_revisions(
            worklist_path, replacement_path, output_root=tmp_path / "reviewed"
        )


def _resolve_fixture(primary: Path, tmp_path: Path) -> Path:
    adjudication = tmp_path / "adjudication"
    prepare_article_adjudication(primary, output_root=adjudication)
    write_json(
        adjudication / "returns" / "R3.json",
        {"reviewer_code": "R3", "results": [_result("R3", accept=False)]},
    )
    resolved_path = tmp_path / "resolved.json"
    resolve_article_claim_reviews(
        primary, adjudication_root=adjudication, output_path=resolved_path
    )
    return resolved_path


def test_invalid_source_reaches_unsampled_downstream_paragraph(tmp_path: Path) -> None:
    primary, plan, _ = _review_fixture(tmp_path, shared=True)
    resolved = _resolve_fixture(primary, tmp_path)
    worklist = compile_article_revision_worklist(primary, resolved, plan)
    assert worklist["affected_paragraphs"] == 2
    indirect = next(row for row in worklist["worklist"] if row["paragraph_id"] == "PAR-0001")
    assert indirect["propagated_from_other_paragraph"] is True
    assert indirect["directives"][0]["packet_id"] == "AR1"


def test_rejects_edited_resolved_verdict_and_original_draft(tmp_path: Path) -> None:
    primary, plan, draft = _review_fixture(tmp_path)
    resolved_path = _resolve_fixture(primary, tmp_path)
    resolved = read_json(resolved_path)
    resolved["resolved"][0]["replacement"] = "An unreviewed edit."
    write_json(resolved_path, resolved)
    with pytest.raises(ValueError, match="differ from validated"):
        compile_article_revision_worklist(primary, resolved_path, plan)
    resolve_article_claim_reviews(
        primary, adjudication_root=tmp_path / "adjudication", output_path=resolved_path
    )
    draft.write_bytes(draft.read_bytes() + b"Unreviewed append.")
    with pytest.raises(ValueError, match="Draft changed"):
        compile_article_revision_worklist(primary, resolved_path, plan)


def test_rejects_unassigned_reviewer_return(tmp_path: Path) -> None:
    primary, _, _ = _review_fixture(tmp_path)
    write_json(
        primary / "returns" / "R9.json",
        {"reviewer_code": "R9", "results": [_result("R9", accept=True)]},
    )
    with pytest.raises(ValueError, match="Unassigned"):
        prepare_article_adjudication(primary, output_root=tmp_path / "adjudication")


def test_no_change_articles_are_copied_without_newline_normalization(tmp_path: Path) -> None:
    primary, plan, draft = _review_fixture(tmp_path)
    write_json(
        primary / "returns" / "R2.json",
        {"reviewer_code": "R2", "results": [_result("R2", accept=True)]},
    )
    resolved_path = tmp_path / "resolved.json"
    resolve_article_claim_reviews(primary, output_path=resolved_path)
    worklist_path = tmp_path / "worklist.json"
    worklist = compile_article_revision_worklist(
        primary, resolved_path, plan, output_path=worklist_path
    )
    assert worklist["affected_paragraphs"] == 0
    replacements = tmp_path / "replacements.json"
    write_json(replacements, {"replacements": []})
    applied = apply_controlled_paragraph_revisions(
        worklist_path, replacements, output_root=tmp_path / "output"
    )
    assert len(applied["articles"]) == 1
    assert Path(applied["articles"][0]["reviewed_path"]).read_bytes() == draft.read_bytes()


def test_exact_topic_sign_flip_does_not_inflate_sample_size() -> None:
    result = exact_topic_sign_flip([1.0] * 8)
    assert result["enumerated_sign_patterns"] == 256
    assert result["one_sided_p"] == 1 / 256
    assert exact_topic_sign_flip([0.0] * 8)["one_sided_p"] == 1.0


def test_routing_freeze_precedes_outcomes_and_counts_real_dual_review_time(tmp_path: Path) -> None:
    primary, _, _ = _review_fixture(tmp_path, with_returns=False)
    routing_path = tmp_path / "routing.json"
    plan = build_article_review_routing_plan(
        primary, output_path=routing_path, uniform_repetitions=3
    )
    assert len(plan["routes"]) == 16
    for reviewer, accept in (("R1", True), ("R2", False)):
        write_json(
            primary / "returns" / f"{reviewer}.json",
            {"reviewer_code": reviewer, "results": [_result(reviewer, accept=accept)]},
        )
    with pytest.raises(ValueError, match="before any reviewer outcomes"):
        build_article_review_routing_plan(primary, output_path=tmp_path / "late.json")
    resolved = _resolve_fixture(primary, tmp_path)
    replay = analyze_article_review_routing(routing_path, resolved)
    assert replay["actionable_claims"] == 1
    assert replay["routes"][0]["actual_dual_review_plus_adjudication_seconds"] == 36


def test_independent_blind_pre_post_packets_and_analysis(tmp_path: Path) -> None:
    primary, plan, _ = _review_fixture(tmp_path)
    resolved = _resolve_fixture(primary, tmp_path)
    worklist_path = tmp_path / "worklist.json"
    worklist = compile_article_revision_worklist(primary, resolved, plan, output_path=worklist_path)
    replacements_path = tmp_path / "replacements.json"
    write_json(
        replacements_path,
        {
            "replacements": [
                {
                    "dataset_id": "topic",
                    "paragraph_id": "PAR-0000",
                    "original_paragraph_sha256": worklist["worklist"][0]["paragraph_sha256"],
                    "replacement_text": "Calibrated claim PH-1 with REF-1.",
                }
            ]
        },
    )
    reviewed_root = tmp_path / "reviewed"
    apply_controlled_paragraph_revisions(
        worklist_path, replacements_path, output_root=reviewed_root
    )
    roster_path = tmp_path / "independent_roster.json"
    write_json(
        roster_path,
        {
            "reviewers": [
                {
                    "reviewer_id": f"E{n}",
                    "domain_qualified": True,
                    "eligible_topics": ["topic"],
                    "conflicted_topics": [],
                    "independent_from_article_production": True,
                }
                for n in range(3)
            ]
        },
    )
    evaluation_root = tmp_path / "evaluation"
    manifest = build_revision_evaluation_packets(
        primary,
        worklist_path,
        replacements_path,
        reviewed_root / "revision_manifest.json",
        roster_path,
        output_root=evaluation_root,
    )
    assert manifest["packets"] == 2
    assert manifest["primary_reviews_required"] == 4
    internal = read_json(evaluation_root / "internal_manifest.json")
    returned = {}
    for packet in internal["internal_packets"]:
        public = read_json(evaluation_root / "packets" / "article" / f"{packet['packet_id']}.json")
        assert "version" not in public and "pair_id" not in public
        for reviewer in packet["primary_reviewers"]:
            row = _result(reviewer, accept=packet["version"] == "post")
            row["packet_id"] = packet["packet_id"]
            returned.setdefault(reviewer, []).append(row)
    for reviewer, rows in returned.items():
        write_json(
            evaluation_root / "returns" / f"{reviewer}.json",
            {"reviewer_code": reviewer, "results": rows},
        )
    evaluated_path = tmp_path / "evaluated.json"
    resolve_article_claim_reviews(evaluation_root, output_path=evaluated_path)
    analysis = analyze_revision_evaluation(evaluation_root, evaluated_path)
    assert analysis["primary_test"]["mean_difference"] == 1.0
    assert analysis["primary_test"]["topics"] == 1
    assert analysis["primary_test"]["one_sided_p"] == 0.5
