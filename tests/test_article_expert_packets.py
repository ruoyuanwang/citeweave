from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from citeweave.article_expert_evaluation import ARTICLE_CONDITIONS
from citeweave.article_expert_packets import (
    CLAIM_STRATA,
    FIGURE_ACCESS_MODE,
    MAX_CONDITION_MEAN_WORD_RATIO,
    MAX_WITHIN_TOPIC_WORD_RATIO,
    assess_article_packet_readiness,
    build_article_expert_packets,
    build_article_intake_template,
    build_evaluator_roster_template,
)
from citeweave.io import read_json, sha256_file, write_json


def _article(condition: str) -> str:
    sections = (
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    )
    body = "\n\n".join(
        f"## {section}\nEvidence PH-AAAAAAAAAAAAAAAA and REF-BBBBBBBBBBBBBBBB. "
        f"Condition: {condition}."
        for section in sections
    )
    claims = "\n".join(
        f"Registered claim {index} cites REF-BBBBBBBBBBBBBBBB."
        for index in range(20)
    )
    return f"{body}\n\n{claims}\n"


def _production_record(
    condition: str, article_sha256: str, pack_sha256: str, figure_sha256: str
) -> dict[str, object]:
    record: dict[str, object] = {
        "condition": condition,
        "article_sha256": article_sha256,
        "writer_pack_sha256": pack_sha256,
        "figure_sha256": figure_sha256,
        "writer_input_sha256": pack_sha256,
        "figure_access_mode": FIGURE_ACCESS_MODE,
        "writer_rendered_figure_access": False,
        "draft_sha256": "d" * 64,
        "posthoc_figure_insertion_receipt_sha256": "i" * 64,
        "producer_id": "producer",
        "started_at": "2026-08-24T01:00:00Z",
        "completed_at": "2026-08-24T02:00:00Z",
        "no_cross_condition_draft_access": True,
    }
    if condition == "human_same_evidence":
        record.update(
            domain_qualified=True,
            system_builder=False,
            machine_drafts_visible=False,
            writer_input_delivery_log_sha256="f" * 64,
        )
    elif condition == "one_shot_llm":
        record.update(
            generation_requests=1,
            iterative_revision=False,
            graph_operator_access=False,
            input_modality="text_only_structured_graph",
            text_request_sha256="m" * 64,
        )
    else:
        record.update(
            graph_program_used=True,
            validated_review_pipeline_used=True,
            review_event_log_sha256="c" * 64,
            input_modality="text_only_structured_graph",
            text_request_sha256="m" * 64,
        )
    return record


def _complete_inputs(root: Path) -> tuple[Path, Path]:
    packs = root / "packs"
    records = []
    for topic_index in range(8):
        topic_id = f"topic_{topic_index}"
        figure_path = packs / topic_id / "graph.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure_path.write_bytes(f"figure-{topic_id}".encode())
        pack_path = packs / topic_id / "evidence_pack.json"
        write_json(
            pack_path,
            {
                "writing_brief": {"permitted_range": [1, 10_000]},
                "posthoc_figure_commitment": {
                    "figure_sha256": sha256_file(figure_path)
                },
            },
        )
        records.append(
            {
                "dataset_id": topic_id,
                "pack": str(pack_path.resolve()),
                "pack_sha256": sha256_file(pack_path),
                "evaluation_figure": str(figure_path.resolve()),
                "evaluation_figure_sha256": sha256_file(figure_path),
            }
        )
    pack_manifest = root / "pack_manifest.json"
    write_json(
        pack_manifest,
        {
            "status": "text_only_same_evidence_writer_inputs_ready",
            "records": records,
        },
    )
    intake_path = root / "article_intake.json"
    intake = build_article_intake_template(pack_manifest, output_path=intake_path)
    for row in intake["articles"]:
        stem = f"{row['topic_id']}__{row['condition']}"
        article_path = root / "articles" / f"{stem}.md"
        article_path.parent.mkdir(parents=True, exist_ok=True)
        article_path.write_text(_article(row["condition"]), encoding="utf-8")
        article_text = article_path.read_text(encoding="utf-8")
        article_sha256 = sha256_file(article_path)
        inventory_path = root / "inventories" / f"{stem}.json"
        claims = []
        global_index = 0
        for stratum in CLAIM_STRATA:
            for claim_index in range(4):
                claim_text = (
                    f"Registered claim {global_index} cites REF-BBBBBBBBBBBBBBBB."
                )
                start_char = article_text.index(claim_text)
                claims.append(
                    {
                        "candidate_id": f"{stratum}_{claim_index}",
                        "text": claim_text,
                        "start_char": start_char,
                        "end_char": start_char + len(claim_text),
                        "section": "Results",
                        "eligible_strata": [stratum],
                        "evidence_tokens": ["REF-BBBBBBBBBBBBBBBB"],
                    }
                )
                global_index += 1
        write_json(
            inventory_path,
            {
                "article_sha256": article_sha256,
                "abstractor_id": "blinded_abstractor",
                "condition_blinded_to_abstractor": True,
                "claims": claims,
            },
        )
        production_path = root / "production" / f"{stem}.json"
        write_json(
            production_path,
            _production_record(
                row["condition"],
                article_sha256,
                row["writer_pack_sha256"],
                row["evaluation_figure_sha256"],
            ),
        )
        row["article_path"] = str(article_path.resolve())
        row["claim_inventory_path"] = str(inventory_path.resolve())
        row["production_record_path"] = str(production_path.resolve())
    intake["status"] = "same_evidence_articles_frozen"
    write_json(intake_path, intake)

    roster_path = root / "evaluator_roster.json"
    evaluators = []
    topics = [f"topic_{index}" for index in range(8)]
    for index in range(4):
        evaluators.append(
            {
                "evaluator_id": f"expert_{index}",
                "role": "domain_expert" if index < 2 else "methods_expert",
                "eligible_topics": topics,
                "conflicted_topics": [],
                "conflicts_declared": True,
                "independent_from_article_production": True,
                "independent_from_system_development": True,
                "review_or_revision_topics": [],
            }
        )
    write_json(roster_path, {"evaluators": evaluators})
    return intake_path, roster_path


def test_templates_report_real_missing_inputs(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        pack = tmp_path / f"pack_{index}.json"
        write_json(pack, {"topic": index})
        records.append(
            {
                "dataset_id": f"topic_{index}",
                "pack": str(pack.resolve()),
                "pack_sha256": sha256_file(pack),
                "evaluation_figure": str(pack.resolve()),
                "evaluation_figure_sha256": sha256_file(pack),
            }
        )
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "status": "text_only_same_evidence_writer_inputs_ready",
            "records": records,
        },
    )
    intake_path = tmp_path / "intake.json"
    roster_path = tmp_path / "roster.json"
    intake = build_article_intake_template(manifest, output_path=intake_path)
    build_evaluator_roster_template(
        [row["dataset_id"] for row in records], output_path=roster_path
    )
    readiness = assess_article_packet_readiness(intake_path, roster_path)
    assert len(intake["articles"]) == 24
    assert readiness["status"] == "blocked"
    assert any("lacks article_path" in gap for gap in readiness["gaps"])
    assert any("four conflict-free evaluators" in gap for gap in readiness["gaps"])


def test_builds_blinded_balanced_packets(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    assert assess_article_packet_readiness(intake_path, roster_path)["status"] == "ready"
    output = tmp_path / "packets"
    manifest = build_article_expert_packets(
        intake_path, roster_path, output_dir=output
    )
    assert manifest["status"] == "expert_packets_ready"
    assert manifest["articles"] == 24
    controls = manifest["article_quality_controls"]
    assert controls["status"] == "pre_review_controls_passed"
    assert controls["length_parity"]["status"] == "passed"
    assert (
        controls["length_parity"]["observed_max_within_topic_word_ratio"]
        <= MAX_WITHIN_TOPIC_WORD_RATIO
    )
    assert (
        controls["length_parity"]["observed_condition_mean_word_ratio"]
        <= MAX_CONDITION_MEAN_WORD_RATIO
    )
    assert len(controls["articles"]) == 24
    assert all(row["candidate_claims"] == 20 for row in controls["articles"])
    assignment_counts = Counter()
    pair_counts = Counter()
    for row in manifest["assignments"]:
        for condition in row["conditions"]:
            assignment_counts[(row["topic_id"], condition)] += 1
        for left_index, left in enumerate(row["conditions"]):
            for right in row["conditions"][left_index + 1 :]:
                pair_counts[(row["topic_id"], frozenset((left, right)))] += 1
    assert set(assignment_counts.values()) == {3}
    assert set(pair_counts.values()) == {2}

    claim_rater_counts = Counter()
    article_strata: dict[str, dict[str, str]] = {}
    for record in manifest["evaluator_packets"]:
        packet_text = Path(record["path"]).read_text(encoding="utf-8")
        packet = json.loads(packet_text)
        assert all(condition not in packet_text for condition in ARTICLE_CONDITIONS)
        for topic in packet["topics"]:
            for article in topic["articles"]:
                article_strata.setdefault(article["article_code"], {})
                for claim in article["claims"]:
                    claim_rater_counts[(article["article_code"], claim["claim_id"])] += 1
                    article_strata[article["article_code"]][claim["claim_id"]] = claim[
                        "stratum"
                    ]
                blinded = Path(article["article_path"]).read_text(encoding="utf-8")
                assert "PH-AAAAAAAAAAAAAAAA" not in blinded
                assert "REF-BBBBBBBBBBBBBBBB" not in blinded
                assert "EV-" in blinded
    assert set(claim_rater_counts.values()) == {2}
    assert all(len(claims) == 20 for claims in article_strata.values())
    assert all(
        Counter(claims.values()) == {stratum: 4 for stratum in CLAIM_STRATA}
        for claims in article_strata.values()
    )


def test_readiness_rejects_missing_text_request_receipt(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    intake = read_json(intake_path)
    row = next(
        item for item in intake["articles"] if item["condition"] == "one_shot_llm"
    )
    production_path = Path(row["production_record_path"])
    production = read_json(production_path)
    del production["text_request_sha256"]
    write_json(production_path, production)
    readiness = assess_article_packet_readiness(intake_path, roster_path)
    assert readiness["status"] == "blocked"
    assert any("text_request_sha256" in gap for gap in readiness["gaps"])


def test_readiness_rejects_cross_role_or_producer_evaluators(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    roster = read_json(roster_path)
    roster["evaluators"][0]["review_or_revision_topics"] = ["topic_0"]
    roster["evaluators"][1]["independent_from_system_development"] = False
    roster["evaluators"][2]["evaluator_id"] = "producer"
    write_json(roster_path, roster)

    readiness = assess_article_packet_readiness(intake_path, roster_path)

    assert readiness["status"] == "blocked"
    assert readiness["evaluator_coverage"]["topic_0"]["status"] == "blocked"
    assert any("topic_0 requires four conflict-free evaluators" in gap for gap in readiness["gaps"])


def test_readiness_reports_every_undercovered_topic(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    roster = read_json(roster_path)
    roster["evaluators"][0]["conflicts_declared"] = False
    write_json(roster_path, roster)

    readiness = assess_article_packet_readiness(intake_path, roster_path)

    blocked_topics = [
        topic
        for topic, row in readiness["evaluator_coverage"].items()
        if row["status"] == "blocked"
    ]
    assert blocked_topics == [f"topic_{index}" for index in range(8)]
    assert sum("requires four conflict-free evaluators" in gap for gap in readiness["gaps"]) == 8


def test_rejects_unblinded_claim_abstractor(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    intake = read_json(intake_path)
    inventory_path = Path(intake["articles"][0]["claim_inventory_path"])
    inventory = read_json(inventory_path)
    inventory["condition_blinded_to_abstractor"] = False
    write_json(inventory_path, inventory)
    try:
        build_article_expert_packets(
            intake_path, roster_path, output_dir=tmp_path / "packets"
        )
    except ValueError as error:
        assert "condition-blinded" in str(error)
    else:
        raise AssertionError("Unblinded claim abstraction must be rejected")


def test_readiness_rejects_within_topic_length_advantage(tmp_path: Path) -> None:
    intake_path, roster_path = _complete_inputs(tmp_path)
    intake = read_json(intake_path)
    row = next(
        item
        for item in intake["articles"]
        if item["topic_id"] == "topic_0"
        and item["condition"] == "citeweave_graph_review"
    )
    article_path = Path(row["article_path"])
    article_path.write_text(
        article_path.read_text(encoding="utf-8") + (" additional" * 120),
        encoding="utf-8",
    )
    article_sha256 = sha256_file(article_path)
    inventory_path = Path(row["claim_inventory_path"])
    inventory = read_json(inventory_path)
    inventory["article_sha256"] = article_sha256
    write_json(inventory_path, inventory)
    production_path = Path(row["production_record_path"])
    production = read_json(production_path)
    production["article_sha256"] = article_sha256
    write_json(production_path, production)

    readiness = assess_article_packet_readiness(intake_path, roster_path)
    assert readiness["status"] == "blocked"
    assert readiness["length_parity"]["status"] == "failed"
    assert any("longest/shortest" in gap for gap in readiness["gaps"])
