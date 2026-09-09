from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from .article_expert_evaluation import ARTICLE_CONDITIONS
from .io import read_json, sha256_file, write_json

CLAIM_STRATA = (
    "results",
    "discussion",
    "graph_derived",
    "numerical",
    "causal_risk",
)
REQUIRED_SECTIONS = (
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
)
METADATA_LINE = re.compile(
    r"^\s*(?:author|generator|model|condition|system|produced by|created by)\s*:",
    re.IGNORECASE,
)
EVIDENCE_TOKEN = re.compile(r"\b(?:PH|REF)-[A-Za-z0-9_-]+\b")
FIGURE_ACCESS_MODE = "withheld_during_drafting_inserted_posthoc"
MAX_WITHIN_TOPIC_WORD_RATIO = 1.10
MAX_CONDITION_MEAN_WORD_RATIO = 1.05


def _opaque_id(prefix: str, *parts: str, seed: int) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:12].upper()}"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def _length_parity_summary(
    records: list[dict[str, Any]],
    *,
    max_within_topic_ratio: float = MAX_WITHIN_TOPIC_WORD_RATIO,
    max_condition_mean_ratio: float = MAX_CONDITION_MEAN_WORD_RATIO,
) -> dict[str, Any]:
    by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    by_condition: dict[str, list[int]] = defaultdict(list)
    for record in records:
        topic_id = str(record["topic_id"])
        condition = str(record["condition"])
        word_count = int(record["word_count"])
        if word_count <= 0:
            raise ValueError(f"Article has no countable words: {topic_id}/{condition}")
        if condition in by_topic[topic_id]:
            raise ValueError(f"Duplicate article length record: {topic_id}/{condition}")
        by_topic[topic_id][condition] = word_count
        by_condition[condition].append(word_count)

    gaps = []
    topic_summaries = {}
    for topic_id, counts in sorted(by_topic.items()):
        missing = sorted(set(ARTICLE_CONDITIONS) - set(counts))
        if missing:
            gaps.append(f"{topic_id} lacks length records for {missing}")
            continue
        shortest = min(counts.values())
        longest = max(counts.values())
        ratio = longest / shortest
        topic_summaries[topic_id] = {
            "word_counts": dict(sorted(counts.items())),
            "longest_to_shortest_ratio": ratio,
            "passes": ratio <= max_within_topic_ratio + 1e-12,
        }
        if ratio > max_within_topic_ratio + 1e-12:
            gaps.append(
                f"{topic_id} longest/shortest word-count ratio {ratio:.6f} exceeds "
                f"{max_within_topic_ratio:.2f}"
            )

    condition_means = {
        condition: sum(values) / len(values)
        for condition, values in sorted(by_condition.items())
        if values
    }
    if set(condition_means) == set(ARTICLE_CONDITIONS):
        condition_mean_ratio = max(condition_means.values()) / min(
            condition_means.values()
        )
        if condition_mean_ratio > max_condition_mean_ratio + 1e-12:
            gaps.append(
                "Across-topic condition-mean word-count ratio "
                f"{condition_mean_ratio:.6f} exceeds {max_condition_mean_ratio:.2f}"
            )
    else:
        condition_mean_ratio = None
        gaps.append("Length parity requires all three registered conditions")

    return {
        "status": "passed" if not gaps else "failed",
        "max_within_topic_word_ratio": max_within_topic_ratio,
        "max_condition_mean_word_ratio": max_condition_mean_ratio,
        "observed_max_within_topic_word_ratio": max(
            (row["longest_to_shortest_ratio"] for row in topic_summaries.values()),
            default=None,
        ),
        "observed_condition_mean_word_ratio": condition_mean_ratio,
        "condition_mean_word_counts": condition_means,
        "topics": topic_summaries,
        "gaps": gaps,
    }


def _density_summary(
    *, text: str, inventory: dict[str, Any], word_count: int
) -> dict[str, Any]:
    evidence_mentions = EVIDENCE_TOKEN.findall(text)
    candidate_texts = [
        re.sub(r"\s+", " ", str(row.get("text") or "")).strip().casefold()
        for row in inventory.get("claims", [])
        if str(row.get("text") or "").strip()
    ]
    unique_candidates = len(set(candidate_texts))
    scale = 1000.0 / word_count
    return {
        "word_count": word_count,
        "evidence_token_mentions": len(evidence_mentions),
        "unique_evidence_tokens": len(set(evidence_mentions)),
        "candidate_claims": len(candidate_texts),
        "unique_candidate_claims": unique_candidates,
        "evidence_mentions_per_1000_words": len(evidence_mentions) * scale,
        "unique_evidence_tokens_per_1000_words": len(set(evidence_mentions)) * scale,
        "candidate_claims_per_1000_words": len(candidate_texts) * scale,
        "duplicate_candidate_claim_rate": (
            1.0 - unique_candidates / len(candidate_texts) if candidate_texts else 0.0
        ),
    }


def build_article_intake_template(
    writer_pack_manifest_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    manifest = read_json(writer_pack_manifest_path)
    if manifest.get("status") != "text_only_same_evidence_writer_inputs_ready":
        raise ValueError(
            "Article intake requires audited text-only writer inputs, not image-bearing packs"
        )
    rows = []
    for record in sorted(manifest["records"], key=lambda row: row["dataset_id"]):
        for condition in ARTICLE_CONDITIONS:
            rows.append(
                {
                    "topic_id": record["dataset_id"],
                    "condition": condition,
                    "writer_pack": record["pack"],
                    "writer_pack_sha256": record["pack_sha256"],
                    "evaluation_figure": record["evaluation_figure"],
                    "evaluation_figure_sha256": record[
                        "evaluation_figure_sha256"
                    ],
                    "article_path": None,
                    "claim_inventory_path": None,
                    "production_record_path": None,
                }
            )
    template = {
        "schema_version": 1,
        "status": "awaiting_same_evidence_articles",
        "writer_pack_manifest": str(writer_pack_manifest_path.resolve()),
        "writer_pack_manifest_sha256": sha256_file(writer_pack_manifest_path),
        "required_articles": len(rows),
        "articles": rows,
    }
    write_json(output_path, template)
    return template


def build_evaluator_roster_template(
    topic_ids: list[str],
    *,
    output_path: Path,
) -> dict[str, Any]:
    template = {
        "schema_version": 1,
        "status": "awaiting_expert_recruitment",
        "topic_ids": sorted(topic_ids),
        "eligibility_contract": {
            "minimum_eligible_evaluators_per_topic": 4,
            "minimum_domain_experts_per_topic": 2,
            "conflicts_excluded_before_assignment": True,
            "independent_from_article_production": True,
            "independent_from_system_development": True,
            "prior_claim_review_or_revision_excluded_by_topic": True,
        },
        "required_evaluator_fields": [
            "evaluator_id",
            "role",
            "eligible_topics",
            "conflicted_topics",
            "conflicts_declared",
            "independent_from_article_production",
            "independent_from_system_development",
            "review_or_revision_topics",
        ],
        "evaluators": [],
    }
    write_json(output_path, template)
    return template


def _validate_production_record(
    record: dict[str, Any],
    *,
    condition: str,
    article_sha256: str,
    writer_pack_sha256: str,
    figure_sha256: str,
) -> list[str]:
    errors = []
    shared = {
        "condition": condition,
        "article_sha256": article_sha256,
        "writer_pack_sha256": writer_pack_sha256,
        "figure_sha256": figure_sha256,
        "writer_input_sha256": writer_pack_sha256,
        "figure_access_mode": FIGURE_ACCESS_MODE,
        "writer_rendered_figure_access": False,
        "no_cross_condition_draft_access": True,
    }
    for field, expected in shared.items():
        if record.get(field) != expected:
            errors.append(f"production record {field!r} must equal {expected!r}")
    for field in (
        "producer_id",
        "started_at",
        "completed_at",
        "draft_sha256",
        "posthoc_figure_insertion_receipt_sha256",
    ):
        if not record.get(field):
            errors.append(f"production record lacks {field}")
    if condition == "human_same_evidence":
        expected = {
            "domain_qualified": True,
            "system_builder": False,
            "machine_drafts_visible": False,
        }
        if not record.get("writer_input_delivery_log_sha256"):
            errors.append(
                "human production record lacks writer_input_delivery_log_sha256"
            )
    elif condition == "one_shot_llm":
        expected = {
            "generation_requests": 1,
            "iterative_revision": False,
            "graph_operator_access": False,
        }
        expected.update(input_modality="text_only_structured_graph")
        for field in ("text_request_sha256",):
            if not record.get(field):
                errors.append(f"one-shot production record lacks {field}")
    else:
        expected = {
            "graph_program_used": True,
            "validated_review_pipeline_used": True,
            "input_modality": "text_only_structured_graph",
        }
        if not record.get("review_event_log_sha256"):
            errors.append("graph-review production record lacks review_event_log_sha256")
        for field in ("text_request_sha256",):
            if not record.get(field):
                errors.append(f"graph-review production record lacks {field}")
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"production record {field!r} must equal {value!r}")
    return errors


def _sample_claims(
    inventory: dict[str, Any],
    *,
    topic_id: str,
    article_sha256: str,
    article_text: str,
    seed: int,
) -> list[dict[str, Any]]:
    if inventory.get("article_sha256") != article_sha256:
        raise ValueError("Claim inventory is not bound to the submitted article bytes")
    if not inventory.get("abstractor_id"):
        raise ValueError("Claim inventory lacks an accountable abstractor_id")
    if inventory.get("condition_blinded_to_abstractor") is not True:
        raise ValueError("Claim abstractor must be condition-blinded")
    candidates = inventory.get("claims", [])
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or None in candidate_ids:
        raise ValueError("Claim candidate IDs must be present and unique")

    slots = [f"slot:{stratum}:{index}" for stratum in CLAIM_STRATA for index in range(4)]
    ordered = sorted(
        candidates,
        key=lambda row: _opaque_id(
            "ORDER", topic_id, article_sha256, str(row["candidate_id"]), seed=seed
        ),
    )
    candidate_by_id = {}
    slot_candidates: dict[str, list[str]] = {slot: [] for slot in slots}
    for candidate in ordered:
        candidate_id = str(candidate["candidate_id"])
        text = str(candidate.get("text", "")).strip()
        if not text:
            raise ValueError(f"Empty claim candidate: {candidate_id}")
        start = candidate.get("start_char")
        end = candidate.get("end_char")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or article_text[start:end] != text
        ):
            raise ValueError(
                f"Claim candidate is not an exact article span: {candidate_id}"
            )
        eligible = set(candidate.get("eligible_strata", []))
        if not eligible or not eligible.issubset(CLAIM_STRATA):
            raise ValueError(f"Invalid claim strata for {candidate_id}")
        evidence_tokens = candidate.get("evidence_tokens", [])
        if any(
            not isinstance(token, str)
            or not EVIDENCE_TOKEN.fullmatch(token)
            or token not in article_text
            for token in evidence_tokens
        ):
            raise ValueError(
                f"Claim candidate references evidence absent from article: {candidate_id}"
            )
        candidate_by_id[candidate_id] = candidate
        for stratum in CLAIM_STRATA:
            if stratum in eligible:
                for index in range(4):
                    slot_candidates[f"slot:{stratum}:{index}"].append(candidate_id)

    candidate_to_slot: dict[str, str] = {}

    def augment(slot: str, visited: set[str]) -> bool:
        for candidate_id in slot_candidates[slot]:
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            previous_slot = candidate_to_slot.get(candidate_id)
            if previous_slot is None or augment(previous_slot, visited):
                candidate_to_slot[candidate_id] = slot
                return True
        return False

    ordered_slots = sorted(
        slots,
        key=lambda slot: (
            len(slot_candidates[slot]),
            CLAIM_STRATA.index(slot.split(":")[1]),
            int(slot.rsplit(":", 1)[1]),
        ),
    )
    unmatched = list(ordered_slots)
    while unmatched:
        next_unmatched = [slot for slot in unmatched if not augment(slot, set())]
        if len(next_unmatched) == len(unmatched):
            break
        unmatched = next_unmatched
    slot_to_candidate = {slot: candidate for candidate, slot in candidate_to_slot.items()}
    if unmatched or any(slot not in slot_to_candidate for slot in slots):
        counts = Counter(
            stratum
            for candidate in candidates
            for stratum in set(candidate.get("eligible_strata", []))
            if stratum in CLAIM_STRATA
        )
        raise ValueError(
            "Claim inventory cannot supply 4 unique claims per registered stratum; "
            f"eligible counts={dict(counts)}"
        )
    selected = []
    for stratum in CLAIM_STRATA:
        for index in range(4):
            candidate = candidate_by_id[slot_to_candidate[f"slot:{stratum}:{index}"]]
            selected.append(
                {
                    "claim_id": _opaque_id(
                        "CLM",
                        topic_id,
                        article_sha256,
                        str(candidate["candidate_id"]),
                        seed=seed,
                    ),
                    "candidate_id": candidate["candidate_id"],
                    "stratum": stratum,
                    "text": candidate["text"],
                    "section": candidate.get("section"),
                    "evidence_tokens": candidate.get("evidence_tokens", []),
                }
            )
    if len({row["candidate_id"] for row in selected}) != 20:
        raise AssertionError("Matching produced duplicate claim candidates")
    return selected


def _blind_article(text: str, *, topic_id: str, seed: int) -> tuple[str, dict[str, str]]:
    lines = text.splitlines()
    if lines[:1] == ["---"]:
        try:
            end = lines.index("---", 1)
            lines = lines[end + 1 :]
        except ValueError:
            pass
    lines = [line for line in lines if not METADATA_LINE.match(line)]
    normalized = "\n".join(lines).strip() + "\n"
    token_map = {
        token: _opaque_id("EV", topic_id, token, seed=seed)
        for token in sorted(set(EVIDENCE_TOKEN.findall(normalized)))
    }
    for raw, opaque in token_map.items():
        normalized = normalized.replace(raw, opaque)
    for condition in ARTICLE_CONDITIONS:
        normalized = re.sub(
            re.escape(condition),
            "[generator label removed]",
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized, token_map


def _select_topic_assignments(
    roster: dict[str, Any],
    topic_ids: list[str],
    *,
    seed: int,
    producer_ids_by_topic: dict[str, set[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    evaluators = roster.get("evaluators", [])
    identifiers = [row.get("evaluator_id") for row in evaluators]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("Evaluator IDs must be nonempty strings")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluator IDs must be present and unique")
    producer_ids_by_topic = producer_ids_by_topic or {}
    assignments = {}
    for topic_id in topic_ids:
        producer_ids = producer_ids_by_topic.get(topic_id, set())
        eligible = [
            row
            for row in evaluators
            if topic_id in row.get("eligible_topics", [])
            and topic_id not in row.get("conflicted_topics", [])
            and row.get("conflicts_declared") is True
            and row.get("independent_from_article_production") is True
            and row.get("independent_from_system_development") is True
            and topic_id not in row.get("review_or_revision_topics", [])
            and row.get("evaluator_id") not in producer_ids
        ]
        domain = [row for row in eligible if row.get("role") == "domain_expert"]
        if len(eligible) < 4 or len(domain) < 2:
            raise ValueError(
                f"Topic {topic_id} requires four conflict-free evaluators, including "
                "two domain experts"
            )
        generator = Random(_opaque_id("SEED", topic_id, seed=seed))
        domain = sorted(domain, key=lambda row: row["evaluator_id"])
        generator.shuffle(domain)
        selected = domain[:2]
        remaining = [row for row in eligible if row not in selected]
        remaining.sort(key=lambda row: row["evaluator_id"])
        generator.shuffle(remaining)
        selected.extend(remaining[:2])
        generator.shuffle(selected)
        blocks = [
            list(ARTICLE_CONDITIONS),
            [ARTICLE_CONDITIONS[0], ARTICLE_CONDITIONS[1]],
            [ARTICLE_CONDITIONS[0], ARTICLE_CONDITIONS[2]],
            [ARTICLE_CONDITIONS[1], ARTICLE_CONDITIONS[2]],
        ]
        assignments[topic_id] = [
            {"evaluator": evaluator, "conditions": block}
            for evaluator, block in zip(selected, blocks, strict=True)
        ]
    return assignments


def _producer_ids_by_topic(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    producer_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        production_path = row.get("production_record_path")
        if not production_path or not Path(production_path).is_file():
            continue
        record = read_json(Path(production_path))
        producer_id = record.get("producer_id")
        if isinstance(producer_id, str) and producer_id.strip():
            producer_ids[str(row["topic_id"])].add(producer_id)
    return producer_ids


def assess_article_packet_readiness(
    intake_path: Path,
    roster_path: Path,
) -> dict[str, Any]:
    intake = read_json(intake_path)
    roster = read_json(roster_path)
    rows = intake.get("articles", [])
    topics = sorted({row.get("topic_id") for row in rows if row.get("topic_id")})
    expected = {(topic, condition) for topic in topics for condition in ARTICLE_CONDITIONS}
    observed = {(row.get("topic_id"), row.get("condition")) for row in rows}
    gaps = []
    length_records = []
    if len(topics) != 8 or observed != expected or len(rows) != 24:
        gaps.append("intake must contain exactly 8 topics × 3 registered conditions")
    for row in rows:
        missing = False
        for field in ("article_path", "claim_inventory_path", "production_record_path"):
            path = row.get(field)
            if not path or not Path(path).is_file():
                gaps.append(f"{row.get('topic_id')}:{row.get('condition')} lacks {field}")
                missing = True
        pack_path = Path(str(row.get("writer_pack") or ""))
        if not pack_path.is_file():
            gaps.append(
                f"{row.get('topic_id')}:{row.get('condition')} lacks writer_pack"
            )
            continue
        if sha256_file(pack_path) != row.get("writer_pack_sha256"):
            gaps.append(
                f"{row.get('topic_id')}:{row.get('condition')} writer_pack hash mismatch"
            )
            continue
        figure_path = Path(str(row.get("evaluation_figure") or ""))
        if not figure_path.is_file():
            gaps.append(
                f"{row.get('topic_id')}:{row.get('condition')} lacks evaluation_figure"
            )
            continue
        if sha256_file(figure_path) != row.get("evaluation_figure_sha256"):
            gaps.append(
                f"{row.get('topic_id')}:{row.get('condition')} evaluation_figure hash mismatch"
            )
            continue
        if missing:
            continue
        article_path = Path(row["article_path"])
        article_text = article_path.read_text(encoding="utf-8")
        word_count = _word_count(article_text)
        pack = read_json(pack_path)
        permitted_range = (pack.get("writing_brief") or {}).get("permitted_range")
        if (
            not isinstance(permitted_range, list)
            or len(permitted_range) != 2
            or not permitted_range[0] <= word_count <= permitted_range[1]
        ):
            gaps.append(
                f"{row.get('topic_id')}:{row.get('condition')} word count {word_count} "
                "is outside the shared writing brief"
            )
        length_records.append(
            {
                "topic_id": row["topic_id"],
                "condition": row["condition"],
                "word_count": word_count,
            }
        )
        production_errors = _validate_production_record(
            read_json(Path(row["production_record_path"])),
            condition=str(row["condition"]),
            article_sha256=sha256_file(article_path),
            writer_pack_sha256=str(row["writer_pack_sha256"]),
            figure_sha256=str(row["evaluation_figure_sha256"]),
        )
        gaps.extend(
            f"{row.get('topic_id')}:{row.get('condition')} {error}"
            for error in production_errors
        )
    length_parity = (
        _length_parity_summary(length_records)
        if length_records
        else {
            "status": "failed",
            "gaps": ["No complete article lengths are available"],
        }
    )
    if len(length_records) == len(rows):
        gaps.extend(length_parity["gaps"])
    producer_ids_by_topic = _producer_ids_by_topic(rows)
    evaluator_coverage = {}
    assignment_errors = []
    for topic in topics:
        try:
            assignments = _select_topic_assignments(
                roster,
                [topic],
                seed=20260824,
                producer_ids_by_topic=producer_ids_by_topic,
            )
            evaluator_coverage[topic] = {
                "status": "ready",
                "selected_evaluators": len(assignments[topic]),
                "article_producer_ids_excluded": len(
                    producer_ids_by_topic.get(topic, set())
                ),
            }
        except ValueError as error:
            assignment_errors.append(str(error))
            evaluator_coverage[topic] = {
                "status": "blocked",
                "reason": str(error),
                "article_producer_ids_excluded": len(
                    producer_ids_by_topic.get(topic, set())
                ),
            }
    gaps.extend(assignment_errors)
    return {
        "schema_version": 1,
        "status": "ready" if not gaps else "blocked",
        "topics": len(topics),
        "article_rows": len(rows),
        "eligible_evaluators": len(roster.get("evaluators", [])),
        "evaluator_coverage": evaluator_coverage,
        "length_parity": length_parity,
        "gaps": gaps,
    }


def build_article_expert_packets(
    intake_path: Path,
    roster_path: Path,
    *,
    output_dir: Path,
    seed: int = 20260824,
) -> dict[str, Any]:
    intake = read_json(intake_path)
    roster = read_json(roster_path)
    rows = intake.get("articles", [])
    topics = sorted({row["topic_id"] for row in rows})
    expected = {(topic, condition) for topic in topics for condition in ARTICLE_CONDITIONS}
    observed = {(row["topic_id"], row["condition"]) for row in rows}
    if len(topics) != 8 or len(rows) != 24 or observed != expected:
        raise ValueError("Confirmatory intake requires exactly 8 topics × 3 conditions")
    output_dir.mkdir(parents=True, exist_ok=True)
    producer_ids_by_topic = _producer_ids_by_topic(rows)
    assignments = _select_topic_assignments(
        roster,
        topics,
        seed=seed,
        producer_ids_by_topic=producer_ids_by_topic,
    )
    prepared = {}
    topic_figures: dict[str, dict[str, str]] = {}
    for row in rows:
        article_path = Path(row["article_path"])
        pack_path = Path(row["writer_pack"])
        inventory_path = Path(row["claim_inventory_path"])
        production_path = Path(row["production_record_path"])
        figure_path = Path(row["evaluation_figure"])
        for path in (
            article_path,
            pack_path,
            inventory_path,
            production_path,
            figure_path,
        ):
            if not path.is_file():
                raise ValueError(f"Required packet input does not exist: {path}")
        if sha256_file(pack_path) != row["writer_pack_sha256"]:
            raise ValueError(f"Writer pack changed after intake freeze: {pack_path}")
        figure_sha256 = sha256_file(figure_path)
        if figure_sha256 != row["evaluation_figure_sha256"]:
            raise ValueError(f"Evaluation figure changed after intake freeze: {figure_path}")
        previous_figure = topic_figures.get(row["topic_id"])
        current_figure = {
            "source_path": str(figure_path.resolve()),
            "sha256": figure_sha256,
        }
        if previous_figure is not None and previous_figure != current_figure:
            raise ValueError(
                f"All conditions must share one exact figure for topic {row['topic_id']}"
            )
        topic_figures[row["topic_id"]] = current_figure
        article_sha256 = sha256_file(article_path)
        text = article_path.read_text(encoding="utf-8")
        pack = read_json(pack_path)
        word_count = _word_count(text)
        low, high = pack["writing_brief"]["permitted_range"]
        missing_sections = [
            section
            for section in REQUIRED_SECTIONS
            if not re.search(
                rf"^#+\s+{re.escape(section)}\s*$",
                text,
                re.IGNORECASE | re.MULTILINE,
            )
        ]
        if not low <= word_count <= high:
            raise ValueError(
                f"Article word count outside shared brief: {article_path} ({word_count})"
            )
        if missing_sections:
            raise ValueError(f"Article lacks required sections: {missing_sections}")
        production_errors = _validate_production_record(
            read_json(production_path),
            condition=row["condition"],
            article_sha256=article_sha256,
            writer_pack_sha256=row["writer_pack_sha256"],
            figure_sha256=figure_sha256,
        )
        if production_errors:
            raise ValueError("; ".join(production_errors))
        inventory = read_json(inventory_path)
        claims = _sample_claims(
            inventory,
            topic_id=row["topic_id"],
            article_sha256=article_sha256,
            article_text=text,
            seed=seed,
        )
        blinded, token_map = _blind_article(text, topic_id=row["topic_id"], seed=seed)
        blinded_claims = []
        for claim in claims:
            claim_text = claim["text"]
            for raw, opaque in token_map.items():
                claim_text = claim_text.replace(raw, opaque)
            for condition in ARTICLE_CONDITIONS:
                claim_text = re.sub(
                    re.escape(condition),
                    "[generator label removed]",
                    claim_text,
                    flags=re.IGNORECASE,
                )
            blinded_claims.append(
                {
                    "claim_id": claim["claim_id"],
                    "stratum": claim["stratum"],
                    "text": claim_text,
                    "section": claim["section"],
                    "evidence_tokens": [
                        token_map.get(token, token) for token in claim["evidence_tokens"]
                    ],
                }
            )
        article_code = _opaque_id(
            "ART", row["topic_id"], row["condition"], article_sha256, seed=seed
        )
        blinded_path = output_dir / "articles" / f"{article_code}.md"
        blinded_path.parent.mkdir(parents=True, exist_ok=True)
        blinded_path.write_text(blinded, encoding="utf-8")
        prepared[(row["topic_id"], row["condition"])] = {
            "article_code": article_code,
            "article_path": str(blinded_path.resolve()),
            "article_sha256": sha256_file(blinded_path),
            "source_article_sha256": article_sha256,
            "word_count": word_count,
            "density": _density_summary(
                text=text, inventory=inventory, word_count=word_count
            ),
            "claims": blinded_claims,
            "token_map": token_map,
        }

    length_parity = _length_parity_summary(
        [
            {
                "topic_id": topic_id,
                "condition": condition,
                "word_count": article["word_count"],
            }
            for (topic_id, condition), article in prepared.items()
        ]
    )
    if length_parity["status"] != "passed":
        raise ValueError("; ".join(length_parity["gaps"]))

    blinded_figures = {}
    for topic_id, figure in topic_figures.items():
        suffix = Path(figure["source_path"]).suffix.lower() or ".png"
        figure_code = _opaque_id("FIG", topic_id, figure["sha256"], seed=seed)
        blinded_figure_path = output_dir / "figures" / f"{figure_code}{suffix}"
        blinded_figure_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure["source_path"], blinded_figure_path)
        if sha256_file(blinded_figure_path) != figure["sha256"]:
            raise AssertionError("Blinded figure copy changed the frozen bytes")
        blinded_figures[topic_id] = {
            "figure_path": str(blinded_figure_path.resolve()),
            "figure_sha256": figure["sha256"],
        }

    evaluator_packets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignment_rows = []
    article_reviewers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for topic_id, blocks in assignments.items():
        for block in blocks:
            for condition in block["conditions"]:
                article_reviewers[(topic_id, condition)].append(
                    block["evaluator"]["evaluator_id"]
                )
    for key, reviewers in article_reviewers.items():
        if len(reviewers) != 3 or len(set(reviewers)) != 3:
            raise AssertionError(f"Article does not have three unique reviewers: {key}")
        reviewers.sort()

    for topic_id, blocks in assignments.items():
        for block in blocks:
            evaluator_id = block["evaluator"]["evaluator_id"]
            articles = []
            for condition in block["conditions"]:
                article = prepared[(topic_id, condition)]
                reviewers = article_reviewers[(topic_id, condition)]
                assigned_claims = [
                    claim
                    for claim_index, claim in enumerate(article["claims"])
                    if evaluator_id != reviewers[claim_index % len(reviewers)]
                ]
                articles.append({**article, "assigned_claims": assigned_claims})
            generator = Random(_opaque_id("ORDER", evaluator_id, topic_id, seed=seed))
            generator.shuffle(articles)
            evaluator_packets[evaluator_id].append(
                {
                    "topic_code": _opaque_id("TOP", topic_id, seed=seed),
                    **blinded_figures[topic_id],
                    "articles": [
                        {
                            "article_code": article["article_code"],
                            "article_path": article["article_path"],
                            "article_sha256": article["article_sha256"],
                            "claims": article["assigned_claims"],
                        }
                        for article in articles
                    ],
                }
            )
            assignment_rows.append(
                {
                    "topic_id": topic_id,
                    "evaluator_id": evaluator_id,
                    "evaluator_role": block["evaluator"]["role"],
                    "conditions": block["conditions"],
                    "article_codes": [article["article_code"] for article in articles],
                }
            )
    packet_records = []
    for evaluator_id, topic_packets in evaluator_packets.items():
        path = output_dir / "evaluators" / f"{evaluator_id}.json"
        packet = {
            "schema_version": 1,
            "evaluator_id": evaluator_id,
            "hidden_fields_excluded": ["condition", "author_identity", "automatic_scores"],
            "topics": topic_packets,
        }
        write_json(path, packet)
        packet_records.append(
            {"evaluator_id": evaluator_id, "path": str(path.resolve()), "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": 1,
        "status": "expert_packets_ready",
        "seed": seed,
        "intake_sha256": sha256_file(intake_path),
        "evaluator_roster_sha256": sha256_file(roster_path),
        "topics": len(topics),
        "articles": len(prepared),
        "claims_per_article": 20,
        "claim_stratum_quota": 4,
        "article_quality_controls": {
            "status": "pre_review_controls_passed",
            "length_parity": length_parity,
            "density_metrics_are_descriptive_only": True,
            "articles": [
                {
                    "topic_id": topic_id,
                    "condition": condition,
                    "article_code": article["article_code"],
                    **article["density"],
                }
                for (topic_id, condition), article in sorted(prepared.items())
            ],
        },
        "assignment_design": (
            "Per topic: one evaluator rates all three conditions and three evaluators "
            "rate one distinct condition pair; each article has three raters and each "
            "condition pair has two co-raters."
        ),
        "assignments": assignment_rows,
        "evaluator_packets": packet_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
