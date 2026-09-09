from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

import yaml

from .io import read_json, sha256_file, write_json

EVIDENCE_TOKEN = re.compile(r"\b(?:PH|REF)-[A-Za-z0-9_-]+\b")
HEADING = re.compile(r"^#+[ \t]+([^\r\n]+?)[ \t]*\r?$", re.MULTILINE)
SENTENCE = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)
CAUSAL_RISK = re.compile(
    r"\b(?:cause[ds]?|causal|drive[sn]?|led to|leads? to|result(?:s|ed)? in|"
    r"determin(?:e|es|ed)|transform(?:s|ed)?|proves?|demonstrates?)\b",
    re.IGNORECASE,
)
NUMERIC = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")


def validate_article_claim_review_protocol(
    protocol_path: Path,
    freeze_path: Path,
    machine_plan_path: Path,
    amendment_path: Path | None = None,
    amendment_freeze_path: Path | None = None,
) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    protocol_hash = sha256_file(protocol_path)
    if freeze.get("sha256") != protocol_hash:
        raise RuntimeError("Article claim review protocol differs from its freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != freeze.get("protocol_id"):
        raise RuntimeError("Article claim review protocol identity mismatch")
    plan_hash = sha256_file(machine_plan_path)
    if (
        protocol.get("prerequisites", {}).get("machine_generation_plan", {}).get("sha256")
        != plan_hash
    ):
        raise RuntimeError("Article claim review protocol does not bind this plan")
    implementation = dict(protocol.get("implementation") or {})
    if (amendment_path is None) != (amendment_freeze_path is None):
        raise RuntimeError("Both amendment and amendment freeze are required")
    if amendment_path is not None:
        amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
        amendment_freeze = read_json(amendment_freeze_path)
        if (
            amendment_freeze.get("sha256") != sha256_file(amendment_path)
            or amendment.get("base_protocol_sha256") != protocol_hash
            or amendment.get("machine_plan_sha256") != plan_hash
            or amendment_freeze.get("amendment_id") != amendment.get("amendment_id")
        ):
            raise RuntimeError("Article review amendment identity/hash mismatch")
        implementation.update(amendment.get("implementation_overrides") or {})
    for label, artifact in implementation.items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Article claim review implementation mismatch: {label}")
    return protocol


def _opaque(prefix: str, *parts: str, seed: int) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode()
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16].upper()}"


def _article_paragraphs(article: str) -> list[dict[str, Any]]:
    paragraphs = []
    cursor = 0
    headings = list(HEADING.finditer(article))
    for heading_index, heading in enumerate(headings):
        section = heading.group(1).strip()
        content_start = heading.end()
        content_end = (
            headings[heading_index + 1].start()
            if heading_index + 1 < len(headings)
            else len(article)
        )
        content = article[content_start:content_end]
        for block_match in re.finditer(
            r"\S(?:.*?\S)?(?=\r?\n[ \t]*\r?\n|[ \t\r\n]*\Z)", content, re.DOTALL
        ):
            block = block_match.group(0)
            if not EVIDENCE_TOKEN.search(block):
                continue
            paragraph_id = f"PAR-{cursor:04d}"
            cursor += 1
            paragraph_hash = hashlib.sha256(block.encode()).hexdigest()
            absolute_start = content_start + block_match.start()
            paragraphs.append(
                {
                    "paragraph_id": paragraph_id,
                    "paragraph_sha256": paragraph_hash,
                    "section": section,
                    "text": block,
                    "start_char": absolute_start,
                    "end_char": content_start + block_match.end(),
                }
            )
    return paragraphs


def _claim_candidates(article: str) -> list[dict[str, Any]]:
    candidates = []
    for paragraph in _article_paragraphs(article):
        if paragraph["section"].casefold() not in {"results", "discussion"}:
            continue
        for sentence_index, match in enumerate(SENTENCE.finditer(paragraph["text"])):
            text = match.group(0).strip()
            tokens = sorted(set(EVIDENCE_TOKEN.findall(text)))
            phenomenon_ids = [token for token in tokens if token.startswith("PH-")]
            if not phenomenon_ids:
                continue
            absolute_start = paragraph["start_char"] + match.start()
            absolute_start += len(match.group(0)) - len(match.group(0).lstrip())
            candidates.append(
                {
                    "candidate_id": (f"{paragraph['paragraph_id']}:S{sentence_index:02d}"),
                    "text": text,
                    "section": paragraph["section"],
                    "paragraph_id": paragraph["paragraph_id"],
                    "paragraph_sha256": paragraph["paragraph_sha256"],
                    "paragraph_context": paragraph["text"],
                    "start_char": absolute_start,
                    "end_char": absolute_start + len(text),
                    "evidence_tokens": tokens,
                    "phenomenon_ids": phenomenon_ids,
                    "risk_features": {
                        "causal_language": bool(CAUSAL_RISK.search(text)),
                        "numeric": bool(NUMERIC.search(EVIDENCE_TOKEN.sub("", text))),
                        "multiple_phenomena": len(phenomenon_ids) > 1,
                        "discussion_interpretation": (
                            paragraph["section"].casefold() == "discussion"
                        ),
                    },
                }
            )
    return candidates


def select_article_review_claims(
    article: str,
    *,
    phenomenon_ids: list[str],
    claims: int = 20,
    minimum_per_phenomenon: int = 2,
    seed: int = 20260827,
) -> list[dict[str, Any]]:
    candidates = _claim_candidates(article)

    def score(row: dict[str, Any]) -> tuple[int, str]:
        risk = row["risk_features"]
        value = (
            4 * int(risk["causal_language"])
            + 2 * int(risk["numeric"])
            + 2 * int(risk["multiple_phenomena"])
            + int(risk["discussion_interpretation"])
        )
        tie = _opaque("ORDER", row["candidate_id"], row["text"], seed=seed)
        return -value, tie

    by_phenomenon = {
        phenomenon_id: sorted(
            [row for row in candidates if phenomenon_id in row["phenomenon_ids"]],
            key=score,
        )
        for phenomenon_id in phenomenon_ids
    }
    selected: list[dict[str, Any]] = []
    selected_ids = set()
    for phenomenon_id in phenomenon_ids:
        available = by_phenomenon[phenomenon_id]
        for row in available:
            if (
                sum(phenomenon_id in item["phenomenon_ids"] for item in selected)
                >= minimum_per_phenomenon
            ):
                break
            if row["candidate_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["candidate_id"])
            if (
                sum(phenomenon_id in item["phenomenon_ids"] for item in selected)
                >= minimum_per_phenomenon
            ):
                break
        if (
            sum(phenomenon_id in item["phenomenon_ids"] for item in selected)
            < minimum_per_phenomenon
        ):
            raise ValueError(
                f"Draft lacks {minimum_per_phenomenon} distinct claims for {phenomenon_id}"
            )
    for row in sorted(candidates, key=score):
        if len(selected) >= claims:
            break
        if row["candidate_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["candidate_id"])
    if len(selected) != claims:
        raise ValueError(f"Draft requires {claims} unique reviewable claims, found {len(selected)}")
    return sorted(selected, key=lambda row: (row["start_char"], row["candidate_id"]))


def _reviewers_for_topic(
    roster: dict[str, Any], topic_id: str, *, seed: int
) -> tuple[list[str], str]:
    identities = [row.get("reviewer_id") for row in roster.get("reviewers", [])]
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("Every reviewer needs a nonempty identity")
    if len(identities) != len(set(identities)):
        raise ValueError("Reviewer identities must be unique")
    eligible = [
        row
        for row in roster.get("reviewers", [])
        if topic_id in row.get("eligible_topics", [])
        and topic_id not in row.get("conflicted_topics", [])
        and row.get("domain_qualified") is True
    ]
    if len(eligible) < 3:
        raise ValueError(
            f"Topic {topic_id} requires three conflict-free domain-qualified reviewers"
        )
    generator = Random(_opaque("REVIEWERS", topic_id, seed=seed))
    eligible.sort(key=lambda row: row["reviewer_id"])
    generator.shuffle(eligible)
    return [eligible[0]["reviewer_id"], eligible[1]["reviewer_id"]], eligible[2]["reviewer_id"]


def build_article_claim_review_packets(
    machine_plan_path: Path,
    roster_path: Path,
    *,
    output_root: Path,
    seed: int = 20260827,
) -> dict[str, Any]:
    plan = read_json(machine_plan_path)
    roster = read_json(roster_path)
    citeweave_cells = [row for row in plan["cells"] if row["condition"] == "citeweave_graph_review"]
    if len(citeweave_cells) != 8 or len({row["dataset_id"] for row in citeweave_cells}) != 8:
        raise ValueError("Article claim review requires eight distinct CiteWeave cells")
    if (output_root / "internal_manifest.json").exists():
        existing = read_json(output_root / "internal_manifest.json")
        if existing.get("machine_plan_sha256") != sha256_file(machine_plan_path) or existing.get(
            "roster_sha256"
        ) != sha256_file(roster_path):
            raise ValueError("Frozen article review inputs changed")
        for record in existing["packet_records"]:
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise ValueError("Frozen article review packet changed")
        return read_json(output_root / "manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)
    assignments: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"article": []})
    adjudicators = {}
    packet_records = []
    internal_packets = []
    for cell in sorted(citeweave_cells, key=lambda row: row["dataset_id"]):
        record_path = Path(cell["output_dir"]) / "execution_record.json"
        draft_path = Path(cell["output_dir"]) / "draft.md"
        if not record_path.is_file() or not draft_path.is_file():
            raise ValueError(f"CiteWeave draft is missing: {cell['dataset_id']}")
        execution = read_json(record_path)
        if execution.get("status") != "draft_ready_for_human_review":
            raise ValueError(f"CiteWeave draft is not review-ready: {cell['dataset_id']}")
        if execution.get("draft_sha256") != sha256_file(draft_path):
            raise ValueError(f"CiteWeave draft hash mismatch: {cell['dataset_id']}")
        writer_input_path = Path(cell["writer_input"])
        if cell.get("writer_input_sha256") != sha256_file(writer_input_path):
            raise ValueError(f"Writer input hash mismatch: {cell['dataset_id']}")
        writer_input = read_json(writer_input_path)
        phenomenon_index = {row["phenomenon_id"]: row for row in writer_input["graph_phenomena"]}
        source_index = {row["reference_id"]: row for row in writer_input["representative_sources"]}
        article = draft_path.read_bytes().decode("utf-8")
        selected = select_article_review_claims(
            article,
            phenomenon_ids=sorted(phenomenon_index),
            seed=seed,
        )
        primary_reviewers, adjudicator = _reviewers_for_topic(roster, cell["dataset_id"], seed=seed)
        adjudicators[cell["dataset_id"]] = adjudicator
        for claim in selected:
            packet_id = _opaque(
                "AR",
                cell["dataset_id"],
                execution["draft_sha256"],
                claim["candidate_id"],
                seed=seed,
            )
            linked_phenomena = [
                phenomenon_index[token]
                for token in claim["phenomenon_ids"]
                if token in phenomenon_index
            ]
            linked_refs = sorted(
                {token for token in claim["evidence_tokens"] if token in source_index}
                | {
                    reference_id
                    for phenomenon in linked_phenomena
                    for reference_id in phenomenon["reference_ids"]
                }
            )
            packet = {
                "schema_version": 1,
                "packet_id": packet_id,
                "review_layer": "article",
                "topic_code": _opaque("TOPIC", cell["dataset_id"], seed=seed),
                "claim_id": _opaque("CLAIM", cell["dataset_id"], claim["candidate_id"], seed=seed),
                "claim": claim["text"],
                "paragraph_context": claim["paragraph_context"],
                "section": claim["section"],
                "paragraph_id": claim["paragraph_id"],
                "paragraph_sha256": claim["paragraph_sha256"],
                "risk_features": claim["risk_features"],
                "phenomena": linked_phenomena,
                "sources": [source_index[value] for value in linked_refs],
                "allowed_decisive_evidence_ids": sorted(
                    set(claim["evidence_tokens"])
                    | set(linked_refs)
                    | {
                        evidence_id
                        for phenomenon in linked_phenomena
                        for evidence_id in phenomenon["graph_evidence_ids"]
                    }
                ),
                "review_questions": {
                    "factual_supported": "Is the claim factually supported by the supplied graph result and excerpts?",
                    "interpretation_calibrated": "Does the wording stay within the allowed interpretation contract?",
                    "alternative_adequate": "Is an adequate alternative explanation or failure condition provided where needed?",
                    "evidence_sufficient": "Are the decisive evidence dependencies sufficient?",
                    "action": "accept, rewrite, reject_claim, or abstain",
                },
            }
            packet_path = output_root / "packets" / "article" / f"{packet_id}.json"
            write_json(packet_path, packet)
            packet_records.append(
                {
                    "packet_id": packet_id,
                    "path": str(packet_path.resolve()),
                    "sha256": sha256_file(packet_path),
                }
            )
            internal_packets.append(
                {
                    "packet_id": packet_id,
                    "dataset_id": cell["dataset_id"],
                    "draft_sha256": execution["draft_sha256"],
                    "draft_path": str(draft_path.resolve()),
                    "paragraph_sha256": claim["paragraph_sha256"],
                    "candidate_id": claim["candidate_id"],
                    "start_char": claim["start_char"],
                    "end_char": claim["end_char"],
                    "paragraph_id": claim["paragraph_id"],
                    "evidence_tokens": claim["evidence_tokens"],
                    "primary_reviewers": primary_reviewers,
                    "adjudicator": adjudicator,
                }
            )
            for reviewer in primary_reviewers:
                assignments[reviewer]["article"].append(packet_id)
    public_manifest = {
        "schema_version": 1,
        "status": "article_claim_review_packets_ready",
        "machine_plan_sha256": sha256_file(machine_plan_path),
        "roster_sha256": sha256_file(roster_path),
        "topics": 8,
        "claims_per_topic": 20,
        "packets": len(packet_records),
        "primary_reviews_required": len(packet_records) * 2,
        "assignment_rule": "Every claim receives two independent primary reviews; disagreements only are adjudicated by the frozen third reviewer.",
        "packet_records": packet_records,
    }
    write_json(output_root / "manifest.json", public_manifest)
    write_json(
        output_root / "internal_manifest.json",
        {
            **public_manifest,
            "assignments": dict(assignments),
            "topic_adjudicators": adjudicators,
            "internal_packets": internal_packets,
        },
    )
    write_json(
        output_root / "reviewer_roster_freeze.json",
        {
            "schema_version": 1,
            "roster_sha256": sha256_file(roster_path),
            "machine_plan_sha256": sha256_file(machine_plan_path),
            "outcome_count_at_assignment": 0,
        },
    )
    return public_manifest


def build_article_reviewer_roster_template(
    topic_ids: list[str], *, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("Refusing to overwrite an existing reviewer roster")
    roster = {
        "schema_version": 1,
        "status": "awaiting_real_article_reviewers",
        "topic_ids": sorted(topic_ids),
        "reviewer_contract": {
            "minimum_conflict_free_domain_qualified_reviewers_per_topic": 3,
            "two_primary_plus_one_adjudicator": True,
            "synthetic_or_proxy_reviewers_prohibited": True,
        },
        "reviewers": [],
    }
    write_json(output_path, roster)
    return roster


def assess_article_claim_review_readiness(
    machine_plan_path: Path,
    roster_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    plan = read_json(machine_plan_path)
    cells = [
        row for row in plan.get("cells", []) if row.get("condition") == "citeweave_graph_review"
    ]
    topics = sorted({str(row.get("dataset_id")) for row in cells})
    missing_drafts = []
    invalid_drafts = []
    for cell in cells:
        output_dir = Path(cell["output_dir"])
        draft_path = output_dir / "draft.md"
        record_path = output_dir / "execution_record.json"
        if not draft_path.is_file() or not record_path.is_file():
            missing_drafts.append(cell["dataset_id"])
            continue
        record = read_json(record_path)
        if record.get("status") != "draft_ready_for_human_review" or record.get(
            "draft_sha256"
        ) != sha256_file(draft_path):
            invalid_drafts.append(cell["dataset_id"])
    roster = read_json(roster_path) if roster_path.is_file() else {"reviewers": []}
    reviewers = roster.get("reviewers", [])
    coverage = {}
    for topic in topics:
        eligible = sorted(
            {
                str(row["reviewer_id"])
                for row in reviewers
                if topic in row.get("eligible_topics", [])
                and topic not in row.get("conflicted_topics", [])
                and row.get("domain_qualified") is True
                and row.get("reviewer_id")
            }
        )
        coverage[topic] = {
            "eligible_conflict_free_domain_reviewers": eligible,
            "count": len(eligible),
            "ready": len(eligible) >= 3,
        }
    reasons = []
    reviewer_ids = [row.get("reviewer_id") for row in reviewers]
    if any(not isinstance(value, str) or not value.strip() for value in reviewer_ids):
        reasons.append("reviewer identities must be nonempty strings")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        reasons.append("duplicate reviewer identities are prohibited")
    if len(cells) != 8 or len(topics) != 8:
        reasons.append("exactly eight distinct CiteWeave article cells are required")
    if missing_drafts:
        reasons.append(f"{len(missing_drafts)} CiteWeave drafts are missing")
    if invalid_drafts:
        reasons.append(f"{len(invalid_drafts)} CiteWeave drafts fail status/hash validation")
    undercovered = [topic for topic, row in coverage.items() if not row["ready"]]
    if undercovered:
        reasons.append(f"{len(undercovered)} topics lack three qualified conflict-free reviewers")
    result = {
        "schema_version": 1,
        "status": "ready" if not reasons else "blocked",
        "machine_plan_sha256": sha256_file(machine_plan_path),
        "roster_sha256": sha256_file(roster_path) if roster_path.is_file() else None,
        "citeweave_cells": len(cells),
        "distinct_topics": len(topics),
        "real_reviewers_registered": len(reviewers),
        "missing_drafts": sorted(missing_drafts),
        "invalid_drafts": sorted(invalid_drafts),
        "reviewer_coverage": coverage,
        "blocking_reasons": reasons,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
