from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_json, write_json

CARD = re.compile(r"CARD:([A-Za-z0-9_-]+)")
WORK = re.compile(r"WORK:([A-Za-z0-9_.:/-]+)")
HEADING = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|\b)")
RISK = re.compile(
    r"\b(?:indicates?|suggests?|reveals?|demonstrates?|proves?|matur(?:e|ing)|"
    r"dominant|resilien(?:t|ce)|redundan(?:t|cy)|fragile|indispensable|driven by)\b",
    re.IGNORECASE,
)
GRAPH_METRIC = re.compile(
    r"\b(?:community|hub|bridge|path|edge|weighted degree|growth ratio|"
    r"connected component|external-edge share|betweenness)\b",
    re.IGNORECASE,
)


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value), set())
    return set()


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"^##\s+(?:\d+\.\s*)?{re.escape(name)}\s*$([\s\S]*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1) if match else ""


def verify_phenomenon_article(
    *, article_path: Path, cards_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    text = article_path.read_text(encoding="utf-8")
    payload = read_json(cards_path)
    valid_cards = {card["card_id"] for card in payload["cards"]}
    valid_works = {
        work["work_id"]
        for card in payload["cards"]
        for work in card["representative_works"]
    }
    answer_labels = {
        value.casefold()
        for card in payload["cards"]
        for value in _strings(card.get("verified_answer"))
        if len(value) >= 4
    }
    used_cards = set(CARD.findall(text))
    used_works = set(WORK.findall(text))
    sections = {
        name: _section(text, name)
        for name in (
            "Abstract",
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
            "Limitations",
            "Conclusion",
        )
    }
    claim_text = "\n\n".join(
        sections[name] for name in ("Abstract", "Results", "Discussion", "Conclusion")
    )
    auditable_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", claim_text)
        if (
            NUMBER.search(paragraph)
            or RISK.search(paragraph)
            or (
                GRAPH_METRIC.search(paragraph)
                and any(label in paragraph.casefold() for label in answer_labels)
            )
        )
        and not paragraph.lstrip().startswith("#")
    ]
    uncited = [
        paragraph
        for paragraph in auditable_paragraphs
        if not CARD.search(paragraph) and not WORK.search(paragraph)
    ]
    gates = {
        "word_count_2000_4000": 2000 <= len(text.split()) <= 4000,
        "all_required_sections": all(sections.values()),
        "all_cards_used": valid_cards <= used_cards,
        "no_unknown_cards": not (used_cards - valid_cards),
        "at_least_three_works_used": len(used_works & valid_works) >= 3,
        "no_unknown_works": not (used_works - valid_works),
        "cards_in_results": len(set(CARD.findall(sections["Results"]))) >= 4,
        "cards_in_discussion": len(set(CARD.findall(sections["Discussion"]))) >= 3,
        "cards_in_conclusion": len(set(CARD.findall(sections["Conclusion"]))) >= 3,
        "has_provenance_map": "provenance map" in text.casefold(),
        "auditable_paragraph_citation_rate_at_least_0_8": (
            1.0 - len(uncited) / len(auditable_paragraphs) >= 0.8
            if auditable_paragraphs
            else False
        ),
    }
    result = {
        "schema_version": 1,
        "article": str(article_path.resolve()),
        "words": len(text.split()),
        "headings": HEADING.findall(text),
        "valid_card_ids": sorted(valid_cards),
        "used_card_ids": sorted(used_cards),
        "unknown_card_ids": sorted(used_cards - valid_cards),
        "missing_card_ids": sorted(valid_cards - used_cards),
        "valid_work_ids": sorted(valid_works),
        "used_work_ids": sorted(used_works),
        "unknown_work_ids": sorted(used_works - valid_works),
        "card_citations": len(CARD.findall(text)),
        "work_citations": len(WORK.findall(text)),
        "auditable_paragraphs": len(auditable_paragraphs),
        "uncited_auditable_paragraphs": len(uncited),
        "citation_rate": (
            1.0 - len(uncited) / len(auditable_paragraphs) if auditable_paragraphs else 0.0
        ),
        "uncited_examples": uncited[:10],
        "gates": gates,
        "passed": all(gates.values()),
    }
    if output_path:
        write_json(output_path, result)
    return result
