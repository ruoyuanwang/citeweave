from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .article_generation_diagnostic import TOKEN_PATTERN, WORD_PATTERN

TOP_LEVEL_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


@dataclass
class _Paragraph:
    section: str
    paragraph_index: int
    sentences: list[str]


def _sections(article: str) -> list[tuple[str, str]]:
    matches = list(TOP_LEVEL_HEADING.finditer(article))
    if not matches:
        raise ValueError("No level-two article sections found")
    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(article)
        sections.append((match.group(1).strip(), article[start:end].strip()))
    return sections


def _paragraphs(article: str) -> list[_Paragraph]:
    parsed = []
    for section, body in _sections(article):
        for index, paragraph in enumerate(re.split(r"\n\s*\n", body), start=1):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            sentences = [value.strip() for value in SENTENCE_BOUNDARY.split(paragraph)]
            parsed.append(_Paragraph(section, index, sentences))
    return parsed


def _render(paragraphs: list[_Paragraph]) -> str:
    by_section: dict[str, list[_Paragraph]] = {}
    order = []
    for paragraph in paragraphs:
        if paragraph.section not in by_section:
            by_section[paragraph.section] = []
            order.append(paragraph.section)
        by_section[paragraph.section].append(paragraph)
    return (
        "\n\n".join(
            f"## {section}\n\n"
            + "\n\n".join(" ".join(row.sentences) for row in by_section[section])
            for section in order
        )
        + "\n"
    )


def optimize_article_word_budget(
    article: str,
    *,
    low: int = 2700,
    high: int = 3300,
    target: int = 3250,
) -> tuple[str, dict[str, Any]]:
    if not (0 < low <= target <= high):
        raise ValueError("Invalid article word-budget bounds")
    original_words = len(WORD_PATTERN.findall(article))
    original_tokens = set(TOKEN_PATTERN.findall(article))
    if original_words < low:
        raise ValueError("Deterministic deletion cannot repair an underlength article")
    if original_words <= high:
        return article, {
            "schema_version": 1,
            "status": "already_within_budget",
            "original_word_count": original_words,
            "optimized_word_count": original_words,
            "removed_sentences": [],
            "evidence_token_set_preserved": True,
        }

    paragraphs = _paragraphs(article)
    section_priority = {
        "Results": 0,
        "Discussion": 1,
        "Introduction": 2,
        "Methods": 3,
        "Limitations": 4,
        "Abstract": 5,
        "Conclusion": 6,
    }
    candidates = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        if len(paragraph.sentences) <= 2:
            continue
        for sentence_index, sentence in enumerate(paragraph.sentences):
            if sentence_index == 0 or TOKEN_PATTERN.search(sentence):
                continue
            words = len(WORD_PATTERN.findall(sentence))
            candidates.append(
                (
                    -words,
                    section_priority.get(paragraph.section, 99),
                    paragraph_index,
                    sentence_index,
                    sentence,
                )
            )
    candidates.sort()
    removed: list[dict[str, Any]] = []
    removed_indices: dict[int, set[int]] = {}
    current_words = original_words
    for negative_words, _, paragraph_index, sentence_index, sentence in candidates:
        if current_words <= target:
            break
        paragraph = paragraphs[paragraph_index]
        already_removed = removed_indices.setdefault(paragraph_index, set())
        if len(paragraph.sentences) - len(already_removed) <= 2:
            continue
        already_removed.add(sentence_index)
        words = -negative_words
        current_words -= words
        removed.append(
            {
                "section": paragraph.section,
                "paragraph_index": paragraph.paragraph_index,
                "sentence_index": sentence_index + 1,
                "word_count": words,
                "sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
            }
        )
    for paragraph_index, indices in removed_indices.items():
        paragraphs[paragraph_index].sentences = [
            sentence
            for index, sentence in enumerate(paragraphs[paragraph_index].sentences)
            if index not in indices
        ]
    optimized = _render(paragraphs)
    optimized_words = len(WORD_PATTERN.findall(optimized))
    optimized_tokens = set(TOKEN_PATTERN.findall(optimized))
    if optimized_words > high or optimized_words < low:
        raise ValueError(
            f"Deterministic sentence deletion could not meet budget: {optimized_words}"
        )
    if optimized_tokens != original_tokens:
        raise ValueError("Article budget optimization changed the evidence-token set")
    return optimized, {
        "schema_version": 1,
        "status": "deterministic_sentence_budget_optimized",
        "original_word_count": original_words,
        "optimized_word_count": optimized_words,
        "target_word_count": target,
        "removed_sentence_count": len(removed),
        "removed_word_count": original_words - optimized_words,
        "removed_sentences": removed,
        "evidence_token_set_preserved": True,
    }
