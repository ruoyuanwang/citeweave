from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from .article_generation_diagnostic import WORD_PATTERN, hierarchical_section_text

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|\b)")
RISK_PATTERN = re.compile(
    r"\b(?:indicates?|suggests?|reveals?|demonstrates?|proves?|driven by|dominant|"
    r"hot topic|mature|nascent|phase transition|influential|impactful)\b",
    re.IGNORECASE,
)
CALIBRATION_PATTERN = re.compile(
    r"\b(?:may|might|could|cannot|uncertain|contingent|alternative|limitation|"
    r"depends? on|does not establish|should not be interpreted)\b",
    re.IGNORECASE,
)
EVIDENCE_TOKEN_PATTERN = re.compile(r"\b(?:PH|REF)-[A-Za-z0-9_-]+\b")
MATCHED_SECTIONS = ("Abstract", "Results", "Discussion", "Conclusion")


def matched_section_view(article: str) -> str:
    """Return the sections shared by generated and cached human reference reports."""

    chunks = []
    for section in MATCHED_SECTIONS:
        body = hierarchical_section_text(article, section)
        if body:
            chunks.append(f"## {section}\n\n{body}")
    return "\n\n".join(chunks).strip()


def _prose_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for value in re.split(r"\n\s*\n", text):
        value = value.strip()
        if not value or value.startswith(("#", "|", "```")):
            continue
        paragraphs.append(" ".join(value.split()))
    return paragraphs


def _sentences(paragraphs: list[str]) -> list[str]:
    values = []
    for paragraph in paragraphs:
        values.extend(
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY.split(paragraph)
            if sentence.strip()
        )
    return values


def _msttr(tokens: list[str], window: int = 100) -> float:
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    scores = [
        len(set(tokens[start : start + window])) / window
        for start in range(0, len(tokens) - window + 1, window)
    ]
    return statistics.fmean(scores)


def _normalized_sentence(sentence: str) -> str:
    return " ".join(token.casefold() for token in WORD_PATTERN.findall(sentence))


def _shingles(paragraph: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in WORD_PATTERN.findall(paragraph)]
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _near_duplicate_paragraph_rate(paragraphs: list[str]) -> tuple[int, int, float]:
    shingles = [_shingles(paragraph) for paragraph in paragraphs]
    pairs = 0
    duplicates = 0
    for left in range(len(shingles)):
        if not shingles[left]:
            continue
        for right in range(left + 1, len(shingles)):
            if not shingles[right]:
                continue
            pairs += 1
            union = shingles[left] | shingles[right]
            similarity = len(shingles[left] & shingles[right]) / len(union)
            if similarity >= 0.8:
                duplicates += 1
    return duplicates, pairs, duplicates / pairs if pairs else 0.0


def diagnose_article_text(text: str) -> dict[str, Any]:
    paragraphs = _prose_paragraphs(text)
    sentences = _sentences(paragraphs)
    tokens = [token.casefold() for token in WORD_PATTERN.findall(" ".join(paragraphs))]
    word_count = len(tokens)
    sentence_lengths = [len(WORD_PATTERN.findall(sentence)) for sentence in sentences]
    paragraph_lengths = [len(WORD_PATTERN.findall(paragraph)) for paragraph in paragraphs]
    eligible_sentences = [
        _normalized_sentence(sentence)
        for sentence in sentences
        if len(WORD_PATTERN.findall(sentence)) >= 8
    ]
    sentence_counts = Counter(eligible_sentences)
    exact_duplicate_excess = sum(count - 1 for count in sentence_counts.values() if count > 1)
    near_duplicates, paragraph_pairs, near_duplicate_rate = (
        _near_duplicate_paragraph_rate(paragraphs)
    )
    paragraph_mean = statistics.fmean(paragraph_lengths) if paragraph_lengths else 0.0
    paragraph_cv = (
        statistics.pstdev(paragraph_lengths) / paragraph_mean
        if len(paragraph_lengths) > 1 and paragraph_mean
        else 0.0
    )
    per_thousand = 1000 / word_count if word_count else 0.0
    evidence_tokens = EVIDENCE_TOKEN_PATTERN.findall(text)
    return {
        "characters": len(text),
        "words": word_count,
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "mean_sentence_words": (
            statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
        ),
        "sentence_words_p90": (
            sorted(sentence_lengths)[max(0, math.ceil(0.9 * len(sentence_lengths)) - 1)]
            if sentence_lengths
            else 0
        ),
        "mean_paragraph_words": paragraph_mean,
        "paragraph_length_cv": paragraph_cv,
        "msttr_100": _msttr(tokens),
        "numeric_mentions_per_1000_words": len(NUMBER_PATTERN.findall(text))
        * per_thousand,
        "interpretive_risk_markers_per_1000_words": len(RISK_PATTERN.findall(text))
        * per_thousand,
        "calibration_markers_per_1000_words": len(CALIBRATION_PATTERN.findall(text))
        * per_thousand,
        "exact_duplicate_sentence_excess_rate": (
            exact_duplicate_excess / len(eligible_sentences) if eligible_sentences else 0.0
        ),
        "near_duplicate_paragraph_pairs": near_duplicates,
        "eligible_paragraph_pairs": paragraph_pairs,
        "near_duplicate_paragraph_pair_rate": near_duplicate_rate,
        "evidence_token_mentions": len(evidence_tokens),
        "unique_evidence_tokens": len(set(evidence_tokens)),
    }


def summarize_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one diagnostic row is required")
    excluded = {"article_id", "condition", "path", "source_sha256", "view_sha256"}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "articles": len(rows),
        "mean": {
            key: statistics.fmean(float(row[key]) for row in rows) for key in numeric_keys
        },
        "median": {
            key: statistics.median(float(row[key]) for row in rows) for key in numeric_keys
        },
    }
