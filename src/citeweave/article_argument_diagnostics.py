from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from .article_generation_diagnostic import WORD_PATTERN, hierarchical_section_text

MATCHED_SECTIONS = ("Abstract", "Results", "Discussion", "Conclusion")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
REF_TOKEN_PATTERN = re.compile(r"\bREF-[A-Za-z0-9_-]+\b")
NUMERIC_CITATION_PATTERN = re.compile(r"\[(?:\s*\d+[a-z]?(?:\s*[-–—,;]\s*)?)+\]")
AUTHOR_YEAR_PATTERN = re.compile(
    r"\((?:[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?[,]?\s+)?(?:19|20)\d{2}[a-z]?\)"
)
NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|\b)")

GENERIC_SCAFFOLD_PATTERN = re.compile(
    r"\b(?:this study|the present study|this analysis|the analysis|these findings|"
    r"this finding|this result|these results|as documented|as reported|as evidenced|"
    r"in the observed (?:network|graph)|overall|taken together)\b",
    re.IGNORECASE,
)
CONTRAST_PATTERN = re.compile(
    r"\b(?:however|whereas|in contrast|by contrast|yet|although|nevertheless|"
    r"rather than|despite|on the other hand)\b",
    re.IGNORECASE,
)
ALTERNATIVE_PATTERN = re.compile(
    r"\b(?:alternative explanation|alternatively|could instead|may instead|"
    r"might instead|another possibility|confound(?:er|ing)?|artifact|artefact)\b",
    re.IGNORECASE,
)
FAILURE_PATTERN = re.compile(
    r"\b(?:fails? when|failure condition|under (?:a |the )?different|sensitive to|"
    r"depends? on|contingent on|would not hold|may not hold|breaks? down|"
    r"limited by|subject to)\b",
    re.IGNORECASE,
)
MECHANISM_PATTERN = re.compile(
    r"\b(?:mechanis(?:m|tic)|pathway|mediates?|modulates?|inhibits?|activates?|"
    r"drives?|regulates?|degradation|diffusion|adsorption|oxidation|reduction|redox|"
    r"molecular|cellular|electrochemical|thermal|kinetic|transport|interaction)\b",
    re.IGNORECASE,
)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    }
)


def _paragraphs(text: str) -> list[str]:
    return [
        " ".join(value.split())
        for value in re.split(r"\n\s*\n", text)
        if value.strip() and not value.lstrip().startswith(("#", "|", "```"))
    ]


def _sentences(paragraphs: list[str]) -> list[str]:
    return [
        sentence.strip()
        for paragraph in paragraphs
        for sentence in SENTENCE_BOUNDARY.split(paragraph)
        if sentence.strip()
    ]


def _tokens(text: str, *, content_only: bool = False) -> list[str]:
    values = [token.casefold() for token in WORD_PATTERN.findall(text)]
    if content_only:
        return [token for token in values if token not in STOPWORDS and len(token) > 2]
    return values


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _paragraph_similarity(paragraphs: list[str]) -> tuple[list[float], list[str]]:
    vectors = [Counter(_tokens(paragraph, content_only=True)) for paragraph in paragraphs]
    similarities = [
        _cosine(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    openings = [" ".join(_tokens(paragraph, content_only=True)[:6]) for paragraph in paragraphs]
    return similarities, [opening for opening in openings if opening]


def _source_markers(sentence: str) -> set[str]:
    markers = set(REF_TOKEN_PATTERN.findall(sentence))
    for match in NUMERIC_CITATION_PATTERN.findall(sentence):
        markers.update(f"numeric:{value}" for value in re.findall(r"\d+[a-z]?", match))
    markers.update(f"author_year:{match.casefold()}" for match in AUTHOR_YEAR_PATTERN.findall(sentence))
    return markers


def _normalized_entropy(counts: list[int]) -> float:
    positive = [count for count in counts if count > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    entropy = -sum((count / total) * math.log(count / total) for count in positive)
    return entropy / math.log(len(positive))


def diagnose_argument_structure(text: str) -> dict[str, Any]:
    """Compute preregisterable surface proxies for argument structure.

    These metrics describe textual form. They do not establish factuality, novelty,
    scientific value, or human preference.
    """

    paragraphs = _paragraphs(text)
    sentences = _sentences(paragraphs)
    words = _tokens(" ".join(paragraphs))
    per_thousand = 1000 / len(words) if words else 0.0
    sentence_lengths = [len(_tokens(sentence)) for sentence in sentences]
    sentence_mean = statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
    sentence_cv = (
        statistics.pstdev(sentence_lengths) / sentence_mean
        if len(sentence_lengths) > 1 and sentence_mean
        else 0.0
    )

    section_words = {
        section: len(_tokens(hierarchical_section_text(text, section)))
        for section in MATCHED_SECTIONS
    }
    section_total = sum(section_words.values())
    section_shares = {
        section: count / section_total if section_total else 0.0
        for section, count in section_words.items()
    }

    similarities, openings = _paragraph_similarity(paragraphs)
    opening_counts = Counter(openings)
    repeated_opening_excess = sum(
        count - 1 for count in opening_counts.values() if count > 1
    )

    marker_sets = [_source_markers(sentence) for sentence in sentences]
    source_bearing = sum(bool(markers) for markers in marker_sets)
    multisource = sum(len(markers) >= 2 for markers in marker_sets)
    distinct_markers = set().union(*marker_sets) if marker_sets else set()

    return {
        "words": len(words),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "sentence_length_cv": sentence_cv,
        "section_word_counts": section_words,
        "section_word_shares": section_shares,
        "normalized_section_entropy": _normalized_entropy(list(section_words.values())),
        "largest_section_share": max(section_shares.values(), default=0.0),
        "mean_paragraph_pair_cosine": (
            statistics.fmean(similarities) if similarities else 0.0
        ),
        "max_paragraph_pair_cosine": max(similarities, default=0.0),
        "paragraph_pairs_cosine_ge_0_5_rate": (
            sum(value >= 0.5 for value in similarities) / len(similarities)
            if similarities
            else 0.0
        ),
        "repeated_six_content_word_opening_excess_rate": (
            repeated_opening_excess / len(openings) if openings else 0.0
        ),
        "source_marker_mentions_per_1000_words": (
            sum(len(markers) for markers in marker_sets) * per_thousand
        ),
        "distinct_source_markers": len(distinct_markers),
        "source_bearing_sentence_fraction": (
            source_bearing / len(sentences) if sentences else 0.0
        ),
        "multi_source_sentence_fraction": (
            multisource / len(sentences) if sentences else 0.0
        ),
        "multi_source_share_among_source_bearing_sentences": (
            multisource / source_bearing if source_bearing else 0.0
        ),
        "generic_scaffolding_markers_per_1000_words": (
            len(GENERIC_SCAFFOLD_PATTERN.findall(text)) * per_thousand
        ),
        "contrast_markers_per_1000_words": (
            len(CONTRAST_PATTERN.findall(text)) * per_thousand
        ),
        "alternative_explanation_markers_per_1000_words": (
            len(ALTERNATIVE_PATTERN.findall(text)) * per_thousand
        ),
        "failure_condition_markers_per_1000_words": (
            len(FAILURE_PATTERN.findall(text)) * per_thousand
        ),
        "mechanism_markers_per_1000_words": (
            len(MECHANISM_PATTERN.findall(text)) * per_thousand
        ),
        "numeric_mentions_per_1000_words": len(NUMBER_PATTERN.findall(text))
        * per_thousand,
    }
