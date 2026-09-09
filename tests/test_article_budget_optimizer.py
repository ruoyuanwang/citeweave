from __future__ import annotations

from citeweave.article_budget_optimizer import optimize_article_word_budget
from citeweave.article_generation_diagnostic import TOKEN_PATTERN, WORD_PATTERN


def _article() -> str:
    sections = []
    for section in (
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    ):
        paragraphs = []
        for paragraph in range(4):
            sentences = [
                f"Essential {section} paragraph {paragraph} PH-{paragraph} REF-{paragraph} remains.",
                "Generic contextual material repeats background concepts and can be removed safely.",
                "Another long generic sentence elaborates wording without adding registered evidence identifiers.",
                "Closing generic interpretation stays available when one sentence is removed.",
            ]
            paragraphs.append(" ".join(sentences))
        sections.append(f"## {section}\n\n" + "\n\n".join(paragraphs))
    return "\n\n".join(sections) + "\n"


def test_optimizer_deletes_whole_non_evidence_sentences_and_preserves_tokens() -> None:
    article = _article()
    original_words = len(WORD_PATTERN.findall(article))
    optimized, audit = optimize_article_word_budget(
        article,
        low=450,
        high=500,
        target=490,
    )
    assert 450 <= len(WORD_PATTERN.findall(optimized)) <= 500
    assert len(WORD_PATTERN.findall(optimized)) < original_words
    assert set(TOKEN_PATTERN.findall(optimized)) == set(TOKEN_PATTERN.findall(article))
    assert audit["removed_sentence_count"] > 0
    assert audit["evidence_token_set_preserved"] is True


def test_optimizer_leaves_in_range_article_unchanged() -> None:
    article = _article()
    words = len(WORD_PATTERN.findall(article))
    optimized, audit = optimize_article_word_budget(
        article, low=words - 1, high=words + 1, target=words
    )
    assert optimized == article
    assert audit["status"] == "already_within_budget"
