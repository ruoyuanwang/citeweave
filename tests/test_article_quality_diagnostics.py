from citeweave.article_quality_diagnostics import (
    diagnose_article_text,
    matched_section_view,
    summarize_diagnostics,
)


def test_matched_section_view_excludes_unshared_sections() -> None:
    article = """# Title

## Abstract

Abstract sentence.

## Introduction

Introduction sentence.

## Results

### Nested result

Result sentence.

## Discussion

Discussion sentence.

## Limitations

Limitation sentence.

## Conclusion

Conclusion sentence.
"""
    view = matched_section_view(article)
    assert "Abstract sentence" in view
    assert "Nested result" in view
    assert "Introduction sentence" not in view
    assert "Limitation sentence" not in view


def test_diagnostics_detect_repetition_and_markers() -> None:
    repeated = "This repeated sentence contains enough words for deterministic duplicate detection."
    text = (
        "## Results\n\n"
        + repeated
        + " "
        + repeated
        + " It may indicate 42 influential nodes (PH-a; REF-b).\n\n"
        + repeated
    )
    result = diagnose_article_text(text)
    assert result["words"] > 0
    assert result["exact_duplicate_sentence_excess_rate"] > 0
    assert result["numeric_mentions_per_1000_words"] > 0
    assert result["interpretive_risk_markers_per_1000_words"] > 0
    assert result["calibration_markers_per_1000_words"] > 0
    assert result["unique_evidence_tokens"] == 2


def test_summary_reports_mean_and_median() -> None:
    rows = [
        {"article_id": "a", "words": 100, "msttr_100": 0.5},
        {"article_id": "b", "words": 300, "msttr_100": 0.7},
    ]
    summary = summarize_diagnostics(rows)
    assert summary["articles"] == 2
    assert summary["mean"]["words"] == 200
    assert summary["median"]["msttr_100"] == 0.6
