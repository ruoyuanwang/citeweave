import pytest

from citeweave.article_argument_diagnostics import diagnose_argument_structure


def test_argument_diagnostic_detects_structure_and_source_integration() -> None:
    text = """## Abstract

This study reports a mechanism with support (REF-a; REF-b).

## Results

However, transport depends on temperature [1, 2].

However, transport depends on pressure [1, 3].

## Discussion

An alternative explanation is adsorption, which may not hold under a different model.

## Conclusion

Taken together, the redox pathway is contingent on sampling (REF-c).
"""
    result = diagnose_argument_structure(text)
    assert result["normalized_section_entropy"] > 0
    assert result["source_bearing_sentence_fraction"] > 0
    assert result["multi_source_sentence_fraction"] > 0
    assert result["contrast_markers_per_1000_words"] > 0
    assert result["alternative_explanation_markers_per_1000_words"] > 0
    assert result["failure_condition_markers_per_1000_words"] > 0
    assert result["mechanism_markers_per_1000_words"] > 0


def test_argument_diagnostic_detects_lexical_template_reuse() -> None:
    paragraph = (
        "Registered evidence connects alpha beta gamma delta epsilon zeta to a result."
    )
    text = f"## Results\n\n{paragraph}\n\n{paragraph}\n\n## Discussion\n\nShort contrast."
    result = diagnose_argument_structure(text)
    assert result["max_paragraph_pair_cosine"] == pytest.approx(1.0)
    assert result["paragraph_pairs_cosine_ge_0_5_rate"] > 0
    assert result["repeated_six_content_word_opening_excess_rate"] > 0


def test_argument_diagnostic_handles_empty_text() -> None:
    result = diagnose_argument_structure("")
    assert result["words"] == 0
    assert result["largest_section_share"] == 0
    assert result["source_bearing_sentence_fraction"] == 0
