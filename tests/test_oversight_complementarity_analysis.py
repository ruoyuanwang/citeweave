import pytest

from citeweave.oversight_complementarity_analysis import (
    OversightComplementarityError,
    analyze_complementarity_cases,
)


def _cases() -> list[dict[str, object]]:
    rows = []
    for dataset in range(8):
        for case in range(12):
            gold = "qualify"
            first = "qualify" if case < 6 else "supported"
            second = "qualify" if case < 8 else "supported"
            rows.append(
                {
                    "case_id": f"D{dataset}-C{case}",
                    "dataset_id": f"D{dataset}",
                    "gold_verdict": gold,
                    "ai_verdict": "supported",
                    "final_verdict": "qualify" if case < 10 else "supported",
                    "team_review_seconds": 30.0,
                    "primary_decisions": [
                        {
                            "reviewer_id": f"R{case % 6}",
                            "verdict": first,
                            "review_seconds": 10.0,
                        },
                        {
                            "reviewer_id": f"R{(case + 1) % 6}",
                            "verdict": second,
                            "review_seconds": 10.0,
                        },
                    ],
                }
            )
    return rows


def test_complementarity_requires_both_registered_superiority_contrasts() -> None:
    result = analyze_complementarity_cases(_cases())
    assert result["cases"] == 96
    assert result["registered_tests"]["C1_team_minus_unaudited_ai"]["passed"]
    assert result["registered_tests"]["C2_team_minus_single_primary"]["passed"]
    assert result["joint_complementarity_passed"] is True
    assert result["pooled_diagnostics"]["team_rescue_cases"] > 0


def test_complementarity_rejects_unbalanced_panel() -> None:
    with pytest.raises(OversightComplementarityError, match="balanced held-out"):
        analyze_complementarity_cases(_cases()[:-1])


def test_complementarity_rejects_impossible_team_time() -> None:
    rows = _cases()
    rows[0]["team_review_seconds"] = 1.0
    with pytest.raises(OversightComplementarityError, match="Team time"):
        analyze_complementarity_cases(rows)
