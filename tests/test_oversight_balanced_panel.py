from citeweave.oversight_balanced_panel import (
    _choose_generation,
    _choose_reject_ids,
)


def _score(verdict: str) -> dict[str, object]:
    if verdict == "reject":
        return {"answer_exact": False, "abstain": False}
    return {
        "answer_exact": True,
        "abstain": False,
        "evidence_precision": 0.8,
        "evidence_recall": 1.0,
        "has_required_limitation": True,
    }


def test_reject_subset_is_balanced_and_retains_mandatory_cases() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "case_id": f"C{index}",
                "issue_type": f"I{index % 4}",
                "task_type": f"T{index % 5}",
                "available_gold_verdicts": (
                    ["reject"] if index < 2 else ["qualify", "reject"]
                ),
            }
        )
    selected = _choose_reject_ids(rows, seed=7)
    assert len(selected) == 6
    assert {"C0", "C1"} <= selected


def test_generation_selection_prefers_strongest_available_condition() -> None:
    generations = [
        {"condition": "flat_bm25", "score": _score("reject")},
        {"condition": "flat_program", "score": _score("reject")},
        {"condition": "graph_program", "score": _score("qualify")},
    ]
    selected = _choose_generation(generations, target_verdict="reject")
    assert selected["condition"] == "flat_program"
