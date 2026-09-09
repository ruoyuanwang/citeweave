import pandas as pd
import pytest

from citeweave.temporal_evidence_control import (
    add_shared_temporal_context,
    build_temporal_annex,
    control_item_id,
)


def test_annex_contains_raw_counts_and_rules_but_no_selected_answer():
    trends = pd.DataFrame(
        [
            {"keyword": word, "year": year, "documents": year - 2017}
            for word in ("a", "b")
            for year in range(2018, 2026)
        ]
    )
    annex = build_temporal_annex("toy", trends)
    assert annex["records"][0]["year_document_counts"] == list(range(1, 9))
    assert annex["public_task_definition"]["early_years"] == [2018, 2019, 2020]
    assert all(set(record) == {"keyword", "year_document_counts"} for record in annex["records"])
    assert "emerging_label" not in annex and "verified_answer" not in annex


def test_all_methods_get_identical_annex_without_mutating_source():
    context = {"rows": [{"node_id": "n"}]}
    annex = {"records": [{"keyword": "a"}]}
    first = add_shared_temporal_context(context, annex)
    second = add_shared_temporal_context({"operator_trace": []}, annex)
    assert first["shared_temporal_evidence"] == second["shared_temporal_evidence"]
    first["shared_temporal_evidence"]["records"].clear()
    assert annex["records"] and second["shared_temporal_evidence"]["records"]
    assert "shared_temporal_evidence" not in context
    with pytest.raises(ValueError):
        add_shared_temporal_context(second, annex)


def test_control_ids_are_separate_and_stable():
    assert control_item_id("a") == control_item_id("a")
    assert control_item_id("a") != control_item_id("b")
    assert control_item_id("a") != "a"
