import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "temporal_parity", Path(__file__).parents[1] / "scripts/audit_temporal_evidence_parity.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_window_swap_preserves_marginals_but_changes_time_assignment():
    trends = pd.DataFrame(
        [{"keyword": "a", "year": year, "documents": year - 2017} for year in range(2018, 2026)]
    )
    changed, mapping = module.swap_windows(trends)
    assert changed.documents.sum() == trends.documents.sum()
    assert set(changed.year) == set(trends.year)
    assert mapping[2018] == 2023
    assert mapping[2023] == 2018
    assert not changed.equals(trends)
    restored, _ = module.swap_windows(changed)
    assert restored.equals(trends)


def test_short_history_cannot_create_overlapping_window_witness():
    with pytest.raises(ValueError):
        module.swap_windows(pd.DataFrame({"year": [2020, 2021, 2022]}))
