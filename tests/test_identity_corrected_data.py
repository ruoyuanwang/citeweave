import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "data_audit", Path(__file__).parents[1] / "scripts" / "audit_identity_corrected_data.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_detects_cross_work_unknown_and_known_identity_drift(tmp_path):
    old = pd.DataFrame(
        {
            "work_id": ["w1", "w2", "w1"],
            "author_id": ["openalex-author:None", "openalex-author:None", "known:1"],
            "position": [1, 1, 2],
        }
    )
    new = old.copy()
    new["author_id"] = ["openalex-author-occurrence:a", "openalex-author-occurrence:b", "known:1"]
    old_path, new_path = tmp_path / "old.parquet", tmp_path / "new.parquet"
    old.to_parquet(old_path)
    new.to_parquet(new_path)
    assert module.compare_memberships(old_path, new_path)["passed"]
    new.loc[1, "author_id"] = new.loc[0, "author_id"]
    new.to_parquet(new_path)
    assert not module.compare_memberships(old_path, new_path)["passed"]
    new.loc[1, "author_id"] = "openalex-author-occurrence:b"
    new.loc[2, "author_id"] = "known:2"
    new.to_parquet(new_path)
    result = module.compare_memberships(old_path, new_path)
    assert not result["passed"]
    assert result["known_author_membership_multiset_differences"] == 2
