from __future__ import annotations

import pytest

from citeweave.formal_protocol_amendment import effective_query_datasets


def test_query_amendment_changes_only_permitted_query_fields() -> None:
    protocol = {
        "datasets": [
            {"id": "a", "keywords": ["old"], "year_from": 2010},
            {"id": "b", "keywords": ["stable"], "year_from": 2010},
        ]
    }
    amendment = {
        "changes": {
            "datasets": {
                "a": {"keywords": ["new"], "rationale": "query review"}
            }
        }
    }
    effective = effective_query_datasets(protocol, amendment=amendment)
    assert effective[0]["keywords"] == ["new"]
    assert effective[1]["keywords"] == ["stable"]
    assert protocol["datasets"][0]["keywords"] == ["old"]


def test_query_amendment_rejects_design_changes() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        effective_query_datasets(
            {"datasets": [{"id": "a", "keywords": ["old"]}]},
            amendment={
                "changes": {"datasets": {"a": {"year_from": 2020}}}
            },
        )
