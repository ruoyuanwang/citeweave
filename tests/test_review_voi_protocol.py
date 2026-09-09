from __future__ import annotations

from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file


def test_sequential_voi_protocol_and_development_panel_are_hash_bound() -> None:
    protocol_path = Path(
        "experiments/article_quality_v2/"
        "article_claim_review_amendment_003_sequential_dependency_voi.yml"
    )
    freeze_path = protocol_path.with_name(
        "article_claim_review_amendment_003_sequential_dependency_voi_freeze.json"
    )
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    assert freeze["sha256"] == sha256_file(protocol_path)
    assert freeze["formal_human_outcomes"] == 0
    assert freeze["development_human_outcomes"] == 0
    for artifact in protocol["implementation"].values():
        assert sha256_file(Path(artifact["path"])) == artifact["sha256"]
    panel = read_json(Path(protocol["development_schema_audit"]["path"]))
    assert sha256_file(Path(protocol["development_schema_audit"]["path"])) == (
        protocol["development_schema_audit"]["sha256"]
    )
    assert panel["claim_features"] == 80
    assert panel["human_outcomes_present"] == 0
