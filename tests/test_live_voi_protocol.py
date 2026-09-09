from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "article_quality_v3"
    / "live_sequential_voi_review_development_protocol.yml"
)
FREEZE = PROTOCOL.with_name(
    "live_sequential_voi_review_development_protocol_freeze.json"
)


def test_live_voi_protocol_is_frozen_before_outcomes_and_binds_implementation() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    freeze = read_json(FREEZE)
    assert freeze["sha256"] == sha256_file(PROTOCOL)
    assert protocol["scientific_role"] == {
        "development_only": True,
        "formal_outcome_collection": False,
        "confirmatory_replacement": False,
        "effectiveness_claim_permitted": False,
    }
    assert protocol["timing_disclosure"]["real_live_voi_human_outcomes_before_freeze"] == 0
    assert freeze["human_outcomes_at_freeze"] == 0
    for artifact in protocol["implementation"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]


def test_live_voi_protocol_preserves_static_formal_collection() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    separation = protocol["separation_from_formal_experiment"]
    assert any("static independent assignments" in line for line in separation)
    assert any("offline replay" in line for line in separation)
