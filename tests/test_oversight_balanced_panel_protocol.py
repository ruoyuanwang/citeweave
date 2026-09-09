from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    ROOT
    / "experiments"
    / "human_review_v2"
    / "review_policy_amendment_008_balanced_real_candidate_panel.yml"
)
FREEZE = AMENDMENT.with_name(
    "review_policy_amendment_008_balanced_real_candidate_panel_freeze.json"
)


def test_balanced_panel_amendment_and_bound_artifacts_are_immutable() -> None:
    amendment = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    freeze = read_json(FREEZE)
    assert freeze["sha256"] == sha256_file(AMENDMENT)
    assert freeze["human_outcomes_inspected"] is False
    for receipt in amendment["implementation"].values():
        assert sha256_file(ROOT / receipt["path"]) == receipt["sha256"]
    panel = amendment["balanced_panel"]
    assert sha256_file(ROOT / panel["task_selection_path"]) == panel[
        "task_selection_sha256"
    ]
    assert sha256_file(ROOT / panel["packet_manifest_path"]) == panel[
        "packet_manifest_sha256"
    ]
    assert sha256_file(ROOT / panel["private_construction_audit_path"]) == panel[
        "private_construction_audit_sha256"
    ]
