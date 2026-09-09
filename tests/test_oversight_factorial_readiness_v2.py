from __future__ import annotations

from pathlib import Path

import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_factorial_assignment import build_factorial_oversight_assignment
from citeweave.oversight_factorial_readiness_v2 import (
    assess_factorial_oversight_readiness_v2,
)
from tests.test_oversight_factorial_assignment import (
    AMENDMENT,
    AMENDMENT_FREEZE,
    _factorial_inputs,
)

_IMPLEMENTATIONS = {
    "capability_freeze": "src/citeweave/oversight_capability_freeze.py",
    "heldout_packet_builder": "src/citeweave/oversight_heldout_packets.py",
    "factorial_assignment": "src/citeweave/oversight_factorial_assignment.py",
    "factorial_outcomes": "src/citeweave/oversight_factorial_outcomes.py",
    "factorial_analysis": "src/citeweave/oversight_factorial_analysis.py",
    "server_timed_review_ui": "src/citeweave/review_ui.py",
    "readiness_v2": "src/citeweave/oversight_factorial_readiness_v2.py",
    "capability_freeze_cli": "scripts/freeze_integrated_oversight_capabilities.py",
    "heldout_packet_cli": "scripts/prepare_oversight_heldout_packets.py",
    "factorial_assignment_cli": "scripts/prepare_factorial_oversight_assignment.py",
    "factorial_outcomes_cli": "scripts/run_oversight_factorial_outcomes.py",
    "factorial_analysis_cli": "scripts/analyze_complementary_oversight_factorial.py",
    "readiness_v2_cli": "scripts/audit_factorial_oversight_readiness_v2.py",
}


def _frozen_amendment(tmp_path: Path) -> tuple[Path, Path]:
    amendment = tmp_path / "amendment.yml"
    payload = {
        "schema_version": 1,
        "amendment_id": "review_policy_amendment_005_integrated_capability_and_factorial_execution",
        "human_outcomes_inspected": False,
        "heldout_outcomes_inspected": False,
        "previous_amendment_sha256": sha256_file(AMENDMENT),
        "implementation": {
            name: {"path": path, "sha256": sha256_file(Path(path))}
            for name, path in _IMPLEMENTATIONS.items()
        },
    }
    amendment.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    write_json(
        freeze,
        {
            "amendment_id": payload["amendment_id"],
            "sha256": sha256_file(amendment),
            "human_outcomes_inspected": False,
            "heldout_outcomes_inspected": False,
        },
    )
    return amendment, freeze


def _ready_panel(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    manifest, registry, capability, observations = _factorial_inputs(tmp_path)
    assignment_root = tmp_path / "assignment"
    build_factorial_oversight_assignment(
        packet_manifest_path=manifest,
        reviewer_registry_path=registry,
        capability_freeze_path=capability,
        combined_observations_path=observations,
        amendment_path=AMENDMENT,
        amendment_freeze_path=AMENDMENT_FREEZE,
        output_root=assignment_root,
        minimum_capability_lower_95=0.4,
    )
    amendment, freeze = _frozen_amendment(tmp_path)
    paths = {
        "packet_manifest_path": manifest,
        "reviewer_registry_path": registry,
        "capability_freeze_path": capability,
        "combined_observations_path": observations,
        "assignment_root": assignment_root,
        "amendment_path": amendment,
        "amendment_freeze_path": freeze,
        "previous_amendment_path": AMENDMENT,
    }
    return assess_factorial_oversight_readiness_v2(**paths), paths


def test_exact_factorial_panel_passes_strict_readiness(tmp_path: Path) -> None:
    result, _ = _ready_panel(tmp_path)
    assert result["status"] == "ready"
    assert not result["blocking_reasons"]
    assert result["counts"]["cases"] == 96
    assert result["counts"]["double_review_overlap_fraction"] >= 0.5


def test_route_hash_manipulation_fails_closed(tmp_path: Path) -> None:
    result, paths = _ready_panel(tmp_path)
    assert result["status"] == "ready"
    assignment_path = paths["assignment_root"] / "assignment_manifest.json"
    payload = read_json(assignment_path)
    payload["case_routes"][0]["packet_arm"] = "adversarial_two_sided"
    write_json(assignment_path, payload)
    result = assess_factorial_oversight_readiness_v2(**paths)
    assert result["status"] == "blocked"
    assert "route_predecision_hashes_valid" in result["blocking_reasons"]
