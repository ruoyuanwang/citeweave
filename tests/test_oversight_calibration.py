from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_calibration import (
    CALIBRATION_ISSUES,
    OversightCalibrationError,
    build_multidimensional_calibration_packets,
)


def _prototype_panel(root: Path) -> Path:
    records = []
    for dataset_index in range(8):
        dataset_id = f"dataset_{dataset_index}"
        for case_index in range(5):
            case_id = f"PH-{dataset_index}-{case_index}"
            public_path = root / "public" / f"{case_id}.json"
            write_json(
                public_path,
                {
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "domain": dataset_id,
                    "claim": (
                        "Registered question: inspect the path.\n"
                        'Candidate structural answer: {"hops": 2, "path": ["a", "b", "c"]}'
                    ),
                    "evidence_set_a": [{"evidence_id": "E1"}],
                    "evidence_set_b": [{"evidence_id": "E2"}],
                    "operator_trace": [
                        {"operator": "path_projection", "hops": 2, "path": ["a", "b", "c"]}
                    ],
                    "alternative_explanations": ["The result depends on graph construction."],
                    "forbidden_inferences": ["causality"],
                },
            )
            records.append(
                {
                    "case_id": case_id,
                    "dataset_id": dataset_id,
                    "public_path": str(public_path.relative_to(root)),
                    "public_sha256": sha256_file(public_path),
                }
            )
    manifest = root / "manifest.json"
    write_json(manifest, {"human_outcomes_inspected": False, "records": records})
    return manifest


def test_builds_balanced_multidimensional_training_panel(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    manifest_path = _prototype_panel(source_root)
    output_root = tmp_path / "calibration"
    manifest = build_multidimensional_calibration_packets(
        prototype_manifest_path=manifest_path,
        output_root=output_root,
    )
    assert manifest["cases"] == 96
    assert manifest["datasets"] == 8
    assert manifest["confirmatory_exclusion"] is True
    assert manifest["internal_validity_counts"] == {"valid": 48, "invalid": 48}
    assert set(manifest["issue_counts"]) == set(CALIBRATION_ISSUES)
    assert set(manifest["issue_counts"].values()) == {24}
    for row in manifest["records"]:
        public = read_json(output_root / row["public_path"])
        internal = read_json(output_root / row["internal_path"])
        assert "expected_verdict" not in public
        assert internal["expected_verdict"] in {"valid", "invalid"}
        assert sha256_file(output_root / row["public_path"]) == row["public_sha256"]


def test_rejects_panel_marked_as_post_outcome(tmp_path: Path) -> None:
    manifest_path = _prototype_panel(tmp_path / "source")
    manifest = read_json(manifest_path)
    manifest["human_outcomes_inspected"] = True
    write_json(manifest_path, manifest)
    with pytest.raises(OversightCalibrationError, match="not pre-outcome"):
        build_multidimensional_calibration_packets(
            prototype_manifest_path=manifest_path,
            output_root=tmp_path / "calibration",
        )
