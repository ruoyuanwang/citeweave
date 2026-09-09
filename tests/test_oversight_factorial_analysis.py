from __future__ import annotations

import pytest

from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_factorial_analysis import (
    ASSIGNMENT_ARMS,
    PACKET_ARMS,
    OversightFactorialOutcome,
    analyze_oversight_factorial,
)
from scripts.analyze_complementary_oversight_factorial import main


def _panel(*, datasets: int = 8) -> list[OversightFactorialOutcome]:
    rows = []
    for dataset in range(datasets):
        for packet in PACKET_ARMS:
            for assignment in ASSIGNMENT_ARMS:
                for replicate in range(3):
                    correct = packet == "adversarial_two_sided" or replicate > 0
                    seconds = 90.0 if assignment == "capability_cost_router" else 180.0
                    rows.append(
                        OversightFactorialOutcome(
                            case_id=(
                                f"D{dataset}-{packet}-{assignment}-{replicate}"
                            ),
                            dataset_id=f"D{dataset}",
                            packet_arm=packet,
                            assignment_arm=assignment,
                            adjudicated_final_correct=correct,
                            review_seconds=seconds,
                        )
                    )
    return rows


def test_factorial_uses_eight_equal_weight_dataset_effects() -> None:
    result = analyze_oversight_factorial(_panel(), bootstrap_samples=100)
    assert result["cases"] == 96
    assert result["datasets"] == 8
    assert result["dataset_weighting"] == "equal_dataset"
    assert result["registered_contrasts"]["O1"]["exact"]["datasets"] == 8
    assert result["registered_contrasts"]["O1"]["exact"][
        "p_value_one_sided"
    ] == pytest.approx(1 / 256)
    assert result["registered_contrasts"]["O2"]["exact"][
        "p_value_one_sided"
    ] == pytest.approx(1 / 256)


def test_factorial_rejects_fewer_than_eight_datasets() -> None:
    with pytest.raises(ValueError, match="at least 8 datasets"):
        analyze_oversight_factorial(_panel(datasets=7), bootstrap_samples=20)


def test_factorial_rejects_missing_arm_within_dataset() -> None:
    rows = [
        row
        for row in _panel()
        if not (
            row.dataset_id == "D0"
            and row.packet_arm == "standard_claim_first"
            and row.assignment_arm == "qualified_random"
        )
    ]
    with pytest.raises(ValueError, match="fewer than 12 cases|all four arms"):
        analyze_oversight_factorial(rows, bootstrap_samples=20)


def test_factorial_rejects_within_dataset_arm_imbalance_below_three() -> None:
    rows = _panel()
    removed = next(
        row
        for row in rows
        if row.dataset_id == "D0"
        and row.packet_arm == "standard_claim_first"
        and row.assignment_arm == "qualified_random"
        and row.case_id.endswith("-2")
    )
    donor = next(
        row
        for row in rows
        if row.dataset_id == "D0"
        and row.assignment_arm == "capability_cost_router"
    )
    replacement = OversightFactorialOutcome(
        **{**donor.__dict__, "case_id": "D0-extra-donor-arm"}
    )
    imbalanced = [row for row in rows if row != removed] + [replacement]
    with pytest.raises(ValueError, match="fewer than 3 cases"):
        analyze_oversight_factorial(imbalanced, bootstrap_samples=20)


def test_factorial_rejects_duplicate_case_or_nonpositive_time() -> None:
    rows = _panel()
    with pytest.raises(ValueError, match="exactly one final outcome"):
        analyze_oversight_factorial(rows + [rows[0]], bootstrap_samples=20)
    bad = [*rows[:-1], OversightFactorialOutcome(**{**rows[-1].__dict__, "review_seconds": 0})]
    with pytest.raises(ValueError, match="positive"):
        analyze_oversight_factorial(bad, bootstrap_samples=20)


def test_analyzer_binds_frozen_packets_and_assignments(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packets.json"
    assignment = tmp_path / "assignments.json"
    input_path = tmp_path / "outcomes.json"
    output = tmp_path / "analysis.json"
    write_json(packet, {"status": "frozen"})
    write_json(assignment, {"status": "frozen"})
    write_json(
        input_path,
        {
            "packet_manifest_sha256": sha256_file(packet),
            "assignment_manifest_sha256": sha256_file(assignment),
            "records": [row.__dict__ for row in _panel()],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_complementary_oversight_factorial.py",
            "--input",
            str(input_path),
            "--packet-manifest",
            str(packet),
            "--assignment-manifest",
            str(assignment),
            "--amendment",
            "experiments/human_review_v2/review_policy_amendment_003_factorial_cluster_inference.yml",
            "--amendment-freeze",
            "experiments/human_review_v2/review_policy_amendment_003_factorial_cluster_inference_freeze.json",
            "--output",
            str(output),
        ],
    )
    main()
    result = read_json(output)
    assert result["cases"] == 96
    assert result["packet_manifest_sha256"] == sha256_file(packet)
