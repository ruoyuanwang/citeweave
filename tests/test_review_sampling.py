from __future__ import annotations

from pathlib import Path

from src.citeweave.io import write_json
from src.citeweave.review_sampling import build_sampled_review_manifest


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "packets"
    records = []
    index = 0
    for condition in ("flat", "graph"):
        for task_type, complexity in (("path", 2), ("counterfactual", 4)):
            for exact in (False, True):
                for repeat in range(3):
                    index += 1
                    factual_id = f"F{index:03d}"
                    semantic_id = f"S{index:03d}" if exact else None
                    write_json(
                        root / "packets" / "factual" / f"{factual_id}.json",
                        {
                            "packet_id": factual_id,
                            "task_type": task_type,
                            "complexity": complexity,
                        },
                    )
                    if semantic_id:
                        write_json(
                            root / "packets" / "semantic" / f"{semantic_id}.json",
                            {
                                "packet_id": semantic_id,
                                "task_type": task_type,
                                "complexity": complexity,
                            },
                        )
                    records.append(
                        {
                            "dataset_id": f"D{repeat}",
                            "item_id": f"I{index}",
                            "condition": condition,
                            "factual_packet_id": factual_id,
                            "semantic_packet_id": semantic_id,
                            "deterministic_answer_exact": exact,
                            "deterministic_evidence_f1": 1.0 if exact else 0.0,
                        }
                    )
    manifest = {
        "schema_version": 1,
        "reviewers": ["A", "B"],
        "records": records,
        "assignments": {},
        "factual_packets": 24,
        "semantic_packets": 12,
    }
    return manifest, root


def test_stratified_partial_overlap_is_deterministic_and_disjoint(
    tmp_path: Path,
) -> None:
    manifest, root = _fixture(tmp_path)
    kwargs = {
        "packet_root": root,
        "factual_per_reviewer": 8,
        "semantic_per_reviewer": 6,
        "overlap_fraction": 0.5,
        "seed": 17,
    }
    first = build_sampled_review_manifest(manifest, **kwargs)
    second = build_sampled_review_manifest(manifest, **kwargs)
    assert first == second
    for layer, expected_common in (("factual", 4), ("semantic", 3)):
        left = set(first["assignments"]["A"][layer])
        right = set(first["assignments"]["B"][layer])
        assert len(left) == len(right) == kwargs[f"{layer}_per_reviewer"]
        assert len(left & right) == expected_common
        assert first["sampling"]["layers"][layer]["common_double_review"] == expected_common
    assert first["sampling"]["online_policy_leakage_prohibited"] is True
    factual_strata = first["sampling"]["layers"]["factual"]["strata"]
    assert all(row["sampled_unique"] > 0 for row in factual_strata)


def test_overlap_increases_when_population_cannot_support_disjoint_assignments(
    tmp_path: Path,
) -> None:
    manifest, root = _fixture(tmp_path)
    sampled = build_sampled_review_manifest(
        manifest,
        packet_root=root,
        factual_per_reviewer=20,
        semantic_per_reviewer=10,
        overlap_fraction=0.0,
        seed=3,
    )
    assert sampled["sampling"]["layers"]["factual"]["common_double_review"] == 16
    assert sampled["sampling"]["layers"]["semantic"]["common_double_review"] == 8
