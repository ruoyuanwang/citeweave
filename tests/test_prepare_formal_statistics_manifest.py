from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from citeweave.formal_statistics import (
    ADAPTIVE_CONDITIONS,
    GRAPH_COMPARISONS,
    REPORT_COMPARISONS,
    analyze_formal_experiment,
)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_formal_statistics_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_formal_statistics_manifest",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LOCKED = [f"locked_topic_{index}" for index in range(6)]
DEVELOPMENT = ["development_topic_a", "development_topic_b"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolved(topic: str, condition_a: str, condition_b: str) -> dict:
    return {
        "packet_id": f"packet-{topic}",
        "sample_id": f"{topic}:sample",
        "source": "dual_consensus",
        "conflicts": [],
        "conditions": {
            condition_a: {
                "supported_claims": 4,
                "unsupported_claims": 0,
                "claim_count": 4,
                "mean_completeness": 5.0,
            },
            condition_b: {
                "supported_claims": 2,
                "unsupported_claims": 2,
                "claim_count": 4,
                "mean_completeness": 3.0,
            },
        },
        "preference": condition_a,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    references = tmp_path / "human_references.yml"
    references.write_text(
        yaml.safe_dump(
            {
                "references": [
                    *[
                        {"id": topic, "role": "development"}
                        for topic in DEVELOPMENT
                    ],
                    *[{"id": topic, "role": "locked"} for topic in LOCKED],
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_root = tmp_path / "report_resolved"
    graph_root = tmp_path / "graph_resolved"
    for root, contracts in (
        (report_root, REPORT_COMPARISONS),
        (graph_root, GRAPH_COMPARISONS),
    ):
        for comparison, (condition_a, condition_b) in contracts.items():
            for topic in LOCKED:
                _write_jsonl(
                    root
                    / comparison
                    / topic
                    / MODULE.RESOLVED_FILENAME,
                    [_resolved(topic, condition_a, condition_b)],
                )
    adaptive_root = tmp_path / "adaptive"
    adaptive_root.mkdir()
    for topic in LOCKED:
        (adaptive_root / f"{topic}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "topic_id": topic,
                    "comparison_contract": {
                        "baseline_original": (
                            "independent quality evaluation of the untouched candidate"
                        ),
                        "post_review_conditions": list(
                            MODULE.POST_REVIEW_CONDITIONS
                        ),
                        "topic_role": "locked",
                        "formal_results_used": True,
                    },
                    "conditions": {
                        condition: (
                            {
                                "items": 10,
                                "review_requests": 0,
                                "final_quality_passed": 8,
                                "auto_accepts": 10,
                                "unsafe_auto_accepts": 2,
                            }
                            if condition == "baseline_original"
                            else {
                                "items": 10,
                                "review_requests": 6,
                                "final_quality_passed": 9,
                                "auto_accepts": 4,
                                "unsafe_auto_accepts": 1,
                            }
                        )
                        for condition in ADAPTIVE_CONDITIONS
                    },
                }
            ),
            encoding="utf-8",
        )
    return {
        "references": references,
        "report_root": report_root,
        "graph_root": graph_root,
        "adaptive_root": adaptive_root,
        "output": tmp_path / "results" / "formal_statistics_manifest.json",
    }


def test_refuses_nonformal_adaptive_contract(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    target = paths["adaptive_root"] / f"{LOCKED[0]}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["comparison_contract"]["formal_results_used"] = False
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        MODULE.ManifestBuildError,
        match="not locked formal output",
    ):
        MODULE.build_manifest(
            references_path=paths["references"],
            report_root=paths["report_root"],
            graph_root=paths["graph_root"],
            adaptive_root=paths["adaptive_root"],
            output_path=paths["output"],
        )


def _build(paths: dict[str, Path]) -> dict:
    return MODULE.build_manifest(
        references_path=paths["references"],
        report_root=paths["report_root"],
        graph_root=paths["graph_root"],
        adaptive_root=paths["adaptive_root"],
        output_path=paths["output"],
    )


def test_builds_manifest_compatible_with_formal_analyzer(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = _build(paths)

    assert manifest["topics"] == LOCKED
    assert not any(topic in manifest["topics"] for topic in DEVELOPMENT)
    assert MODULE.write_idempotent(paths["output"], manifest)
    assert not MODULE.write_idempotent(paths["output"], manifest)
    assert not Path(
        manifest["report_comparisons"][0]["files"][LOCKED[0]]
    ).is_absolute()

    summary = analyze_formal_experiment(
        paths["output"],
        bootstrap_samples=100,
        seed=7,
    )
    assert summary["topic_clusters"] == 6


def test_refuses_development_topic_in_resolved_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    comparison = next(iter(REPORT_COMPARISONS))
    extra = paths["report_root"] / comparison / DEVELOPMENT[0]
    extra.mkdir()

    with pytest.raises(MODULE.ManifestBuildError, match="extra=.*development"):
        _build(paths)


def test_refuses_missing_locked_topic_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    comparison = next(iter(GRAPH_COMPARISONS))
    (
        paths["graph_root"]
        / comparison
        / LOCKED[0]
        / MODULE.RESOLVED_FILENAME
    ).unlink()

    with pytest.raises(MODULE.ManifestBuildError, match="Missing resolved"):
        _build(paths)


def test_refuses_wrong_resolved_conditions(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    comparison, (condition_a, condition_b) = next(iter(REPORT_COMPARISONS.items()))
    path = (
        paths["report_root"]
        / comparison
        / LOCKED[0]
        / MODULE.RESOLVED_FILENAME
    )
    row = _resolved(LOCKED[0], condition_a, condition_b)
    row["conditions"]["wrong_condition"] = row["conditions"].pop(condition_b)
    _write_jsonl(path, [row])

    with pytest.raises(MODULE.ManifestBuildError, match="conditions keys differ"):
        _build(paths)


def test_refuses_extra_adaptive_topic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["adaptive_root"] / f"{DEVELOPMENT[0]}.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.ManifestBuildError, match="extra=.*development"):
        _build(paths)


def test_refuses_baseline_original_that_contains_review_interventions(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    path = paths["adaptive_root"] / f"{LOCKED[0]}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conditions"]["baseline_original"]["review_requests"] = 1
    payload["conditions"]["baseline_original"]["auto_accepts"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        MODULE.ManifestBuildError,
        match="untouched baseline counts do not reconcile",
    ):
        _build(paths)


def test_refuses_overwriting_different_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = _build(paths)
    assert MODULE.write_idempotent(paths["output"], manifest)
    changed = {**manifest, "version": 2}

    with pytest.raises(MODULE.ManifestBuildError, match="different existing"):
        MODULE.write_idempotent(paths["output"], changed)
