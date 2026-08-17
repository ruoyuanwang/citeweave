from __future__ import annotations

import json
from pathlib import Path

import pytest

from citeweave.formal_statistics import (
    GRAPH_COMPARISONS,
    POST_REVIEW_CONDITIONS,
    REPORT_COMPARISONS,
    FormalStatisticsError,
    analyze_formal_experiment,
    holm_adjust,
    write_formal_statistics,
)


def _resolved(
    sample: str,
    condition_a: str,
    condition_b: str,
    *,
    preference: str | None = None,
) -> dict:
    return {
        "packet_id": f"JP{sample:0>20}"[-22:],
        "sample_id": sample,
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
        "preference": preference or condition_a,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_manifest(tmp_path: Path) -> Path:
    topics = ["topic_a", "topic_b"]
    report_contract = REPORT_COMPARISONS
    graph_contract = GRAPH_COMPARISONS

    def panel_specs(contracts: dict[str, tuple[str, str]]) -> list[dict]:
        specs = []
        for name, (condition_a, condition_b) in contracts.items():
            files = {}
            for topic_index, topic in enumerate(topics):
                path = tmp_path / name / f"{topic}.jsonl"
                rows = [
                    _resolved(
                        f"{topic_index}{sample_index}",
                        condition_a,
                        condition_b,
                    )
                    for sample_index in range(2)
                ]
                _write_jsonl(path, rows)
                files[topic] = str(path.relative_to(tmp_path))
            specs.append(
                {
                    "name": name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "files": files,
                }
            )
        return specs

    adaptive_files = {}
    for topic in topics:
        path = tmp_path / "adaptive" / f"{topic}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "topic_id": topic,
                    "comparison_contract": {
                        "baseline_original": (
                            "independent quality evaluation of the untouched candidate"
                        ),
                        "post_review_conditions": list(POST_REVIEW_CONDITIONS),
                        "topic_role": "locked",
                        "formal_results_used": True,
                    },
                    "conditions": {
                        "baseline_original": {
                            "items": 10,
                            "review_requests": 0,
                            "final_quality_passed": 7,
                            "auto_accepts": 10,
                            "unsafe_auto_accepts": 3,
                        },
                        "always_review": {
                            "items": 10,
                            "review_requests": 10,
                            "final_quality_passed": 10,
                            "auto_accepts": 0,
                            "unsafe_auto_accepts": 0,
                        },
                        "static_review": {
                            "items": 10,
                            "review_requests": 6,
                            "final_quality_passed": 9,
                            "auto_accepts": 4,
                            "unsafe_auto_accepts": 1,
                        },
                        "adaptive_review": {
                            "items": 10,
                            "review_requests": 3,
                            "final_quality_passed": 10,
                            "auto_accepts": 7,
                            "unsafe_auto_accepts": 0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        adaptive_files[topic] = str(path.relative_to(tmp_path))
    manifest = {
        "version": 1,
        "topics": topics,
        "report_comparisons": panel_specs(report_contract),
        "graph_comparisons": panel_specs(graph_contract),
        "adaptive_results": adaptive_files,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_formal_statistics_computes_registered_metrics_and_outputs(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    summary = analyze_formal_experiment(
        manifest, bootstrap_samples=100, seed=7
    )

    assert summary["topic_clusters"] == 2
    full = next(
        panel for panel in summary["panels"] if panel["comparison"] == "full_vs_oneshot"
    )
    assert full["conditions"]["citeweave_full"]["ucr"]["estimate"] == 0
    assert full["conditions"]["citeweave_full"]["completeness"]["estimate"] == 5
    assert full["effects"]["ucr_reduction"]["estimate"] == 0.5
    assert full["pairwise_for_condition_a"]["wins"] == 4
    assert full["pairwise_for_condition_a"]["win_rate"]["estimate"] == 1
    assert summary["human_quality_gap"]["completeness_difference"]["estimate"] == 2
    assert len(summary["graph_primary_holm"]) == 3
    assert (
        summary["adaptive"]["conditions"]["adaptive_review"]["metrics"]["rrr"][
            "estimate"
        ]
        == 0.3
    )
    assert (
        summary["adaptive"]["conditions"]["adaptive_review"]["metrics"]["fqpr"][
            "estimate"
        ]
        == 1
    )
    assert (
        summary["adaptive"]["conditions"]["always_review"]["metrics"][
            "unsafe_auto_accept_rate"
        ]["estimate"]
        is None
    )
    assert (
        summary["adaptive"]["original_to_post_review"]["adaptive_review"][
            "absolute_quality_error_rate_reduction"
        ]
        == pytest.approx(0.3)
    )

    output = tmp_path / "output"
    write_formal_statistics(summary, output)
    assert (output / "formal_statistics.json").is_file()
    assert "No report-text exact-match proxy" in (
        output / "formal_results.md"
    ).read_text(encoding="utf-8")
    assert "Original-to-post-review quality-error reduction" in (
        output / "formal_results.md"
    ).read_text(encoding="utf-8")
    assert "win_rate" in (output / "formal_metrics.csv").read_text(encoding="utf-8")
    assert (output / "graph_holm.csv").is_file()


def test_refuses_missing_topic_file(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["report_comparisons"][0]["files"]["topic_b"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FormalStatisticsError, match="missing=.*topic_b"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_refuses_missing_condition_in_resolved_judgment(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / manifest["graph_comparisons"][0]["files"]["topic_a"]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    del rows[0]["conditions"]["no_rag"]
    _write_jsonl(path, rows)

    with pytest.raises(FormalStatisticsError, match="conditions keys differ"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_refuses_duplicate_samples(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / manifest["report_comparisons"][0]["files"]["topic_a"]
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    _write_jsonl(path, [row, row])

    with pytest.raises(FormalStatisticsError, match="duplicate sample_id"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_refuses_unpaired_graph_samples(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / manifest["graph_comparisons"][2]["files"]["topic_a"]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["sample_id"] = "different"
    _write_jsonl(path, rows)

    with pytest.raises(FormalStatisticsError, match="not a paired subset"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_refuses_adaptive_missing_condition(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / manifest["adaptive_results"]["topic_a"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["conditions"]["static_review"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalStatisticsError, match="missing=.*static_review"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_refuses_adaptive_development_output_in_formal_statistics(
    tmp_path: Path,
) -> None:
    manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = tmp_path / manifest["adaptive_results"]["topic_a"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comparison_contract"]["formal_results_used"] = False
    payload["comparison_contract"]["topic_role"] = "development"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalStatisticsError, match="not locked formal output"):
        analyze_formal_experiment(manifest_path, bootstrap_samples=100)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
