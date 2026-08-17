from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


def _module():
    path = Path(__file__).parents[1] / "scripts" / "generate_final_english_report.py"
    spec = importlib.util.spec_from_file_location("final_report_generator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _metric(value: float, low: float, high: float) -> dict[str, object]:
    return {
        "estimate": value,
        "cluster_bootstrap_95_ci": [low, high],
    }


def _panel(
    comparison: str,
    condition_a: str,
    condition_b: str,
    *,
    a_ucr: float,
    b_ucr: float,
) -> dict[str, object]:
    return {
        "family": "graph" if comparison.startswith("graph") else "report",
        "comparison": comparison,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "conditions": {
            condition_a: {
                "ucr": _metric(a_ucr, max(0, a_ucr - 0.02), a_ucr + 0.02),
                "completeness": _metric(4.5, 4.3, 4.7),
            },
            condition_b: {
                "ucr": _metric(b_ucr, max(0, b_ucr - 0.02), b_ucr + 0.02),
                "completeness": _metric(4.0, 3.8, 4.2),
            },
        },
        "pairwise_for_condition_a": {"wins": 5, "ties": 1, "losses": 0},
        "effects": {
            "ucr_reduction": _metric(b_ucr - a_ucr, b_ucr - a_ucr - 0.02, b_ucr - a_ucr + 0.02)
        },
    }


def _fixture(root: Path) -> tuple[Path, list[str], list[str], list[dict[str, object]]]:
    datasets = []
    locked = []
    development = []
    evidence: list[dict[str, object]] = []
    for index in range(8):
        dataset_id = f"topic_{index}"
        role = "development" if index < 2 else "locked"
        (development if role == "development" else locked).append(dataset_id)
        datasets.append(
            {
                "id": dataset_id,
                "role": role,
                "topic": f"Topic {index}",
                "year_from": 2000 + index,
                "year_to": 2005 + index,
            }
        )
        audit_root = root / "experiments" / "formal_workspaces" / dataset_id / "audit"
        harvest = audit_root / "harvest_manifest.json"
        processing = audit_root / "processing_manifest.json"
        evidence_manifest = audit_root / "evidence_preparation_manifest.json"
        _write_json(
            harvest,
            {
                "status": "complete",
                "received_records": 100 + index,
                "unique_records": 99 + index,
                "duplicate_records": 1,
            },
        )
        _write_json(
            processing,
            {"status": "complete", "records_processed": 99 + index},
        )
        _write_json(
            evidence_manifest,
            {
                "passed": True,
                "evidence_items": 72,
                "graph": {"nodes": 40 + index, "edges": 80 + index},
            },
        )
        evidence.extend(
            {
                "item": f"dataset:{dataset_id}:{name}",
                "status": "complete",
                "evidence": [path.relative_to(root).as_posix()],
            }
            for name, path in (
                ("harvest", harvest),
                ("processing", processing),
                ("graph", evidence_manifest),
            )
        )
    registry = root / "experiments" / "registry.yml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump({"status": "frozen", "datasets": datasets}, sort_keys=False),
        encoding="utf-8",
    )

    panels = [
        _panel("full_vs_oneshot", "citeweave_full", "structured_one_shot", a_ucr=0.1, b_ucr=0.2),
        _panel(
            "full_vs_human",
            "citeweave_full",
            "published_human_reference",
            a_ucr=0.1,
            b_ucr=0.12,
        ),
        _panel(
            "oneshot_vs_human",
            "structured_one_shot",
            "published_human_reference",
            a_ucr=0.2,
            b_ucr=0.12,
        ),
        _panel("graph_vs_no", "graph_rag", "no_rag", a_ucr=0.05, b_ucr=0.4),
        _panel("graph_vs_flat", "graph_rag", "flat_structured", a_ucr=0.05, b_ucr=0.06),
        _panel("graph_vs_figure", "graph_rag", "figure_vlm", a_ucr=0.05, b_ucr=0.2),
    ]
    conditions = {}
    for name, reviews, passes, auto, unsafe in (
        ("baseline_original", 0, 24, 30, 6),
        ("always_review", 30, 30, 0, 0),
        ("static_review", 20, 29, 10, 1),
        ("adaptive_review", 15, 29, 15, 1),
    ):
        conditions[name] = {
            "counts": {
                "items": 30,
                "review_requests": reviews,
                "final_quality_passed": passes,
                "auto_accepts": auto,
                "unsafe_auto_accepts": unsafe,
            },
            "metrics": {
                "rrr": _metric(reviews / 30, reviews / 30, reviews / 30),
                "fqpr": _metric(passes / 30, passes / 30, passes / 30),
                "unsafe_auto_accept_rate": {
                    "estimate": None if not auto else unsafe / auto,
                    "cluster_bootstrap_95_ci": None
                    if not auto
                    else [unsafe / auto, unsafe / auto],
                },
            },
        }
    original_to_post = {}
    for name, passes in (
        ("always_review", 30),
        ("static_review", 29),
        ("adaptive_review", 29),
    ):
        post_error = 1 - passes / 30
        original_to_post[name] = {
            "baseline_original_quality_error_rate": 0.2,
            "post_review_quality_error_rate": post_error,
            "absolute_quality_error_rate_reduction": 0.2 - post_error,
            "relative_quality_error_rate_reduction": (0.2 - post_error) / 0.2,
        }
    statistics = {
        "version": 1,
        "topics": locked,
        "topic_clusters": 6,
        "bootstrap": {
            "method": "topic-cluster bootstrap",
            "samples": 10_000,
            "confidence_level": 0.95,
            "seed": 20260806,
        },
        "panels": panels,
        "graph_primary_holm": [
            {
                "comparison": name,
                "raw_p_value": 0.03125,
                "holm_adjusted_p_value": 0.09375,
                "reject_at_0_05": False,
            }
            for name in ("graph_vs_no", "graph_vs_flat", "graph_vs_figure")
        ],
        "adaptive": {
            "conditions": conditions,
            "original_to_post_review": original_to_post,
        },
    }
    statistics_root = root / "experiments" / "formal_statistics"
    _write_json(statistics_root / "formal_statistics.json", statistics)
    _write_json(root / "experiments" / "formal_statistics_manifest.json", {"topics": locked})
    (statistics_root / "formal_metrics.csv").write_text("metric,value\nx,1\n", encoding="utf-8")
    (statistics_root / "graph_holm.csv").write_text("comparison,p\nx,1\n", encoding="utf-8")
    (statistics_root / "formal_results.md").write_text("# Results\n", encoding="utf-8")
    return registry, locked, development, evidence


def _audit(evidence: list[dict[str, object]], final_status: str) -> dict[str, object]:
    return {
        "checks": [
            *evidence,
            {
                "item": "formal_statistics",
                "status": "complete",
                "evidence": [],
            },
            {
                "item": "final_english_end_to_end_report",
                "status": final_status,
                "evidence": [],
            },
        ]
    }


def test_refuses_when_any_nonfinal_requirement_is_not_complete(tmp_path: Path) -> None:
    module = _module()
    registry, _, _, evidence = _fixture(tmp_path)
    evidence[0]["status"] = "incomplete"
    with pytest.raises(module.FinalReportError, match="non-final requirements"):
        module.generate(
            root=tmp_path,
            registry_path=registry,
            graph_run_id="formal_v2_nonthinking_20260806",
            report_path=tmp_path / "experiments/final_report/end_to_end_report.md",
            manifest_path=tmp_path / "experiments/final_report/manifest.json",
            audit=_audit(evidence, "incomplete"),
        )
    assert not (tmp_path / "experiments/final_report").exists()


def test_generates_substantive_report_and_is_byte_idempotent(tmp_path: Path) -> None:
    module = _module()
    registry, locked, _, evidence = _fixture(tmp_path)
    report = tmp_path / "experiments/final_report/end_to_end_report.md"
    manifest = tmp_path / "experiments/final_report/manifest.json"
    kwargs = {
        "root": tmp_path,
        "registry_path": registry,
        "graph_run_id": "formal_v2_nonthinking_20260806",
        "report_path": report,
        "manifest_path": manifest,
    }

    assert module.generate(**kwargs, audit=_audit(evidence, "incomplete")) == "created"
    original_report = report.read_bytes()
    original_manifest = manifest.read_bytes()
    value = json.loads(original_manifest)
    assert value["status"] == "complete"
    assert value["language"] == "English"
    assert value["locked_topic_ids"] == locked
    assert len(value["dataset_ids"]) == 8
    assert value["report_sha256"] == module._sha256(report)
    assert any(path.endswith("harvest_manifest.json") for path in value["source_hashes"])
    text = original_report.decode()
    assert len(text.split()) >= 500
    for heading in (
        "Structured-one-shot benchmark",
        "published human bibliometric studies",
        "Graph grounding",
        "Adaptive review",
        "Limitations and negative results",
        "reproducibility",
    ):
        assert heading.lower() in text.lower()
    assert "only the exact system-flagged excerpt" in text
    assert "within the flagged span" in text

    assert module.generate(**kwargs, audit=_audit(evidence, "complete")) == "unchanged"
    assert report.read_bytes() == original_report
    assert manifest.read_bytes() == original_manifest


def test_refuses_to_overwrite_different_existing_output(tmp_path: Path) -> None:
    module = _module()
    registry, _, _, evidence = _fixture(tmp_path)
    report = tmp_path / "experiments/final_report/end_to_end_report.md"
    manifest = tmp_path / "experiments/final_report/manifest.json"
    kwargs = {
        "root": tmp_path,
        "registry_path": registry,
        "graph_run_id": "formal_v2_nonthinking_20260806",
        "report_path": report,
        "manifest_path": manifest,
    }
    module.generate(**kwargs, audit=_audit(evidence, "incomplete"))
    original_manifest = manifest.read_bytes()
    report.write_text("different\n", encoding="utf-8")

    with pytest.raises(module.FinalReportError, match="differs"):
        module.generate(**kwargs, audit=_audit(evidence, "complete"))
    assert report.read_text(encoding="utf-8") == "different\n"
    assert manifest.read_bytes() == original_manifest


def test_rejects_partial_prior_output_before_rendering(tmp_path: Path) -> None:
    module = _module()
    registry, _, _, evidence = _fixture(tmp_path)
    report = tmp_path / "experiments/final_report/end_to_end_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("partial", encoding="utf-8")
    with pytest.raises(module.FinalReportError, match="partial or invalid"):
        module.generate(
            root=tmp_path,
            registry_path=registry,
            graph_run_id="formal_v2_nonthinking_20260806",
            report_path=report,
            manifest_path=tmp_path / "experiments/final_report/manifest.json",
            audit=_audit(evidence, "incomplete"),
        )
