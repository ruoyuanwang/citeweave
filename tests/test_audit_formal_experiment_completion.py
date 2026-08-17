from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _module():
    path = Path(__file__).parents[1] / "scripts" / "audit_formal_experiment_completion.py"
    spec = importlib.util.spec_from_file_location("completion_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registry(path: Path) -> list[dict[str, object]]:
    datasets = []
    for index in range(8):
        role = "development" if index < 2 else "locked"
        datasets.append(
            {
                "id": f"topic_{index}",
                "role": role,
                "source": "openalex",
                "query_status": "frozen",
                "search_scope": "title_abstract",
                "max_records": None,
            }
        )
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({"status": "frozen", "datasets": datasets}),
        encoding="utf-8",
    )
    return datasets


def test_partial_pass_markers_cannot_claim_completion(tmp_path: Path) -> None:
    module = _module()
    registry = tmp_path / "experiments" / "registry.yml"
    datasets = _registry(registry)
    first = datasets[0]["id"]
    audit = tmp_path / "experiments" / "formal_workspaces" / str(first) / "audit"
    audit.mkdir(parents=True)
    (audit / "harvest_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source": "openalex",
                "received_records": 1,
                "unique_records": 1,
                "duplicate_records": 0,
                "staged_path": "staged/source_records.jsonl.gz",
                "staged_sha256": "fake",
                "slices": [],
            }
        ),
        encoding="utf-8",
    )
    (audit / "acquisition_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "source": "openalex",
                "truncated": False,
                "failed_pages": 0,
                "received_records": 1,
                "unique_records": 1,
                "raw_sha256": [],
            }
        ),
        encoding="utf-8",
    )

    report = module.Inspector(tmp_path, registry).run("formal_v2_nonthinking_20260806")

    assert report["all_complete"] is False
    harvest = next(item for item in report["checks"] if item["item"] == f"dataset:{first}:harvest")
    assert harvest["status"] == "invalid"
    assert any("cursor" in reason for reason in harvest["reasons"])


def test_development_and_rejected_judging_do_not_count_as_formal(
    tmp_path: Path,
) -> None:
    module = _module()
    registry = tmp_path / "experiments" / "registry.yml"
    _registry(registry)
    for root_name in ("development_judging", "rejected_judge_calibration"):
        directory = tmp_path / "experiments" / root_name / "full_vs_oneshot" / "resolved_v9"
        directory.mkdir(parents=True)
        (directory / "resolved_judgments.jsonl").write_text(
            '{"packet_id":"fake","sample_id":"topic_2"}\n', encoding="utf-8"
        )
        (directory / "judge_metrics.json").write_text("{}\n", encoding="utf-8")

    inspector = module.Inspector(tmp_path, registry)
    inspector.registry()
    inspector.judging_family("report", module.REPORT_COMPARISONS)

    assert all(check.status == "incomplete" for check in inspector.checks[1:])
    assert all(
        "development" not in evidence and "rejected" not in evidence
        for check in inspector.checks[1:]
        for evidence in check.evidence
    )


def test_old_graph_run_cannot_substitute_for_frozen_v2(tmp_path: Path) -> None:
    module = _module()
    registry = tmp_path / "experiments" / "registry.yml"
    datasets = _registry(registry)
    topic = str(datasets[0]["id"])
    old = tmp_path / "experiments" / "formal_runs" / topic / "formal_v1_20260806" / "no_rag"
    old.mkdir(parents=True)
    (old / "run_manifest.json").write_text('{"run_id":"formal_v1_20260806"}', encoding="utf-8")
    (old / "items.jsonl").write_text('{"status":"complete"}\n', encoding="utf-8")

    inspector = module.Inspector(tmp_path, registry)
    inspector.registry()
    inspector.graph_runs(datasets[0], "formal_v2_nonthinking_20260806")

    graph_checks = inspector.checks[1:]
    assert len(graph_checks) == 3
    assert all(check.status == "incomplete" for check in graph_checks)
    assert all("formal_v1" not in evidence for check in graph_checks for evidence in check.evidence)


def test_completed_graph_answers_without_mechanical_score_are_incomplete(
    tmp_path: Path,
) -> None:
    module = _module()
    registry = tmp_path / "experiments" / "registry.yml"
    datasets = _registry(registry)
    dataset = str(datasets[0]["id"])
    run_id = "formal_v2_nonthinking_20260806"
    benchmark = (
        tmp_path
        / "experiments"
        / "formal_workspaces"
        / dataset
        / "evidence"
        / "formal_graph_experiment"
        / "benchmark.json"
    )
    benchmark.parent.mkdir(parents=True)
    item_id = f"{dataset}:citation:network_size"
    benchmark.write_text(
        json.dumps([{"item_id": item_id}], ensure_ascii=False),
        encoding="utf-8",
    )
    run = (
        tmp_path
        / "experiments"
        / "formal_runs"
        / dataset
        / run_id
        / "no_rag"
    )
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_id": dataset,
                "condition": "no_rag",
                "profile": {"model": "deepseek-v4-pro"},
                "benchmark_sha256": module.Inspector.sha(benchmark),
            }
        ),
        encoding="utf-8",
    )
    (run / "items.jsonl").write_text(
        json.dumps({"item_id": item_id, "status": "complete"}) + "\n",
        encoding="utf-8",
    )

    inspector = module.Inspector(tmp_path, registry)
    inspector.registry()
    inspector.graph_runs(datasets[0], run_id)

    check = next(
        item
        for item in inspector.checks
        if item.item == f"dataset:{dataset}:graph_text:no_rag"
    )
    assert check.status == "incomplete"
    assert "mechanical score is missing" in " ".join(check.reasons)


def test_statistics_paths_are_resolved_from_manifest_directory(tmp_path: Path) -> None:
    module = _module()
    registry = tmp_path / "experiments" / "registry.yml"
    datasets = _registry(registry)
    locked = [str(item["id"]) for item in datasets if item["role"] == "locked"]
    experiments = tmp_path / "experiments"
    input_path = experiments / "inputs" / "resolved_judgments.jsonl"
    adaptive_path = experiments / "inputs" / "adaptive.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}\n", encoding="utf-8")
    adaptive_path.write_text("{}\n", encoding="utf-8")

    files = {topic: "inputs/resolved_judgments.jsonl" for topic in locked}
    manifest = {
        "topics": locked,
        "report_comparisons": [{"name": "report", "files": files}],
        "graph_comparisons": [{"name": "graph", "files": files}],
        "adaptive_results": {topic: "inputs/adaptive.json" for topic in locked},
    }
    (experiments / "formal_statistics_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = experiments / "formal_statistics"
    output.mkdir()
    (output / "formal_statistics.json").write_text(
        json.dumps(
            {
                "topics": locked,
                "topic_clusters": 6,
                "bootstrap": {
                    "samples": 10_000,
                    "method": "topic-cluster bootstrap",
                    "confidence_level": 0.95,
                },
                "graph_primary_holm": [
                    {"holm_adjusted_p_value": 0.1},
                    {"holm_adjusted_p_value": 0.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("formal_metrics.csv", "graph_holm.csv", "formal_results.md"):
        (output / name).write_text("ok\n", encoding="utf-8")

    inspector = module.Inspector(tmp_path, registry)
    inspector.registry()
    inspector.statistics()

    assert inspector.checks[-1].status == "complete"
