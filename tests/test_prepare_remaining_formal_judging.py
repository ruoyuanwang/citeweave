from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_remaining_formal_judging.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_remaining_formal_judging", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_human_evidence_chunks_remove_controller_condition_identity() -> None:
    items = MODULE._human_evidence_items(
        "# Evidence\n\n"
        "This packet is paired only with the corresponding published human reference.\n\n"
        "The article reports 100 included records."
    )

    serialized = json.dumps(items).casefold()
    assert "published human reference" not in serialized
    assert "published_human_reference" not in serialized
    assert "paired source document" in serialized
    assert "100 included records" in serialized


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "project"
    experiments = root / "experiments"
    references_path = experiments / "human_references.yml"
    development = ["development_a", "development_b"]
    locked = [f"locked_{index}" for index in range(1, 7)]
    references = [
        {"id": dataset_id, "role": "development"} for dataset_id in development
    ] + [{"id": dataset_id, "role": "locked"} for dataset_id in locked]
    references_path.parent.mkdir(parents=True)
    references_path.write_text(
        yaml.safe_dump({"references": references}, sort_keys=False),
        encoding="utf-8",
    )
    freeze_path = experiments / "judge_calibration_freeze.json"
    _json(
        freeze_path,
        {
            "status": "frozen_after_development_calibration",
            "formal_results_used": False,
            "development_topics": development,
            "locked_topic_count": 6,
            "report_rubric_version": "report-frozen-v1",
            "graph_rubric_version": "graph-frozen-v1",
            "rules": {
                "supported_requires_evidence_id": True,
                "paired_corpora_are_scored_against_their_own_evidence": True,
                "human_reference_counts_are_not_gold_for_system_corpora": True,
                "independent_judges": 2,
                "blind_adjudication_on_conflict": True,
                "judges_may_modify_pipeline_or_artifacts": False,
            },
        },
    )
    reports_root = experiments / "formal_reports"
    workspaces_root = experiments / "formal_workspaces"
    human_root = experiments / "human_outputs"
    runs_root = experiments / "formal_runs"
    vision_root = experiments / "vision_outputs"
    run_id = "test-run"
    report_words = " ".join(f"word{index}" for index in range(140))

    for dataset_id in development + locked:
        evidence_path = (
            workspaces_root / dataset_id / "evidence" / "evidence_items.json"
        )
        _json(
            evidence_path,
            [{"evidence_id": "E001", "statement": f"Evidence for {dataset_id}."}],
        )
        evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        for condition in ("structured_one_shot", "citeweave_full"):
            directory = reports_root / dataset_id / condition
            directory.mkdir(parents=True)
            report = f"# System report\n{report_words}"
            report_path = directory / "report.md"
            report_path.write_text(report, encoding="utf-8")
            _json(
                directory / "completion.json",
                {
                    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    "evidence_sha256": evidence_hash,
                },
            )
        human_directory = human_root / dataset_id
        human_directory.mkdir(parents=True)
        (human_directory / "reference_report.md").write_text(
            f"# Reference report\n{report_words}", encoding="utf-8"
        )
        (human_directory / "reference_evidence.md").write_text(
            "# Methods\nSearch strategy and included records.\n\n"
            "# Results\nReported bibliometric patterns.",
            encoding="utf-8",
        )

        item_id = f"{dataset_id}:network:network_size"
        experiment_root = (
            workspaces_root / dataset_id / "evidence" / "formal_graph_experiment"
        )
        context_path = experiment_root / "contexts" / "network_size.json"
        _json(context_path, {"nodes": 3, "links": 2})
        _json(
            experiment_root / "benchmark.json",
            [
                {
                    "item_id": item_id,
                    "question": "How many nodes and links are displayed?",
                    "answerable": True,
                    "answer_contract": {"nodes": "integer", "links": "integer"},
                    "gold_answer": {"nodes": 3, "links": 2},
                    "figure_eligible": True,
                }
            ],
        )
        _json(
            experiment_root / "manifest.json",
            {
                "contexts": [
                    {
                        "item_id": item_id,
                        "graph_path": str(
                            context_path.relative_to(workspaces_root / dataset_id)
                        ).replace("\\", "/"),
                    }
                ]
            },
        )
        for condition in ("no_rag", "flat_structured", "graph_rag"):
            items_path = runs_root / dataset_id / run_id / condition / "items.jsonl"
            items_path.parent.mkdir(parents=True)
            response = {
                "item_id": item_id,
                "abstain": False,
                "answer": {"nodes": 3, "links": 2},
                "explanation": "The visible evidence states both values.",
            }
            row = {
                "item_id": item_id,
                "status": "complete",
                "response": {
                    "choices": [{"message": {"content": json.dumps(response)}}]
                },
            }
            items_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        _json(
            vision_root / f"{dataset_id}.json",
            {
                "visible_only": True,
                "results": [
                    {
                        "item_id": item_id,
                        "abstain": False,
                        "answer": {"nodes": 3, "links": 2},
                        "explanation": "The image visibly prints both values.",
                    }
                ],
            },
        )
    return {
        "references_path": references_path,
        "freeze_path": freeze_path,
        "reports_root": reports_root,
        "workspaces_root": workspaces_root,
        "human_root": human_root,
        "runs_root": runs_root,
        "vision_root": vision_root,
        "output_root": experiments / "formal_judging_ready",
        "run_id": run_id,
    }


def _prepare(paths: dict[str, Path]):
    return MODULE.prepare(**paths, maximum_words=120, seed=42)


def test_prepares_only_six_locked_topics_and_neutral_worklists(tmp_path: Path):
    paths = _fixture(tmp_path)
    result = _prepare(paths)
    output = paths["output_root"]

    assert result["publish_status"] == "created"
    assert result["formal_topic_count"] == 6
    assert result["development_topics_excluded"] == ["development_a", "development_b"]
    report_packets = (output / "packets/eval_a/report/RT01/packets.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(report_packets.splitlines()) == 6
    assert "development_a" not in report_packets
    assert "development_b" not in report_packets
    human_input = (output / "controller/inputs/report/full_vs_human.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"evidence_id":"H001"' in human_input
    assert '"evidence_id":"H002"' in human_input

    for judge_id in ("eval_a", "eval_b"):
        worklist = json.loads(
            (output / f"neutral_worklists/{judge_id}.json").read_text(encoding="utf-8")
        )
        serialized = json.dumps(worklist)
        assert worklist["contains_condition_identities"] is False
        assert worklist["controller_metadata_in_scope"] is False
        assert "secret" not in serialized
        for condition in (
            "structured_one_shot",
            "citeweave_full",
            "published_human_reference",
            "no_rag",
            "flat_structured",
            "graph_rag",
            "figure_vlm",
        ):
            assert condition not in serialized
        judge_root = output / "packets" / judge_id
        assert not list(judge_root.rglob("*secret*"))
        assert not list(judge_root.rglob("*map*"))

    manifest = json.loads(
        (output / "controller/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["report_main_comparisons"] == [
        "full_vs_oneshot",
        "full_vs_human",
    ]
    assert manifest["report_supplementary_comparisons"] == ["oneshot_vs_human"]
    assert {task["packet_count"] for task in manifest["task_map"][:3]} == {6}
    assert all(task["input_sha256"] for task in manifest["task_map"])


def test_identical_rerun_is_idempotent_but_changed_output_fails_closed(tmp_path: Path):
    paths = _fixture(tmp_path)
    _prepare(paths)
    assert _prepare(paths)["publish_status"] == "already_identical"

    packet_path = (
        paths["output_root"] / "packets/eval_a/report/RT01/packets.jsonl"
    )
    packet_path.write_text(packet_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite different"):
        _prepare(paths)


def test_requires_all_eight_topics_before_publishing(tmp_path: Path):
    paths = _fixture(tmp_path)
    missing = paths["vision_root"] / "development_a.json"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="Required formal artifact is missing"):
        _prepare(paths)
    assert not paths["output_root"].exists()


def test_rejects_condition_leak_before_publishing(tmp_path: Path):
    paths = _fixture(tmp_path)
    item_path = (
        paths["runs_root"]
        / "locked_1"
        / paths["run_id"]
        / "graph_rag"
        / "items.jsonl"
    )
    row = json.loads(item_path.read_text(encoding="utf-8"))
    content = json.loads(row["response"]["choices"][0]["message"]["content"])
    content["explanation"] = "This reveals graph_rag."
    row["response"]["choices"][0]["message"]["content"] = json.dumps(content)
    item_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Condition-name leakage"):
        _prepare(paths)
    assert not paths["output_root"].exists()
