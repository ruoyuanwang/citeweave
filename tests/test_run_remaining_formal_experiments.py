from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_remaining_formal_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("run_remaining_formal_experiments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_command_plan_covers_all_remaining_conditions_without_caps(tmp_path: Path):
    dataset_id = "topic-a"
    steps, skipped = MODULE.build_plan(
        registry=tmp_path / "frozen.yml",
        dataset_ids=[dataset_id],
        workspaces_root=tmp_path / "workspaces",
        reports_root=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        packets_root=tmp_path / "packets",
        text_profile=tmp_path / "deepseek_v4_pro.json",
        skip_vlm_packets=False,
    )

    assert skipped == []
    assert [step.stage for step in steps] == [
        "process",
        "visualize",
        "evidence_and_graph_grounding",
        "report:structured_one_shot",
        "report:citeweave_full",
        "graph:no_rag",
        "graph:flat_structured",
        "graph:graph_rag",
        "vlm_packet",
    ]
    graph_steps = [step for step in steps if step.kind == "graph_text"]
    assert len(graph_steps) == 3
    for step in graph_steps:
        assert "--execute" in step.command
        assert step.command[step.command.index("--run-id") + 1] == (
            "formal_v2_nonthinking_20260806"
        )
        assert step.command[step.command.index("--text-profile") + 1] == str(
            tmp_path / "deepseek_v4_pro.json"
        )
        assert not any("max_records" in argument or "--limit" == argument for argument in step.command)


def test_plan_skips_only_verified_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(MODULE, "_completed_or_pending", lambda **_kwargs: True)
    monkeypatch.setattr(MODULE, "verify_report_completion", lambda *_args: True)
    monkeypatch.setattr(MODULE, "verify_graph_run_completion", lambda *_args: True)
    monkeypatch.setattr(MODULE, "verify_vlm_packet", lambda *_args: True)

    steps, skipped = MODULE.build_plan(
        registry=tmp_path / "frozen.yml",
        dataset_ids=["topic-a"],
        workspaces_root=tmp_path / "workspaces",
        reports_root=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        packets_root=tmp_path / "packets",
        text_profile=tmp_path / "profile.json",
        skip_vlm_packets=False,
    )

    assert steps == []
    assert {item["stage"] for item in skipped} == {
        "process",
        "visualize",
        "evidence",
        "graph_grounding",
        "report:structured_one_shot",
        "report:citeweave_full",
        "graph:no_rag",
        "graph:flat_structured",
        "graph:graph_rag",
        "vlm_packet",
    }


def test_invalid_completed_report_fails_closed(tmp_path: Path):
    dataset_id = "topic-a"
    condition = "structured_one_shot"
    evidence = tmp_path / "workspaces" / dataset_id / "evidence" / "evidence_items.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("[]", encoding="utf-8")
    condition_dir = tmp_path / "reports" / dataset_id / condition
    call_dir = condition_dir / "calls" / "01_structured_one_shot"
    call_dir.mkdir(parents=True)
    (call_dir / "call.json").write_text("{}", encoding="utf-8")
    report = condition_dir / "report.md"
    report.write_text("English report", encoding="utf-8")
    (condition_dir / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_id": dataset_id,
                "condition": condition,
                "model": "deepseek-v4-pro",
                "report_language": "English",
                "call_count": 1,
                "report_sha256": "tampered",
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.FormalOrchestrationError, match="report hash mismatch"):
        MODULE.verify_report_completion(
            tmp_path / "reports",
            dataset_id,
            condition,
            evidence,
        )


def test_execute_plan_stops_at_first_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[list[str]] = []

    def fail_first(command, **_kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(
        MODULE,
        "verify_bulk_harvest",
        lambda _workspace: {"passed": True},
    )
    monkeypatch.setattr(MODULE.subprocess, "run", fail_first)
    steps = [
        MODULE.PlannedStep("topic-a", "process", "pipeline", ("python", "one"), "test"),
        MODULE.PlannedStep("topic-a", "visualize", "pipeline", ("python", "two"), "test"),
    ]

    with pytest.raises(subprocess.CalledProcessError):
        MODULE.execute_plan(
            steps,
            dataset_ids=["topic-a"],
            workspaces_root=tmp_path / "workspaces",
            reports_root=tmp_path / "reports",
            runs_root=tmp_path / "runs",
            packets_root=tmp_path / "packets",
            text_profile=tmp_path / "profile.json",
        )
    assert calls == [["python", "one"]]


def test_all_harvests_are_gated_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checked: list[str] = []
    subprocess_calls: list[list[str]] = []

    def harvest(workspace: Path):
        checked.append(workspace.name)
        return {"passed": workspace.name != "topic-b"}

    monkeypatch.setattr(MODULE, "verify_bulk_harvest", harvest)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda command, **_kwargs: subprocess_calls.append(command),
    )
    step = MODULE.PlannedStep(
        "topic-a", "process", "pipeline", ("python", "run"), "test"
    )

    with pytest.raises(MODULE.FormalOrchestrationError, match="topic-b"):
        MODULE.execute_plan(
            [step],
            dataset_ids=["topic-a", "topic-b"],
            workspaces_root=tmp_path / "workspaces",
            reports_root=tmp_path / "reports",
            runs_root=tmp_path / "runs",
            packets_root=tmp_path / "packets",
            text_profile=tmp_path / "profile.json",
        )
    assert checked == ["topic-a", "topic-b"]
    assert subprocess_calls == []
