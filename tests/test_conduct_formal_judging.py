from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from citeweave.formal_statistics import analyze_formal_experiment
from citeweave.judge_protocol import collect_reference_ids

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = _load(
    "prepare_remaining_for_conductor_test",
    ROOT / "scripts" / "prepare_remaining_formal_judging.py",
)
PREPARE_TEST = _load(
    "prepare_remaining_fixture_for_conductor_test",
    ROOT / "tests" / "test_prepare_remaining_formal_judging.py",
)
CONDUCTOR = _load(
    "conduct_formal_judging",
    ROOT / "scripts" / "conduct_formal_judging.py",
)
STATISTICS_MANIFEST = _load(
    "prepare_formal_statistics_for_conductor_test",
    ROOT / "scripts" / "prepare_formal_statistics_manifest.py",
)


def _allowed(packet: dict, slot: str) -> list[str]:
    evidence = packet["canonical_evidence"]
    if isinstance(evidence, dict) and evidence.get("policy", "").startswith(
        "Evaluate each anonymous report"
    ):
        evidence = evidence[f"candidate_{slot.casefold()}_evidence"]
    return collect_reference_ids(evidence)


def _write_returns(ready: Path, returns: Path) -> None:
    manifest = json.loads(
        (ready / "controller" / "manifest.json").read_text(encoding="utf-8")
    )
    for task in manifest["task_map"]:
        task_id = task["task_id"]
        for judge_id in ("eval_a", "eval_b"):
            packet_path = ready / task[f"{judge_id}_path"]
            packets = [
                json.loads(line)
                for line in packet_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            results = []
            for packet in packets:
                candidates = []
                for slot in ("A", "B"):
                    allowed = _allowed(packet, slot)
                    assert allowed
                    candidates.append(
                        {
                            "slot": slot,
                            "claims": [
                                {
                                    "claim": (
                                        "The anonymous candidate makes a claim "
                                        "addressable in the visible evidence."
                                    ),
                                    "verdict": "supported",
                                    "evidence_ids": [allowed[0]],
                                }
                            ],
                            "completeness_score": 4,
                        }
                    )
                results.append(
                    {
                        "packet_id": packet["packet_id"],
                        "judge_id": judge_id,
                        "candidates": candidates,
                        "preference": "tie",
                        "rationale": (
                            "Both anonymous candidates are equally complete and "
                            "addressable in the visible evidence."
                        ),
                    }
                )
            output = returns / judge_id / task_id / "judgments.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in results
                ),
                encoding="utf-8",
            )


def _conduct(paths: dict[str, Path], returns: Path):
    experiments = paths["references_path"].parent
    return CONDUCTOR.conduct(
        ready_root=paths["output_root"],
        returns_root=returns,
        references_path=paths["references_path"],
        report_root=experiments / "formal_judging",
        graph_root=experiments / "formal_graph_judging",
        report_split_root=experiments / "formal_judging" / "report_resolved",
        graph_split_root=experiments / "formal_judging" / "graph_resolved",
    )


def test_conductor_bridges_ready_exchange_to_locked_topic_statistics(tmp_path: Path):
    paths = PREPARE_TEST._fixture(tmp_path)
    PREPARE.prepare(**paths, maximum_words=120, seed=42)
    returns = paths["references_path"].parent / "formal_judge_returns"

    waiting = _conduct(paths, returns)
    assert set(waiting["statuses"].values()) == {"awaiting_independent_judges"}
    for judge_id in ("eval_a", "eval_b"):
        assignment = json.loads(
            (returns / judge_id / "assignment.json").read_text(encoding="utf-8")
        )
        assert assignment["contains_condition_identities"] is False
        assert assignment["may_modify_pipeline_or_source_artifacts"] is False
        serialized = json.dumps(assignment)
        for condition in (
            "citeweave_full",
            "structured_one_shot",
            "published_human_reference",
            "graph_rag",
            "no_rag",
            "flat_structured",
            "figure_vlm",
        ):
            assert condition not in serialized

    _write_returns(paths["output_root"], returns)
    completed = _conduct(paths, returns)
    assert completed["complete"] is True

    experiments = paths["references_path"].parent
    locked = {f"locked_{index}" for index in range(1, 7)}
    for family, comparisons in {
        "report_resolved": {
            "full_vs_oneshot",
            "full_vs_human",
            "oneshot_vs_human",
        },
        "graph_resolved": {"graph_vs_no", "graph_vs_flat", "graph_vs_figure"},
    }.items():
        root = experiments / "formal_judging" / family
        assert {path.name for path in root.iterdir()} == comparisons
        for comparison in comparisons:
            assert {path.name for path in (root / comparison).iterdir()} == locked

    adaptive = experiments / "formal_adaptive_topic_counts"
    for topic in locked:
        payload = {
            "topic_id": topic,
            "comparison_contract": {
                "topic_role": "locked",
                "formal_results_used": True,
                "post_review_conditions": [
                    "always_review",
                    "static_review",
                    "adaptive_review",
                ],
            },
            "conditions": {
                "baseline_original": {
                    "items": 1,
                    "review_requests": 0,
                    "final_quality_passed": 1,
                    "auto_accepts": 1,
                    "unsafe_auto_accepts": 0,
                },
                **{
                    condition: {
                        "items": 1,
                        "review_requests": 1,
                        "final_quality_passed": 1,
                        "auto_accepts": 0,
                        "unsafe_auto_accepts": 0,
                    }
                    for condition in (
                        "always_review",
                        "static_review",
                        "adaptive_review",
                    )
                },
            },
        }
        path = adaptive / f"{topic}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    statistics_manifest_path = experiments / "formal_statistics_manifest.json"
    statistics_manifest = STATISTICS_MANIFEST.build_manifest(
        references_path=paths["references_path"],
        report_root=experiments / "formal_judging" / "report_resolved",
        graph_root=experiments / "formal_judging" / "graph_resolved",
        adaptive_root=adaptive,
        output_path=statistics_manifest_path,
    )
    STATISTICS_MANIFEST.write_idempotent(
        statistics_manifest_path,
        statistics_manifest,
    )
    summary = analyze_formal_experiment(statistics_manifest_path)
    assert {panel["comparison"] for panel in summary["panels"]} == {
        "full_vs_oneshot",
        "full_vs_human",
        "oneshot_vs_human",
        "graph_vs_no",
        "graph_vs_flat",
        "graph_vs_figure",
    }

    assert _conduct(paths, returns)["complete"] is True


def test_conductor_rejects_tampered_ready_inventory(tmp_path: Path):
    paths = PREPARE_TEST._fixture(tmp_path)
    PREPARE.prepare(**paths, maximum_words=120, seed=42)
    packet = (
        paths["output_root"] / "packets" / "eval_a" / "report" / "RT01"
        / "packets.jsonl"
    )
    packet.write_text(packet.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    returns = paths["references_path"].parent / "formal_judge_returns"
    try:
        _conduct(paths, returns)
    except CONDUCTOR.ConductorError as exc:
        assert "inventory" in str(exc)
    else:
        raise AssertionError("Tampered ready exchange was accepted")
