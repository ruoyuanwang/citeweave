from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY = ROOT / "experiments" / "formal_judging_ready"
DEFAULT_RETURNS = ROOT / "experiments" / "formal_judge_returns"
DEFAULT_REFERENCES = ROOT / "experiments" / "human_references.yml"
DEFAULT_REPORT_ROOT = ROOT / "experiments" / "formal_judging"
DEFAULT_GRAPH_ROOT = ROOT / "experiments" / "formal_graph_judging"
DEFAULT_REPORT_SPLIT = DEFAULT_REPORT_ROOT / "report_resolved"
DEFAULT_GRAPH_SPLIT = DEFAULT_REPORT_ROOT / "graph_resolved"
SCORER = ROOT / "scripts" / "score_judge_results.py"
SPLITTER = ROOT / "scripts" / "split_resolved_judgments_by_topic.py"

TASK_CONTRACT = {
    "RT01": ("report", "full_vs_oneshot"),
    "RT02": ("report", "full_vs_human"),
    "RT03": ("report", "oneshot_vs_human"),
    "GT01": ("graph", "graph_vs_no"),
    "GT02": ("graph", "graph_vs_flat"),
    "GT03": ("graph", "graph_vs_figure"),
}
JUDGE_RUBRIC = [
    (
        "Use only the anonymous candidates and canonical evidence visible in each "
        "packet; do not browse, call APIs, use hidden files, or use outside knowledge."
    ),
    (
        "For each candidate, enumerate its material factual, numeric, relational, "
        "methodological, and interpretive claims at a consistent claim granularity."
    ),
    (
        "Mark each claim supported, contradicted, or not_in_evidence. Every supported "
        "claim must cite one or more evidence IDs visible for that candidate; never "
        "invent an evidence ID."
    ),
    (
        "For paired-evidence packets, assess each candidate only against its own paired "
        "evidence and never treat cross-corpus count differences as errors."
    ),
    (
        "Assign completeness_score from 1 to 5 for how fully and cautiously the "
        "candidate answers the question within the visible evidence."
    ),
    (
        "Choose A, B, or tie by evidence-bounded factuality, completeness, explanation, "
        "caution, and usefulness—not by presumed system identity or writing style."
    ),
]
RESULT_SCHEMA = {
    "packet_id": "copy exactly from the packet",
    "judge_id": "the assigned judge_id",
    "candidates": [
        {
            "slot": "A",
            "claims": [
                {
                    "claim": "one material claim",
                    "verdict": "supported|contradicted|not_in_evidence",
                    "evidence_ids": ["visible evidence IDs; required when supported"],
                }
            ],
            "completeness_score": "integer 1..5",
        },
        {
            "slot": "B",
            "claims": "same structure as slot A",
            "completeness_score": "integer 1..5",
        },
    ],
    "preference": "A|B|tie",
    "rationale": "concise evidence-bounded rationale",
}


class ConductorError(ValueError):
    """Raised when the formal blind-Judge exchange fails closed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConductorError(f"Cannot read valid JSON: {path}") from exc


def _write_identical(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ConductorError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_identical(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ConductorError(f"Required source artifact is missing: {source}")
    content = source.read_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise ConductorError(f"Refusing to overwrite different artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _locked_topics(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("references"), list):
        raise ConductorError("Reference registry must contain a references list")
    locked = [
        str(item["id"])
        for item in value["references"]
        if item.get("role") == "locked"
    ]
    development = [
        str(item["id"])
        for item in value["references"]
        if item.get("role") == "development"
    ]
    if len(locked) != 6 or len(set(locked)) != 6 or len(development) != 2:
        raise ConductorError("Formal judging requires exactly 2 development + 6 locked topics")
    return locked


def _validate_ready(
    ready_root: Path,
    references_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = ready_root / "controller" / "manifest.json"
    inventory_path = ready_root / "controller" / "artifact_inventory.json"
    manifest = _read_json(manifest_path)
    inventory = _read_json(inventory_path)
    locked = _locked_topics(references_path)
    if (
        manifest.get("status") != "ready_for_independent_blind_judging"
        or manifest.get("formal_locked_topics") != locked
        or manifest.get("formal_topic_count") != 6
        or len(manifest.get("development_topics_excluded", [])) != 2
    ):
        raise ConductorError("Ready manifest is not the exact frozen six-topic exchange")
    if manifest.get("report_main_comparisons") != [
        "full_vs_oneshot",
        "full_vs_human",
    ]:
        raise ConductorError("Primary report comparison contract differs")
    if manifest.get("report_supplementary_comparisons") != ["oneshot_vs_human"]:
        raise ConductorError("Supplementary report comparison contract differs")
    if manifest.get("graph_comparisons") != [
        "graph_vs_no",
        "graph_vs_flat",
        "graph_vs_figure",
    ]:
        raise ConductorError("Graph comparison contract differs")

    expected_inventory = inventory.get("files_excluding_this_inventory")
    if not isinstance(expected_inventory, dict):
        raise ConductorError("Ready artifact inventory is malformed")
    actual_inventory = {
        path.relative_to(ready_root).as_posix(): _sha(path)
        for path in sorted(ready_root.rglob("*"))
        if path.is_file() and path != inventory_path
    }
    if actual_inventory != expected_inventory:
        raise ConductorError("Ready artifact inventory hash or coverage mismatch")

    tasks = manifest.get("task_map")
    if not isinstance(tasks, list):
        raise ConductorError("Ready manifest lacks task_map")
    by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        if task_id in by_id:
            raise ConductorError(f"Duplicate formal task: {task_id}")
        if task_id not in TASK_CONTRACT:
            raise ConductorError(f"Unexpected formal task: {task_id}")
        domain, comparison = TASK_CONTRACT[task_id]
        if task.get("domain") != domain or task.get("comparison") != comparison:
            raise ConductorError(f"Formal task contract differs: {task_id}")
        counts = task.get("topic_packet_counts")
        if (
            not isinstance(counts, dict)
            or set(counts) != set(locked)
            or any(not isinstance(count, int) or count <= 0 for count in counts.values())
        ):
            raise ConductorError(f"Task does not cover all six locked topics: {task_id}")
        if domain == "report" and set(counts.values()) != {1}:
            raise ConductorError(f"Report task must contain one item per topic: {task_id}")
        for label in ("eval_a", "eval_b"):
            packet_path = ready_root / str(task[f"{label}_path"])
            if _sha(packet_path) != task[f"{label}_sha256"]:
                raise ConductorError(f"{task_id} {label} packet hash mismatch")
        map_path = ready_root / str(task["secret_map_path"])
        if _sha(map_path) != task["secret_map_sha256"]:
            raise ConductorError(f"{task_id} secret map hash mismatch")
        by_id[task_id] = task
    if set(by_id) != set(TASK_CONTRACT):
        raise ConductorError(
            f"Formal tasks differ: missing={sorted(set(TASK_CONTRACT) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(TASK_CONTRACT))}"
        )
    return manifest, by_id


def _prepare_assignments(
    ready_root: Path,
    returns_root: Path,
    tasks: dict[str, dict[str, Any]],
) -> None:
    for judge_id in ("eval_a", "eval_b"):
        work = []
        for task_id in TASK_CONTRACT:
            task = tasks[task_id]
            packet_path = ready_root / str(task[f"{judge_id}_path"])
            output_path = returns_root / judge_id / task_id / "judgments.jsonl"
            work.append(
                {
                    "task_id": task_id,
                    "domain": task["domain"],
                    "packet_file": str(packet_path.resolve()),
                    "packet_sha256": task[f"{judge_id}_sha256"],
                    "packet_count": task["packet_count"],
                    "output_file": str(output_path.resolve()),
                }
            )
        assignment = {
            "schema_version": 1,
            "judge_id": judge_id,
            "role": "independent_blind_evaluation_only",
            "contains_condition_identities": False,
            "controller_metadata_in_scope": False,
            "may_modify_pipeline_or_source_artifacts": False,
            "write_scope": "Only the six output_file paths listed below.",
            "rubric": JUDGE_RUBRIC,
            "result_schema": RESULT_SCHEMA,
            "output_format": "One JSON object per input packet, in input order, as JSONL.",
            "tasks": work,
        }
        _write_identical(returns_root / judge_id / "assignment.json", assignment)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _resolved_is_bound(
    resolved_dir: Path,
    *,
    judge_a: Path,
    judge_b: Path,
    packets_a: Path,
    packets_b: Path,
    blind_map: Path,
    adjudications: Path | None,
) -> bool:
    required = [
        resolved_dir / "resolved_judgments.jsonl",
        resolved_dir / "judge_metrics.json",
        resolved_dir / "scoring_manifest.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    manifest = _read_json(resolved_dir / "scoring_manifest.json")
    expected = {
        "judge_a": _sha(judge_a),
        "judge_b": _sha(judge_b),
        "packets_a": _sha(packets_a),
        "packets_b": _sha(packets_b),
        "blind_map": _sha(blind_map),
        **(
            {"adjudications": _sha(adjudications)}
            if adjudications is not None
            else {}
        ),
    }
    if (
        manifest.get("role") != "read_only_blind_evaluation_and_adjudication"
        or manifest.get("judges_may_modify_pipeline_or_source_artifacts") is not False
        or manifest.get("input_sha256") != expected
    ):
        raise ConductorError(f"Resolved scoring provenance differs: {resolved_dir}")
    return True


def _split(
    *,
    resolved: Path,
    references: Path,
    output_root: Path,
) -> None:
    completed = _run(
        [
            sys.executable,
            str(SPLITTER),
            "--input",
            str(resolved),
            "--references",
            str(references),
            "--output-root",
            str(output_root),
        ]
    )
    if completed.returncode:
        raise ConductorError(
            f"Formal topic split failed for {output_root}: "
            f"{completed.stdout}{completed.stderr}"
        )


def conduct(
    *,
    ready_root: Path,
    returns_root: Path,
    references_path: Path,
    report_root: Path,
    graph_root: Path,
    report_split_root: Path,
    graph_split_root: Path,
) -> dict[str, Any]:
    manifest, tasks = _validate_ready(ready_root, references_path)
    _prepare_assignments(ready_root, returns_root, tasks)
    statuses: dict[str, str] = {}

    for task_id, (domain, comparison) in TASK_CONTRACT.items():
        task = tasks[task_id]
        judge_a = returns_root / "eval_a" / task_id / "judgments.jsonl"
        judge_b = returns_root / "eval_b" / task_id / "judgments.jsonl"
        if not judge_a.is_file() or not judge_b.is_file():
            statuses[task_id] = "awaiting_independent_judges"
            continue

        packets_a = ready_root / str(task["eval_a_path"])
        packets_b = ready_root / str(task["eval_b_path"])
        blind_map = ready_root / str(task["secret_map_path"])
        family_root = report_root if domain == "report" else graph_root
        comparison_root = family_root / comparison
        resolved_dir = comparison_root / "resolved_v1"
        returned_adjudications = (
            returns_root / "adjudicator" / task_id / "adjudications.jsonl"
        )
        adjudications = returned_adjudications if returned_adjudications.is_file() else None

        already_resolved = _resolved_is_bound(
            resolved_dir,
            judge_a=judge_a,
            judge_b=judge_b,
            packets_a=packets_a,
            packets_b=packets_b,
            blind_map=blind_map,
            adjudications=adjudications,
        )
        if not already_resolved:
            command = [
                sys.executable,
                str(SCORER),
                "--judge-a",
                str(judge_a),
                "--judge-b",
                str(judge_b),
                "--packets-a",
                str(packets_a),
                "--packets-b",
                str(packets_b),
                "--blind-map",
                str(blind_map),
                "--output-dir",
                str(resolved_dir),
            ]
            if adjudications is not None:
                command.extend(["--adjudications", str(adjudications)])
            completed = _run(command)
            if completed.returncode:
                conflict_path = resolved_dir / "adjudication_packets.jsonl"
                if conflict_path.is_file() and "require blind adjudication" in (
                    completed.stdout + completed.stderr
                ):
                    assignment = {
                        "schema_version": 1,
                        "judge_id": "adjudicator",
                        "role": "blind_conflict_resolution_only",
                        "contains_condition_identities": False,
                        "controller_metadata_in_scope": False,
                        "may_modify_pipeline_or_source_artifacts": False,
                        "packet_file": str(conflict_path.resolve()),
                        "packet_sha256": _sha(conflict_path),
                        "output_file": str(returned_adjudications.resolve()),
                        "write_scope": "Only output_file may be written.",
                        "rubric": [
                            *JUDGE_RUBRIC,
                            (
                                "Resolve only the listed conflicts by independently "
                                "checking both candidates and prior anonymous judgments "
                                "against the visible evidence."
                            ),
                        ],
                        "result_schema": {
                            **RESULT_SCHEMA,
                            "judge_id": "adjudicator",
                        },
                        "output_format": (
                            "One JSON object per adjudication packet, in input order, "
                            "as JSONL."
                        ),
                    }
                    _write_identical(
                        returns_root / "adjudicator" / task_id / "assignment.json",
                        assignment,
                    )
                    statuses[task_id] = "awaiting_blind_adjudication"
                    continue
                raise ConductorError(
                    f"Formal scoring failed for {task_id}: "
                    f"{completed.stdout}{completed.stderr}"
                )

        _copy_identical(judge_a, comparison_root / "judge_a.jsonl")
        _copy_identical(judge_b, comparison_root / "judge_b.jsonl")
        _copy_identical(blind_map, comparison_root / "secret_blind_map.json")
        _copy_identical(packets_a, comparison_root / "eval_a_packets.jsonl")
        _copy_identical(packets_b, comparison_root / "eval_b_packets.jsonl")
        if adjudications is not None:
            _copy_identical(adjudications, resolved_dir / "adjudications.jsonl")
        exchange_manifest = {
            "schema_version": 1,
            "task_id": task_id,
            "domain": domain,
            "comparison": comparison,
            "formal_locked_topics": manifest["formal_locked_topics"],
            "independent_judges": ["eval_a", "eval_b"],
            "adjudicator_used": adjudications is not None,
            "judges_may_modify_pipeline_or_source_artifacts": False,
            "source_sha256": {
                "judge_a": _sha(judge_a),
                "judge_b": _sha(judge_b),
                "eval_a_packets": _sha(packets_a),
                "eval_b_packets": _sha(packets_b),
                "secret_blind_map": _sha(blind_map),
                "scoring_manifest": _sha(resolved_dir / "scoring_manifest.json"),
            },
        }
        _write_identical(comparison_root / "exchange_manifest.json", exchange_manifest)

        split_base = report_split_root if domain == "report" else graph_split_root
        _split(
            resolved=resolved_dir / "resolved_judgments.jsonl",
            references=references_path,
            output_root=split_base / comparison,
        )
        statuses[task_id] = "complete"

    return {
        "schema_version": 1,
        "formal_locked_topics": manifest["formal_locked_topics"],
        "statuses": statuses,
        "complete": all(status == "complete" for status in statuses.values()),
        "judge_assignments": {
            judge_id: str((returns_root / judge_id / "assignment.json").resolve())
            for judge_id in ("eval_a", "eval_b")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convey immutable blind packets to independent Judge return locations, "
            "validate/score their read-only outputs, prepare blind adjudication tasks, "
            "and split resolved results into the six locked formal topics. No model is called."
        )
    )
    parser.add_argument("--ready-root", type=Path, default=DEFAULT_READY)
    parser.add_argument("--returns-root", type=Path, default=DEFAULT_RETURNS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--report-split-root", type=Path, default=DEFAULT_REPORT_SPLIT)
    parser.add_argument("--graph-split-root", type=Path, default=DEFAULT_GRAPH_SPLIT)
    args = parser.parse_args()
    try:
        result = conduct(
            ready_root=args.ready_root,
            returns_root=args.returns_root,
            references_path=args.references,
            report_root=args.report_root,
            graph_root=args.graph_root,
            report_split_root=args.report_split_root,
            graph_split_root=args.graph_split_root,
        )
    except ConductorError as exc:
        raise SystemExit(f"Formal Judge conductor refused the exchange: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
