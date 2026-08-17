from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from citeweave.judge_protocol import (
    BlindPacket,
    canonical_json,
    prepare_blind_pair,
    prepare_dual_evidence_blind_pair,
    scan_condition_leaks,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "human_references.yml"
DEFAULT_FREEZE = ROOT / "experiments" / "judge_calibration_freeze.json"
DEFAULT_REPORTS = ROOT / "experiments" / "formal_reports"
DEFAULT_WORKSPACES = ROOT / "experiments" / "formal_workspaces"
DEFAULT_HUMAN = ROOT / "experiments" / "human_outputs"
DEFAULT_RUNS = ROOT / "experiments" / "formal_runs"
DEFAULT_VISION = ROOT / "experiments" / "vision_outputs"
DEFAULT_OUTPUT = ROOT / "experiments" / "formal_judging_ready"
DEFAULT_RUN_ID = "formal_v2_nonthinking_20260806"

REPORT_CONDITIONS = ("structured_one_shot", "citeweave_full")
GRAPH_CONDITIONS = ("no_rag", "flat_structured", "graph_rag")
EXPECTED_ROLES = {"development": 2, "locked": 6}
REPORT_TASKS = {
    "RT01": {
        "comparison": "full_vs_oneshot",
        "condition_a": "citeweave_full",
        "condition_b": "structured_one_shot",
        "evidence_mode": "shared",
        "analysis_role": "primary",
    },
    "RT02": {
        "comparison": "full_vs_human",
        "condition_a": "citeweave_full",
        "condition_b": "published_human_reference",
        "evidence_mode": "paired",
        "analysis_role": "primary",
    },
    "RT03": {
        "comparison": "oneshot_vs_human",
        "condition_a": "structured_one_shot",
        "condition_b": "published_human_reference",
        "evidence_mode": "paired",
        "analysis_role": "supplementary",
    },
}
GRAPH_TASKS = {
    "GT01": {
        "comparison": "graph_vs_no",
        "condition_a": "graph_rag",
        "condition_b": "no_rag",
        "evidence_mode": "shared",
    },
    "GT02": {
        "comparison": "graph_vs_flat",
        "condition_a": "graph_rag",
        "condition_b": "flat_structured",
        "evidence_mode": "shared",
    },
    "GT03": {
        "comparison": "graph_vs_figure",
        "condition_a": "graph_rag",
        "condition_b": "figure_vlm",
        "evidence_mode": "shared",
    },
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_value(value: Any) -> str:
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required formal artifact is missing: {path}") from None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required formal artifact is missing: {path}") from None
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain one JSON object")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _track(path: Path, root: Path, provenance: dict[str, str]) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required formal artifact is missing: {path}")
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = resolved.as_posix()
    provenance[label] = _sha_bytes(resolved.read_bytes())


def _load_registry(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("references"), list):
        raise TypeError("Human-reference registry must contain a references list")
    references = value["references"]
    ids = [str(item.get("id", "")) for item in references]
    if any(not dataset_id for dataset_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Human-reference IDs must be non-empty and unique")
    roles = [str(item.get("role", "")) for item in references]
    counts = {role: roles.count(role) for role in set(roles)}
    if counts != EXPECTED_ROLES:
        raise ValueError(
            f"Formal registry must contain exactly {EXPECTED_ROLES}; observed {counts}"
        )
    development = [item["id"] for item in references if item["role"] == "development"]
    locked = [item["id"] for item in references if item["role"] == "locked"]
    return references, development, locked


def _load_freeze(
    path: Path,
    *,
    development: list[str],
    locked: list[str],
) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("status") != "frozen_after_development_calibration":
        raise ValueError("Judge calibration is not frozen")
    if value.get("formal_results_used") is not False:
        raise ValueError("Calibration freeze must predate formal results")
    if value.get("development_topics") != development:
        raise ValueError("Frozen development topics differ from the registry")
    if value.get("locked_topic_count") != len(locked) or len(locked) != 6:
        raise ValueError("Frozen locked-topic count must equal the six formal topics")
    for field in ("report_rubric_version", "graph_rubric_version"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"Calibration freeze is missing {field}")
    rules = value.get("rules")
    if not isinstance(rules, dict):
        raise TypeError("Calibration freeze rules must be an object")
    required_rules = {
        "supported_requires_evidence_id": True,
        "paired_corpora_are_scored_against_their_own_evidence": True,
        "human_reference_counts_are_not_gold_for_system_corpora": True,
        "independent_judges": 2,
        "blind_adjudication_on_conflict": True,
        "judges_may_modify_pipeline_or_artifacts": False,
    }
    if any(rules.get(key) != expected for key, expected in required_rules.items()):
        raise ValueError("Calibration freeze rules differ from the formal Judge contract")
    return value


def _completed_report(
    report_root: Path,
    dataset_id: str,
    condition: str,
    *,
    provenance: dict[str, str],
    source_root: Path,
) -> tuple[str, str]:
    directory = report_root / dataset_id / condition
    report_path = directory / "report.md"
    completion_path = directory / "completion.json"
    _track(report_path, source_root, provenance)
    _track(completion_path, source_root, provenance)
    completion = _read_json(completion_path)
    report_bytes = report_path.read_bytes()
    if completion.get("report_sha256") != _sha_bytes(report_bytes):
        raise ValueError(f"Report hash mismatch: {report_path}")
    evidence_hash = completion.get("evidence_sha256")
    if not isinstance(evidence_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
        raise ValueError(f"Invalid report evidence hash: {completion_path}")
    return report_bytes.decode("utf-8"), evidence_hash


def _anonymize(markdown: str) -> str:
    value = re.sub(
        r"(?im)^#\s+.*$",
        "# Anonymous Bibliometric Report",
        markdown,
        count=1,
    )
    value = re.sub(r"(?i)\bCiteWeave\b", "the reporting system", value)
    value = re.sub(
        r"(?i)\b(structured[_ -]?one[_ -]?shot|published human reference)\b",
        "the report",
        value,
    )
    return value.strip()


def _matched_length(left: str, right: str, maximum_words: int) -> tuple[str, str, int]:
    left_words = left.split()
    right_words = right.split()
    budget = min(len(left_words), len(right_words), maximum_words)
    if budget < 100:
        raise ValueError("Matched report comparison has fewer than 100 words")
    return " ".join(left_words[:budget]), " ".join(right_words[:budget]), budget


def _human_evidence_items(markdown: str) -> list[dict[str, Any]]:
    # The evidence packet's provenance preamble may name the experimental
    # condition in plain language ("published human reference"). Preserve the
    # evidence itself while removing that controller-only identity before any
    # Judge-visible chunks are constructed.
    markdown = re.sub(
        r"(?i)\bpublished[\s_-]+human[\s_-]+reference\b",
        "paired source document",
        markdown,
    )
    items: list[dict[str, Any]] = []
    section = "Unsectioned evidence"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        statement = "\n\n".join(buffer).strip()
        buffer.clear()
        if statement:
            items.append(
                {
                    "evidence_id": f"H{len(items) + 1:03d}",
                    "source_type": "reference_article",
                    "section": section,
                    "statement": statement,
                }
            )

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or "Untitled section"
        elif line:
            buffer.append(line)
        else:
            flush()
    flush()
    if not items:
        raise ValueError("Published reference evidence contains no citable text")
    if [item["evidence_id"] for item in items] != [
        f"H{index:03d}" for index in range(1, len(items) + 1)
    ]:
        raise AssertionError("Human evidence IDs are not a contiguous H### sequence")
    return items


def _parse_completion_content(value: str) -> str:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return value
    if not isinstance(parsed, dict):
        raise TypeError("Graph completion content must be one JSON object")
    return canonical_json(parsed)


def _text_candidates(
    path: Path,
    expected_ids: set[str],
) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for row in _read_jsonl(path):
        if row.get("status") != "complete":
            continue
        item_id = str(row.get("item_id", ""))
        if item_id in candidates:
            raise ValueError(f"Duplicate completed graph item in {path}: {item_id}")
        try:
            content = str(row["response"]["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Malformed completed graph response: {path}:{item_id}") from exc
        candidates[item_id] = _parse_completion_content(content)
    if set(candidates) != expected_ids:
        missing = sorted(expected_ids - set(candidates))
        extra = sorted(set(candidates) - expected_ids)
        raise ValueError(f"Graph output coverage mismatch in {path}; missing={missing}, extra={extra}")
    return candidates


def _vision_candidates(path: Path, expected_ids: set[str]) -> dict[str, str]:
    value = _read_json(path)
    if value.get("visible_only") is not True:
        raise ValueError(f"Figure/VLM output is not marked visible_only: {path}")
    results = value.get("results")
    if not isinstance(results, list):
        raise TypeError(f"Figure/VLM output lacks a results list: {path}")
    candidates: dict[str, str] = {}
    for item in results:
        item_id = str(item.get("item_id", ""))
        if item_id in candidates:
            raise ValueError(f"Duplicate Figure/VLM item in {path}: {item_id}")
        candidates[item_id] = canonical_json(item)
    if set(candidates) != expected_ids:
        missing = sorted(expected_ids - set(candidates))
        extra = sorted(set(candidates) - expected_ids)
        raise ValueError(
            f"Figure/VLM output coverage mismatch in {path}; missing={missing}, extra={extra}"
        )
    return candidates


def _graph_contexts(workspace: Path, expected_ids: set[str]) -> dict[str, Any]:
    experiment_root = workspace / "evidence" / "formal_graph_experiment"
    manifest_path = experiment_root / "manifest.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("contexts")
    if not isinstance(rows, list):
        raise TypeError(f"Graph context manifest lacks contexts: {manifest_path}")
    result: dict[str, Any] = {}
    workspace_resolved = workspace.resolve()
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if item_id in result:
            raise ValueError(f"Duplicate graph context: {item_id}")
        path = (workspace / str(row.get("graph_path", ""))).resolve()
        if workspace_resolved not in path.parents:
            raise ValueError(f"Graph context escapes its workspace: {path}")
        result[item_id] = _read_json(path)
    if set(result) != expected_ids:
        raise ValueError(f"Graph context coverage mismatch for {workspace.name}")
    return result


def _build_sources(
    *,
    references: list[dict[str, Any]],
    locked: list[str],
    reports_root: Path,
    workspaces_root: Path,
    human_root: Path,
    runs_root: Path,
    vision_root: Path,
    run_id: str,
    maximum_words: int,
    source_root: Path,
    provenance: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, int]]:
    report_rows = {task["comparison"]: [] for task in REPORT_TASKS.values()}
    graph_rows = {task["comparison"]: [] for task in GRAPH_TASKS.values()}
    matched_lengths: dict[str, int] = {}
    locked_set = set(locked)

    for reference in references:
        dataset_id = str(reference["id"])
        reports: dict[str, str] = {}
        evidence_hashes: dict[str, str] = {}
        for condition in REPORT_CONDITIONS:
            report, evidence_hash = _completed_report(
                reports_root,
                dataset_id,
                condition,
                provenance=provenance,
                source_root=source_root,
            )
            reports[condition] = _anonymize(report)
            evidence_hashes[condition] = evidence_hash
        if len(set(evidence_hashes.values())) != 1:
            raise ValueError(f"{dataset_id}: report conditions used different evidence")

        evidence_path = workspaces_root / dataset_id / "evidence" / "evidence_items.json"
        _track(evidence_path, source_root, provenance)
        if _sha_bytes(evidence_path.read_bytes()) != next(iter(evidence_hashes.values())):
            raise ValueError(f"{dataset_id}: frozen Evidence Bundle hash mismatch")
        system_evidence = _read_json(evidence_path)

        human_report_path = human_root / dataset_id / "reference_report.md"
        human_evidence_path = human_root / dataset_id / "reference_evidence.md"
        _track(human_report_path, source_root, provenance)
        _track(human_evidence_path, source_root, provenance)
        human_report = _anonymize(human_report_path.read_text(encoding="utf-8"))
        human_evidence = _human_evidence_items(
            human_evidence_path.read_text(encoding="utf-8")
        )

        workspace = workspaces_root / dataset_id
        benchmark_path = workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
        context_manifest_path = (
            workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
        )
        _track(benchmark_path, source_root, provenance)
        _track(context_manifest_path, source_root, provenance)
        benchmark = _read_json(benchmark_path)
        if not isinstance(benchmark, list) or not benchmark:
            raise ValueError(f"{dataset_id}: formal graph benchmark is empty")
        benchmark_by_id = {str(item.get("item_id", "")): item for item in benchmark}
        if "" in benchmark_by_id or len(benchmark_by_id) != len(benchmark):
            raise ValueError(f"{dataset_id}: graph benchmark item IDs are invalid")
        expected_ids = set(benchmark_by_id)
        if any(not item_id.startswith(f"{dataset_id}:") for item_id in expected_ids):
            raise ValueError(f"{dataset_id}: graph benchmark contains a foreign item")
        contexts = _graph_contexts(workspace, expected_ids)
        context_manifest = _read_json(context_manifest_path)
        for context_record in context_manifest["contexts"]:
            _track(
                workspace / str(context_record["graph_path"]),
                source_root,
                provenance,
            )

        graph_candidates = {}
        for condition in GRAPH_CONDITIONS:
            items_path = runs_root / dataset_id / run_id / condition / "items.jsonl"
            _track(items_path, source_root, provenance)
            graph_candidates[condition] = _text_candidates(items_path, expected_ids)
        figure_ids = {
            item_id
            for item_id, item in benchmark_by_id.items()
            if item.get("figure_eligible") is True
        }
        vision_path = vision_root / f"{dataset_id}.json"
        _track(vision_path, source_root, provenance)
        figure_candidates = _vision_candidates(vision_path, figure_ids)

        # All eight topics reach this point. Only locked topics may enter formal inputs.
        if dataset_id not in locked_set:
            continue

        question = (
            "Which anonymous report provides the stronger evidence-bounded bibliometric "
            "synthesis in coverage, explanation, structure, caution, and overall usefulness?"
        )
        report_rows["full_vs_oneshot"].append(
            {
                "sample_id": dataset_id,
                "question": question,
                "canonical_evidence": system_evidence,
                "candidates": {
                    "citeweave_full": reports["citeweave_full"],
                    "structured_one_shot": reports["structured_one_shot"],
                },
            }
        )
        for condition, comparison in (
            ("citeweave_full", "full_vs_human"),
            ("structured_one_shot", "oneshot_vs_human"),
        ):
            system_text, human_text, word_count = _matched_length(
                reports[condition], human_report, maximum_words
            )
            matched_lengths[f"{dataset_id}:{condition}"] = word_count
            report_rows[comparison].append(
                {
                    "sample_id": dataset_id,
                    "question": question,
                    "candidates": {
                        condition: system_text,
                        "published_human_reference": human_text,
                    },
                    "evidence_by_condition": {
                        condition: system_evidence,
                        "published_human_reference": human_evidence,
                    },
                }
            )

        for item_id, item in benchmark_by_id.items():
            evidence = {
                "item_id": item_id,
                "answerable": item["answerable"],
                "answer_contract": item["answer_contract"],
                "gold_answer": item["gold_answer"],
                "graph_evidence": contexts[item_id],
            }
            base = {
                "sample_id": item_id,
                "question": item["question"],
                "canonical_evidence": evidence,
            }
            graph_rows["graph_vs_no"].append(
                {
                    **base,
                    "candidates": {
                        "graph_rag": graph_candidates["graph_rag"][item_id],
                        "no_rag": graph_candidates["no_rag"][item_id],
                    },
                }
            )
            graph_rows["graph_vs_flat"].append(
                {
                    **base,
                    "candidates": {
                        "graph_rag": graph_candidates["graph_rag"][item_id],
                        "flat_structured": graph_candidates["flat_structured"][item_id],
                    },
                }
            )
            if item_id in figure_ids:
                graph_rows["graph_vs_figure"].append(
                    {
                        **base,
                        "candidates": {
                            "graph_rag": graph_candidates["graph_rag"][item_id],
                            "figure_vlm": figure_candidates[item_id],
                        },
                    }
                )

    development_set = {item["id"] for item in references} - locked_set
    for rows in [*report_rows.values(), *graph_rows.values()]:
        if any(
            any(
                str(row["sample_id"]).startswith(f"{dataset_id}:")
                or row["sample_id"] == dataset_id
                for dataset_id in development_set
            )
            for row in rows
        ):
            raise AssertionError("A development topic entered formal Judge inputs")
    if any(len(rows) != 6 for rows in report_rows.values()):
        raise AssertionError("Each report comparison must contain exactly six locked topics")
    return report_rows, graph_rows, matched_lengths


def _validate_packet(packet: BlindPacket, conditions: list[str]) -> None:
    leaks = scan_condition_leaks(packet.model_dump(mode="json"), conditions)
    if leaks:
        raise ValueError(f"Condition-name leakage in {packet.packet_id}: {leaks}")
    visible = {
        "sample_id": packet.sample_id,
        "judge_id": packet.judge_id,
        "rubric_version": packet.rubric_version,
        "question": packet.question,
        "canonical_evidence": packet.canonical_evidence,
        "candidate_a": packet.candidate_a,
        "candidate_b": packet.candidate_b,
    }
    if packet.content_sha256 != _sha_value(visible):
        raise ValueError(f"Packet content hash mismatch: {packet.packet_id}")


def _prepare_task(
    *,
    stage: Path,
    domain: str,
    task_id: str,
    task: dict[str, str],
    rows: list[dict[str, Any]],
    rubric_version: str,
    seed: int,
    formal_topics: list[str],
) -> dict[str, Any]:
    condition_a = task["condition_a"]
    condition_b = task["condition_b"]
    prepare = (
        prepare_dual_evidence_blind_pair
        if task["evidence_mode"] == "paired"
        else prepare_blind_pair
    )
    packets_a = []
    packets_b = []
    mappings = []
    sample_ids = set()
    topic_counts = {dataset_id: 0 for dataset_id in formal_topics}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample in {domain}/{task_id}: {sample_id}")
        sample_ids.add(sample_id)
        matching_topics = [
            dataset_id
            for dataset_id in formal_topics
            if sample_id == dataset_id or sample_id.startswith(f"{dataset_id}:")
        ]
        if len(matching_topics) != 1:
            raise ValueError(
                f"Formal packet sample does not map to exactly one locked topic: {sample_id}"
            )
        topic_counts[matching_topics[0]] += 1
        packet_a, packet_b, mapping = prepare(
            row,
            condition_a=condition_a,
            condition_b=condition_b,
            rubric_version=rubric_version,
            seed=seed,
        )
        _validate_packet(packet_a, [condition_a, condition_b])
        _validate_packet(packet_b, [condition_a, condition_b])
        packets_a.append(packet_a.model_dump(mode="json"))
        packets_b.append(packet_b.model_dump(mode="json"))
        mappings.append(mapping.model_dump(mode="json"))
    if any(count == 0 for count in topic_counts.values()):
        raise ValueError(
            f"{domain}/{task_id} does not cover all six locked topics: {topic_counts}"
        )
    if domain == "report" and any(count != 1 for count in topic_counts.values()):
        raise ValueError(
            f"Report task {task_id} must contain one packet per locked topic: {topic_counts}"
        )

    input_path = stage / "controller" / "inputs" / domain / f"{task['comparison']}.jsonl"
    map_path = stage / "controller" / "secret_maps" / domain / f"{task_id}.json"
    packet_a_path = stage / "packets" / "eval_a" / domain / task_id / "packets.jsonl"
    packet_b_path = stage / "packets" / "eval_b" / domain / task_id / "packets.jsonl"
    _write_jsonl(input_path, rows)
    _write_json(map_path, mappings)
    _write_jsonl(packet_a_path, packets_a)
    _write_jsonl(packet_b_path, packets_b)
    return {
        "task_id": task_id,
        "domain": domain,
        "comparison": task["comparison"],
        "analysis_role": task.get("analysis_role", "primary"),
        "evidence_mode": task["evidence_mode"],
        "condition_a": condition_a,
        "condition_b": condition_b,
        "packet_count": len(rows),
        "topic_packet_counts": topic_counts,
        "input_path": input_path.relative_to(stage).as_posix(),
        "input_sha256": _sha_bytes(input_path.read_bytes()),
        "secret_map_path": map_path.relative_to(stage).as_posix(),
        "secret_map_sha256": _sha_bytes(map_path.read_bytes()),
        "eval_a_path": packet_a_path.relative_to(stage).as_posix(),
        "eval_a_sha256": _sha_bytes(packet_a_path.read_bytes()),
        "eval_b_path": packet_b_path.relative_to(stage).as_posix(),
        "eval_b_sha256": _sha_bytes(packet_b_path.read_bytes()),
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha_bytes(path.read_bytes())
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _publish_idempotently(stage: Path, output_root: Path) -> str:
    if not output_root.exists():
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_root)
        return "created"
    if not output_root.is_dir():
        raise FileExistsError(f"Formal Judge output is not a directory: {output_root}")
    expected = _tree_hashes(stage)
    observed = _tree_hashes(output_root)
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            path for path in set(expected) & set(observed) if expected[path] != observed[path]
        )
        raise FileExistsError(
            "Refusing to overwrite different formal Judge artifacts; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return "already_identical"


def prepare(
    *,
    references_path: Path,
    freeze_path: Path,
    reports_root: Path,
    workspaces_root: Path,
    human_root: Path,
    runs_root: Path,
    vision_root: Path,
    output_root: Path,
    run_id: str = DEFAULT_RUN_ID,
    maximum_words: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    source_root = references_path.resolve().parents[1]
    references, development, locked = _load_registry(references_path)
    freeze = _load_freeze(freeze_path, development=development, locked=locked)
    provenance: dict[str, str] = {}
    _track(references_path, source_root, provenance)
    _track(freeze_path, source_root, provenance)
    report_rows, graph_rows, matched_lengths = _build_sources(
        references=references,
        locked=locked,
        reports_root=reports_root,
        workspaces_root=workspaces_root,
        human_root=human_root,
        runs_root=runs_root,
        vision_root=vision_root,
        run_id=run_id,
        maximum_words=maximum_words,
        source_root=source_root,
        provenance=provenance,
    )

    temp_parent = output_root.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=temp_parent))
    try:
        task_manifests = []
        for task_id, task in REPORT_TASKS.items():
            task_manifests.append(
                _prepare_task(
                    stage=stage,
                    domain="report",
                    task_id=task_id,
                    task=task,
                    rows=report_rows[task["comparison"]],
                    rubric_version=freeze["report_rubric_version"],
                    seed=seed,
                    formal_topics=locked,
                )
            )
        for task_id, task in GRAPH_TASKS.items():
            task_manifests.append(
                _prepare_task(
                    stage=stage,
                    domain="graph",
                    task_id=task_id,
                    task=task,
                    rows=graph_rows[task["comparison"]],
                    rubric_version=freeze["graph_rubric_version"],
                    seed=seed,
                    formal_topics=locked,
                )
            )

        for judge_id in ("eval_a", "eval_b"):
            neutral_tasks = []
            for task in task_manifests:
                packet_path = stage / task[f"{judge_id}_path"]
                neutral_tasks.append(
                    {
                        "task_id": task["task_id"],
                        "domain": task["domain"],
                        "packet_file": packet_path.relative_to(stage).as_posix(),
                        "packet_count": task["packet_count"],
                        "packet_sha256": task[f"{judge_id}_sha256"],
                        "judge_id": judge_id,
                        "output_filename": "judgments.jsonl",
                    }
                )
            worklist = {
                "schema_version": 1,
                "judge_id": judge_id,
                "role": "independent_blind_evaluation_only",
                "may_modify_pipeline_or_artifacts": False,
                "contains_condition_identities": False,
                "controller_metadata_in_scope": False,
                "tasks": neutral_tasks,
            }
            forbidden = [
                task["condition_a"] for task in task_manifests
            ] + [task["condition_b"] for task in task_manifests]
            leaks = scan_condition_leaks(worklist, forbidden)
            if leaks:
                raise AssertionError(f"Neutral worklist leaked conditions: {leaks}")
            _write_json(stage / "neutral_worklists" / f"{judge_id}.json", worklist)

        manifest = {
            "schema_version": 1,
            "status": "ready_for_independent_blind_judging",
            "run_id": run_id,
            "seed": seed,
            "all_topics_verified": [item["id"] for item in references],
            "development_topics_excluded": development,
            "formal_locked_topics": locked,
            "formal_topic_count": len(locked),
            "maximum_words": maximum_words,
            "matched_word_counts": matched_lengths,
            "calibration_freeze_sha256": _sha_bytes(freeze_path.read_bytes()),
            "rubrics": {
                "report": freeze["report_rubric_version"],
                "graph": freeze["graph_rubric_version"],
            },
            "report_main_comparisons": ["full_vs_oneshot", "full_vs_human"],
            "report_supplementary_comparisons": ["oneshot_vs_human"],
            "graph_comparisons": ["graph_vs_no", "graph_vs_flat", "graph_vs_figure"],
            "human_evidence_contract": "contiguous_H###_chunks",
            "task_map": task_manifests,
            "source_artifact_sha256": dict(sorted(provenance.items())),
            "judge_facing_roots": {
                "eval_a": "packets/eval_a",
                "eval_b": "packets/eval_b",
            },
            "controller_only_root": "controller",
        }
        _write_json(stage / "controller" / "manifest.json", manifest)
        inventory = _tree_hashes(stage)
        _write_json(
            stage / "controller" / "artifact_inventory.json",
            {
                "schema_version": 1,
                "files_excluding_this_inventory": inventory,
                "aggregate_sha256": _sha_value(inventory),
            },
        )
        publish_status = _publish_idempotently(stage, output_root)
        result = dict(manifest)
        result["publish_status"] = publish_status
        result["output_root"] = str(output_root)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all eight completed topics, then prepare deterministic blind Judge "
            "inputs for only the six frozen locked topics. This command never calls a model."
        )
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--workspaces-root", type=Path, default=DEFAULT_WORKSPACES)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--vision-root", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--maximum-words", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(
        references_path=args.references,
        freeze_path=args.freeze,
        reports_root=args.reports_root,
        workspaces_root=args.workspaces_root,
        human_root=args.human_root,
        runs_root=args.runs_root,
        vision_root=args.vision_root,
        output_root=args.output_root,
        run_id=args.run_id,
        maximum_words=args.maximum_words,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
