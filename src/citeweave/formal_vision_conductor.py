from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .formal_protocol import verify_frozen_query_registry
from .io import atomic_write_bytes, sha256_file

EXPECTED_PACKET_KEYS = {
    "schema_version",
    "packet_role",
    "dataset_id",
    "generator",
    "main_inference",
    "visible_only",
    "prohibited_inputs",
    "instructions",
    "benchmark_sha256",
    "items",
    "output_schema",
    "packet_sha256",
}
EXPECTED_OUTPUT_KEYS = {
    "schema_version",
    "packet_sha256",
    "dataset_id",
    "generator_role",
    "visible_only",
    "results",
}
EXPECTED_RESULT_KEYS = {
    "item_id",
    "abstain",
    "answer",
    "explanation",
}
EXPECTED_PROHIBITED_INPUTS = {
    "gold_answer",
    "graph JSON",
    "flat structured context",
    "human reference output",
    "other condition output",
}
GENERATOR_ROLE = "codex_visual_subagent"
RUN_ID = "formal_v2_nonthinking_20260806"


class VisionConductorError(ValueError):
    """Raised when the formal visible-only exchange must fail closed."""


@dataclass(frozen=True)
class ValidatedPacket:
    dataset_id: str
    packet_path: Path
    packet_sha256: str
    packet_file_sha256: str
    benchmark_file_sha256: str
    figure_manifest_path: Path
    figure_manifest_sha256: str
    items: list[dict[str, Any]]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _embedded_packet_hash(packet: dict[str, Any]) -> str:
    without_hash = {
        key: value for key, value in packet.items() if key != "packet_sha256"
    }
    return hashlib.sha256(_canonical(without_hash)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisionConductorError(f"Cannot read valid JSON: {path}") from exc


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_idempotent(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise VisionConductorError(
                f"Refusing to overwrite a different formal artifact: {path}"
            )
        return
    atomic_write_bytes(path, payload)


def load_frozen_topics(registry_path: Path) -> list[dict[str, str]]:
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VisionConductorError(
            f"Cannot read frozen registry: {registry_path}"
        ) from exc
    if not isinstance(registry, dict):
        raise VisionConductorError("Frozen registry must be a mapping")
    try:
        verify_frozen_query_registry(registry)
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionConductorError("Frozen registry verification failed") from exc
    datasets = registry["datasets"]
    topics = [
        {"dataset_id": str(item["id"]), "role": str(item.get("role", ""))}
        for item in datasets
    ]
    ids = [item["dataset_id"] for item in topics]
    roles = [item["role"] for item in topics]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise VisionConductorError(
            "Formal Figure/VLM exchange requires exactly eight unique topics"
        )
    if roles.count("development") != 2 or roles.count("locked") != 6:
        raise VisionConductorError(
            "Formal Figure/VLM exchange requires 2 development and 6 locked topics"
        )
    return topics


def _manifest_artifacts(
    manifest: Any,
    *,
    manifest_path: Path,
) -> dict[Path, dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("figures"), list
    ):
        raise VisionConductorError(
            f"Figure manifest has no figures list: {manifest_path}"
        )
    by_path: dict[Path, dict[str, Any]] = {}
    for artifact in manifest["figures"]:
        if not isinstance(artifact, dict):
            raise VisionConductorError(
                f"Figure manifest contains a non-object artifact: {manifest_path}"
            )
        png = artifact.get("png")
        digest = artifact.get("png_sha256")
        if not isinstance(png, str) or not isinstance(digest, str):
            raise VisionConductorError(
                f"Figure manifest artifact lacks PNG/hash: {manifest_path}"
            )
        resolved = Path(png).resolve()
        if resolved in by_path:
            raise VisionConductorError(
                f"Duplicate PNG in figure manifest: {resolved}"
            )
        by_path[resolved] = artifact
    return by_path


def _safe_packet_projection(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise VisionConductorError("Figure/VLM packet item must be an object")
    expected = {
        "item_id",
        "task_type",
        "question",
        "answer_contract",
        "figure_path",
    }
    if set(item) != expected:
        raise VisionConductorError(
            f"Figure/VLM packet item keys differ: {sorted(set(item))}"
        )
    item_id = item.get("item_id")
    question = item.get("question")
    task_type = item.get("task_type")
    contract = item.get("answer_contract")
    figure_path = item.get("figure_path")
    if (
        not isinstance(item_id, str)
        or not item_id
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(task_type, str)
        or not task_type
        or not isinstance(contract, dict)
        or not contract
        or not isinstance(figure_path, str)
        or not figure_path
    ):
        raise VisionConductorError(f"Malformed Figure/VLM item: {item_id!r}")
    if any(
        not isinstance(key, str)
        or not key
        or value not in {"integer", "number", "string", "boolean"}
        for key, value in contract.items()
    ):
        raise VisionConductorError(f"Unsupported answer contract: {item_id}")
    return {
        "item_id": item_id,
        "task_type": task_type,
        "question": question,
        "answer_contract": contract,
        "figure_path": figure_path,
    }


def validate_packet(
    *,
    dataset_id: str,
    packet_root: Path,
    workspace_root: Path,
) -> ValidatedPacket:
    packet_path = (packet_root / f"{dataset_id}.json").resolve()
    workspace = (workspace_root / dataset_id).resolve()
    benchmark_path = (
        workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
    )
    manifest_path = workspace / "figures" / "figure_manifest.json"
    packet = _read_json(packet_path)
    benchmark = _read_json(benchmark_path)
    manifest = _read_json(manifest_path)
    if not isinstance(packet, dict) or set(packet) != EXPECTED_PACKET_KEYS:
        keys = sorted(packet) if isinstance(packet, dict) else []
        raise VisionConductorError(
            f"{dataset_id}: packet top-level keys differ: {keys}"
        )
    if (
        packet.get("schema_version") != 1
        or packet.get("packet_role") != "cross_model_figure_vlm_generation"
        or packet.get("dataset_id") != dataset_id
        or packet.get("generator") != "independent_codex_visual_subagent"
        or packet.get("main_inference") is not False
        or packet.get("visible_only") is not True
        or set(packet.get("prohibited_inputs", [])) != EXPECTED_PROHIBITED_INPUTS
    ):
        raise VisionConductorError(f"{dataset_id}: packet role contract differs")
    if packet.get("packet_sha256") != _embedded_packet_hash(packet):
        raise VisionConductorError(f"{dataset_id}: invalid embedded packet SHA-256")
    if (
        not benchmark_path.is_file()
        or packet.get("benchmark_sha256") != sha256_file(benchmark_path)
    ):
        raise VisionConductorError(f"{dataset_id}: benchmark hash differs")
    if not isinstance(benchmark, list):
        raise VisionConductorError(f"{dataset_id}: benchmark must be a list")

    packet_items_raw = packet.get("items")
    if not isinstance(packet_items_raw, list) or not packet_items_raw:
        raise VisionConductorError(f"{dataset_id}: packet has no items")
    packet_items = [_safe_packet_projection(item) for item in packet_items_raw]
    item_ids = [item["item_id"] for item in packet_items]
    if len(item_ids) != len(set(item_ids)):
        raise VisionConductorError(f"{dataset_id}: duplicate packet item IDs")
    eligible = [
        _safe_packet_projection(
            {
                "item_id": item.get("item_id"),
                "task_type": item.get("task_type"),
                "question": item.get("question"),
                "answer_contract": item.get("answer_contract"),
                "figure_path": item.get("figure_path"),
            }
        )
        for item in benchmark
        if isinstance(item, dict) and item.get("figure_eligible") is True
    ]
    if packet_items != eligible:
        raise VisionConductorError(
            f"{dataset_id}: packet differs from exact figure-eligible benchmark order"
        )

    output_schema = packet.get("output_schema")
    if (
        not isinstance(output_schema, dict)
        or set(output_schema) != EXPECTED_OUTPUT_KEYS
        or output_schema.get("schema_version") != 1
        or output_schema.get("packet_sha256") != "<copy packet_sha256>"
        or output_schema.get("dataset_id") != dataset_id
        or output_schema.get("generator_role") != GENERATOR_ROLE
        or output_schema.get("visible_only") is not True
        or not isinstance(output_schema.get("results"), list)
        or len(output_schema["results"]) != 1
        or set(output_schema["results"][0]) != EXPECTED_RESULT_KEYS
    ):
        raise VisionConductorError(f"{dataset_id}: strict output schema differs")

    artifact_by_path = _manifest_artifacts(manifest, manifest_path=manifest_path)
    safe_items: list[dict[str, Any]] = []
    figures_root = (workspace / "figures").resolve()
    for item in packet_items:
        figure_path = Path(item["figure_path"]).resolve()
        try:
            figure_path.relative_to(figures_root)
        except ValueError as exc:
            raise VisionConductorError(
                f"{dataset_id}: figure escapes workspace figure directory"
            ) from exc
        if figure_path.suffix.lower() != ".png" or not figure_path.is_file():
            raise VisionConductorError(
                f"{dataset_id}: missing PNG figure: {figure_path}"
            )
        artifact = artifact_by_path.get(figure_path)
        if artifact is None:
            raise VisionConductorError(
                f"{dataset_id}: figure is absent from figure manifest: {figure_path}"
            )
        expected_figure_sha = artifact["png_sha256"]
        if sha256_file(figure_path) != expected_figure_sha:
            raise VisionConductorError(
                f"{dataset_id}: figure SHA-256 differs: {figure_path}"
            )
        safe_items.append(
            {
                **item,
                "figure_path": str(figure_path),
                "figure_sha256": expected_figure_sha,
            }
        )

    return ValidatedPacket(
        dataset_id=dataset_id,
        packet_path=packet_path,
        packet_sha256=str(packet["packet_sha256"]),
        packet_file_sha256=sha256_file(packet_path),
        benchmark_file_sha256=sha256_file(benchmark_path),
        figure_manifest_path=manifest_path,
        figure_manifest_sha256=sha256_file(manifest_path),
        items=safe_items,
    )


def _validate_answer(
    *,
    item_id: str,
    abstain: bool,
    answer: Any,
    contract: dict[str, str],
) -> None:
    if abstain:
        if answer is not None:
            raise VisionConductorError(
                f"{item_id}: abstaining result must have answer=null"
            )
        return
    if not isinstance(answer, dict) or set(answer) != set(contract):
        raise VisionConductorError(
            f"{item_id}: answer keys differ from exact answer contract"
        )
    for key, expected_type in contract.items():
        value = answer[key]
        valid = {
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
        }[expected_type]
        if not valid:
            raise VisionConductorError(
                f"{item_id}: answer field {key!r} is not {expected_type}"
            )
        if expected_type in {"integer", "number"} and value < 0:
            raise VisionConductorError(
                f"{item_id}: answer field {key!r} must be non-negative"
            )


def validate_output(
    output_path: Path,
    packet: ValidatedPacket,
) -> dict[str, Any]:
    output = _read_json(output_path)
    if not isinstance(output, dict) or set(output) != EXPECTED_OUTPUT_KEYS:
        keys = sorted(output) if isinstance(output, dict) else []
        raise VisionConductorError(
            f"{packet.dataset_id}: output top-level keys differ: {keys}"
        )
    if (
        output.get("schema_version") != 1
        or output.get("packet_sha256") != packet.packet_sha256
        or output.get("dataset_id") != packet.dataset_id
        or output.get("generator_role") != GENERATOR_ROLE
        or output.get("visible_only") is not True
    ):
        raise VisionConductorError(
            f"{packet.dataset_id}: output identity/visibility contract differs"
        )
    results = output.get("results")
    if not isinstance(results, list):
        raise VisionConductorError(f"{packet.dataset_id}: results must be a list")
    expected_ids = [item["item_id"] for item in packet.items]
    result_ids = [
        item.get("item_id") if isinstance(item, dict) else None for item in results
    ]
    if result_ids != expected_ids:
        raise VisionConductorError(
            f"{packet.dataset_id}: result IDs/order/coverage differ from packet"
        )
    for result, source in zip(results, packet.items, strict=True):
        if not isinstance(result, dict) or set(result) != EXPECTED_RESULT_KEYS:
            raise VisionConductorError(
                f"{source['item_id']}: strict result schema differs"
            )
        abstain = result.get("abstain")
        explanation = result.get("explanation")
        if not isinstance(abstain, bool):
            raise VisionConductorError(
                f"{source['item_id']}: abstain must be boolean"
            )
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 1000
        ):
            raise VisionConductorError(
                f"{source['item_id']}: explanation must be 1..1000 characters"
            )
        _validate_answer(
            item_id=source["item_id"],
            abstain=abstain,
            answer=result.get("answer"),
            contract=source["answer_contract"],
        )
    return output


def _assignment(
    *,
    packet: ValidatedPacket,
    registry_sha256: str,
    return_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assignment_role": "independent_codex_visible_only_figure_evaluation",
        "dataset_id": packet.dataset_id,
        "generator_role": GENERATOR_ROLE,
        "visible_only": True,
        "main_inference": False,
        "registry_sha256": registry_sha256,
        "source_packet_sha256": packet.packet_sha256,
        "source_packet_file_sha256": packet.packet_file_sha256,
        "figure_manifest_sha256": packet.figure_manifest_sha256,
        "controller_metadata_in_scope": False,
        "may_read_unlisted_workspace_files": False,
        "may_modify_pipeline_or_source_artifacts": False,
        "prohibited_inputs": sorted(EXPECTED_PROHIBITED_INPUTS),
        "instructions": [
            (
                "Use only the PNG files explicitly listed in items. Do not inspect "
                "the workspace, packet source, benchmark, graph data, reports, gold "
                "answers, or outputs from any other condition."
            ),
            (
                "Visually inspect each PNG and answer only its paired question. If "
                "the requested values are not established by the visible image, "
                "abstain with answer=null."
            ),
            (
                "Return exactly one result for every item in the listed order. Do "
                "not add fields, commentary, Markdown, or modify any other file."
            ),
        ],
        "items": packet.items,
        "return_path": str(return_path.resolve()),
        "write_scope": "Only return_path may be created or replaced by the assigned visual subagent.",
        "strict_output_schema": {
            "schema_version": 1,
            "packet_sha256": packet.packet_sha256,
            "dataset_id": packet.dataset_id,
            "generator_role": GENERATOR_ROLE,
            "visible_only": True,
            "results": [
                {
                    "item_id": "<copy exact item_id in input order>",
                    "abstain": "<boolean>",
                    "answer": (
                        "<object matching answer_contract exactly, or null iff abstain>"
                    ),
                    "explanation": "<brief visible-image basis, 1..1000 characters>",
                }
            ],
        },
    }


def validate_all_packets(
    *,
    registry_path: Path,
    packet_root: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, str]], dict[str, ValidatedPacket]]:
    topics = load_frozen_topics(registry_path)
    packets = {
        topic["dataset_id"]: validate_packet(
            dataset_id=topic["dataset_id"],
            packet_root=packet_root,
            workspace_root=workspace_root,
        )
        for topic in topics
    }
    return topics, packets


def prepare_assignments(
    *,
    registry_path: Path,
    packet_root: Path,
    workspace_root: Path,
    output_root: Path,
    exchange_root: Path,
    returns_root: Path,
) -> dict[str, Any]:
    topics, packets = validate_all_packets(
        registry_path=registry_path,
        packet_root=packet_root,
        workspace_root=workspace_root,
    )
    registry_sha256 = sha256_file(registry_path)
    assignments: list[dict[str, Any]] = []
    complete: list[str] = []
    for topic in topics:
        dataset_id = topic["dataset_id"]
        packet = packets[dataset_id]
        official_output = (output_root / f"{dataset_id}.json").resolve()
        return_path = (returns_root / f"{dataset_id}.json").resolve()
        assignment_path = (exchange_root / dataset_id / "assignment.json").resolve()
        if official_output.exists():
            official = validate_output(official_output, packet)
            if return_path.exists():
                returned = validate_output(return_path, packet)
                if _canonical(returned) != _canonical(official):
                    raise VisionConductorError(
                        f"{dataset_id}: return differs from immutable official output"
                    )
            complete.append(dataset_id)
            continue
        assignment = _assignment(
            packet=packet,
            registry_sha256=registry_sha256,
            return_path=return_path,
        )
        _write_idempotent(assignment_path, assignment)
        assignments.append(
            {
                "dataset_id": dataset_id,
                "role": topic["role"],
                "assignment_path": str(assignment_path),
                "assignment_sha256": sha256_file(assignment_path),
                "return_path": str(return_path),
                "item_count": len(packet.items),
            }
        )
    return {
        "status": "assignments_ready" if assignments else "all_outputs_complete",
        "formal_topic_count": len(topics),
        "complete_output_count": len(complete),
        "missing_output_count": len(assignments),
        "complete_datasets": complete,
        "assignments": assignments,
        "score_command": score_command(
            topics=topics,
            output_root=output_root,
        ),
    }


def score_command(
    *,
    topics: list[dict[str, str]],
    output_root: Path,
    run_id: str = RUN_ID,
) -> list[str]:
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "score_formal_graph_runs.py"
        ),
    ]
    for topic in topics:
        command.extend(["--dataset", topic["dataset_id"]])
    command.extend(
        [
            "--run-id",
            run_id,
            "--vision-root",
            str(output_root.resolve()),
            "--include-vision",
        ]
    )
    return command


def import_returns(
    *,
    registry_path: Path,
    packet_root: Path,
    workspace_root: Path,
    output_root: Path,
    returns_root: Path,
    run_score: bool,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    topics, packets = validate_all_packets(
        registry_path=registry_path,
        packet_root=packet_root,
        workspace_root=workspace_root,
    )
    pending_writes: list[tuple[Path, bytes]] = []
    for topic in topics:
        dataset_id = topic["dataset_id"]
        packet = packets[dataset_id]
        official_path = (output_root / f"{dataset_id}.json").resolve()
        return_path = (returns_root / f"{dataset_id}.json").resolve()
        if official_path.exists():
            official = validate_output(official_path, packet)
            if return_path.exists():
                returned = validate_output(return_path, packet)
                if _canonical(returned) != _canonical(official):
                    raise VisionConductorError(
                        f"{dataset_id}: return differs from immutable official output"
                    )
            continue
        if not return_path.is_file():
            raise VisionConductorError(
                f"{dataset_id}: missing visual-subagent return: {return_path}"
            )
        returned = validate_output(return_path, packet)
        pending_writes.append((official_path, _json_bytes(returned)))

    # Every return is validated before the first official artifact is published.
    for target, payload in pending_writes:
        if target.exists():
            if target.read_bytes() != payload:
                raise VisionConductorError(
                    f"Refusing to overwrite different official output: {target}"
                )
            continue
        atomic_write_bytes(target, payload)

    output_hashes: dict[str, str] = {}
    for topic in topics:
        dataset_id = topic["dataset_id"]
        official_path = (output_root / f"{dataset_id}.json").resolve()
        validate_output(official_path, packets[dataset_id])
        output_hashes[dataset_id] = sha256_file(official_path)

    command = score_command(topics=topics, output_root=output_root, run_id=run_id)
    score_status = "prepared"
    if run_score:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise VisionConductorError(
                "Mechanical Figure/VLM scoring failed: "
                f"stdout={completed.stdout[-2000:]!r} "
                f"stderr={completed.stderr[-2000:]!r}"
            )
        score_status = "complete"
    return {
        "status": "all_outputs_complete",
        "formal_topic_count": len(topics),
        "newly_imported_count": len(pending_writes),
        "output_sha256": output_hashes,
        "mechanical_score_status": score_status,
        "score_command": command,
    }
