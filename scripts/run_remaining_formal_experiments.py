from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from citeweave.formal_graph_experiment import (
    verify_formal_graph_grounding,
)
from citeweave.formal_protocol import verify_frozen_query_registry
from citeweave.harvest_acceptance import verify_bulk_harvest
from citeweave.io import read_json, sha256_file
from citeweave.large_scale_evidence import verify_large_scale_evidence
from citeweave.processing_acceptance import verify_large_processing
from citeweave.visual_acceptance import verify_visualization

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets_openalex_title_abstract.yml"
DEFAULT_WORKSPACES = ROOT / "experiments" / "formal_workspaces"
DEFAULT_REPORTS = ROOT / "experiments" / "formal_reports"
DEFAULT_RUNS = ROOT / "experiments" / "formal_runs"
DEFAULT_PACKETS = ROOT / "experiments" / "vision_packets"
DEFAULT_TEXT_PROFILE = ROOT / "experiments" / "provider_profiles" / "deepseek_v4_pro.json"
FORMAL_GRAPH_RUN_ID = "formal_v2_nonthinking_20260806"
TEXT_GRAPH_CONDITIONS = ("no_rag", "flat_structured", "graph_rag")
REPORT_CONDITIONS = ("structured_one_shot", "citeweave_full")

StepKind = Literal["pipeline", "report", "graph_text", "vlm_packet"]


class FormalOrchestrationError(RuntimeError):
    """Raised when an immutable formal artifact fails its contract."""


@dataclass(frozen=True)
class PlannedStep:
    dataset_id: str
    stage: str
    kind: StepKind
    command: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "stage": self.stage,
            "kind": self.kind,
            "reason": self.reason,
            "command": list(self.command),
        }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_pass(stage: str, dataset_id: str, result: dict[str, Any]) -> None:
    if not result.get("passed"):
        raise FormalOrchestrationError(
            f"{dataset_id}: {stage} acceptance failed: "
            f"{json.dumps(result, ensure_ascii=False, default=str)}"
        )


def load_frozen_dataset_ids(registry_path: Path) -> list[str]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    verify_frozen_query_registry(registry)
    entries = registry["datasets"]
    dataset_ids = [str(item["id"]) for item in entries]
    if len(dataset_ids) != 8 or len(set(dataset_ids)) != 8:
        raise FormalOrchestrationError(
            "The frozen formal registry must contain exactly eight unique datasets."
        )
    return dataset_ids


def _pipeline_command(
    registry: Path,
    dataset_id: str,
    stage: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "run_formal_pipeline.py"),
        "--registry",
        str(registry),
        "--dataset",
        dataset_id,
        "--stage",
        stage,
    )


def _report_command(
    dataset_id: str,
    condition: str,
    evidence_path: Path,
    reports_root: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "run_report_conditions.py"),
        "--dataset-id",
        dataset_id,
        "--evidence",
        str(evidence_path),
        "--output-root",
        str(reports_root),
        "--condition",
        condition,
        "--model",
        "deepseek-v4-pro",
        "--resume",
    )


def _graph_command(
    dataset_id: str,
    condition: str,
    text_profile: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "run_formal_graph_experiment.py"),
        "--dataset",
        dataset_id,
        "--condition",
        condition,
        "--text-profile",
        str(text_profile),
        "--run-id",
        FORMAL_GRAPH_RUN_ID,
        "--execute",
    )


def _packet_command(
    dataset_id: str,
    workspaces_root: Path,
    packets_root: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "prepare_codex_figure_vlm_packet.py"),
        "--dataset",
        dataset_id,
        "--workspace-root",
        str(workspaces_root),
        "--output-root",
        str(packets_root),
    )


def _completed_or_pending(
    *,
    dataset_id: str,
    stage: str,
    marker: Path,
    verifier: Callable[[Path], dict[str, Any]],
    workspace: Path,
    resumable_status: bool = False,
) -> bool:
    """Return True only for a verified complete artifact.

    Missing markers and explicitly incomplete processing checkpoints are resumable.
    A marker claiming completion is immutable and therefore fails closed if verification
    does not pass.
    """
    if not marker.exists():
        return False
    if resumable_status:
        manifest = read_json(marker)
        if manifest.get("status") != "complete":
            return False
    result = verifier(workspace)
    _require_pass(stage, dataset_id, result)
    return True


def verify_report_completion(
    reports_root: Path,
    dataset_id: str,
    condition: str,
    evidence_path: Path,
) -> bool:
    condition_dir = reports_root / dataset_id / condition
    completion_path = condition_dir / "completion.json"
    if not completion_path.exists():
        return False
    completion = read_json(completion_path)
    report_path = condition_dir / "report.md"
    expected_calls = 1 if condition == "structured_one_shot" else 4
    call_files = sorted((condition_dir / "calls").glob("*/call.json"))
    errors: list[str] = []
    if completion.get("status") != "complete":
        errors.append("completion status is not complete")
    if completion.get("dataset_id") != dataset_id:
        errors.append("dataset id mismatch")
    if completion.get("condition") != condition:
        errors.append("condition mismatch")
    if completion.get("model") != "deepseek-v4-pro":
        errors.append("model mismatch")
    if completion.get("report_language") != "English":
        errors.append("report language is not English")
    if completion.get("call_count") != expected_calls or len(call_files) != expected_calls:
        errors.append("completed call count mismatch")
    if not report_path.is_file():
        errors.append("report is missing")
    elif completion.get("report_sha256") != sha256_file(report_path):
        errors.append("report hash mismatch")
    if not evidence_path.is_file():
        errors.append("evidence is missing")
    elif completion.get("evidence_sha256") != sha256_file(evidence_path):
        errors.append("evidence hash mismatch")
    if errors:
        raise FormalOrchestrationError(
            f"{dataset_id}: immutable report condition {condition} is invalid: "
            + "; ".join(errors)
        )
    return True


def _latest_graph_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    completed: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = (str(record["condition"]), str(record["item_id"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FormalOrchestrationError(
                f"Malformed graph checkpoint at {path}:{line_number}"
            ) from exc
        if key in completed:
            raise FormalOrchestrationError(
                f"Graph checkpoint contains a record after completion: {key}"
            )
        latest[key] = record
        if record.get("status") == "complete":
            completed.add(key)
    return latest


def verify_graph_run_completion(
    runs_root: Path,
    workspace: Path,
    dataset_id: str,
    condition: str,
    text_profile: Path,
) -> bool:
    run_dir = runs_root / dataset_id / FORMAL_GRAPH_RUN_ID / condition
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "items.jsonl"
    if not manifest_path.exists():
        if checkpoint_path.exists():
            raise FormalOrchestrationError(
                f"{dataset_id}: graph checkpoint exists without a run manifest: {condition}"
            )
        return False

    manifest = read_json(manifest_path)
    benchmark_path = workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
    grounding_path = workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
    profile = read_json(text_profile)
    benchmark = read_json(benchmark_path)
    expected_ids = [str(item["item_id"]) for item in benchmark]
    errors: list[str] = []
    expected_fields = {
        "run_id": FORMAL_GRAPH_RUN_ID,
        "dataset_id": dataset_id,
        "condition": condition,
        "prompt_version": "formal-graph-qa-v2-nonthinking",
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            errors.append(f"{field} mismatch")
    actual_profile = manifest.get("profile") or {}
    for field in ("profile_id", "base_url", "model", "api_key_env", "modality"):
        if actual_profile.get(field) != profile.get(field):
            errors.append(f"profile {field} mismatch")
    if manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
        errors.append("benchmark hash mismatch")
    if manifest.get("grounding_manifest_sha256") != sha256_file(grounding_path):
        errors.append("grounding manifest hash mismatch")
    if manifest.get("item_ids") != expected_ids or manifest.get("items") != len(expected_ids):
        errors.append("benchmark item contract mismatch")
    if errors:
        raise FormalOrchestrationError(
            f"{dataset_id}: immutable graph run manifest is invalid for {condition}: "
            + "; ".join(errors)
        )
    if not checkpoint_path.exists():
        return False
    latest = _latest_graph_records(checkpoint_path)
    expected_keys = {(condition, item_id) for item_id in expected_ids}
    unexpected = set(latest) - expected_keys
    if unexpected:
        raise FormalOrchestrationError(
            f"{dataset_id}: graph checkpoint has unexpected items for {condition}: "
            f"{sorted(unexpected)}"
        )
    return all(
        latest.get(key, {}).get("status") == "complete" for key in expected_keys
    )


def verify_vlm_packet(
    packets_root: Path,
    workspace: Path,
    dataset_id: str,
) -> bool:
    packet_path = packets_root / f"{dataset_id}.json"
    if not packet_path.exists():
        return False
    packet = read_json(packet_path)
    benchmark_path = workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
    benchmark = read_json(benchmark_path)
    expected_ids = [
        str(item["item_id"]) for item in benchmark if item.get("figure_eligible")
    ]
    unsigned = dict(packet)
    recorded_hash = unsigned.pop("packet_sha256", None)
    errors: list[str] = []
    if packet.get("dataset_id") != dataset_id:
        errors.append("dataset id mismatch")
    if packet.get("benchmark_sha256") != sha256_file(benchmark_path):
        errors.append("benchmark hash mismatch")
    if [str(item.get("item_id")) for item in packet.get("items", [])] != expected_ids:
        errors.append("eligible item ids mismatch")
    if recorded_hash != _sha(unsigned):
        errors.append("packet hash mismatch")
    if errors:
        raise FormalOrchestrationError(
            f"{dataset_id}: immutable Figure/VLM packet is invalid: " + "; ".join(errors)
        )
    return True


def build_plan(
    *,
    registry: Path,
    dataset_ids: list[str],
    workspaces_root: Path,
    reports_root: Path,
    runs_root: Path,
    packets_root: Path,
    text_profile: Path,
    skip_vlm_packets: bool,
) -> tuple[list[PlannedStep], list[dict[str, str]]]:
    steps: list[PlannedStep] = []
    skipped: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        workspace = workspaces_root / dataset_id
        process_done = _completed_or_pending(
            dataset_id=dataset_id,
            stage="processing",
            marker=workspace / "audit" / "processing_manifest.json",
            verifier=verify_large_processing,
            workspace=workspace,
            resumable_status=True,
        )
        if process_done:
            skipped.append({"dataset_id": dataset_id, "stage": "process"})
        else:
            steps.append(
                PlannedStep(
                    dataset_id,
                    "process",
                    "pipeline",
                    _pipeline_command(registry, dataset_id, "process"),
                    "missing or resumable incomplete processing artifact",
                )
            )

        visual_done = _completed_or_pending(
            dataset_id=dataset_id,
            stage="visualization",
            marker=workspace / "figures" / "figure_manifest.json",
            verifier=verify_visualization,
            workspace=workspace,
        )
        if visual_done:
            skipped.append({"dataset_id": dataset_id, "stage": "visualize"})
        else:
            steps.append(
                PlannedStep(
                    dataset_id,
                    "visualize",
                    "pipeline",
                    _pipeline_command(registry, dataset_id, "visualize"),
                    "visualization artifact is missing",
                )
            )

        evidence_marker = workspace / "audit" / "evidence_preparation_manifest.json"
        evidence_done = _completed_or_pending(
            dataset_id=dataset_id,
            stage="evidence",
            marker=evidence_marker,
            verifier=verify_large_scale_evidence,
            workspace=workspace,
        )
        graph_marker = workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
        graph_done = _completed_or_pending(
            dataset_id=dataset_id,
            stage="formal graph grounding",
            marker=graph_marker,
            verifier=verify_formal_graph_grounding,
            workspace=workspace,
        )
        if evidence_done:
            skipped.append({"dataset_id": dataset_id, "stage": "evidence"})
        if graph_done:
            skipped.append({"dataset_id": dataset_id, "stage": "graph_grounding"})
        if not evidence_done:
            if graph_done:
                raise FormalOrchestrationError(
                    f"{dataset_id}: graph grounding exists without verified evidence"
                )
            steps.append(
                PlannedStep(
                    dataset_id,
                    "evidence_and_graph_grounding",
                    "pipeline",
                    _pipeline_command(registry, dataset_id, "evidence"),
                    "evidence and graph-grounding artifacts are missing",
                )
            )
        elif not graph_done:
            steps.append(
                PlannedStep(
                    dataset_id,
                    "graph_grounding",
                    "pipeline",
                    _pipeline_command(registry, dataset_id, "graph"),
                    "verified evidence exists but graph grounding is missing",
                )
            )

        evidence_path = workspace / "evidence" / "evidence_items.json"
        for condition in REPORT_CONDITIONS:
            if verify_report_completion(
                reports_root, dataset_id, condition, evidence_path
            ):
                skipped.append({"dataset_id": dataset_id, "stage": f"report:{condition}"})
            else:
                steps.append(
                    PlannedStep(
                        dataset_id,
                        f"report:{condition}",
                        "report",
                        _report_command(
                            dataset_id,
                            condition,
                            evidence_path,
                            reports_root,
                        ),
                        "report condition is missing or resumable incomplete",
                    )
                )

        for condition in TEXT_GRAPH_CONDITIONS:
            if verify_graph_run_completion(
                runs_root,
                workspace,
                dataset_id,
                condition,
                text_profile,
            ):
                skipped.append({"dataset_id": dataset_id, "stage": f"graph:{condition}"})
            else:
                steps.append(
                    PlannedStep(
                        dataset_id,
                        f"graph:{condition}",
                        "graph_text",
                        _graph_command(dataset_id, condition, text_profile),
                        "graph text condition is missing or resumable incomplete",
                    )
                )

        if skip_vlm_packets:
            skipped.append({"dataset_id": dataset_id, "stage": "vlm_packet:user_skipped"})
        elif verify_vlm_packet(packets_root, workspace, dataset_id):
            skipped.append({"dataset_id": dataset_id, "stage": "vlm_packet"})
        else:
            steps.append(
                PlannedStep(
                    dataset_id,
                    "vlm_packet",
                    "vlm_packet",
                    _packet_command(dataset_id, workspaces_root, packets_root),
                    "Figure/VLM packet is missing",
                )
            )
    return steps, skipped


def _verify_after_step(
    step: PlannedStep,
    *,
    workspaces_root: Path,
    reports_root: Path,
    runs_root: Path,
    packets_root: Path,
    text_profile: Path,
) -> None:
    workspace = workspaces_root / step.dataset_id
    if step.kind == "pipeline":
        if step.stage == "process":
            _require_pass("processing", step.dataset_id, verify_large_processing(workspace))
        elif step.stage == "visualize":
            _require_pass("visualization", step.dataset_id, verify_visualization(workspace))
        elif step.stage == "evidence_and_graph_grounding":
            _require_pass("evidence", step.dataset_id, verify_large_scale_evidence(workspace))
            _require_pass(
                "formal graph grounding",
                step.dataset_id,
                verify_formal_graph_grounding(workspace),
            )
        elif step.stage == "graph_grounding":
            _require_pass(
                "formal graph grounding",
                step.dataset_id,
                verify_formal_graph_grounding(workspace),
            )
    elif step.kind == "report":
        condition = step.stage.removeprefix("report:")
        if not verify_report_completion(
            reports_root,
            step.dataset_id,
            condition,
            workspace / "evidence" / "evidence_items.json",
        ):
            raise FormalOrchestrationError("Report subprocess returned without completion.")
    elif step.kind == "graph_text":
        condition = step.stage.removeprefix("graph:")
        if not verify_graph_run_completion(
            runs_root,
            workspace,
            step.dataset_id,
            condition,
            text_profile,
        ):
            raise FormalOrchestrationError("Graph subprocess returned without completion.")
    elif not verify_vlm_packet(packets_root, workspace, step.dataset_id):
        raise FormalOrchestrationError("Packet subprocess returned without an artifact.")


def execute_plan(
    steps: list[PlannedStep],
    *,
    dataset_ids: list[str],
    workspaces_root: Path,
    reports_root: Path,
    runs_root: Path,
    packets_root: Path,
    text_profile: Path,
) -> None:
    # Gate every selected corpus before starting any downstream work.
    for dataset_id in dataset_ids:
        _require_pass(
            "harvest",
            dataset_id,
            verify_bulk_harvest(workspaces_root / dataset_id),
        )
    environment = os.environ.copy()
    for step in steps:
        subprocess.run(
            list(step.command),
            cwd=ROOT,
            env=environment,
            check=True,
        )
        _verify_after_step(
            step,
            workspaces_root=workspaces_root,
            reports_root=reports_root,
            runs_root=runs_root,
            packets_root=packets_root,
            text_profile=text_profile,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the remaining immutable formal experiment stages after full harvest."
        )
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-vlm-packets", action="store_true")
    parser.add_argument("--text-profile", type=Path, default=DEFAULT_TEXT_PROFILE)
    args = parser.parse_args()

    registry = args.registry.resolve()
    frozen_ids = load_frozen_dataset_ids(registry)
    requested = list(dict.fromkeys(args.dataset or frozen_ids))
    unknown = sorted(set(requested) - set(frozen_ids))
    if unknown:
        raise SystemExit(f"Unknown dataset ids: {unknown}")
    # Preserve frozen registry order even if --dataset is repeated in another order.
    selected = [dataset_id for dataset_id in frozen_ids if dataset_id in requested]
    text_profile = args.text_profile.resolve()
    if not text_profile.is_file():
        raise SystemExit(f"Text provider profile is missing: {text_profile}")

    steps, skipped = build_plan(
        registry=registry,
        dataset_ids=selected,
        workspaces_root=DEFAULT_WORKSPACES,
        reports_root=DEFAULT_REPORTS,
        runs_root=DEFAULT_RUNS,
        packets_root=DEFAULT_PACKETS,
        text_profile=text_profile,
        skip_vlm_packets=args.skip_vlm_packets,
    )
    summary = {
        "dry_run": args.dry_run,
        "registry": str(registry),
        "frozen_dataset_count": len(frozen_ids),
        "selected_dataset_ids": selected,
        "formal_graph_run_id": FORMAL_GRAPH_RUN_ID,
        "text_profile": str(text_profile),
        "planned_steps": [step.as_dict() for step in steps],
        "verified_skips": skipped,
        "vlm_output_policy": (
            "Packets only; Figure/VLM answers remain a Codex visual-subagent task."
        ),
        "execution_preflight": (
            "A dry run does not claim harvest acceptance. A real run verifies every "
            "selected harvest before starting its first downstream subprocess."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    execute_plan(
        steps,
        dataset_ids=selected,
        workspaces_root=DEFAULT_WORKSPACES,
        reports_root=DEFAULT_REPORTS,
        runs_root=DEFAULT_RUNS,
        packets_root=DEFAULT_PACKETS,
        text_profile=text_profile,
    )


if __name__ == "__main__":
    main()
