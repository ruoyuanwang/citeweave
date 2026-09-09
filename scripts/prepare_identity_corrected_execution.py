"""Qualify corrected panels without API calls; promotion is a separate explicit action.

The legacy integrity hold is intentionally retained. Only the corrected controller
accepts a promotion receipt. Preparing or finishing indexes cannot authorize API use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
EXECUTION = BASE / "formal_v3_identity_corrected_execution"
CONFIG = EXECUTION / "config.json"
FREEZE = EXECUTION / "input_freeze.json"
QUALIFICATION = EXECUTION / "qualification.json"
PROMOTION = EXECUTION / "promotion.json"
HOLD = BASE / "formal_v3_author_identity_integrity_hold.json"
CORRECTION = BASE / "formal_v3_amendment_004_author_identity.yml"
CORRECTION_SHA = "1d58984df27e71f37b2db6cc11aaa5f59446212adc24220d7a41743c587e622c"
TOKENIZER_IO_AMENDMENT = BASE / "formal_v3_execution_amendment_009_utf8_tokenizer_io.yml"
TOKENIZER_IO_AMENDMENT_SHA = (
    "b2ca5554c230abf7792707cc96f37d6653bf4b384916256694bc68b30858a16d"
)
TOKENIZER_IO_AMENDMENT_FREEZE = (
    BASE / "formal_v3_execution_amendment_009_utf8_tokenizer_io_freeze.json"
)
FAILURE_EVIDENCE = (
    EXECUTION / "failures" / "qualification_failure_20260901_1413"
)
PRIOR_INPUT_FREEZE_SHA = (
    "6fe178ffdda0b37894cdef5795a03d3412415cc992c1249598f9a7f7b3257a2b"
)
PANELS = ("primary", "replication", "extension")
RESULT_ROOTS = tuple(
    BASE / name
    for name in (
        "formal_v3_runs",
        "formal_v3_replication_runs",
        "formal_v3_complexity_extension_runs",
    )
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def immutable_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def bind(paths) -> dict[str, str]:
    result = {}
    for value in sorted({Path(p).resolve() for p in paths}):
        if value.name.lower() == "apikey.md":
            raise ValueError("Credential material must never enter execution bindings")
        result[str(value)] = sha256_file(value)
    return result


def verify(bindings: dict[str, str]) -> None:
    for value, expected in bindings.items():
        path = Path(value)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Bound input drift: {path}")


def require_no_outcomes(roots=RESULT_ROOTS) -> None:
    # These directories are dedicated exclusively to provider outcomes.
    for root in roots:
        if root.exists() and any(p.is_file() for p in root.rglob("*")):
            raise ValueError(f"Prospective preparation/promotion requires empty outcomes: {root}")


def option(command: list[str], key: str) -> Path:
    return (ROOT / command[command.index(key) + 1]).resolve()


def validate_command(command: list[str]) -> None:
    if not command or "--execute" in command or "--api-key-file" in command:
        raise ValueError("Preparation commands must not invoke provider execution")
    allowed = {
        "audit_formal_v3_readiness.py",
        "audit_formal_v3_replication_readiness.py",
        "audit_formal_v3_complexity_extension_readiness.py",
        "audit_identity_corrected_data.py",
        "audit_identity_corrected_reuse.py",
        "run_formal_v3_panel.py",
        "run_formal_v3_replication_panel.py",
        "run_formal_v3_complexity_extension_panel.py",
    }
    if Path(command[0]).name not in allowed:
        raise ValueError("Unapproved preparation command")


def run(command: list[str], label: str) -> None:
    validate_command(command)
    print(json.dumps({"stage": label, "started_at_utc": now()}), flush=True)
    with (EXECUTION / f"{label}.log").open("w", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            shell=False,
        )


def artifact_bindings(manifest_path: Path) -> dict[str, str]:
    payload = read_json(manifest_path)
    if not payload.get("passed"):
        raise ValueError(f"Manifest not passed: {manifest_path}")
    result = bind([manifest_path])
    for item in payload.get("artifacts", []):
        path = (manifest_path.parent / item["path"]).resolve()
        result[str(path)] = item["sha256"]
    for item in payload.get("embedding_indexes", []):
        path = (ROOT / item["manifest_path"]).resolve()
        result[str(path)] = item["manifest_sha256"]
        result[str(path.parent / "embeddings.npy")] = item["embeddings_sha256"]
    probe = payload.get("api_usage_probe_artifact")
    if probe:
        result[str((manifest_path.parent / probe["path"]).resolve())] = probe["sha256"]
    verify(result)
    return result


def corrected_source_bindings(config: dict) -> dict[str, str]:
    result = {}
    primary = option(config["readiness_commands"][0], "--benchmark-root")
    records = read_json(primary / "construction_manifest.json")["records"]
    if len(records) != 4:
        raise ValueError("Four corrected primary topics required")
    for record in records:
        folder = primary / record["dataset_id"]
        receipt_path = folder / "author_identity_rebuild_receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt["status"] != "rebuilt_not_promoted"
            or receipt["author_identity_correction_sha256"] != CORRECTION_SHA
            or receipt["benchmark_sha256"] != record["benchmark_sha256"]
            or receipt["code_sha256"]
            != sha256_file(ROOT / "src/citeweave/author_benchmark_rebuild.py")
            or receipt["task_comparisons"]["unchanged_tasks"] != 20
            or receipt["task_comparisons"]["rebuilt_author_tasks"] != 5
            or receipt["operator_replay"]["valid_tasks"] != 5
            or receipt["operator_replay"]["invalid_tasks"] != 0
            or len(receipt["independent_arithmetic"]) != 3
            or not all(r["passed"] for r in receipt["independent_arithmetic"])
        ):
            raise ValueError(f"Corrected task evidence failed: {folder.name}")
        result.update(receipt["source_sha256"])
        result[str((folder / "benchmark.json").resolve())] = record["benchmark_sha256"]
        result.update(bind([receipt_path]))
    verify(result)
    return result


def freeze_inputs() -> None:
    if FREEZE.exists():
        verify(read_json(FREEZE)["bindings"])
        return
    require_no_outcomes()
    if sha256_file(CORRECTION) != CORRECTION_SHA or not HOLD.is_file():
        raise ValueError("Frozen correction and legacy integrity hold required")
    tokenizer_io_freeze = read_json(TOKENIZER_IO_AMENDMENT_FREEZE)
    if (
        sha256_file(TOKENIZER_IO_AMENDMENT) != TOKENIZER_IO_AMENDMENT_SHA
        or tokenizer_io_freeze.get("sha256") != TOKENIZER_IO_AMENDMENT_SHA
        or tokenizer_io_freeze.get("prior_input_freeze_sha256")
        != PRIOR_INPUT_FREEZE_SHA
    ):
        raise ValueError("Frozen UTF-8 tokenizer I/O amendment required")
    prior_freeze = FAILURE_EVIDENCE / "input_freeze.json"
    if sha256_file(prior_freeze) != PRIOR_INPUT_FREEZE_SHA:
        raise ValueError("Archived prior input freeze drift")
    config = read_json(CONFIG)
    if config["automatic_api_resume"] is not False:
        raise ValueError("Amendment forbids automatic API resume")
    paths = {
        CONFIG,
        CORRECTION,
        HOLD,
        TOKENIZER_IO_AMENDMENT,
        TOKENIZER_IO_AMENDMENT_FREEZE,
    }
    paths.update(p for p in FAILURE_EVIDENCE.glob("*") if p.is_file())
    paths.update(BASE.glob("formal_v3*.yml"))
    paths.update(BASE.glob("formal_v3*_freeze.json"))
    paths.update((BASE / "formal_v3_identity_corrected_statistics_v2").glob("*.*"))
    ignore = {
        "--execution-root",
        "--results-root",
        "--output",
        "--output-plan",
        "--readiness",
        "--neural-manifest",
    }
    commands = config["readiness_commands"] + config["plan_commands"]
    for command in commands:
        validate_command(command)
        paths.add(ROOT / command[0])
        for index in range(1, len(command), 2):
            if command[index] in ignore:
                continue
            path = ROOT / command[index + 1]
            if path.is_dir():
                paths.update(p for p in path.rglob("*") if p.is_file())
            else:
                paths.add(path)
    script_names = (
        "prepare_identity_corrected_execution.py",
        "audit_identity_corrected_reuse.py",
        "audit_identity_corrected_data.py",
        "run_identity_corrected_formal_pipeline.ps1",
        "qualify_identity_corrected_after_neural.ps1",
        "run_identity_corrected_neural_prerequisites.ps1",
        "build_neural_dense_sidecars.py",
        "run_graph_discovery_experiment.py",
        "audit_formal_panel_terminal.py",
        "merge_formal_v3_complexity_extension_results.py",
        "analyze_formal_v3_results.py",
        "analyze_formal_v3_scale_replication.py",
        "analyze_formal_v3_complexity_extension.py",
    )
    paths.update(ROOT / "scripts" / name for name in script_names)
    module_names = (
        "io.py",
        "graph_discovery.py",
        "formal_request.py",
        "token_budget.py",
        "formal_protocol_amendment.py",
        "neural_dense_runtime.py",
        "neural_index_audit.py",
        "panel_terminal_audit.py",
        "author_benchmark_rebuild.py",
        "transform.py",
        "bulk_processing.py",
        "processing_acceptance.py",
    )
    paths.update(ROOT / "src/citeweave" / name for name in module_names)
    bindings = bind(paths)
    bindings.update(corrected_source_bindings(config))
    tokenizer = option(config["readiness_commands"][0], "--tokenizer-manifest")
    bindings.update(artifact_bindings(tokenizer))
    immutable_json(
        FREEZE,
        {
            "schema_version": 1,
            "created_at_utc": now(),
            "automatic_api_resume": False,
            "bindings": bindings,
        },
    )


def prepare() -> None:
    if QUALIFICATION.exists():
        verify_qualification()
        return
    verify(read_json(FREEZE)["bindings"])
    require_no_outcomes()
    config = read_json(CONFIG)
    data_path = EXECUTION / "data_readiness.json"
    run(
        [
            "scripts/audit_identity_corrected_data.py",
            "--corrected-root",
            "experiments/formal_v3_identity_corrected_workspaces",
            "--output",
            str(data_path),
        ],
        "data_readiness",
    )
    if read_json(data_path)["status"] != "scoped_data_invariants_passed":
        raise ValueError("Fresh data invariants failed")
    runtime_bindings = {}
    for name, command in zip(PANELS, config["readiness_commands"], strict=True):
        if "--neural-manifest" in command:
            runtime_bindings.update(artifact_bindings(option(command, "--neural-manifest")))
        run(command, f"{name}_readiness")
        if read_json(option(command, "--output"))["status"] != "ready":
            raise ValueError(f"{name} readiness failed")
    for name, command, count_key, count in zip(
        PANELS,
        config["plan_commands"],
        ("calls", "calls", "incremental_calls"),
        (800, 120, 312),
        strict=True,
    ):
        run(command, f"{name}_plan")
        plan = read_json(option(command, "--output-plan"))
        if plan["status"] != "ready_to_execute" or plan[count_key] != count:
            raise ValueError(f"{name} plan count/status mismatch")
    reuse_path = EXECUTION / "reuse_preflight.json"
    reuse_command = ["scripts/audit_identity_corrected_reuse.py"]
    for name, command in zip(PANELS, config["readiness_commands"], strict=True):
        reuse_command += [
            f"--{name}-root",
            str(option(command, "--benchmark-root")),
            f"--{name}-readiness",
            str(option(command, "--output")),
        ]
    run(reuse_command + ["--output", str(reuse_path)], "reuse_preflight")
    if read_json(reuse_path)["status"] != "all_360_reuse_identities_equal":
        raise ValueError("Reuse identity preflight failed")
    files = [FREEZE, data_path, reuse_path]
    files += [option(c, "--output") for c in config["readiness_commands"]]
    files += [option(c, "--output-plan") for c in config["plan_commands"]]
    runtime_bindings.update(bind(files))
    verify(read_json(FREEZE)["bindings"])
    verify(runtime_bindings)
    require_no_outcomes()
    immutable_json(
        QUALIFICATION,
        {
            "schema_version": 1,
            "status": "qualified_awaiting_explicit_promotion",
            "created_at_utc": now(),
            "automatic_api_resume": False,
            "formal_provider_outcomes": 0,
            "bindings": runtime_bindings,
            "next_action": "Inspect readiness and reuse audits, then explicitly promote this qualification hash.",
        },
    )


def verify_qualification() -> None:
    qualification = read_json(QUALIFICATION)
    if qualification.get("status") != "qualified_awaiting_explicit_promotion":
        raise ValueError("Invalid qualification status")
    verify(qualification["bindings"])
    verify(read_json(FREEZE)["bindings"])


def promote(reviewed_sha: str, reviewed_by: str) -> None:
    if not reviewed_by.strip() or reviewed_sha != sha256_file(QUALIFICATION):
        raise ValueError("Explicit reviewer and exact reviewed qualification hash required")
    verify_qualification()
    require_no_outcomes()
    archive = EXECUTION / "legacy_integrity_hold_archived.json"
    if archive.exists():
        if sha256_file(archive) != sha256_file(HOLD):
            raise ValueError("Legacy hold archive drift")
    else:
        archive.write_bytes(HOLD.read_bytes())
    immutable_json(
        PROMOTION,
        {
            "schema_version": 1,
            "status": "explicitly_promoted_corrected_inputs_only",
            "created_at_utc": now(),
            "qualification_sha256": reviewed_sha,
            "reviewed_by": reviewed_by,
            "review_type": "execution_integrity_not_article_human_evaluation",
            "archived_hold_sha256": sha256_file(archive),
            "legacy_entrypoint_remains_blocked": True,
            "correction_amendment_sha256": CORRECTION_SHA,
            "tokenizer_io_amendment_sha256": TOKENIZER_IO_AMENDMENT_SHA,
        },
    )


def verify_promoted() -> None:
    promotion = read_json(PROMOTION)
    if (
        promotion.get("status") != "explicitly_promoted_corrected_inputs_only"
        or promotion.get("qualification_sha256") != sha256_file(QUALIFICATION)
        or promotion.get("correction_amendment_sha256") != CORRECTION_SHA
        or promotion.get("tokenizer_io_amendment_sha256")
        != TOKENIZER_IO_AMENDMENT_SHA
        or promotion.get("archived_hold_sha256")
        != sha256_file(EXECUTION / "legacy_integrity_hold_archived.json")
    ):
        raise ValueError("Explicit promotion receipt invalid")
    verify_qualification()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze-only", "prepare", "verify-qualified", "promote", "verify-promoted"):
        group.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--reviewed-qualification-sha256")
    parser.add_argument("--reviewed-by")
    args = parser.parse_args()
    if args.freeze_only:
        freeze_inputs()
    elif args.prepare:
        prepare()
    elif args.verify_qualified:
        verify_qualification()
    elif args.promote:
        promote(args.reviewed_qualification_sha256 or "", args.reviewed_by or "")
    else:
        verify_promoted()
    print(json.dumps({"status": "ok", "finished_at_utc": now()}), flush=True)


if __name__ == "__main__":
    main()
