"""Materialize and freeze all 96 budget-matched control requests, with no API calls."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from citeweave.formal_request import build_messages, canonical_sha256, task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json
from citeweave.temporal_evidence_control import add_shared_temporal_context
from citeweave.token_budget import CommandTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
OUTPUT = BASE / "temporal_shared_evidence_control_v1"
QUALIFIED = BASE / "formal_v3_identity_corrected_execution/qualification.json"
FREEZE = OUTPUT / "execution_inputs_freeze.json"


def verify(bindings):
    for name, expected in bindings.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Control input drift: {name}")


def verify_annex_retention(messages, annex):
    rendered = messages[1]["content"].split("CONTEXT:\n", 1)[1]
    actual = json.loads(rendered)["shared_temporal_evidence"]
    if actual != annex:
        raise ValueError("Shared temporal evidence was changed or truncated")


def main():
    if FREEZE.exists():
        payload = read_json(FREEZE)
        verify(payload["bindings"])
        print(json.dumps({"status": payload["status"], "cells": 96}))
        return
    if not QUALIFIED.is_file():
        print(
            json.dumps(
                {"status": "blocked", "reason": "corrected_original_panels_not_yet_qualified"}
            )
        )
        raise SystemExit(2)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_identity_corrected_execution.py"),
            "--verify-qualified",
        ],
        cwd=ROOT,
        check=True,
    )
    design_freeze = OUTPUT / "protocol_freeze.json"
    bindings = read_json(design_freeze)["bindings"]
    verify(bindings)
    protocol = read_json(OUTPUT / "protocol.json")
    plan = read_json(OUTPUT / "source_plan.json")
    readiness_path = BASE / "formal_v3_identity_corrected_execution/extension_readiness.json"
    readiness = read_json(readiness_path)
    if readiness["status"] != "ready":
        raise ValueError("Original extension is not ready")
    tokenizer_path = Path(readiness["tokenizer_manifest_path"])
    tokenizer_payload = read_json(tokenizer_path)
    if tokenizer_payload != readiness["tokenizer_manifest"]:
        raise ValueError("Tokenizer identity drift")
    if (
        protocol["total_context_token_budget"] != tokenizer_payload["context_token_budget"]
        or protocol["model"] != tokenizer_payload["model"]
    ):
        raise ValueError("Shared control must retain original model and token budget")
    tokenizer = CommandTokenizer(tokenizer_payload, manifest_dir=tokenizer_path.parent)
    neural_manifest_path = Path(readiness["neural_dense_manifest_path"])
    neural = read_json(neural_manifest_path)
    if neural != readiness["neural_dense_manifest"]:
        raise ValueError("Neural manifest differs from original readiness")
    sidecars = {}
    for artifact in neural["artifacts"]:
        if artifact["artifact_type"] != "neural_context_sidecar":
            continue
        path = (neural_manifest_path.parent / artifact["path"]).resolve()
        bindings[str(path)] = artifact["sha256"]
        sidecars[artifact["dataset_id"]] = read_json(path)
    verify(bindings)
    requests, benchmarks = [], {}
    for dataset in plan["records"]:
        topic = dataset["dataset_id"]
        source_path = Path(dataset["source_benchmark"])
        if sha256_file(source_path) != dataset["source_benchmark_sha256"]:
            raise ValueError("Source benchmark drift")
        source_tasks = {t["item_id"]: t for t in read_json(source_path)["tasks"]}
        if sidecars[topic]["source_benchmark_sha256"] != sha256_file(source_path):
            raise ValueError("Neural sidecar not bound to source benchmark")
        annex = read_json(Path(dataset["annex_path"]))
        benchmarks[topic] = []
        for item in dataset["tasks"]:
            source = source_tasks[item["source_item_id"]]
            if canonical_sha256(source) != item["source_task_sha256"]:
                raise ValueError("Control source task changed")
            task = copy.deepcopy(source)
            task["item_id"] = item["item_id"]
            task["source_item_id"] = item["source_item_id"]
            task["evidence_ids"] = list(
                dict.fromkeys([*source["evidence_ids"], annex["evidence_id"]])
            )
            task["contexts"] = {}
            task.pop("context_hashes", None)
            for condition in protocol["conditions"]:
                original = (
                    sidecars[topic]["contexts"][item["source_item_id"]]
                    if condition == "flat_neural_dense"
                    else source["contexts"][condition]
                )
                task["contexts"][condition] = add_shared_temporal_context(original, annex)
            benchmarks[topic].append(task)
            for condition in protocol["conditions"]:
                messages, audit = build_messages(
                    task,
                    condition,
                    tokenizer=tokenizer,
                    token_budget=protocol["total_context_token_budget"],
                )
                verify_annex_retention(messages, annex)
                requests.append(
                    {
                        "dataset_id": topic,
                        "item_id": task["item_id"],
                        "source_item_id": item["source_item_id"],
                        "scale": task["scale"],
                        "condition": condition,
                        "messages": messages,
                        "task_payload_sha256": task_payload_sha256(task),
                        "messages_sha256": canonical_sha256(messages),
                        "annex_sha256": dataset["annex_sha256"],
                        "budget_audit": audit,
                    }
                )
        print(json.dumps({"topic": topic, "prepared_cells": len(requests)}), flush=True)
    if len(requests) != 96 or len({(r["item_id"], r["condition"]) for r in requests}) != 96:
        raise ValueError("Expected exactly 96 unique requests")
    request_path, benchmark_path = OUTPUT / "requests.json", OUTPUT / "benchmarks.json"
    write_json(request_path, {"schema_version": 1, "cells": requests})
    write_json(benchmark_path, {"schema_version": 1, "datasets": benchmarks})
    for path in (
        request_path,
        benchmark_path,
        design_freeze,
        QUALIFIED,
        readiness_path,
        tokenizer_path,
        neural_manifest_path,
        Path(__file__).resolve(),
        ROOT / "src/citeweave/formal_request.py",
        ROOT / "src/citeweave/token_budget.py",
        ROOT / "src/citeweave/temporal_evidence_control.py",
    ):
        bindings[str(path)] = sha256_file(path)
    verify(bindings)
    write_json(
        FREEZE,
        {
            "schema_version": 1,
            "status": "96_requests_frozen_no_provider_execution",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "bindings": bindings,
            "automatic_api_resume": False,
            "next_gate": "Explicit original-panel promotion plus a frozen control executor is required before API execution.",
        },
    )


if __name__ == "__main__":
    main()
