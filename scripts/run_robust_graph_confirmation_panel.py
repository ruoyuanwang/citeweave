from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from citeweave.confirmation_panel import (
    _freeze_request_identities,
    _verify_tokenizer,
)
from citeweave.formal_request import canonical_sha256
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            for separator in ("=", ":"):
                key, found, candidate = line.partition(separator)
                if found and key.strip().casefold() == "deepseek" and candidate.strip():
                    return candidate.strip()
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def _balance(base_url: str, key: str) -> dict[str, Any]:
    response = httpx.get(
        base_url.rstrip("/") + "/user/balance",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "is_available": payload.get("is_available") is True,
        "balances": [
            {
                field: row.get(field)
                for field in (
                    "currency",
                    "total_balance",
                    "granted_balance",
                    "topped_up_balance",
                )
            }
            for row in payload.get("balance_infos") or []
        ],
    }


def validate_confirmation_inputs(
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[str, ...]]:
    freeze = read_json(freeze_path)
    if freeze.get("status") != "ready_for_independent_confirmation_readiness_audit":
        raise ValueError("Post-selection freeze is not ready")
    if freeze.get("confirmatory") is not True:
        raise ValueError("Execution freeze is not confirmatory")
    protocol_path = Path(freeze["protocol"])
    if sha256_file(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("Confirmation protocol changed after panel construction")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    for raw_path, expected in freeze.get("implementation_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Confirmation implementation drift: {raw_path}")
    for raw_path, expected in freeze.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Confirmation source drift: {raw_path}")
    construction_path = Path(freeze["construction_manifest"])
    request_path = Path(freeze["request_identity_manifest"])
    if sha256_file(construction_path) != freeze["construction_manifest_sha256"]:
        raise ValueError("Confirmation construction manifest drift")
    if sha256_file(request_path) != freeze["request_identity_manifest_sha256"]:
        raise ValueError("Confirmation request identity manifest drift")
    construction = read_json(construction_path)
    requests = read_json(request_path)
    if (
        construction.get("confirmatory") is not True
        or construction.get("topics") != 8
        or construction.get("tasks") != 40
        or construction.get("planned_calls") != 200
        or construction.get("robustness_cases") != 144
    ):
        raise ValueError("Confirmation construction panel size mismatch")
    for record in construction["records"]:
        benchmark = Path(record["benchmark"])
        if not benchmark.is_file() or sha256_file(benchmark) != record["benchmark_sha256"]:
            raise ValueError(f"Confirmation benchmark drift: {record['dataset_id']}")
        for case in record["robustness_cases"]:
            path = Path(case["path"])
            if not path.is_file() or sha256_file(path) != case["sha256"]:
                raise ValueError(
                    f"Confirmation robustness case drift: {record['dataset_id']}"
                )
    tokenizer_path = Path(requests["tokenizer_manifest"])
    if sha256_file(tokenizer_path) != freeze["tokenizer_manifest_sha256"]:
        raise ValueError("Confirmation tokenizer manifest drift")
    _, tokenizer = _verify_tokenizer(tokenizer_path)
    conditions = tuple(protocol["benchmark"]["conditions"])
    rebuilt_rows, rebuilt_summary = _freeze_request_identities(
        construction,
        conditions=conditions,
        model=requests["model"],
        token_budget=int(protocol["benchmark"]["context_token_budget"]),
        tokenizer=tokenizer,
    )
    if canonical_sha256(rebuilt_rows) != canonical_sha256(requests["records"]):
        raise ValueError("Rebuilt confirmation request identities differ from freeze")
    if canonical_sha256(rebuilt_summary) != canonical_sha256(requests["token_summary"]):
        raise ValueError("Rebuilt confirmation token audit differs from freeze")
    if requests.get("cells") != 200:
        raise ValueError("Confirmation request manifest does not contain 200 cells")
    return freeze, construction, requests, conditions


def _audit_progress(
    construction: dict[str, Any], requests: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    expected = {
        (row["dataset_id"], row["item_id"], row["condition"]): row
        for row in requests["records"]
    }
    observed = {}
    parse_failures = 0
    attempt_records = 0
    for record in construction["records"]:
        path = output_root / record["dataset_id"] / "results.json"
        if not path.is_file():
            continue
        payload = read_json(path)
        attempt_records += len(payload.get("attempt_log") or [])
        for row in payload.get("records") or []:
            key = (record["dataset_id"], row["item_id"], row["condition"])
            if key not in expected:
                raise ValueError("Observed confirmation result is outside the frozen panel")
            if row.get("task_payload_sha256") != expected[key]["task_payload_sha256"]:
                raise ValueError("Observed confirmation task identity drift")
            if row.get("request_messages_sha256") != expected[key][
                "request_messages_sha256"
            ]:
                raise ValueError("Observed confirmation message identity drift")
            if key in observed:
                raise ValueError("Duplicate current confirmation cell")
            observed[key] = row
            parse_failures += row.get("status") != "complete"
    complete = sum(row.get("status") == "complete" for row in observed.values())
    return {
        "schema_version": 1,
        "status": "terminal" if complete == 200 and not parse_failures else "incomplete",
        "expected_cells": 200,
        "observed_current_cells": len(observed),
        "complete_cells": complete,
        "parse_failures": parse_failures,
        "archived_attempt_records": attempt_records,
        "all_observed_identities_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    freeze, construction, requests, conditions = validate_confirmation_inputs(
        args.freeze
    )
    plan = {
        "schema_version": 1,
        "status": "validated_confirmatory_execution_plan",
        "confirmatory": True,
        "freeze_sha256": sha256_file(args.freeze),
        "construction_manifest_sha256": freeze["construction_manifest_sha256"],
        "request_identity_manifest_sha256": freeze[
            "request_identity_manifest_sha256"
        ],
        "model": requests["model"],
        "conditions": list(conditions),
        "topics": 8,
        "planned_calls": 200,
        "runs": [
            {
                "dataset_id": record["dataset_id"],
                "benchmark": record["benchmark"],
                "benchmark_sha256": record["benchmark_sha256"],
                "output": str((args.output_root / record["dataset_id"]).resolve()),
                "calls": record["planned_calls"],
            }
            for record in construction["records"]
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "execution_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    key = _api_key(args.api_key_file, args.api_key_env)
    balance = _balance(args.base_url, key)
    write_json(args.output_root / "provider_balance_preflight.json", balance)
    if not balance["is_available"]:
        raise SystemExit("Provider execution blocked: balance endpoint reports unavailable")
    generic = ROOT / "scripts/run_graph_discovery_experiment.py"
    for record in construction["records"]:
        subprocess.run(
            [
                sys.executable,
                str(generic),
                "--benchmark",
                record["benchmark"],
                "--output",
                str(args.output_root / record["dataset_id"]),
                "--api-key-file",
                str(args.api_key_file),
                "--api-key-env",
                args.api_key_env,
                "--base-url",
                args.base_url,
                "--model",
                requests["model"],
                "--conditions",
                *conditions,
                "--execute",
            ],
            cwd=ROOT,
            check=True,
        )
        progress = _audit_progress(construction, requests, args.output_root)
        write_json(args.output_root / "execution_audit.json", progress)
    progress = _audit_progress(construction, requests, args.output_root)
    write_json(args.output_root / "execution_audit.json", progress)
    if progress["status"] != "terminal":
        raise SystemExit("Confirmation execution is incomplete; rerun the same frozen plan")
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
