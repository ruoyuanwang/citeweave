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

from citeweave.io import read_json, sha256_file, write_json
from citeweave.robust_graph_statistics import CONDITIONS

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


def validate_execution_inputs(
    *,
    protocol_path: Path,
    freeze_path: Path,
    readiness_path: Path,
    construction_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze = read_json(freeze_path)
    readiness = read_json(readiness_path)
    construction = read_json(construction_manifest_path)
    protocol_hash = sha256_file(protocol_path)
    if freeze.get("sha256") != protocol_hash:
        raise RuntimeError("Protocol differs from its freeze artifact")
    if readiness.get("status") != "ready_for_development_execution":
        raise RuntimeError("Readiness audit does not permit development execution")
    if readiness.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Readiness protocol identity mismatch")
    if readiness.get("construction_manifest_sha256") != sha256_file(
        construction_manifest_path
    ):
        raise RuntimeError("Readiness construction-manifest identity mismatch")
    if tuple(protocol["panel"]["conditions"]) != CONDITIONS:
        raise RuntimeError("Registered conditions differ from runner contract")
    if construction.get("planned_calls") != 200 or construction.get("topics") != 8:
        raise RuntimeError("Construction manifest is not the registered 200-cell panel")
    records = list(construction.get("records") or [])
    if len(records) != 8:
        raise RuntimeError("Construction manifest must bind exactly eight topic benchmarks")
    for record in records:
        path = Path(record["benchmark"])
        if not path.is_file():
            path = construction_manifest_path.parent / record["dataset_id"] / "benchmark.json"
        if not path.is_file() or sha256_file(path) != record.get("benchmark_sha256"):
            raise RuntimeError(f"Benchmark identity mismatch: {record.get('dataset_id')}")
        record["resolved_benchmark"] = str(path.resolve())
    return readiness, records


def _provider_balance(base_url: str, key: str) -> dict[str, Any]:
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
                for field in ("currency", "total_balance", "granted_balance", "topped_up_balance")
            }
            for row in payload.get("balance_infos") or []
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--construction-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    readiness, records = validate_execution_inputs(
        protocol_path=args.protocol,
        freeze_path=args.freeze,
        readiness_path=args.readiness,
        construction_manifest_path=args.construction_manifest,
    )
    plan = {
        "schema_version": 1,
        "status": "validated_development_execution_plan",
        "confirmatory": False,
        "protocol_sha256": readiness["protocol_sha256"],
        "readiness_sha256": sha256_file(args.readiness),
        "construction_manifest_sha256": sha256_file(args.construction_manifest),
        "model": args.model,
        "conditions": list(CONDITIONS),
        "topics": len(records),
        "planned_calls": sum(int(record["planned_calls"]) for record in records),
        "runs": [
            {
                "dataset_id": record["dataset_id"],
                "benchmark": record["resolved_benchmark"],
                "benchmark_sha256": record["benchmark_sha256"],
                "output": str((args.output_root / record["dataset_id"]).resolve()),
                "calls": record["planned_calls"],
            }
            for record in records
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "execution_plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    key = _api_key(args.api_key_file, args.api_key_env)
    balance = _provider_balance(args.base_url, key)
    write_json(args.output_root / "provider_balance_preflight.json", balance)
    if not balance["is_available"]:
        raise SystemExit("Provider execution blocked: balance endpoint reports unavailable")

    generic_runner = Path(__file__).with_name("run_graph_discovery_experiment.py")
    for record in records:
        command = [
            sys.executable,
            str(generic_runner),
            "--benchmark",
            record["resolved_benchmark"],
            "--output",
            str(args.output_root / record["dataset_id"]),
            "--api-key-file",
            str(args.api_key_file),
            "--api-key-env",
            args.api_key_env,
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--readiness",
            str(args.readiness),
            "--conditions",
            *CONDITIONS,
            "--execute",
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
