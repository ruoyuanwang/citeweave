"""Check all 360 planned cross-panel reuses before any model outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.formal_request import build_messages, canonical_sha256, task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json
from citeweave.token_budget import CommandTokenizer

CONDITIONS = (
    "flat_hybrid",
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
    "graph_program",
)


def identity_difference(actual: dict, expected: dict) -> list[str]:
    return [
        key
        for key in ("task_payload_sha256", "context_sha256", "request_messages_sha256")
        if actual.get(key) != expected.get(key)
    ]


def audit_reuse(
    *,
    primary_root: Path,
    replication_root: Path,
    extension_root: Path,
    primary_readiness: Path,
    replication_readiness: Path,
    extension_readiness: Path,
) -> dict:
    readiness = {
        "primary": read_json(primary_readiness),
        "replication": read_json(replication_readiness),
        "extension": read_json(extension_readiness),
    }
    if any(r.get("status") != "ready" for r in readiness.values()):
        raise ValueError("All three readiness audits must pass before reuse preflight")
    tokenizer_hashes = set()
    for panel in readiness.values():
        path = Path(panel["tokenizer_manifest_path"])
        if read_json(path) != panel["tokenizer_manifest"]:
            raise ValueError("Tokenizer content differs from audited manifest")
        tokenizer_hashes.add(sha256_file(path))
    if len(tokenizer_hashes) != 1:
        raise ValueError("Panels do not share one tokenizer identity")
    extension = readiness["extension"]
    cells = extension.get("cell_identity_manifest") or []
    expected = {(r["dataset_id"], r["item_id"], r["condition"]): r for r in cells}
    if len(cells) != 672 or len(expected) != 672:
        raise ValueError("Extension must contain exactly 672 unique precomputed identities")
    if extension.get("cell_identity_manifest_sha256") != canonical_sha256(cells):
        raise ValueError("Extension cell identity manifest drift")
    tokenizer_path = Path(extension["tokenizer_manifest_path"])
    tokenizer_payload = read_json(tokenizer_path)
    if sha256_file(tokenizer_path) != extension["tokenizer_manifest_sha256"]:
        raise ValueError("Tokenizer manifest drift")
    tokenizer = CommandTokenizer(tokenizer_payload, manifest_dir=tokenizer_path.parent)
    budget = int(tokenizer_payload["context_token_budget"])
    manifest = readiness["primary"]["neural_dense_manifest"]
    manifest_path = Path(readiness["primary"]["neural_dense_manifest_path"])
    sidecars = {}
    for artifact in manifest["artifacts"]:
        if artifact.get("artifact_type") != "neural_context_sidecar":
            continue
        path = manifest_path.parent / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError("Primary neural sidecar drift")
        sidecars[artifact["dataset_id"]] = read_json(path)
    records = []
    construction = read_json(extension_root / "construction_manifest.json")
    for item in construction["records"]:
        topic, panel = item["dataset_id"], item["source_panel"]
        if panel not in {"primary", "replication"}:
            raise ValueError("Unknown source panel")
        source_path = (
            (primary_root if panel == "primary" else replication_root) / topic / "benchmark.json"
        )
        source = read_json(source_path)
        tasks = {r["item_id"]: r for r in source["tasks"]}
        extension_path = extension_root / topic / "benchmark.json"
        if sha256_file(extension_path) != item["benchmark_sha256"]:
            raise ValueError("Extension benchmark drift")
        for task in read_json(extension_path)["tasks"]:
            if task["complexity"] <= 1:
                continue
            original = tasks[task["item_id"]]
            for condition in CONDITIONS if panel == "primary" else ("flat_hybrid", "graph_program"):
                override = None
                if condition == "flat_neural_dense":
                    sidecar = sidecars[topic]
                    if sidecar["source_benchmark_sha256"] != sha256_file(source_path):
                        raise ValueError("Source neural benchmark binding differs")
                    override = sidecar["contexts"][original["item_id"]]
                messages, audit = build_messages(
                    original,
                    condition,
                    context_override=override,
                    tokenizer=tokenizer,
                    token_budget=budget,
                )
                actual = {
                    "task_payload_sha256": task_payload_sha256(original),
                    "context_sha256": audit["context_sha256"],
                    "request_messages_sha256": canonical_sha256(messages),
                }
                key = (topic, task["item_id"], condition)
                differences = identity_difference(actual, expected[key])
                records.append(
                    {
                        "dataset_id": topic,
                        "item_id": task["item_id"],
                        "condition": condition,
                        "source_panel": panel,
                        "passed": not differences,
                        "mismatches": differences,
                        "actual": actual,
                        "expected": {k: expected[key][k] for k in actual},
                    }
                )
        print(
            json.dumps({"topic": topic, "source_panel": panel, "checked_so_far": len(records)}),
            flush=True,
        )
    counts = {
        panel: sum(r["source_panel"] == panel for r in records)
        for panel in ("primary", "replication")
    }
    passed = (
        len(records) == 360
        and len({(r["dataset_id"], r["item_id"], r["condition"]) for r in records}) == 360
        and counts == {"primary": 240, "replication": 120}
        and all(r["passed"] for r in records)
    )
    return {
        "schema_version": 1,
        "status": "all_360_reuse_identities_equal" if passed else "reuse_identity_mismatch",
        "checked_cells": len(records),
        "source_counts": counts,
        "records": records,
        "readiness_sha256": {
            "primary": sha256_file(primary_readiness),
            "replication": sha256_file(replication_readiness),
            "extension": sha256_file(extension_readiness),
        },
        "note": "No provider requests are made. A mismatch blocks execution, not merely the eventual merge.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "primary-root",
        "replication-root",
        "extension-root",
        "primary-readiness",
        "replication-readiness",
        "extension-readiness",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = audit_reuse(**args)
    write_json(output, result)
    if result["status"] != "all_360_reuse_identities_equal":
        raise SystemExit("Cross-panel reuse failed before model execution")


if __name__ == "__main__":
    main()
