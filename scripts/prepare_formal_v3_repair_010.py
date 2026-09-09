from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
EXECUTION = BASE / "formal_v3_identity_corrected_execution"
AMENDMENT = BASE / "formal_v3_execution_amendment_010_nested_operator_budget.yml"
AMENDMENT_FREEZE = (
    BASE / "formal_v3_execution_amendment_010_nested_operator_budget_freeze.json"
)
EXACT_AUDIT = EXECUTION / "repair_010_exact_message_audit.json"
QUALIFICATION = EXECUTION / "repair_010_qualification.json"
PROMOTION = EXECUTION / "repair_010_promotion.json"
PRIOR_FREEZE = EXECUTION / "input_freeze.json"
PRIOR_QUALIFICATION = EXECUTION / "qualification.json"
PRIOR_PROMOTION = EXECUTION / "promotion.json"
TOKEN_BUDGET_CODE = ROOT / "src/citeweave/token_budget.py"
TOKEN_BUDGET_TEST = ROOT / "tests/test_token_budget.py"
AUDITOR = ROOT / "scripts/audit_formal_v3_repair_exact_messages.py"
REPAIR_PREPARER = ROOT / "scripts/prepare_formal_v3_repair_010.py"
CONTROLLER = ROOT / "scripts/run_identity_corrected_formal_pipeline.ps1"

AMENDMENT_SHA256 = "a2307a08df890d1e9a2a79c6c9dab7a9425e5e1e6b8ed5ed4208777d8d668d9d"
TOKEN_BUDGET_SHA256 = "a4db5c95e9110e7c7830c281c82fcb741bca18c7c06cff0f5ff7f305487809c9"
TOKEN_BUDGET_TEST_SHA256 = (
    "cbad6ef19e37eb52538d4b342d6c6f5e07a1604af238a4dea50f498751d0cd6d"
)
PRIOR_FREEZE_SHA256 = "9340907c3c8a6131ab5c92b316227a2f5a6dac8b3f4fe9e9abd23d04c6e660b7"
PRIOR_QUALIFICATION_SHA256 = (
    "42e0a1f46ef5b4f717dc4588a23eddad36348f647363b3447448168e7d87bb4a"
)
PRIOR_PROMOTION_SHA256 = (
    "902e0a021ca99e6791e92ced1607ed71771b7bf6c2b34577e0fe8bf8522d5fa4"
)
EXPECTED_RESULTS = {
    BASE / "formal_v3_runs/federated_learning_healthcare_2016_2025/results.json": (
        "d5508417680dbb35f5c25cc33d3f3fa62d9fd3428f8fc266d87e6882de8e43b7"
    ),
    BASE / "formal_v3_runs/green_hydrogen_electrolysis_2010_2025/results.json": (
        "e758136ed4360f38da3d7afde1b987ed45b5fd260869b11ee3a38d3609e13a0f"
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen artifact drift: {path}")


def _canonical_hash(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _verify_prior_chain_except_repaired_code() -> None:
    _require_hash(PRIOR_FREEZE, PRIOR_FREEZE_SHA256)
    _require_hash(PRIOR_QUALIFICATION, PRIOR_QUALIFICATION_SHA256)
    _require_hash(PRIOR_PROMOTION, PRIOR_PROMOTION_SHA256)
    freeze = read_json(PRIOR_FREEZE)
    repaired = {TOKEN_BUDGET_CODE.resolve(), CONTROLLER.resolve()}
    skipped = 0
    for raw_path, expected in freeze["bindings"].items():
        path = Path(raw_path).resolve()
        if path in repaired:
            skipped += 1
            continue
        _require_hash(path, expected)
    if skipped != len(repaired):
        raise ValueError("Prior freeze is missing one or more repair-scoped code bindings")


def _verify_preserved_results(
    *,
    strict_file_hashes: bool,
    expected_record_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    if strict_file_hashes:
        for path, expected in EXPECTED_RESULTS.items():
            _require_hash(path, expected)
    records = []
    for path in EXPECTED_RESULTS:
        payload = read_json(path)
        if payload.get("attempt_log"):
            raise ValueError(f"Unexpected retry history in preserved results: {path}")
        records.extend(payload.get("records") or [])
    if any(record.get("status", "complete") != "complete" for record in records):
        raise ValueError("Preserved provider result files contain non-complete cells")
    keys = [f"{row['item_id']}\u0000{row['condition']}" for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Preserved provider result files contain duplicate logical cells")
    observed_hashes = {key: _canonical_hash(row) for key, row in zip(keys, records, strict=True)}
    if strict_file_hashes:
        if len(records) != 380:
            raise ValueError("Pre-resume provider results must contain exactly 380 cells")
        for root in (
            BASE / "formal_v3_replication_runs",
            BASE / "formal_v3_complexity_extension_runs",
        ):
            if root.exists() and any(path.is_file() for path in root.rglob("results.json")):
                raise ValueError(f"Unexpected downstream provider result before repair: {root}")
    elif not expected_record_hashes or any(
        observed_hashes.get(key) != expected
        for key, expected in expected_record_hashes.items()
    ):
        raise ValueError("One or more of the 380 preserved provider records changed")
    return observed_hashes


def _verify_exact_audit() -> dict[str, Any]:
    audit = read_json(EXACT_AUDIT)
    if (
        audit.get("status") != "ready_for_repair_qualification"
        or audit.get("scientific_effects_inspected") is not False
        or audit.get("provider_calls") != 0
        or audit.get("logical_cells")
        != {"primary": 800, "replication": 120, "extension": 672}
        or audit.get("assembled_cells") != 1592
        or audit.get("context_budget_violations") != 0
        or audit.get("preserved_complete_cells") != 380
        or audit.get("completed_request_identity_changes") != []
        or audit.get("blocking_reasons") != []
    ):
        raise ValueError("Exhaustive repair exact-message audit did not pass")
    identities = audit.get("cell_identity_manifest") or []
    if len(identities) != 1592:
        raise ValueError("Exact-message identity manifest length mismatch")
    keys = {(row["panel"], row["item_id"], row["condition"]) for row in identities}
    if len(keys) != 1592:
        raise ValueError("Exact-message identity manifest contains duplicates")
    budget = int(audit["context_token_budget"])
    if any(int(row["budgeted_context_tokens"]) > budget for row in identities):
        raise ValueError("Exact-message identity manifest exceeds the context budget")
    return audit


def _base_bindings() -> dict[str, str]:
    paths = {
        AMENDMENT,
        AMENDMENT_FREEZE,
        EXACT_AUDIT,
        PRIOR_FREEZE,
        PRIOR_QUALIFICATION,
        PRIOR_PROMOTION,
        TOKEN_BUDGET_CODE,
        TOKEN_BUDGET_TEST,
        AUDITOR,
        REPAIR_PREPARER,
        CONTROLLER,
        EXECUTION / "primary_plan.json",
        EXECUTION / "replication_plan.json",
        EXECUTION / "extension_plan.json",
        EXECUTION / "primary_readiness.json",
        EXECUTION / "replication_readiness.json",
        EXECUTION / "extension_readiness.json",
    }
    paths.update(EXPECTED_RESULTS)
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def verify_prerequisites(*, strict_result_hashes: bool = True) -> dict[str, Any]:
    _verify_prior_chain_except_repaired_code()
    _require_hash(AMENDMENT, AMENDMENT_SHA256)
    amendment_freeze = read_json(AMENDMENT_FREEZE)
    if amendment_freeze.get("sha256") != AMENDMENT_SHA256:
        raise ValueError("Amendment 010 freeze does not bind the amendment")
    _require_hash(TOKEN_BUDGET_CODE, TOKEN_BUDGET_SHA256)
    _require_hash(TOKEN_BUDGET_TEST, TOKEN_BUDGET_TEST_SHA256)
    if strict_result_hashes:
        _verify_preserved_results(strict_file_hashes=True)
    audit = _verify_exact_audit()
    return audit


def prepare() -> None:
    if QUALIFICATION.exists():
        verify_qualification()
        return
    audit = verify_prerequisites()
    preserved_record_hashes = _verify_preserved_results(strict_file_hashes=True)
    _immutable_json(
        QUALIFICATION,
        {
            "schema_version": 1,
            "status": "qualified_awaiting_explicit_repair_promotion",
            "created_at_utc": _now(),
            "automatic_api_resume": False,
            "scientific_effects_inspected": False,
            "preserved_complete_cells": 380,
            "remaining_primary_cells": 420,
            "replication_cells": 120,
            "extension_incremental_cells": 312,
            "exact_message_logical_cells": 1592,
            "cell_identity_manifest_sha256": audit["cell_identity_manifest_sha256"],
            "preserved_record_hashes": preserved_record_hashes,
            "bindings": _base_bindings(),
            "next_action": (
                "Independently inspect this qualification hash, then explicitly promote it; "
                "promotion alone does not start provider execution."
            ),
        },
    )


def verify_qualification(*, allow_result_growth: bool = False) -> None:
    verify_prerequisites(strict_result_hashes=not allow_result_growth)
    qualification = read_json(QUALIFICATION)
    if (
        qualification.get("status") != "qualified_awaiting_explicit_repair_promotion"
        or qualification.get("automatic_api_resume") is not False
        or qualification.get("scientific_effects_inspected") is not False
        or qualification.get("preserved_complete_cells") != 380
    ):
        raise ValueError("Repair qualification metadata is invalid")
    mutable_results = {path.resolve() for path in EXPECTED_RESULTS}
    for raw_path, expected in qualification.get("bindings", {}).items():
        path = Path(raw_path)
        if allow_result_growth and path.resolve() in mutable_results:
            continue
        _require_hash(path, expected)
    if allow_result_growth:
        _verify_preserved_results(
            strict_file_hashes=False,
            expected_record_hashes=qualification.get("preserved_record_hashes") or {},
        )


def promote(expected_qualification_sha256: str) -> None:
    verify_qualification()
    observed = sha256_file(QUALIFICATION)
    if observed != expected_qualification_sha256:
        raise ValueError("Explicit repair promotion hash does not match qualification")
    if PROMOTION.exists():
        verify_promoted()
        return
    _immutable_json(
        PROMOTION,
        {
            "schema_version": 1,
            "status": "explicitly_promoted_for_incomplete_cells_only",
            "created_at_utc": _now(),
            "automatic_api_resume": False,
            "qualification_sha256": observed,
            "preserved_complete_cells": 380,
            "resume_scope": "incomplete_cells_only",
        },
    )


def verify_promoted() -> None:
    verify_qualification(allow_result_growth=True)
    promotion = read_json(PROMOTION)
    if (
        promotion.get("status") != "explicitly_promoted_for_incomplete_cells_only"
        or promotion.get("automatic_api_resume") is not False
        or promotion.get("qualification_sha256") != sha256_file(QUALIFICATION)
        or promotion.get("preserved_complete_cells") != 380
        or promotion.get("resume_scope") != "incomplete_cells_only"
    ):
        raise ValueError("Repair promotion is invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify-qualified", action="store_true")
    mode.add_argument("--promote-qualification-sha256")
    mode.add_argument("--verify-promoted", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.verify_qualified:
        verify_qualification()
    elif args.promote_qualification_sha256:
        promote(args.promote_qualification_sha256)
    else:
        verify_promoted()


if __name__ == "__main__":
    main()
