from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bulk_acquisition import _read_raw_page, harvest_lock
from .io import (
    atomic_write_bytes,
    load_config,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .models import HarvestManifest, HarvestSlice, ProjectPaths, SourceName


class HarvestRepairError(RuntimeError):
    """Raised when an immutable harvest cannot be safely reopened."""


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _within_workspace(workspace: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def _validate_page_chain(
    paths: ProjectPaths,
    shard: HarvestSlice,
    *,
    page_size: int,
) -> dict[str, Any]:
    if shard.status != "complete":
        raise HarvestRepairError(f"{shard.slice_id}: slice status is not complete")
    if not shard.cursor_exhausted:
        raise HarvestRepairError(f"{shard.slice_id}: slice is not marked cursor-exhausted")
    if shard.cursor is not None:
        raise HarvestRepairError(f"{shard.slice_id}: completed slice cursor is not null")
    if not shard.pages:
        raise HarvestRepairError(f"{shard.slice_id}: slice has no raw pages")

    expected_indices = list(range(1, len(shard.pages) + 1))
    actual_indices = [page.page_index for page in shard.pages]
    if actual_indices != expected_indices:
        raise HarvestRepairError(f"{shard.slice_id}: page indices are not contiguous")
    if sum(page.records for page in shard.pages) != shard.received_records:
        raise HarvestRepairError(f"{shard.slice_id}: page and slice record counts differ")

    expected_cursor = "*"
    compressed_bytes = 0
    for page in shard.pages:
        if page.cursor_in != expected_cursor:
            raise HarvestRepairError(
                f"{shard.slice_id}: cursor chain differs at page {page.page_index}"
            )
        raw_path = paths.root / page.raw_path
        if not _within_workspace(paths.root, raw_path):
            raise HarvestRepairError(
                f"{shard.slice_id}: raw page escapes the workspace: {page.raw_path}"
            )
        if not raw_path.is_file():
            raise HarvestRepairError(
                f"{shard.slice_id}: raw page is missing: {page.raw_path}"
            )
        if raw_path.stat().st_size != page.bytes_compressed:
            raise HarvestRepairError(
                f"{shard.slice_id}: compressed byte count differs: {page.raw_path}"
            )
        try:
            payload = _read_raw_page(raw_path)
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
            raise HarvestRepairError(
                f"{shard.slice_id}: raw page is unreadable: {page.raw_path}"
            ) from exc
        if _canonical_payload_sha256(payload) != page.raw_sha256:
            raise HarvestRepairError(
                f"{shard.slice_id}: raw page hash differs: {page.raw_path}"
            )
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != page.records:
            raise HarvestRepairError(
                f"{shard.slice_id}: payload record count differs: {page.raw_path}"
            )
        payload_cursor = (payload.get("meta") or {}).get("next_cursor")
        if payload_cursor != page.cursor_out:
            raise HarvestRepairError(
                f"{shard.slice_id}: recorded cursor differs from payload: {page.raw_path}"
            )
        expected_cursor = page.cursor_out
        compressed_bytes += page.bytes_compressed

    final_page = shard.pages[-1]
    if final_page.records != page_size:
        raise HarvestRepairError(
            f"{shard.slice_id}: final page is not full "
            f"({final_page.records} != {page_size})"
        )
    if not final_page.cursor_out:
        raise HarvestRepairError(
            f"{shard.slice_id}: final page already has a null next cursor"
        )
    return {
        "slice_id": shard.slice_id,
        "pages_verified": len(shard.pages),
        "records_verified": shard.received_records,
        "compressed_bytes_verified": compressed_bytes,
        "resume_cursor": final_page.cursor_out,
        "final_page_index": final_page.page_index,
        "final_page_records": final_page.records,
        "final_page_sha256": final_page.raw_sha256,
    }


def _load_and_validate(
    workspace: Path,
    *,
    expected_project_id: str,
    slice_ids: tuple[str, ...],
) -> tuple[ProjectPaths, HarvestManifest, list[dict[str, Any]]]:
    paths = ProjectPaths(workspace)
    if paths.root.name != expected_project_id:
        raise HarvestRepairError(
            f"Workspace name {paths.root.name!r} does not match expected project "
            f"{expected_project_id!r}"
        )
    config_path = paths.root / "project.yml"
    if not config_path.is_file():
        raise HarvestRepairError(f"Project configuration is missing: {config_path}")
    config = load_config(config_path)
    if config.project_id != expected_project_id:
        raise HarvestRepairError(
            f"project.yml identifies {config.project_id!r}, not {expected_project_id!r}"
        )

    manifest_path = paths.audit / "harvest_manifest.json"
    if not manifest_path.is_file():
        raise HarvestRepairError(f"Harvest manifest is missing: {manifest_path}")
    harvest = HarvestManifest.model_validate(read_json(manifest_path))
    if harvest.source != SourceName.openalex:
        raise HarvestRepairError("Only OpenAlex harvests may use this repair")
    if harvest.status != "complete":
        raise HarvestRepairError(
            f"Harvest root status must be complete, found {harvest.status!r}"
        )
    if not slice_ids:
        raise HarvestRepairError("At least one target slice is required")
    if len(set(slice_ids)) != len(slice_ids):
        raise HarvestRepairError("Target slice IDs must be unique")

    by_id = {shard.slice_id: shard for shard in harvest.slices}
    missing = sorted(set(slice_ids) - set(by_id))
    if missing:
        raise HarvestRepairError(f"Target slices are absent: {missing}")

    manifest_received = sum(shard.received_records for shard in harvest.slices)
    if manifest_received != harvest.received_records:
        raise HarvestRepairError(
            "Root and slice received-record counts differ "
            f"({harvest.received_records} != {manifest_received})"
        )
    if harvest.unique_records + harvest.duplicate_records != harvest.received_records:
        raise HarvestRepairError(
            "Root unique/duplicate counts do not reconcile with received records"
        )
    if not harvest.staged_path or not harvest.staged_sha256:
        raise HarvestRepairError("Completed harvest lacks a staged artifact binding")
    staged_path = paths.root / harvest.staged_path
    if not _within_workspace(paths.root, staged_path) or not staged_path.is_file():
        raise HarvestRepairError("Bound staged artifact is missing or outside the workspace")
    if sha256_file(staged_path) != harvest.staged_sha256:
        raise HarvestRepairError("Bound staged artifact hash differs")

    eligible = {
        shard.slice_id
        for shard in harvest.slices
        if (
            shard.status == "complete"
            and shard.cursor_exhausted
            and shard.pages
            and shard.pages[-1].records == harvest.page_size
            and bool(shard.pages[-1].cursor_out)
        )
    }
    if eligible != set(slice_ids):
        raise HarvestRepairError(
            "Requested slices must exactly match all prematurely sealed full-page "
            f"OpenAlex slices; requested={sorted(slice_ids)}, eligible={sorted(eligible)}"
        )

    validations = [
        _validate_page_chain(paths, by_id[slice_id], page_size=harvest.page_size)
        for slice_id in slice_ids
    ]
    return paths, harvest, validations


def inspect_openalex_terminal_cursor_repair(
    workspace: Path,
    *,
    expected_project_id: str,
    slice_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Fail-closed preflight for reopening prematurely sealed OpenAlex slices."""

    paths, harvest, validations = _load_and_validate(
        workspace,
        expected_project_id=expected_project_id,
        slice_ids=slice_ids,
    )
    manifest_path = paths.audit / "harvest_manifest.json"
    return {
        "status": "validated",
        "execute": False,
        "project_id": expected_project_id,
        "workspace": str(paths.root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "root_status": harvest.status,
        "page_size": harvest.page_size,
        "target_slices": validations,
        "mutation": {
            "root_status": "partial",
            "slice_status": "pending",
            "cursor": "last verified non-null cursor_out",
            "cursor_exhausted": False,
            "finished_at": None,
        },
        "raw_pages_deleted": 0,
    }


def repair_openalex_terminal_cursors(
    workspace: Path,
    *,
    expected_project_id: str,
    slice_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Atomically reopen exact, fully verified OpenAlex cursor chains."""

    paths = ProjectPaths(workspace)
    with harvest_lock(paths):
        paths, harvest, validations = _load_and_validate(
            workspace,
            expected_project_id=expected_project_id,
            slice_ids=slice_ids,
        )
        manifest_path = paths.audit / "harvest_manifest.json"
        before_bytes = manifest_path.read_bytes()
        before_sha256 = sha256_bytes(before_bytes)
        repair_payload = {
            "protocol": "openalex-terminal-cursor-repair-v1",
            "project_id": expected_project_id,
            "manifest_sha256": before_sha256,
            "slice_ids": sorted(slice_ids),
        }
        repair_id = sha256_bytes(
            json.dumps(
                repair_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )[:24]
        receipt_root = paths.audit / "harvest_manifest_repairs"
        backup_path = receipt_root / f"{repair_id}.before.json"
        receipt_path = receipt_root / f"{repair_id}.json"
        if backup_path.exists() or receipt_path.exists():
            raise HarvestRepairError(
                f"Repair audit artifact already exists for {repair_id}; refusing overwrite"
            )

        by_id = {shard.slice_id: shard for shard in harvest.slices}
        changed_at = datetime.now(UTC)
        for validation in validations:
            shard = by_id[validation["slice_id"]]
            shard.status = "pending"
            shard.cursor = validation["resume_cursor"]
            shard.cursor_exhausted = False
            shard.finished_at = None
            shard.last_error = None
        harvest.status = "partial"
        harvest.updated_at = changed_at
        harvest.warnings.append(
            "Audited repair reopened prematurely sealed OpenAlex cursor chains: "
            f"repair_id={repair_id}, slices={','.join(sorted(slice_ids))}."
        )

        atomic_write_bytes(backup_path, before_bytes)
        write_json(manifest_path, harvest.model_dump(mode="json"))
        after_sha256 = sha256_file(manifest_path)
        receipt = {
            "schema_version": 1,
            "repair_id": repair_id,
            "protocol": "openalex-terminal-cursor-repair-v1",
            "applied_at": changed_at.isoformat(),
            "project_id": expected_project_id,
            "workspace": str(paths.root),
            "manifest": str(manifest_path),
            "manifest_before_sha256": before_sha256,
            "manifest_after_sha256": after_sha256,
            "manifest_backup": str(backup_path),
            "manifest_backup_sha256": sha256_file(backup_path),
            "target_slices": validations,
            "root_status_before": "complete",
            "root_status_after": "partial",
            "raw_pages_deleted": 0,
            "raw_pages_modified": 0,
        }
        write_json(receipt_path, receipt)
        return {
            "status": "repaired",
            "execute": True,
            "repair_id": repair_id,
            "project_id": expected_project_id,
            "manifest": str(manifest_path),
            "manifest_sha256": after_sha256,
            "receipt": str(receipt_path),
            "backup": str(backup_path),
            "target_slices": validations,
            "raw_pages_deleted": 0,
            "next_command": (
                "Run the formal harvest stage to resume exactly these cursor chains."
            ),
        }
