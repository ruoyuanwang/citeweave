from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from citeweave.bulk_acquisition import _save_raw_page
from citeweave.harvest_repair import (
    HarvestRepairError,
    inspect_openalex_terminal_cursor_repair,
    repair_openalex_terminal_cursors,
)
from citeweave.io import atomic_write_bytes, read_json, save_config, sha256_file, write_json
from citeweave.models import (
    AcquisitionPolicy,
    HarvestManifest,
    HarvestPage,
    HarvestSlice,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)

PROJECT_ID = "climate-change-repair-test"
SLICES = ("20120102-20131231", "20140101-20160101")


def _config() -> ProjectConfig:
    return ProjectConfig(
        project_id=PROJECT_ID,
        protocol=SearchProtocol(
            title="Repair test",
            keywords=["climate change"],
            query_mode="all",
            year_from=2012,
            year_to=2016,
            source=SourceName.openalex,
            document_types=["article"],
            max_records=None,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk",
            partition_strategy="adaptive_date",
            target_slice_records=25_000,
            page_size=100,
        ),
    )


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / PROJECT_ID
    paths = ProjectPaths(workspace)
    paths.create()
    save_config(workspace / "project.yml", _config())
    slices: list[HarvestSlice] = []
    for slice_id in SLICES:
        cursor_out = f"resume-{slice_id}"
        payload = {
            "meta": {"count": 100, "next_cursor": cursor_out},
            "results": [
                {"id": f"https://openalex.org/W{slice_id}{index:03d}"}
                for index in range(100)
            ],
        }
        raw_path, digest, compressed_bytes = _save_raw_page(
            paths,
            SourceName.openalex,
            slice_id,
            1,
            payload,
            compress=True,
        )
        slices.append(
            HarvestSlice(
                slice_id=slice_id,
                date_from="2012-01-02" if slice_id == SLICES[0] else "2014-01-01",
                date_to="2013-12-31" if slice_id == SLICES[0] else "2016-01-01",
                expected_records=100,
                cursor_snapshot_expected_records=100,
                cursor_exhausted=True,
                status="complete",
                cursor=None,
                received_records=100,
                pages=[
                    HarvestPage(
                        page_index=1,
                        cursor_in="*",
                        cursor_out=cursor_out,
                        records=100,
                        raw_path=raw_path.relative_to(workspace).as_posix(),
                        raw_sha256=digest,
                        bytes_compressed=compressed_bytes,
                    )
                ],
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        )
    staged_path = paths.staged / "source_records.jsonl.gz"
    atomic_write_bytes(staged_path, b"bound-staged-corpus")
    manifest = HarvestManifest(
        source=SourceName.openalex,
        query_fingerprint="f" * 64,
        query={"search_expression": '"climate change"'},
        status="complete",
        page_size=100,
        partition_strategy="adaptive_date",
        target_slice_records=25_000,
        root_expected_records=200,
        planned_expected_records=200,
        cursor_snapshot_expected_records=200,
        received_records=200,
        unique_records=200,
        duplicate_records=0,
        staged_path=staged_path.relative_to(workspace).as_posix(),
        staged_sha256=sha256_file(staged_path),
        slices=slices,
    )
    write_json(paths.audit / "harvest_manifest.json", manifest.model_dump(mode="json"))
    return workspace


def test_preflight_is_read_only_and_reports_exact_resume_cursors(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "audit" / "harvest_manifest.json"
    before = manifest_path.read_bytes()

    result = inspect_openalex_terminal_cursor_repair(
        workspace,
        expected_project_id=PROJECT_ID,
        slice_ids=SLICES,
    )

    assert result["status"] == "validated"
    assert result["execute"] is False
    assert [row["slice_id"] for row in result["target_slices"]] == list(SLICES)
    assert all(row["final_page_records"] == 100 for row in result["target_slices"])
    assert manifest_path.read_bytes() == before
    assert not (workspace / "audit" / "harvest_manifest_repairs").exists()


def test_execute_reopens_only_verified_slices_and_writes_audit_receipt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "audit" / "harvest_manifest.json"
    before = manifest_path.read_bytes()
    raw_hashes_before = {
        path.relative_to(workspace).as_posix(): sha256_file(path)
        for path in workspace.glob("raw/harvest/**/*.json.gz")
    }

    result = repair_openalex_terminal_cursors(
        workspace,
        expected_project_id=PROJECT_ID,
        slice_ids=SLICES,
    )

    repaired = read_json(manifest_path)
    assert result["status"] == "repaired"
    assert repaired["status"] == "partial"
    by_id = {row["slice_id"]: row for row in repaired["slices"]}
    for slice_id in SLICES:
        assert by_id[slice_id]["status"] == "pending"
        assert by_id[slice_id]["cursor"] == f"resume-{slice_id}"
        assert by_id[slice_id]["cursor_exhausted"] is False
        assert by_id[slice_id]["finished_at"] is None
        assert len(by_id[slice_id]["pages"]) == 1
    assert Path(result["backup"]).read_bytes() == before
    receipt = read_json(Path(result["receipt"]))
    assert receipt["manifest_before_sha256"] == sha256_file(Path(result["backup"]))
    assert receipt["manifest_after_sha256"] == sha256_file(manifest_path)
    assert receipt["raw_pages_deleted"] == 0
    assert receipt["raw_pages_modified"] == 0
    assert {
        path.relative_to(workspace).as_posix(): sha256_file(path)
        for path in workspace.glob("raw/harvest/**/*.json.gz")
    } == raw_hashes_before


def test_repair_refuses_partial_target_set_without_writing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "audit" / "harvest_manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(HarvestRepairError, match="exactly match"):
        repair_openalex_terminal_cursors(
            workspace,
            expected_project_id=PROJECT_ID,
            slice_ids=(SLICES[0],),
        )

    assert manifest_path.read_bytes() == before
    assert not (workspace / "audit" / "harvest_manifest_repairs").exists()


def test_repair_refuses_hash_mismatch_without_writing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "audit" / "harvest_manifest.json"
    manifest = read_json(manifest_path)
    manifest["slices"][0]["pages"][0]["raw_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()

    with pytest.raises(HarvestRepairError, match="raw page hash differs"):
        repair_openalex_terminal_cursors(
            workspace,
            expected_project_id=PROJECT_ID,
            slice_ids=SLICES,
        )

    assert manifest_path.read_bytes() == before
    assert not (workspace / "audit" / "harvest_manifest_repairs").exists()


def test_repair_refuses_wrong_project_identity(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(HarvestRepairError, match="does not match expected project"):
        inspect_openalex_terminal_cursor_repair(
            workspace,
            expected_project_id="another-project",
            slice_ids=SLICES,
        )
