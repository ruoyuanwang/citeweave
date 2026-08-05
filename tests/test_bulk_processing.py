from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bibagent.bulk_processing import process_large_metadata
from bibagent.exceptions import ProcessingError
from bibagent.models import (
    ProcessingPolicy,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)
from bibagent.processing_acceptance import verify_large_processing
from bibagent.transform import Canonicalizer


def _write_gzip_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _config() -> ProjectConfig:
    return ProjectConfig(
        project_id="large-processing-test",
        protocol=SearchProtocol(
            title="Large processing test",
            keywords=["bibliometric"],
            year_from=2020,
            year_to=2025,
            source=SourceName.crossref,
        ),
        processing=ProcessingPolicy(
            mode="disk",
            chunk_size=100,
            duckdb_memory_limit="1GB",
            candidate_pool_size=100,
            edge_row_limit=10_000,
        ),
    )


def test_in_memory_dedup_remaps_loser_relationships():
    base = {
        "title": ["Same normalized title"],
        "published": {"date-parts": [[2024, 1, 1]]},
        "container-title": ["Journal"],
        "reference": [{"DOI": "10.9999/ref.one"}],
    }
    first = deepcopy(base)
    first["URL"] = "https://example.test/first"
    first["author"] = [{"given": "Ada", "family": "One"}]
    second = deepcopy(base)
    second["URL"] = "https://example.test/second"
    second["author"] = [{"given": "Bea", "family": "Two"}]
    second["reference"] = [{"DOI": "10.9999/ref.two"}]

    tables = Canonicalizer("crossref").canonicalize([first, second])

    assert len(tables.works) == 1
    assert tables.authorships["work_id"].nunique() == 1
    assert set(tables.authorships["author_id"]) == set(tables.authors["author_id"])
    assert tables.references["citing_work_id"].nunique() == 1
    assert tables.references["cited_work_id"].nunique() == 2
    assert len(tables.provenance) == 2


def test_disk_processing_global_dedup_and_visualization_contract(tmp_path: Path):
    paths = ProjectPaths(tmp_path)
    paths.create()
    records: list[dict] = []
    for index in range(200):
        records.append(
            {
                "DOI": f"10.1234/large.{index}",
                "URL": f"https://doi.org/10.1234/large.{index}",
                "title": [f"Bibliometric processing theme{index % 12} work {index}"],
                "abstract": "<p>Structured metadata and science mapping.</p>",
                "published": {
                    "date-parts": [[2026 if index == 199 else 2020 + index % 6, 1, 1]]
                },
                "container-title": ["Journal of Structured Evidence"],
                "ISSN": ["1234-5678"],
                "type": "journal-article",
                "author": [
                    {"given": "Author", "family": f"{index % 20}"},
                    {"given": "Collaborator", "family": f"{(index + 1) % 20}"},
                ],
                "subject": [],
                "reference": [
                    {"DOI": f"10.9999/reference.{index % 25}"},
                    {"DOI": f"10.9999/reference.{(index + 1) % 25}"},
                ],
                "is-referenced-by-count": index % 17,
                "references-count": 2,
            }
        )
    duplicate_a = {
        "URL": "https://example.test/duplicate-a",
        "title": ["A title deduplicated across partitions"],
        "published": {"date-parts": [[2024, 2, 1]]},
        "container-title": ["Journal of Structured Evidence"],
        "author": [{"given": "Duplicate", "family": "Alpha"}],
        "subject": ["deduplication"],
        "reference": [{"DOI": "10.9999/reference.alpha"}],
    }
    duplicate_b = deepcopy(duplicate_a)
    duplicate_b["URL"] = "https://example.test/duplicate-b"
    duplicate_b["author"] = [{"given": "Duplicate", "family": "Beta"}]
    duplicate_b["reference"] = [{"DOI": "10.9999/reference.beta"}]
    records.insert(25, duplicate_a)
    records.append(duplicate_b)
    _write_gzip_jsonl(paths.staged / "source_records.jsonl.gz", records)

    partial = process_large_metadata(
        tmp_path,
        _config(),
        resume=False,
        batch_budget=1,
    )
    result = process_large_metadata(tmp_path, _config(), resume=True)
    acceptance = verify_large_processing(tmp_path)
    idempotent = process_large_metadata(tmp_path, _config(), resume=True)
    changed = _config()
    changed = changed.model_copy(
        update={
            "processing": changed.processing.model_copy(update={"edge_row_limit": 9_999})
        }
    )

    assert partial["partial"]
    assert partial["records_processed"] == 100
    assert partial["batches_completed"] == 1
    assert result["resumed"]
    assert result["records_processed"] == 202
    assert result["batches_completed"] == 3
    assert result["quality"]["canonical_records"] == 200
    assert result["quality"]["table_rows"]["provenance"] == 201
    assert result["quality"]["scope_filter"]["excluded_source_records"] == 1
    assert result["quality"]["scope_filter"]["input_records_accounted"] == 202
    assert result["quality"]["keyword_derivation"]["applied"]
    assert result["quality"]["keyword_derivation"]["derived_documents"] == 199
    assert result["quality"]["foreign_key_orphans"] == {
        name: 0 for name in result["quality"]["foreign_key_orphans"]
    }
    assert acceptance["passed"]
    assert acceptance["passed_checks"] == acceptance["total_checks"] == 6
    assert idempotent["already_complete"]
    with pytest.raises(ProcessingError, match="--refinalize"):
        process_large_metadata(tmp_path, changed, resume=True)
    rebuilt = process_large_metadata(tmp_path, changed, resume=True, refinalize=True)
    assert rebuilt["refinalized"]
    assert verify_large_processing(tmp_path)["passed"]
