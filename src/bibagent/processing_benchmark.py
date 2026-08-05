from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

from .bulk_processing import process_large_metadata
from .exceptions import ProcessingError
from .io import save_config, sha256_file, write_json
from .models import (
    ProcessingPolicy,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)
from .processing_acceptance import verify_large_processing


def _synthetic_record(index: int, references_per_document: int) -> dict[str, Any]:
    year = 2020 + index % 5
    return {
        "DOI": f"10.5555/processing-benchmark.{index}",
        "URL": f"https://doi.org/10.5555/processing-benchmark.{index}",
        "title": [f"Scalable bibliometric processing theme {index % 100} document {index}"],
        "abstract": (
            "This benchmark record contains structured metadata for reproducible "
            f"science mapping theme {index % 100}."
        ),
        "published": {"date-parts": [[year, 1 + index % 12, 1]]},
        "container-title": [f"Synthetic Journal {index % 250}"],
        "ISSN": [f"{1000 + index % 9000:04d}-{index % 10}{index % 10}{index % 10}X"],
        "type": "journal-article",
        "language": "en",
        "publisher": "BibAgent Benchmark Press",
        "author": [
            {
                "given": "Author",
                "family": str(index % 5_000),
                "affiliation": [{"name": f"University {index % 500}"}],
            },
            {
                "given": "Collaborator",
                "family": str((index + 1) % 5_000),
                "affiliation": [{"name": f"University {(index + 1) % 500}"}],
            },
        ],
        "subject": [f"theme {index % 100}", f"method {index % 20}"],
        "reference": [
            {"DOI": f"10.7777/benchmark-reference.{(index + offset) % 20_000}"}
            for offset in range(references_per_document)
        ],
        "references-count": references_per_document,
        "is-referenced-by-count": index % 100,
    }


def run_processing_benchmark(
    output: Path,
    *,
    documents: int = 100_000,
    references_per_document: int = 10,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    """Generate and process a reproducible corpus through the full disk pipeline."""
    if documents < 1 or references_per_document < 0:
        raise ValueError("documents must be positive and references_per_document non-negative")
    paths = ProjectPaths(output)
    if (paths.root / "project.yml").exists():
        raise ProcessingError(
            "Benchmark output already contains project.yml; choose a new directory."
        )
    paths.create()
    config = ProjectConfig(
        project_id=f"processing-benchmark-{documents}",
        protocol=SearchProtocol(
            title=f"Synthetic processing benchmark ({documents} records)",
            keywords=["bibliometric processing benchmark"],
            query_mode="phrase",
            year_from=2020,
            year_to=2024,
            source=SourceName.crossref,
            include_references=True,
        ),
        processing=ProcessingPolicy(
            mode="disk",
            chunk_size=chunk_size,
            duckdb_memory_limit="4GB",
            candidate_pool_size=640,
            edge_row_limit=200_000,
            keep_partitions=True,
        ),
    )
    save_config(paths.root / "project.yml", config)
    staged = paths.staged / "source_records.jsonl.gz"
    temporary = staged.with_suffix(staged.suffix + ".tmp")
    started = time.perf_counter()
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for index in range(documents):
            handle.write(
                json.dumps(
                    _synthetic_record(index, references_per_document),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    os.replace(temporary, staged)
    generation_seconds = time.perf_counter() - started

    processing = process_large_metadata(
        paths.root,
        config,
        input_path=staged,
        resume=False,
        chunk_size=chunk_size,
    )
    acceptance = verify_large_processing(paths.root)
    result = {
        "version": 1,
        "documents": documents,
        "references_per_document": references_per_document,
        "expected_reference_rows": documents * references_per_document,
        "chunk_size": chunk_size,
        "staged_path": str(staged),
        "staged_bytes": staged.stat().st_size,
        "staged_sha256": sha256_file(staged),
        "generation_seconds": round(generation_seconds, 3),
        "processing_seconds": processing["elapsed_seconds"],
        "table_rows": processing["quality"]["table_rows"],
        "quality_passed": processing["quality"]["passed"],
        "acceptance_passed": acceptance["passed"],
        "acceptance_checks": f"{acceptance['passed_checks']}/{acceptance['total_checks']}",
    }
    write_json(paths.audit / "processing_benchmark.json", result)
    return result
