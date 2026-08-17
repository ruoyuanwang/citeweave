from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from citeweave.io import save_config, write_json, write_parquet
from citeweave.large_scale_evidence import (
    prepare_large_scale_evidence,
    verify_large_scale_evidence,
)
from citeweave.models import (
    AcquisitionManifest,
    ProcessingPolicy,
    ProjectConfig,
    ProjectPaths,
    SearchProtocol,
    SourceName,
)
from citeweave.transform import Canonicalizer


def test_large_evidence_is_prepared_from_disk_backed_corpus(
    crossref_records: list[dict], tmp_path: Path
) -> None:
    paths = ProjectPaths(tmp_path)
    paths.create()
    config = ProjectConfig(
        project_id="large-evidence-fixture",
        protocol=SearchProtocol(
            title="Fixture",
            keywords=["fixture"],
            year_from=2020,
            year_to=2025,
            source=SourceName.crossref,
        ),
        processing=ProcessingPolicy(mode="disk", duckdb_memory_limit="1GB"),
    )
    save_config(tmp_path / "project.yml", config)
    records = []
    for index in range(12):
        record = dict(crossref_records[index % len(crossref_records)])
        record["DOI"] = f"10.1234/evidence.{index}"
        record["URL"] = f"https://doi.org/10.1234/evidence.{index}"
        record["title"] = [f"Evidence fixture work {index}"]
        record["abstract"] = f"<p>Abstract for evidence fixture {index}.</p>"
        record["published"] = {"date-parts": [[2020 + index % 6, 1, 1]]}
        record["is-referenced-by-count"] = index
        records.append(record)
    tables = Canonicalizer("crossref").canonicalize(records)
    for name in (
        "works",
        "authors",
        "institutions",
        "authorships",
        "sources",
        "keywords",
        "topics",
        "references",
        "provenance",
        "duplicates",
    ):
        write_parquet(paths.canonical / f"{name}.parquet", getattr(tables, name))

    visual = paths.canonical / "visualization"
    annual = (
        tables.works.groupby("year")["work_id"]
        .nunique()
        .rename("documents")
        .reset_index()
    )
    write_parquet(visual / "annual_output.parquet", annual)
    write_parquet(
        visual / "document_types.parquet",
        pd.DataFrame({"document_type": ["journal-article"], "documents": [12]}),
    )
    write_parquet(
        visual / "source_productivity.parquet",
        pd.DataFrame(
            {
                "source_id": ["source:s1"],
                "source_name": ["Fixture Journal"],
                "issn": ["1234-5678"],
                "documents": [12],
                "citations": [66],
            }
        ),
    )
    write_parquet(
        visual / "author_productivity.parquet",
        pd.DataFrame(
            {
                "author_id": ["author:a1"],
                "author_name": ["Ada Fixture"],
                "orcid": [None],
                "documents": [12],
                "citations": [66],
            }
        ),
    )
    write_parquet(
        visual / "institution_productivity.parquet",
        pd.DataFrame(
            {
                "institution_id": ["institution:i1"],
                "institution_name": ["Fixture Institute"],
                "ror": [None],
                "country_code": ["US"],
                "documents": [12],
            }
        ),
    )
    keyword_values = tables.keywords["keyword"].dropna().astype(str).unique().tolist()
    if not keyword_values:
        keyword_values = ["fixture"]
        write_parquet(
            paths.canonical / "keywords.parquet",
            pd.DataFrame(
                {
                    "work_id": tables.works["work_id"],
                    "keyword": ["fixture"] * len(tables.works),
                    "keyword_type": ["fixture"] * len(tables.works),
                    "score": [1.0] * len(tables.works),
                }
            ),
        )
    write_parquet(
        visual / "keyword_occurrences.parquet",
        pd.DataFrame(
            {
                "keyword": keyword_values,
                "keyword_type": ["fixture"] * len(keyword_values),
                "occurrences": [12] * len(keyword_values),
            }
        ),
    )
    write_parquet(
        paths.analyses / "visualization" / "keyword_cooccurrence_nodes.parquet",
        pd.DataFrame(
            {
                "id": ["fixture", "evidence"],
                "label": ["Fixture", "Evidence"],
                "occurrences": [12, 8],
                "degree": [1, 1],
                "weighted_degree": [5.0, 5.0],
                "betweenness": [0.0, 0.0],
                "cluster": [1, 1],
            }
        ),
    )
    write_parquet(
        paths.analyses / "visualization" / "keyword_cooccurrence_edges.parquet",
        pd.DataFrame(
            {
                "source": ["fixture"],
                "target": ["evidence"],
                "weight": [5.0],
                "association_strength": [0.052],
            }
        ),
    )
    write_json(
        paths.analyses / "visualization" / "keyword_cooccurrence_method.json",
        {"candidate_pool": 2, "full_candidate_edge_count": 1},
    )
    write_json(paths.figures / "figure_manifest.json", {"figures": []})
    acquisition = AcquisitionManifest(
        source=SourceName.crossref,
        query={"keywords": ["fixture"]},
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        expected_records=12,
        received_records=12,
        unique_records=12,
        complete=True,
        raw_sha256=["a" * 64],
    )
    write_json(
        paths.audit / "acquisition_manifest.json",
        acquisition.model_dump(mode="json"),
    )
    write_json(
        paths.quality / "processing_report.json",
        {
            "canonical_records": 12,
            "field_coverage": {
                field: {"present": 12, "ratio": 1.0}
                for field in ("title", "year", "doi", "abstract", "cited_by_count")
            },
        },
    )

    result = prepare_large_scale_evidence(tmp_path)
    acceptance = verify_large_scale_evidence(tmp_path)

    assert result["passed"]
    assert result["evidence_items"] > 10
    assert result["graph"]["facts"] == 4
    assert result["graph_qa_available"]
    assert result["scalability"]["full_canonical_relations_loaded_into_pandas"] is False
    assert acceptance["passed"]
