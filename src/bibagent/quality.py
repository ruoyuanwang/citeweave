from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .models import AcquisitionManifest
from .transform import CanonicalTables

KEY_FIELDS = [
    "title",
    "year",
    "doi",
    "abstract",
    "source_id",
    "cited_by_count",
    "reference_count",
]


@dataclass
class QualityReport:
    summary: dict[str, Any]
    field_coverage: pd.DataFrame
    analysis_readiness: pd.DataFrame


def build_quality_report(manifest: AcquisitionManifest, tables: CanonicalTables) -> QualityReport:
    works = tables.works
    total = len(works)
    coverage_rows = []
    for field in KEY_FIELDS:
        present = int(works[field].notna().sum()) if field in works else 0
        coverage_rows.append(
            {
                "field": field,
                "present": present,
                "missing": total - present,
                "coverage": present / total if total else 0.0,
            }
        )
    field_coverage = pd.DataFrame(coverage_rows)
    source_name_present = (
        int(
            works[["work_id", "source_id"]]
            .merge(
                tables.sources[["source_id", "name"]],
                on="source_id",
                how="left",
            )["name"]
            .notna()
            .sum()
        )
        if total
        else 0
    )
    field_coverage = pd.concat(
        [
            field_coverage,
            pd.DataFrame(
                [
                    {
                        "field": "source_name",
                        "present": source_name_present,
                        "missing": total - source_name_present,
                        "coverage": source_name_present / total if total else 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    work_with_author = (
        tables.authorships["work_id"].nunique() if not tables.authorships.empty else 0
    )
    work_with_keyword = tables.keywords["work_id"].nunique() if not tables.keywords.empty else 0
    work_with_reference = (
        tables.references["citing_work_id"].nunique() if not tables.references.empty else 0
    )
    readiness = [
        {
            "analysis": "performance",
            "ready": total > 0 and works["year"].notna().mean() >= 0.8,
            "coverage": float(works["year"].notna().mean()) if total else 0.0,
            "requirement": "year coverage >= 0.80",
        },
        {
            "analysis": "coauthorship",
            "ready": total > 0 and work_with_author / total >= 0.5,
            "coverage": work_with_author / total if total else 0.0,
            "requirement": "works with authors >= 0.50",
        },
        {
            "analysis": "keyword_cooccurrence",
            "ready": total > 0 and work_with_keyword / total >= 0.3,
            "coverage": work_with_keyword / total if total else 0.0,
            "requirement": "works with source or derived keywords >= 0.30",
        },
        {
            "analysis": "citation_network",
            "ready": total > 0 and work_with_reference / total >= 0.1,
            "coverage": work_with_reference / total if total else 0.0,
            "requirement": "works with parsed references >= 0.10",
        },
    ]
    unique_ratio = (
        manifest.unique_records / manifest.expected_records if manifest.expected_records else 0.0
    )
    summary = {
        "source": manifest.source.value,
        "expected_records": manifest.expected_records,
        "received_records": manifest.received_records,
        "unique_source_records": manifest.unique_records,
        "canonical_works": total,
        "duplicate_source_records": manifest.duplicate_records,
        "deduplicated_works": len(tables.duplicates),
        "pages": manifest.pages,
        "acquisition_complete": manifest.complete,
        "truncated": manifest.truncated,
        "unique_to_expected_ratio": unique_ratio,
        "authors": len(tables.authors),
        "institutions": len(tables.institutions),
        "authorship_links": len(tables.authorships),
        "keyword_links": len(tables.keywords),
        "reference_links": len(tables.references),
        "warnings": manifest.warnings,
    }
    return QualityReport(
        summary=summary,
        field_coverage=field_coverage,
        analysis_readiness=pd.DataFrame(readiness),
    )
