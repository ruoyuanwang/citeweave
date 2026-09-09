"""Shared raw temporal evidence and public task definition, without selected answers."""

from __future__ import annotations

import copy
import hashlib

import pandas as pd


def build_temporal_annex(dataset_id: str, trends: pd.DataFrame) -> dict:
    required = {"year", "keyword", "documents"}
    if not required <= set(trends):
        raise ValueError("Raw keyword-year counts required")
    if trends.duplicated(["keyword", "year"]).any():
        raise ValueError("Keyword-year counts must be unique")
    years = sorted(int(year) for year in trends.year.dropna().unique())
    if len(years) < 6:
        raise ValueError("Six observed years required for disjoint windows")
    records = []
    for keyword, group in trends.groupby("keyword", sort=True):
        by_year = {int(row.year): int(row.documents) for row in group.itertuples()}
        if any(count < 0 for count in by_year.values()):
            raise ValueError("Negative document counts")
        records.append(
            {
                "keyword": str(keyword),
                "year_document_counts": [by_year.get(year, 0) for year in years],
            }
        )
    if not 2 <= len(records) <= 15:
        raise ValueError("The original fixed two-to-fifteen-keyword candidate table is required")
    return {
        "schema_version": 1,
        "evidence_id": f"TEMPORAL-TABLE:{dataset_id}",
        "representation": "shared_raw_keyword_year_counts_with_public_selection_definition",
        "year_columns": years,
        "records": records,
        "public_task_definition": {
            "candidate_universe": "All keywords in this table that occur in the supplied graph scale.",
            "early_years": years[:3],
            "recent_years": years[-3:],
            "recent_documents": "Sum document counts over recent_years.",
            "growth_ratio": "(recent_documents + 1) / (early_documents + 1).",
            "eligible_recent_floor": "Sorted candidate recent_documents at zero-based index floor(candidate_count / 2).",
            "emerging_selection": "Among candidates at or above recent floor, maximize growth_ratio, then recent_documents, then choose lexicographically smallest keyword.",
            "established_selection": "Among remaining candidates, maximize full-period weighted_degree, then early_documents, then choose lexicographically smallest keyword.",
            "community_comparison": "Compare the two selected keywords' full-period detected community memberships; community IDs are not research-topic labels.",
        },
        "boundary": "Year counts and the operational definition are shared verbatim by every method. No chosen node, growth ratio, weighted degree, community assignment, or answer is supplied by this annex.",
    }


def add_shared_temporal_context(context: dict, annex: dict) -> dict:
    if "shared_temporal_evidence" in context:
        raise ValueError("Cannot attach the shared annex twice")
    result = copy.deepcopy(context)
    result["shared_temporal_evidence"] = copy.deepcopy(annex)
    return result


def control_item_id(source_item_id: str) -> str:
    return "temporal-shared-v1:" + hashlib.sha256(source_item_id.encode()).hexdigest()[:24]
