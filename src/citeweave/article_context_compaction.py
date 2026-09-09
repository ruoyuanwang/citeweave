from __future__ import annotations

import hashlib
import json
from typing import Any

FULL_SOURCE_SECTIONS = {"Introduction", "Results", "Discussion"}
LIGHT_SOURCE_FIELDS = ("reference_id", "title", "year", "matched_keyword")
FULL_SOURCE_FIELDS = (*LIGHT_SOURCE_FIELDS, "abstract_excerpt")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def section_writer_view(
    writer_input: dict[str, Any], *, section: str
) -> dict[str, Any]:
    if section not in {
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
    }:
        raise ValueError(f"Unsupported article section: {section}")
    source_fields = (
        FULL_SOURCE_FIELDS if section in FULL_SOURCE_SECTIONS else LIGHT_SOURCE_FIELDS
    )
    sources = [
        {field: row[field] for field in source_fields if field in row}
        for row in writer_input["representative_sources"]
    ]
    return {
        "schema_version": 1,
        "status": "deterministic_section_writer_view",
        "source_writer_input_sha256": _canonical_hash(writer_input),
        "dataset_id": writer_input["dataset_id"],
        "writing_brief": writer_input["writing_brief"],
        "condition_contract": writer_input["condition_contract"],
        "graph_phenomena": writer_input["graph_phenomena"],
        "representative_sources": sources,
        "nonvisual_figure_metadata": writer_input["nonvisual_figure_metadata"],
        "view_contract": {
            "section": section,
            "all_graph_phenomena_preserved": True,
            "all_reference_ids_preserved": True,
            "all_abstract_excerpts_preserved": section in FULL_SOURCE_SECTIONS,
            "retrieval_ranking_metadata_omitted": True,
            "rendered_figure_access": False,
        },
    }


def audit_section_writer_view(
    writer_input: dict[str, Any], view: dict[str, Any]
) -> dict[str, Any]:
    original_ph = {
        row["phenomenon_id"]: _canonical_hash(row)
        for row in writer_input["graph_phenomena"]
    }
    view_ph = {
        row["phenomenon_id"]: _canonical_hash(row) for row in view["graph_phenomena"]
    }
    original_sources = {
        row["reference_id"]: row for row in writer_input["representative_sources"]
    }
    view_sources = {
        row["reference_id"]: row for row in view["representative_sources"]
    }
    section = view["view_contract"]["section"]
    excerpts_required = section in FULL_SOURCE_SECTIONS
    excerpt_hashes_preserved = all(
        not excerpts_required
        or _canonical_hash(original_sources[reference_id]["abstract_excerpt"])
        == _canonical_hash(view_sources[reference_id]["abstract_excerpt"])
        for reference_id in original_sources
    )
    gates = {
        "source_hash_bound": view["source_writer_input_sha256"]
        == _canonical_hash(writer_input),
        "dataset_preserved": view["dataset_id"] == writer_input["dataset_id"],
        "writing_brief_preserved": view["writing_brief"]
        == writer_input["writing_brief"],
        "condition_contract_preserved": view["condition_contract"]
        == writer_input["condition_contract"],
        "all_phenomenon_records_byte_equivalent": original_ph == view_ph,
        "all_reference_ids_preserved": set(original_sources) == set(view_sources),
        "full_section_excerpt_hashes_preserved": excerpt_hashes_preserved,
        "rendered_figure_withheld": view["view_contract"]["rendered_figure_access"]
        is False,
    }
    return {
        "schema_version": 1,
        "section": section,
        "passed": all(gates.values()),
        "quality_gates": gates,
        "original_serialized_chars": len(
            json.dumps(writer_input, ensure_ascii=False, separators=(",", ":"))
        ),
        "view_serialized_chars": len(
            json.dumps(view, ensure_ascii=False, separators=(",", ":"))
        ),
        "original_source_chars": len(
            json.dumps(
                writer_input["representative_sources"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        "view_source_chars": len(
            json.dumps(
                view["representative_sources"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    }
