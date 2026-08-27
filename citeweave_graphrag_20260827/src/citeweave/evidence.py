from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .analytics import AnalysisBundle
from .io import write_json
from .models import AcquisitionManifest, EvidenceItem
from .quality import QualityReport
from .transform import CanonicalTables
from .visualization import FigureArtifact


@dataclass
class EvidenceBundle:
    items: list[EvidenceItem]
    graph: nx.DiGraph

    def prompt_packet(
        self,
        max_abstract_chars: int = 900,
        *,
        claim_types: set[str] | None = None,
        evidence_ids: set[str] | None = None,
    ) -> str:
        rows = []
        for item in self.items:
            if claim_types is not None and item.claim_type not in claim_types:
                continue
            if evidence_ids is not None and item.evidence_id not in evidence_ids:
                continue
            value = item.value
            if isinstance(value, str):
                if len(value) > max_abstract_chars:
                    value = value[:max_abstract_chars] + "…"
            elif item.claim_type == "work_content" and isinstance(value, dict):
                value = dict(value)
                abstract = value.get("abstract")
                if isinstance(abstract, str) and len(abstract) > max_abstract_chars:
                    value["abstract"] = abstract[:max_abstract_chars] + "…"
            rows.append(
                {
                    "id": item.evidence_id,
                    "type": item.claim_type,
                    "statement": item.statement,
                    "value": value,
                    "method": item.method,
                    "caveat": item.caveat,
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2, default=str)


def _frame_records(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe records for evidence packets."""
    selected = frame if columns is None else frame[columns]
    if limit is not None:
        selected = selected.head(limit)
    return json.loads(selected.to_json(orient="records", force_ascii=False, date_format="iso"))


def build_evidence(
    manifest: AcquisitionManifest,
    tables: CanonicalTables,
    analyses: AnalysisBundle,
    figures: list[FigureArtifact],
    quality: QualityReport | None = None,
    graph_explanations: list[dict[str, Any]] | None = None,
) -> EvidenceBundle:
    items: list[EvidenceItem] = []

    def add(
        claim_type: str,
        statement: str,
        value: Any,
        artifact: str,
        method: str,
        selector: dict[str, Any] | None = None,
        caveat: str | None = None,
    ) -> None:
        items.append(
            EvidenceItem(
                evidence_id=f"E{len(items) + 1:03d}",
                claim_type=claim_type,
                statement=statement,
                value=value,
                artifact_path=artifact,
                selector=selector or {},
                method=method,
                caveat=caveat,
            )
        )

    add(
        "corpus_size",
        f"The included corpus contains {analyses.summary['documents']} unique works.",
        analyses.summary["documents"],
        "canonical/works.parquet",
        "Count of canonical work_id after deterministic exact deduplication.",
    )
    add(
        "acquisition_completeness",
        (
            f"The source reported {manifest.expected_records} results and "
            f"{manifest.unique_records} unique source records were acquired."
        ),
        {
            "expected": manifest.expected_records,
            "acquired_unique": manifest.unique_records,
            "complete": manifest.complete,
            "truncated": manifest.truncated,
        },
        "audit/acquisition_manifest.json",
        "Cursor pagination continued until the source returned no next cursor.",
        caveat="Completeness is relative to the named source, query, time, and retrieval date.",
    )
    if quality is not None:
        coverage_values = {
            row["field"]: {
                "present": int(row["present"]),
                "missing": int(row["missing"]),
                "coverage_percent": round(float(row["coverage"]) * 100, 2),
                "missing_percent": round((1 - float(row["coverage"])) * 100, 2),
            }
            for _, row in quality.field_coverage.iterrows()
        }
        add(
            "metadata_coverage",
            "Field completeness was measured after normalization for every canonical work.",
            coverage_values,
            "quality/field_coverage.parquet",
            "Non-null canonical values divided by the canonical corpus size.",
            caveat=(
                "Field completeness measures metadata availability, not database recall "
                "or the substantive quality of the publications."
            ),
        )
    if not analyses.annual.empty:
        peak = analyses.annual.loc[analyses.annual["publications"].idxmax()]
        add(
            "time_span",
            (
                f"The corpus spans {analyses.summary['timespan_start']}–"
                f"{analyses.summary['timespan_end']}."
            ),
            {
                "start": analyses.summary["timespan_start"],
                "end": analyses.summary["timespan_end"],
            },
            "analyses/annual.parquet",
            "Minimum and maximum non-missing publication year.",
        )
        add(
            "annual_peak",
            f"Annual output peaked in {int(peak['year'])} with {int(peak['publications'])} works.",
            {"year": int(peak["year"]), "publications": int(peak["publications"])},
            "analyses/annual.parquet",
            "Maximum whole-counted works grouped by publication year.",
            caveat="The final year may be incomplete when retrieval occurs before year end.",
        )
        if analyses.summary.get("annual_growth_rate") is not None:
            add(
                "growth",
                (
                    "Compound annual publication growth over the observed non-zero "
                    f"endpoints is {analyses.summary['annual_growth_rate']:.2f}%."
                ),
                round(float(analyses.summary["annual_growth_rate"]), 2),
                "analyses/annual.parquet",
                "CAGR = (last/first)^(1/year difference)-1.",
                caveat="Endpoint CAGR does not establish a monotonic trend.",
            )
        add(
            "annual_output_series",
            "Annual publication counts are available for every observed year.",
            [
                {
                    "year": int(row["year"]),
                    "publications": int(row["publications"]),
                    "year_over_year_change": (
                        None
                        if index == 0
                        else int(row["publications"])
                        - int(analyses.annual.iloc[index - 1]["publications"])
                    ),
                }
                for index, (_, row) in enumerate(
                    analyses.annual.sort_values("year").reset_index(drop=True).iterrows()
                )
            ],
            "analyses/annual.parquet",
            (
                "Whole-counted canonical works grouped by non-missing publication year; "
                "year-over-year change is the current count minus the preceding count."
            ),
            caveat="The final year may be incomplete when retrieval occurs before year end.",
        )
    for frame, kind, label_col, artifact in (
        (analyses.top_sources, "top_source", "name", "analyses/top_sources.parquet"),
        (analyses.top_authors, "top_author", "name", "analyses/top_authors.parquet"),
        (
            analyses.top_institutions,
            "top_institution",
            "name",
            "analyses/top_institutions.parquet",
        ),
    ):
        if not frame.empty:
            row = frame.iloc[0]
            add(
                kind,
                f"{row[label_col]} ranks first with {int(row['publications'])} publications.",
                {"name": row[label_col], "publications": int(row["publications"])},
                artifact,
                "Whole counting within the included corpus; ties retain source sort order.",
            )
            add(
                f"{kind}_ranking",
                f"The leading {kind.replace('_', ' ')} records are retained for interpretation.",
                _frame_records(
                    frame,
                    columns=[label_col, "publications"],
                    limit=15,
                ),
                artifact,
                "Whole counting within the included corpus; ties retain source sort order.",
                caveat=(
                    "Productivity is descriptive and must not be treated as research "
                    "quality or causal influence."
                ),
            )
    if not analyses.document_types.empty:
        document_type_records = _frame_records(
            analyses.document_types,
            columns=["document_type", "publications"],
        )
        document_type_total = sum(int(row["publications"]) for row in document_type_records)
        for row in document_type_records:
            row["share_percent"] = (
                round(int(row["publications"]) / document_type_total * 100, 2)
                if document_type_total
                else 0.0
            )
        review_family_documents = sum(
            int(row["publications"])
            for row in document_type_records
            if str(row["document_type"]).casefold() in {"review", "systematic review"}
        )
        add(
            "document_type_distribution",
            "Document types were counted across the complete canonical corpus.",
            {
                "types": document_type_records,
                "review_and_systematic_review_documents": review_family_documents,
            },
            "analyses/document_types.parquet",
            "Whole-counted canonical works grouped by source-supplied document type.",
            caveat="Document-type labels inherit source classification practices.",
        )
    add(
        "citation_summary",
        (
            f"Mean citations are {analyses.summary['average_citations']:.2f} and "
            f"the median is {analyses.summary['median_citations']:.2f}."
        ),
        {
            "mean": round(float(analyses.summary["average_citations"]), 2),
            "median": round(float(analyses.summary["median_citations"]), 2),
        },
        "analyses/citation_distribution.parquet",
        "Citation-count field supplied by the selected source at retrieval time.",
        caveat="Citation counts are source- and retrieval-date-dependent.",
    )
    zero_cited = int(
        (pd.to_numeric(tables.works["cited_by_count"], errors="coerce").fillna(0) == 0).sum()
    )
    add(
        "zero_citation_share",
        (
            f"{zero_cited} works have zero source-reported citations, representing "
            f"{zero_cited / len(tables.works) * 100:.2f}% of the corpus."
        ),
        {
            "zero_cited": zero_cited,
            "share_percent": round(zero_cited / len(tables.works) * 100, 2),
        },
        "canonical/works.parquet",
        "Count and percentage of canonical works with cited_by_count equal to zero.",
        caveat="Recent works have had less time to accumulate citations.",
    )
    add(
        "citation_distribution_detail",
        "Citation-distribution statistics include central tendency, upper quantiles, maximum, and zero-cited count.",
        [
            {
                "metric": row["metric"],
                "value": row["value"],
                **(
                    {"percentile": int(str(row["metric"])[1:])}
                    if re.fullmatch(r"p\d+", str(row["metric"]))
                    else {}
                ),
            }
            for row in _frame_records(
                analyses.citation_distribution,
                columns=["metric", "value"],
            )
        ],
        "analyses/citation_distribution.parquet",
        "Descriptive statistics of the source-supplied cited-by count.",
        caveat=(
            "Citation counts are source- and retrieval-date-dependent; papers have "
            "unequal citation windows."
        ),
    )
    if not analyses.top_cited_documents.empty:
        add(
            "top_cited_documents",
            "The most-cited included documents are retained with title, year, DOI, and source-reported citation count.",
            _frame_records(
                analyses.top_cited_documents,
                columns=["title", "year", "doi", "cited_by_count"],
                limit=15,
            ),
            "analyses/top_cited_documents.parquet",
            "Descending source-reported citation count, with publication year as tie-breaker.",
            caveat=(
                "Raw citation counts favor older publications and do not by themselves "
                "measure quality."
            ),
        )
    if not analyses.bradford_sources.empty:
        zone_summary = (
            analyses.bradford_sources.groupby("zone", dropna=False)
            .agg(sources=("source_id", "nunique"), documents=("publications", "sum"))
            .reset_index()
        )
        zone_records = _frame_records(zone_summary)
        zone_documents = [int(row["documents"]) for row in zone_records]
        add(
            "bradford_zone_distribution",
            "Bradford zones summarize source concentration among records with known source names.",
            {
                "zones": zone_records,
                "rounded_document_range": {
                    "lower": int(min(zone_documents) // 50 * 50),
                    "upper": int(math.ceil(max(zone_documents) / 50) * 50),
                    "rounding_unit": 50,
                },
            },
            "analyses/bradford_sources.parquet",
            "Sources ranked by output and divided by cumulative document thirds.",
            caveat="Records without a source name are excluded from Bradford zoning.",
        )
    if not analyses.keyword_trends.empty:
        add(
            "keyword_temporal_dynamics",
            "Year-by-year document counts are available for the globally most frequent keywords.",
            _frame_records(
                analyses.keyword_trends.sort_values(
                    ["global_documents", "keyword", "year"],
                    ascending=[False, True, True],
                ),
                columns=["year", "keyword", "documents", "global_documents"],
                limit=80,
            ),
            "analyses/keyword_trends.parquet",
            "Unique works per normalized keyword and publication year.",
            caveat=(
                "Keyword frequencies depend on source metadata, synonym normalization, "
                "and the incomplete final year."
            ),
        )
    if not analyses.three_field.empty:
        add(
            "three_field_relations",
            "Leading author–source–keyword combinations are retained for relational interpretation.",
            _frame_records(
                analyses.three_field.sort_values("documents", ascending=False),
                columns=["author", "source", "keyword", "documents"],
                limit=30,
            ),
            "analyses/three_field.parquet",
            "Whole-counted links among leading authors, sources, and keywords.",
            caveat="The map is a filtered descriptive view, not a causal pathway.",
        )
    for name, network in analyses.networks.items():
        if network.nodes.empty:
            continue
        top = network.nodes.sort_values(["weighted_degree", "occurrences"], ascending=False).head(5)
        add(
            f"{name}_structure",
            (
                f"The {name.replace('_', ' ')} candidate network contains "
                f"{len(network.nodes)} nodes and {len(network.edges)} edges."
            ),
            {
                "nodes": len(network.nodes),
                "edges": len(network.edges),
                "clusters": int(network.nodes["cluster"].nunique()),
                "top_nodes": [
                    {
                        "label": row["label"],
                        "occurrences": int(row["occurrences"]),
                        "weighted_degree": round(float(row["weighted_degree"]), 3),
                        "cluster": int(row["cluster"]),
                    }
                    for _, row in top.iterrows()
                ],
                "network_parameters": network.metadata,
            },
            f"analyses/network_{name}_nodes.parquet",
            (
                "Candidate selection by occurrence, association-strength edge normalization, "
                "and Louvain community detection."
            ),
            caveat=(
                "Network centrality describes this corpus and parameterization; it does not "
                "establish causal or substantive importance."
            ),
        )
    # Balance citation anchors with recent works so the discussion is not limited
    # to older/highly cited records. The same work is emitted only once.
    works_with_abstract = tables.works.dropna(subset=["abstract"]).assign(
        cited_by_count=lambda frame: pd.to_numeric(frame["cited_by_count"], errors="coerce").fillna(
            0
        )
    )
    highly_cited = works_with_abstract.sort_values(
        ["cited_by_count", "year"], ascending=[False, False]
    ).head(16)
    recent = works_with_abstract.sort_values(
        ["year", "cited_by_count"], ascending=[False, False]
    ).head(12)
    abstract_works = (
        pd.concat([highly_cited, recent], ignore_index=True).drop_duplicates("work_id").head(24)
    )
    author_lookup = (
        tables.authorships.merge(
            tables.authors[["author_id", "name"]],
            on="author_id",
            how="left",
        )
        .dropna(subset=["name"])
        .groupby("work_id")["name"]
        .apply(lambda values: list(dict.fromkeys(values))[:8])
        .to_dict()
        if not tables.authorships.empty
        else {}
    )
    source_lookup = (
        tables.sources.drop_duplicates("source_id").set_index("source_id")["name"].to_dict()
        if not tables.sources.empty
        else {}
    )
    for _, row in abstract_works.iterrows():
        add(
            "work_content",
            f"Metadata and abstract for included work: {row['title']}",
            {
                "work_id": row["work_id"],
                "doi": row.get("doi"),
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "title": row["title"],
                "authors": author_lookup.get(row["work_id"], []),
                "source": source_lookup.get(row.get("source_id")),
                "abstract": row["abstract"],
            },
            "canonical/works.parquet",
            "Source-supplied title and abstract; no full-text inference.",
            caveat="An abstract cannot support claims that require full-text inspection.",
        )
    for figure_number, figure in enumerate(figures, start=1):
        add(
            "figure",
            (
                f"Figure {figure_number} ({figure.name}) was rendered from deterministic "
                "analysis outputs."
            ),
            {
                "figure_number": figure_number,
                "figure_name": figure.name,
                "caption_facts": figure.caption_facts,
            },
            f"figures/{figure.name}.svg",
            "Versioned figure skill with saved parameters and source tables.",
        )
    for explanation in graph_explanations or []:
        if explanation.get("status") != "complete":
            continue
        verified_claims = explanation.get("verified_claims") or []
        if not verified_claims:
            continue
        manuscript_claims = [
            {
                key: claim[key]
                for key in (
                    "claim_id",
                    "type",
                    "statement",
                    "nodes",
                    "communities",
                    "path",
                    "evidence_edges",
                    "verified",
                )
                if key in claim
            }
            for claim in verified_claims
        ]
        figure_name = str(explanation.get("figure_name") or "unknown")
        add(
            "figure_interpretation",
            f"Verified graph-grounded interpretation claims for {figure_name}.",
            {
                "figure_name": figure_name,
                "network_name": explanation.get("network_name"),
                "mode": explanation.get("mode"),
                "verified_claims": manuscript_claims,
                "caveats": explanation.get("caveats") or [],
                "verification": explanation.get("verification") or {},
            },
            "evidence/graph_explanations.json",
            "Graph explanation model followed by deterministic node, community, path, and edge verification.",
            caveat=(
                "Only verified structural relations may support manuscript claims; co-occurrence does not imply causality."
            ),
        )
    method_references = [
        {
            "citation": (
                "Donthu, N., Kumar, S., Mukherjee, D., Pandey, N., & Lim, W. M. "
                "(2021). How to conduct a bibliometric analysis: An overview and "
                "guidelines. Journal of Business Research, 133, 285–296."
            ),
            "doi": "10.1016/j.jbusres.2021.04.070",
            "role": "bibliometric workflow and interpretation guideline",
        },
        {
            "citation": (
                "Aria, M., & Cuccurullo, C. (2017). bibliometrix: An R-tool for "
                "comprehensive science mapping analysis. Journal of Informetrics, "
                "11(4), 959–975."
            ),
            "doi": "10.1016/j.joi.2017.08.007",
            "role": "science-mapping analysis baseline",
        },
        {
            "citation": (
                "van Eck, N. J., & Waltman, L. (2010). Software survey: VOSviewer, "
                "a computer program for bibliometric mapping. Scientometrics, 84, 523–538."
            ),
            "doi": "10.1007/s11192-009-0146-3",
            "role": "bibliometric network visualization baseline",
        },
        {
            "citation": (
                "Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. "
                "(2008). Fast unfolding of communities in large networks. "
                "Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008."
            ),
            "doi": "10.1088/1742-5468/2008/10/P10008",
            "role": "Louvain community-detection method",
        },
    ]
    for reference in method_references:
        add(
            "method_reference",
            reference["citation"],
            reference,
            "evidence/method_references.json",
            "Curated and DOI-verified methodological reference.",
        )

    graph = nx.DiGraph()
    graph.add_node("query_run", kind="query_run", source=manifest.source.value)
    graph.add_node("raw_snapshot", kind="raw_snapshot")
    graph.add_edge("query_run", "raw_snapshot", relation="produced")
    for digest in manifest.raw_sha256:
        node_id = f"raw:{digest}"
        graph.add_node(node_id, kind="raw_page", sha256=digest)
        graph.add_edge("raw_snapshot", node_id, relation="contains")
    graph.add_node("normalization_run", kind="transformation_run")
    graph.add_node("canonical_corpus", kind="corpus", works=len(tables.works))
    graph.add_node("analysis_run", kind="analysis_run")
    graph.add_edge("raw_snapshot", "normalization_run", relation="input_to")
    graph.add_edge("normalization_run", "canonical_corpus", relation="produced")
    graph.add_edge("canonical_corpus", "analysis_run", relation="input_to")
    for item in items:
        artifact_id = "artifact:" + item.artifact_path
        if artifact_id not in graph:
            graph.add_node(
                artifact_id,
                kind="artifact",
                path=item.artifact_path,
            )
            graph.add_edge("analysis_run", artifact_id, relation="produced")
        graph.add_node(
            item.evidence_id,
            kind="evidence",
            claim_type=item.claim_type,
            artifact=item.artifact_path,
        )
        graph.add_edge(artifact_id, item.evidence_id, relation="substantiates")
    return EvidenceBundle(items, graph)


def parse_evidence_ids(text: str, valid_ids: set[str]) -> list[str]:
    found: set[str] = set()
    for bracket in re.findall(r"\[[^\]]*E\d{3}[^\]]*\]", text):
        for start, end in re.findall(r"E(\d{3})\s*[–—-]\s*E(\d{3})", bracket):
            for number in range(int(start), int(end) + 1):
                candidate = f"E{number:03d}"
                if candidate in valid_ids:
                    found.add(candidate)
        for candidate in re.findall(r"E\d{3}", bracket):
            if candidate in valid_ids:
                found.add(candidate)
    return sorted(found)


def bind_claims(bundle: EvidenceBundle, manuscript: str) -> pd.DataFrame:
    """Attach manuscript claim units to evidence nodes and return an audit ledger."""
    valid_ids = {item.evidence_id for item in bundle.items}
    rows = []
    claim_index = 0
    for paragraph in manuscript.split("\n\n"):
        text = paragraph.strip()
        if not text or text.startswith(("#", "|")):
            continue
        evidence_ids = parse_evidence_ids(text, valid_ids)
        if not evidence_ids:
            continue
        claim_index += 1
        claim_id = f"C{claim_index:03d}"
        rows.append(
            {
                "claim_id": claim_id,
                "text": text,
                "evidence_ids": "|".join(evidence_ids),
                "evidence_count": len(evidence_ids),
                "contains_number": bool(re.search(r"\d", text)),
                "supported": True,
            }
        )
        bundle.graph.add_node(claim_id, kind="claim", text=text[:1000])
        for evidence_id in evidence_ids:
            bundle.graph.add_edge(evidence_id, claim_id, relation="supports_claim")
    return pd.DataFrame(rows)


def save_evidence(bundle: EvidenceBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "evidence_items.json",
        [item.model_dump(mode="json") for item in bundle.items],
    )
    write_json(
        out_dir / "method_references.json",
        [item.value for item in bundle.items if item.claim_type == "method_reference"],
    )
    write_json(
        out_dir / "evidence_graph.json",
        {
            "nodes": [{"id": node, **data} for node, data in bundle.graph.nodes(data=True)],
            "edges": [
                {"source": left, "target": right, **data}
                for left, right, data in bundle.graph.edges(data=True)
            ],
        },
    )
    nx.write_graphml(bundle.graph, out_dir / "evidence_graph.graphml")
