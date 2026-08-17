from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .analytics import AnalysisBundle
from .evidence import EvidenceCorpusStats, build_evidence, save_evidence
from .graph_grounding import (
    build_bounded_network_knowledge_graph,
    save_graph_grounding,
)
from .io import load_config, read_json, sha256_file, write_json, write_parquet
from .models import AcquisitionManifest, ProjectConfig, ProjectPaths
from .quality import QualityReport
from .scalable_reporting import _selected_networks
from .transform import CANONICAL_COLUMNS, CanonicalTables
from .visualization import FigureArtifact


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _empty_table(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS[name])


def _bounded_tables(
    connection: duckdb.DuckDBPyConnection,
    canonical: Path,
) -> CanonicalTables:
    works_path = _sql_path(canonical / "works.parquet")
    works = connection.execute(
        f"""
        WITH eligible AS (
          SELECT *,
                 row_number() OVER (
                   ORDER BY coalesce(cited_by_count, 0) DESC, year DESC NULLS LAST, work_id
                 ) AS cited_rank,
                 row_number() OVER (
                   ORDER BY year DESC NULLS LAST, coalesce(cited_by_count, 0) DESC, work_id
                 ) AS recent_rank
          FROM read_parquet('{works_path}')
          WHERE abstract IS NOT NULL AND trim(abstract) <> ''
        )
        SELECT * EXCLUDE (cited_rank, recent_rank)
        FROM eligible
        WHERE cited_rank <= 16 OR recent_rank <= 12
        QUALIFY row_number() OVER (
          ORDER BY least(cited_rank, recent_rank), cited_rank, recent_rank, work_id
        ) <= 24
        """
    ).df()
    work_ids = works["work_id"].astype(str).tolist()
    if work_ids:
        ids = ", ".join("'" + value.replace("'", "''") + "'" for value in work_ids)
        authorships = connection.execute(
            f"""
            SELECT * EXCLUDE (_rank)
            FROM (
              SELECT *, row_number() OVER (
                PARTITION BY work_id ORDER BY position NULLS LAST, author_id
              ) AS _rank
              FROM read_parquet('{_sql_path(canonical / 'authorships.parquet')}')
              WHERE work_id IN ({ids})
            )
            WHERE _rank <= 8
            """
        ).df()
        author_ids = authorships["author_id"].dropna().astype(str).unique().tolist()
        if author_ids:
            quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in author_ids)
            authors = connection.execute(
                f"""
                SELECT * FROM read_parquet('{_sql_path(canonical / 'authors.parquet')}')
                WHERE author_id IN ({quoted})
                """
            ).df()
        else:
            authors = _empty_table("authors")
        source_ids = works["source_id"].dropna().astype(str).unique().tolist()
        if source_ids:
            quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in source_ids)
            sources = connection.execute(
                f"""
                SELECT * FROM read_parquet('{_sql_path(canonical / 'sources.parquet')}')
                WHERE source_id IN ({quoted})
                """
            ).df()
        else:
            sources = _empty_table("sources")
    else:
        authorships = _empty_table("authorships")
        authors = _empty_table("authors")
        sources = _empty_table("sources")
    return CanonicalTables(
        works=works,
        authors=authors,
        institutions=_empty_table("institutions"),
        authorships=authorships,
        sources=sources,
        keywords=_empty_table("keywords"),
        topics=_empty_table("topics"),
        references=_empty_table("references"),
        provenance=_empty_table("provenance"),
        duplicates=pd.DataFrame(columns=["kept_work_id", "removed_work_id", "rule"]),
    )


def _analysis(
    connection: duckdb.DuckDBPyConnection,
    paths: ProjectPaths,
) -> tuple[AnalysisBundle, EvidenceCorpusStats]:
    canonical = paths.canonical
    visual = canonical / "visualization"

    def frame(filename: str, *, limit: int | None = None) -> pd.DataFrame:
        suffix = f" LIMIT {limit}" if limit is not None else ""
        return connection.execute(
            f"SELECT * FROM read_parquet('{_sql_path(visual / filename)}'){suffix}"
        ).df()

    annual = frame("annual_output.parquet").rename(columns={"documents": "publications"})
    annual = annual[["year", "publications"]].sort_values("year").reset_index(drop=True)
    annual["year"] = annual["year"].astype(int)

    def ranking(filename: str, label: str) -> pd.DataFrame:
        return connection.execute(
            f"""
            SELECT * EXCLUDE ({label}, documents),
                   {label} AS name, documents AS publications
            FROM read_parquet('{_sql_path(visual / filename)}')
            WHERE {label} IS NOT NULL AND trim({label}) <> ''
            ORDER BY documents DESC, {label}
            LIMIT 30
            """
        ).df()

    top_sources = ranking("source_productivity.parquet", "source_name")
    top_authors = ranking("author_productivity.parquet", "author_name")
    top_institutions = ranking("institution_productivity.parquet", "institution_name")
    document_types = frame("document_types.parquet").rename(
        columns={"documents": "publications"}
    )
    works = _sql_path(canonical / "works.parquet")
    counts = connection.execute(
        f"""
        SELECT count(*) AS documents,
               count(DISTINCT source_id) AS sources,
               avg(coalesce(cited_by_count, 0)) AS mean_citations,
               median(coalesce(cited_by_count, 0)) AS median_citations,
               count(*) FILTER (WHERE coalesce(cited_by_count, 0) = 0) AS zero_cited
        FROM read_parquet('{works}')
        """
    ).fetchone()
    entity_counts = connection.execute(
        f"""
        SELECT
          (SELECT count(*) FROM read_parquet('{_sql_path(canonical / 'authors.parquet')}')),
          (SELECT count(*) FROM read_parquet('{_sql_path(canonical / 'institutions.parquet')}')),
          (SELECT count(*) FROM read_parquet('{_sql_path(canonical / 'references.parquet')}'))
        """
    ).fetchone()
    citation_distribution = connection.execute(
        f"""
        SELECT * FROM (VALUES
          ('mean', (SELECT avg(coalesce(cited_by_count, 0)) FROM read_parquet('{works}'))),
          ('median', (SELECT median(coalesce(cited_by_count, 0)) FROM read_parquet('{works}'))),
          ('p75', (SELECT quantile_cont(coalesce(cited_by_count, 0), .75) FROM read_parquet('{works}'))),
          ('p90', (SELECT quantile_cont(coalesce(cited_by_count, 0), .90) FROM read_parquet('{works}'))),
          ('p95', (SELECT quantile_cont(coalesce(cited_by_count, 0), .95) FROM read_parquet('{works}'))),
          ('p98', (SELECT quantile_cont(coalesce(cited_by_count, 0), .98) FROM read_parquet('{works}'))),
          ('maximum', (SELECT max(coalesce(cited_by_count, 0)) FROM read_parquet('{works}'))),
          ('zero_cited', (SELECT count(*) FILTER (WHERE coalesce(cited_by_count, 0) = 0) FROM read_parquet('{works}')))
        ) metrics(metric, value)
        """
    ).df()
    top_cited = connection.execute(
        f"""
        SELECT title, year, doi, cited_by_count
        FROM read_parquet('{works}')
        ORDER BY coalesce(cited_by_count, 0) DESC, year DESC NULLS LAST, title
        LIMIT 30
        """
    ).df()
    source_file = _sql_path(visual / "source_productivity.parquet")
    bradford = connection.execute(
        f"""
        WITH ranked AS (
          SELECT *, sum(documents) OVER () AS total,
                 sum(documents) OVER (
                   ORDER BY documents DESC, source_name
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                 ) AS cumulative
          FROM read_parquet('{source_file}')
          WHERE source_name IS NOT NULL AND trim(source_name) <> ''
        ), zoned AS (
          SELECT least(3, floor((cumulative - 1) / greatest(total / 3.0, 1)) + 1)::INTEGER AS zone,
                 documents
          FROM ranked
        )
        SELECT zone, count(*) AS sources, sum(documents) AS documents
        FROM zoned GROUP BY zone ORDER BY zone
        """
    ).df()
    keyword_file = _sql_path(visual / "keyword_occurrences.parquet")
    keyword_trends = connection.execute(
        f"""
        WITH top_keywords AS (
          SELECT keyword, occurrences AS global_documents
          FROM read_parquet('{keyword_file}')
          WHERE keyword IS NOT NULL
          ORDER BY occurrences DESC, keyword
          LIMIT 15
        )
        SELECT works.year, keywords.keyword,
               count(DISTINCT keywords.work_id) AS documents,
               max(top_keywords.global_documents) AS global_documents
        FROM read_parquet('{_sql_path(canonical / 'keywords.parquet')}') keywords
        JOIN top_keywords USING (keyword)
        JOIN read_parquet('{works}') works USING (work_id)
        WHERE works.year IS NOT NULL
        GROUP BY works.year, keywords.keyword
        ORDER BY global_documents DESC, keyword, year
        """
    ).df()
    three_field = connection.execute(
        f"""
        WITH top_authors AS (
          SELECT author_id FROM read_parquet(
            '{_sql_path(visual / 'author_productivity.parquet')}') LIMIT 20
        ), top_sources AS (
          SELECT source_id FROM read_parquet(
            '{_sql_path(visual / 'source_productivity.parquet')}') LIMIT 20
        ), top_keywords AS (
          SELECT keyword FROM read_parquet('{keyword_file}') LIMIT 20
        )
        SELECT max(authors.name) AS author, max(sources.name) AS source,
               keywords.keyword, count(DISTINCT rel.work_id) AS documents
        FROM read_parquet('{_sql_path(canonical / 'authorships.parquet')}') rel
        JOIN top_authors USING (author_id)
        JOIN read_parquet('{works}') works USING (work_id)
        JOIN top_sources USING (source_id)
        JOIN read_parquet('{_sql_path(canonical / 'keywords.parquet')}') keywords USING (work_id)
        JOIN top_keywords USING (keyword)
        LEFT JOIN read_parquet('{_sql_path(canonical / 'authors.parquet')}') authors USING (author_id)
        LEFT JOIN read_parquet('{_sql_path(canonical / 'sources.parquet')}') sources USING (source_id)
        GROUP BY rel.author_id, works.source_id, keywords.keyword
        ORDER BY documents DESC, author, source, keyword
        LIMIT 80
        """
    ).df()
    first_year, last_year = int(annual["year"].min()), int(annual["year"].max())
    first_count, last_count = int(annual.iloc[0]["publications"]), int(annual.iloc[-1]["publications"])
    growth = (
        ((last_count / first_count) ** (1 / (last_year - first_year)) - 1) * 100
        if first_count > 0 and last_year > first_year
        else None
    )
    summary = {
        "documents": int(counts[0]),
        "authors": int(entity_counts[0]),
        "institutions": int(entity_counts[1]),
        "sources": int(counts[1]),
        "references": int(entity_counts[2]),
        "timespan_start": first_year,
        "timespan_end": last_year,
        "annual_growth_rate": growth,
        "average_citations": float(counts[2] or 0),
        "median_citations": float(counts[3] or 0),
        "analysis_scope": "all canonical records; network displays are bounded sparse views",
        "full_adjacency_matrix_materialized": False,
    }
    return (
        AnalysisBundle(
            summary=summary,
            annual=annual,
            top_sources=top_sources,
            top_authors=top_authors,
            top_institutions=top_institutions,
            document_types=document_types[["document_type", "publications"]],
            citation_distribution=citation_distribution,
            top_cited_documents=top_cited,
            bradford_sources=bradford,
            keyword_trends=keyword_trends,
            three_field=three_field,
            networks=_selected_networks(paths),
        ),
        EvidenceCorpusStats(documents=int(counts[0]), zero_cited=int(counts[4])),
    )


def _quality(paths: ProjectPaths) -> QualityReport:
    report = read_json(paths.quality / "processing_report.json")
    total = int(report["canonical_records"])
    coverage = pd.DataFrame(
        [
            {
                "field": field,
                "present": int(value["present"]),
                "missing": total - int(value["present"]),
                "coverage": float(value["ratio"]),
            }
            for field, value in report["field_coverage"].items()
        ]
    )
    readiness = pd.DataFrame(columns=["analysis", "ready", "coverage", "requirement"])
    return QualityReport(summary=report, field_coverage=coverage, analysis_readiness=readiness)


def _figures(paths: ProjectPaths) -> list[FigureArtifact]:
    manifest = read_json(paths.figures / "figure_manifest.json")
    return [
        FigureArtifact(
            name=item["name"],
            png=Path(item["png"]),
            svg=Path(item["svg"]),
            caption_facts=item.get("facts", {}),
            qa={
                key: item.get(key)
                for key in ("png_sha256", "svg_sha256", "width_px", "height_px")
            },
        )
        for item in manifest.get("figures", [])
    ]


def _save_analysis_contract(paths: ProjectPaths, analyses: AnalysisBundle) -> None:
    """Save only bounded aggregates required by evidence selectors and reporting."""
    write_json(paths.analyses / "summary.json", analyses.summary)
    for name in (
        "annual",
        "top_sources",
        "top_authors",
        "top_institutions",
        "document_types",
        "citation_distribution",
        "top_cited_documents",
        "bradford_sources",
        "keyword_trends",
        "three_field",
    ):
        write_parquet(paths.analyses / f"{name}.parquet", getattr(analyses, name))
    for name, network in analyses.networks.items():
        write_parquet(paths.analyses / f"network_{name}_nodes.parquet", network.nodes)
        write_parquet(paths.analyses / f"network_{name}_edges.parquet", network.edges)
        write_json(paths.analyses / f"network_{name}_manifest.json", network.metadata)


def prepare_large_scale_evidence(project: Path) -> dict[str, Any]:
    """Create report evidence and graph QA without loading full relations into pandas."""
    started = time.perf_counter()
    paths = ProjectPaths(project)
    config: ProjectConfig = load_config(paths.root / "project.yml")
    manifest = AcquisitionManifest.model_validate(
        read_json(paths.audit / "acquisition_manifest.json")
    )
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config.processing.duckdb_memory_limit}'")
    connection.execute("SET threads=4")
    try:
        analyses, corpus_stats = _analysis(connection, paths)
        bounded_tables = _bounded_tables(connection, paths.canonical)
    finally:
        connection.close()
    quality = _quality(paths)
    _save_analysis_contract(paths, analyses)
    write_parquet(paths.quality / "field_coverage.parquet", quality.field_coverage)
    write_parquet(paths.quality / "analysis_readiness.parquet", quality.analysis_readiness)
    write_json(paths.quality / "summary.json", quality.summary)
    evidence = build_evidence(
        manifest,
        bounded_tables,
        analyses,
        _figures(paths),
        quality,
        corpus_stats=corpus_stats,
    )
    save_evidence(evidence, paths.evidence)
    graph = build_bounded_network_knowledge_graph(analyses)
    graph_summary = save_graph_grounding(graph, paths.evidence, dataset_id=config.project_id)
    evidence_path = paths.evidence / "evidence_items.json"
    qa_path = paths.evidence / "graph_qa_benchmark.json"
    result = {
        "version": 1,
        "passed": bool(evidence.items),
        "project": str(paths.root),
        "evidence_items": len(evidence.items),
        "evidence_sha256": sha256_file(evidence_path),
        "graph": graph_summary,
        "graph_qa_available": qa_path.exists() and graph_summary["qa_items"] > 0,
        "scalability": {
            "full_canonical_relations_loaded_into_pandas": False,
            "duckdb_full_corpus_scans": True,
            "bounded_abstract_works": len(bounded_tables.works),
            "bounded_network_views": sorted(analyses.networks),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(paths.audit / "evidence_preparation_manifest.json", result)
    return result


def verify_large_scale_evidence(project: Path) -> dict[str, Any]:
    paths = ProjectPaths(project)
    manifest_path = paths.audit / "evidence_preparation_manifest.json"
    evidence_path = paths.evidence / "evidence_items.json"
    graph_facts = paths.evidence / "graph_facts.json"
    checks = {
        "manifest_exists": manifest_path.exists(),
        "evidence_exists": evidence_path.exists(),
        "graph_facts_exists": graph_facts.exists(),
    }
    manifest: dict[str, Any] = read_json(manifest_path) if manifest_path.exists() else {}
    if evidence_path.exists():
        items = read_json(evidence_path)
        checks["evidence_nonempty"] = isinstance(items, list) and bool(items)
        checks["evidence_hash_matches"] = manifest.get("evidence_sha256") == sha256_file(
            evidence_path
        )
        checks["unique_evidence_ids"] = len(
            {item.get("evidence_id") for item in items}
        ) == len(items)
        checks["evidence_artifacts_exist"] = all(
            (paths.root / item["artifact_path"]).exists() for item in items
        )
    else:
        checks.update(
            {
                "evidence_nonempty": False,
                "evidence_hash_matches": False,
                "unique_evidence_ids": False,
                "evidence_artifacts_exist": False,
            }
        )
    if manifest.get("graph_qa_available"):
        checks["graph_qa_exists"] = (paths.evidence / "graph_qa_benchmark.json").exists()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
    }
