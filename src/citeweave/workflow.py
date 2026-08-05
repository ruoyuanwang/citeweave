from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .analytics import AnalysisBundle, NetworkResult, analyze
from .bulk_acquisition import bulk_acquire, harvest_lock, iter_staged_records
from .bulk_processing import process_large_metadata
from .connectors import (
    CrossrefConnector,
    EuropePmcConnector,
    ImportFileConnector,
    OpenAlexConnector,
)
from .evidence import EvidenceBundle, bind_claims, build_evidence, save_evidence
from .exceptions import QualityGateError
from .figure_catalog import figure_caption, figure_section, order_figures
from .generation import (
    GenerationResult,
    _normalize_evidence_tokens,
    finalize_staged_manuscript,
    generate_manuscript,
    save_generation,
    validate_manuscript,
)
from .interoperability import export_all
from .io import (
    atomic_write_bytes,
    load_config,
    read_json,
    save_config,
    sha256_file,
    write_json,
    write_jsonl,
    write_parquet,
)
from .models import (
    AcquisitionManifest,
    EvidenceItem,
    ProjectConfig,
    ProjectPaths,
    SourceName,
)
from .quality import QualityReport, build_quality_report
from .scalable_reporting import build_large_scale_analysis
from .scale import save_scale_plan
from .transform import Canonicalizer, CanonicalTables, derive_keywords
from .visualization import FigureArtifact, render_all
from .word_export import export_word_report


def create_project(root: Path, config: ProjectConfig) -> ProjectPaths:
    paths = ProjectPaths(root)
    paths.create()
    save_config(paths.root / "project.yml", config)
    write_json(
        paths.audit / "state.json",
        {
            "project_id": config.project_id,
            "state": "initialized",
            "updated_at": datetime.now(UTC),
        },
    )
    return paths


def _set_state(paths: ProjectPaths, state: str, detail: dict[str, Any] | None = None) -> None:
    write_json(
        paths.audit / "state.json",
        {
            "state": state,
            "detail": detail or {},
            "updated_at": datetime.now(UTC),
        },
    )


def _connector(config: ProjectConfig, paths: ProjectPaths, *, bulk: bool = False):
    source = config.protocol.source
    connector_options: dict[str, Any] = {}
    if bulk:
        policy = config.acquisition
        default_rates = {
            SourceName.crossref: 2.5 if config.crossref_mailto else 0.8,
            SourceName.openalex: 5.0,
            SourceName.europe_pmc: 4.0,
        }
        connector_options = {
            "max_retries": policy.max_retries,
            "requests_per_second": (policy.requests_per_second or default_rates.get(source)),
        }
    if source == SourceName.crossref:
        return CrossrefConnector(
            paths.raw,
            mailto=config.crossref_mailto,
            **connector_options,
        )
    if source == SourceName.openalex:
        return OpenAlexConnector(paths.raw, **connector_options)
    if source == SourceName.europe_pmc:
        return EuropePmcConnector(paths.raw, **connector_options)
    if source == SourceName.import_file:
        return ImportFileConnector(paths.raw)
    raise NotImplementedError(f"Unsupported source: {source}")


def harvest_project(
    root: Path,
    config: ProjectConfig,
    *,
    resume: bool = True,
    page_budget: int | None = None,
) -> dict[str, Any]:
    """Run only the resumable bulk acquisition stage."""
    paths = ProjectPaths(root)
    paths.create()
    save_config(paths.root / "project.yml", config)
    _set_state(paths, "acquiring_bulk", {"resume": resume})
    connector = _connector(config, paths, bulk=True)
    try:
        try:
            with harvest_lock(paths):
                acquisition = bulk_acquire(
                    paths,
                    config,
                    connector,
                    resume=resume,
                    page_budget=page_budget,
                )
        except BaseException as exc:
            _set_state(
                paths,
                "acquisition_interrupted",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1_000],
                    "checkpoint": str(paths.audit / "harvest_manifest.json"),
                    "resumable": (paths.audit / "harvest_manifest.json").exists(),
                },
            )
            raise
    finally:
        connector.close()
    write_json(
        paths.audit / "acquisition_manifest.json",
        acquisition.manifest.model_dump(mode="json"),
    )
    state = "acquired" if acquisition.manifest.complete else "acquisition_partial"
    _set_state(
        paths,
        state,
        {
            "expected": acquisition.manifest.expected_records,
            "unique": acquisition.manifest.unique_records,
            "pages": acquisition.manifest.pages,
            "staged_path": (
                str(acquisition.staged_path.relative_to(paths.root))
                if acquisition.staged_path
                else None
            ),
        },
    )
    return {
        "project_root": str(paths.root),
        "complete": acquisition.manifest.complete,
        "expected_records": acquisition.manifest.expected_records,
        "received_records": acquisition.manifest.received_records,
        "unique_records": acquisition.manifest.unique_records,
        "pages": acquisition.manifest.pages,
        "raw_files": len(acquisition.raw_paths),
        "staged_path": str(acquisition.staged_path) if acquisition.staged_path else None,
        "manifest": str(paths.audit / "harvest_manifest.json"),
    }


def process_project(
    root: Path,
    config: ProjectConfig,
    *,
    resume: bool = True,
    chunk_size: int | None = None,
    keep_partitions: bool | None = None,
    refinalize: bool = False,
    batch_budget: int | None = None,
) -> dict[str, Any]:
    """Run bounded-memory cleaning, global deduplication and table materialization."""
    paths = ProjectPaths(root)
    paths.create()
    if config.processing.mode == "auto":
        config = config.model_copy(
            update={"processing": config.processing.model_copy(update={"mode": "disk"})}
        )
    save_config(paths.root / "project.yml", config)
    _set_state(
        paths,
        "processing_metadata",
        {"resume": resume, "chunk_size": chunk_size or config.processing.chunk_size},
    )
    try:
        result = process_large_metadata(
            root,
            config,
            resume=resume,
            chunk_size=chunk_size,
            keep_partitions=keep_partitions,
            refinalize=refinalize,
            batch_budget=batch_budget,
        )
    except BaseException as exc:
        _set_state(
            paths,
            "processing_interrupted",
            {
                "error_type": type(exc).__name__,
                "error": str(exc)[:1_000],
                "checkpoint": str(paths.audit / "processing_manifest.json"),
                "resumable": (paths.audit / "processing_manifest.json").exists(),
            },
        )
        raise
    if result.get("partial"):
        _set_state(
            paths,
            "processing_partial",
            {
                "records": result.get("records_processed"),
                "batches": result.get("batches_completed"),
                "checkpoint": str(paths.audit / "processing_manifest.json"),
                "resumable": True,
            },
        )
        return result
    _set_state(
        paths,
        "metadata_structured",
        {
            "records": result.get("records_processed")
            or result.get("quality", {}).get("input_records"),
            "canonical_records": result.get("quality", {}).get("canonical_records"),
            "quality_passed": result.get("quality", {}).get("passed"),
            "manifest": str(paths.audit / "processing_manifest.json"),
        },
    )
    return result


def _save_canonical(paths: ProjectPaths, tables: CanonicalTables) -> None:
    for name, frame in tables.as_dict().items():
        write_parquet(paths.canonical / f"{name}.parquet", frame)


def _save_quality(paths: ProjectPaths, report: QualityReport) -> None:
    write_json(paths.quality / "summary.json", report.summary)
    write_parquet(paths.quality / "field_coverage.parquet", report.field_coverage)
    report.field_coverage.to_csv(paths.quality / "field_coverage.csv", index=False)
    write_parquet(paths.quality / "analysis_readiness.parquet", report.analysis_readiness)
    report.analysis_readiness.to_csv(paths.quality / "analysis_readiness.csv", index=False)


def _save_analyses(paths: ProjectPaths, bundle: AnalysisBundle) -> None:
    write_json(paths.analyses / "summary.json", bundle.summary)
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
        write_parquet(paths.analyses / f"{name}.parquet", getattr(bundle, name))
        getattr(bundle, name).to_csv(paths.analyses / f"{name}.csv", index=False)
    for name, network in bundle.networks.items():
        write_parquet(paths.analyses / f"network_{name}_nodes.parquet", network.nodes)
        write_parquet(paths.analyses / f"network_{name}_edges.parquet", network.edges)
        network.nodes.to_csv(paths.analyses / f"network_{name}_nodes.csv", index=False)
        network.edges.to_csv(paths.analyses / f"network_{name}_edges.csv", index=False)
        write_json(paths.analyses / f"network_{name}_manifest.json", network.metadata)


def _deterministic_manuscript(config: ProjectConfig, evidence: EvidenceBundle) -> GenerationResult:
    citations = {item.claim_type: item for item in evidence.items}
    corpus = citations["corpus_size"]
    complete = citations["acquisition_completeness"]
    lines = [
        f"# {config.protocol.title}：文献计量分析",
        "",
        "## 摘要",
        "",
        (
            f"本研究围绕“{config.protocol.title}”开展可复现的文献计量分析。"
            f"检索采用项目配置中预先登记的年份范围，数据来自 "
            f"{config.protocol.source.value}。经规范化和精确去重后，"
            f"分析语料包含 {corpus.value} 篇文献。[{corpus.evidence_id}]"
        ),
        "",
        "## 1 数据与方法",
        "",
        (
            f"数据采集采用游标分页直至来源不再返回后续游标。数据源报告与实际采集情况为："
            f"{json.dumps(complete.value, ensure_ascii=False)}。[{complete.evidence_id}]"
        ),
        (
            "分析采用全计数描述年度产出、来源、作者和机构分布；网络分析保存候选节点、"
            "原始边权、关联强度、社群结果及可视化筛选参数。网络关系仅用于描述本语料中的"
            "结构联系，不作为因果关系或论文内容结论。"
        ),
        "",
        "## 2 结果",
        "",
    ]
    for claim_type in (
        "time_span",
        "annual_peak",
        "growth",
        "top_source",
        "top_author",
        "top_institution",
        "citation_summary",
        "coauthorship_structure",
        "keyword_cooccurrence_structure",
        "citation_structure",
        "cocitation_structure",
        "bibliographic_coupling_structure",
    ):
        item = citations.get(claim_type)
        if item:
            lines.append(f"{item.statement} [{item.evidence_id}]")
            lines.append("")
    lines += [
        "## 3 讨论与局限",
        "",
        (
            "以上结果描述的是特定检索式、年份范围、数据源和采集时点下的语料结构。"
            "数据库覆盖、缺失元数据、实体消歧和网络阈值均可能影响结论；主题含义和机制性"
            "判断仍需结合代表性文献全文进行人工核验。"
        ),
        "",
        "## 4 结论",
        "",
        (
            "本报告提供了一条从原始元数据到结构化表、分析图、证据项和文本声明的可追溯路径。"
            "其结论应作为文献计量证据，而不是替代系统性阅读。"
        ),
    ]
    manuscript = "\n".join(lines)
    validation = validate_manuscript(manuscript, evidence, strict_structure=False)
    validation["mode"] = "deterministic_template"
    if not validation["valid"]:
        raise QualityGateError(
            "Deterministic manuscript failed evidence validation: "
            + json.dumps(validation, ensure_ascii=False)
        )
    return GenerationResult(
        manuscript=manuscript,
        review=None,
        validation=validation,
        model="deterministic-template",
    )


def _render_html(
    paths: ProjectPaths,
    config: ProjectConfig,
    quality: QualityReport,
    analyses: AnalysisBundle,
    figures: list[FigureArtifact],
    manuscript: str,
) -> Path:
    import html

    import markdown  # type: ignore

    ordered_figures = order_figures(figures)
    figure_numbers = {figure.name: index for index, figure in enumerate(ordered_figures, 1)}
    by_section: dict[str, list[FigureArtifact]] = {}
    for figure in ordered_figures:
        by_section.setdefault(figure_section(figure.name), []).append(figure)

    def render_figure_cards(section: str) -> str:
        return "\n".join(
            f"""
        <figure>
          <img src="../figures/{figure.png.name}" alt="{html.escape(figure_caption(figure.name))}">
          <figcaption>图 {figure_numbers[figure.name]}　{html.escape(figure_caption(figure.name))}
          <span>基于全量规范化元数据与有界稀疏网络绘制。</span></figcaption>
        </figure>
        """
            for figure in by_section.get(section, [])
        )

    enriched_lines: list[str] = []
    active_section: str | None = None
    inserted_sections: set[str] = set()
    for line in manuscript.splitlines():
        heading_match = re.match(r"^###\s+(3\.[1-5])\b", line)
        if heading_match:
            if active_section and active_section not in inserted_sections:
                enriched_lines.extend(
                    [
                        "",
                        '<div class="figure-grid">',
                        render_figure_cards(active_section),
                        "</div>",
                        "",
                    ]
                )
                inserted_sections.add(active_section)
            active_section = heading_match.group(1)
        elif line.startswith("## ") and active_section:
            if active_section not in inserted_sections:
                enriched_lines.extend(
                    [
                        "",
                        '<div class="figure-grid">',
                        render_figure_cards(active_section),
                        "</div>",
                        "",
                    ]
                )
                inserted_sections.add(active_section)
            active_section = None
        enriched_lines.append(line)
    if active_section and active_section not in inserted_sections:
        enriched_lines.extend(
            ["", '<div class="figure-grid">', render_figure_cards(active_section), "</div>"]
        )
    html_body = markdown.markdown(
        "\n".join(enriched_lines),
        extensions=["tables", "fenced_code"],
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(config.protocol.title)} · CiteWeave</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--blue:#2563eb;--paper:#fbfcfe;--line:#dce3ed}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef2f7;color:var(--ink);
font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif;line-height:1.75}}
.shell{{max-width:1280px;margin:0 auto;background:white;min-height:100vh;box-shadow:0 0 50px #cbd5e1}}
header{{padding:56px 7%;background:linear-gradient(135deg,#0f172a,#1e3a8a);color:white}}
header p{{color:#cbd5e1;margin:12px 0 0}} .metrics{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;padding:28px 7%;
background:#f8fafc;border-bottom:1px solid var(--line)}}
.metric{{background:white;border:1px solid var(--line);border-radius:12px;padding:18px}}
.metric strong{{display:block;font-size:26px;color:var(--blue)}} .metric span{{color:var(--muted)}}
main{{padding:38px 7% 80px}} h1,h2,h3{{line-height:1.25}} h2{{margin-top:42px;
padding-bottom:9px;border-bottom:1px solid var(--line)}} p{{max-width:88ch}}
.figure-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
gap:24px;margin:24px 0 34px}}
figure{{margin:0;border:1px solid var(--line);border-radius:14px;padding:12px;background:var(--paper)}}
figure img{{width:100%;height:auto;display:block}} figcaption{{padding:9px 4px 2px;
color:var(--ink);font-size:13px}} figcaption span{{display:block;color:var(--muted);
margin-top:3px}} .notice{{border-left:4px solid #f97316;
background:#fff7ed;padding:14px 18px;border-radius:6px;margin:22px 0}}
@media(max-width:650px){{header,main{{padding-left:22px;padding-right:22px}}
.metrics{{padding-left:22px;padding-right:22px}}.figure-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><div class="shell">
<header><h1>{html.escape(config.protocol.title)}</h1><p>可审计的文献计量研究报告 ·
{config.protocol.source.value} · {config.protocol.year_from}—{config.protocol.year_to}</p></header>
<section class="metrics">
<div class="metric"><strong>{analyses.summary["documents"]:,}</strong><span>规范文献</span></div>
<div class="metric"><strong>{analyses.summary["authors"]:,}</strong><span>作者</span></div>
<div class="metric"><strong>{analyses.summary["sources"]:,}</strong><span>来源</span></div>
<div class="metric"><strong>{analyses.summary["references"]:,}</strong><span>参考文献关系</span></div>
</section>
<main>
<div class="notice">完整性仅相对于指定数据源、检索式、年份范围和采集时点；
引用次数与网络结构不是跨数据库不变事实。</div>
{html_body}
</main></div></body></html>"""
    output = paths.report / "index.html"
    atomic_write_bytes(output, document.encode("utf-8"))
    return output


def _package_manifest(paths: ProjectPaths) -> None:
    records = []
    for path in sorted(paths.root.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            records.append(
                {
                    "path": path.relative_to(paths.root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        paths.audit / "package_manifest.json",
        {
            "created_at": datetime.now(UTC),
            "files": records,
        },
    )


def run_project(
    root: Path,
    config: ProjectConfig,
    *,
    use_llm: bool = False,
    llm_api_key: str | None = None,
    review_rounds: int = 1,
    allow_truncated: bool = False,
) -> dict[str, Any]:
    paths = create_project(root, config)
    _set_state(paths, "acquiring")
    bulk_mode = config.acquisition.mode == "bulk"
    connector = _connector(config, paths, bulk=bulk_mode)
    try:
        if bulk_mode:
            with harvest_lock(paths):
                acquisition = bulk_acquire(paths, config, connector, resume=True)
        else:
            acquisition = connector.acquire(config.protocol)
    finally:
        connector.close()
    write_json(
        paths.audit / "acquisition_manifest.json",
        acquisition.manifest.model_dump(mode="json"),
    )
    if acquisition.staged_path is None:
        write_jsonl(paths.staged / "source_records.jsonl", acquisition.records)
        records_for_normalization = acquisition.records
    else:
        records_for_normalization = iter_staged_records(acquisition.staged_path)
    if acquisition.manifest.truncated and not allow_truncated:
        raise QualityGateError(
            "Acquisition is truncated. Re-run without max_records or explicitly allow truncated analysis."
        )

    _set_state(paths, "normalizing")
    tables = Canonicalizer(config.protocol.source).canonicalize(records_for_normalization)
    tables.keywords = derive_keywords(tables.works, tables.keywords)
    _save_canonical(paths, tables)

    quality = build_quality_report(acquisition.manifest, tables)
    _save_quality(paths, quality)
    if not acquisition.manifest.complete and not allow_truncated:
        raise QualityGateError("Acquisition completeness gate failed.")

    _set_state(paths, "analyzing")
    analyses = analyze(
        tables,
        network_candidate_pool=max(config.visualization_max_nodes * 8, 400),
    )
    save_scale_plan(tables, config, paths.audit / "scale_plan.json")
    _save_analyses(paths, analyses)
    export_all(tables, analyses, paths.analyses / "exports")

    _set_state(paths, "rendering")
    figures = render_all(
        tables,
        analyses,
        paths.figures,
        max_nodes=config.visualization_max_nodes,
        label_budget=config.visualization_label_budget,
        seed=config.random_seed,
    )
    write_json(
        paths.figures / "figure_manifest.json",
        [
            {
                "name": figure.name,
                "png": figure.png.name,
                "svg": figure.svg.name,
                "caption_facts": figure.caption_facts,
                "qa": figure.qa,
            }
            for figure in figures
        ],
    )

    evidence = build_evidence(acquisition.manifest, tables, analyses, figures, quality)
    save_evidence(evidence, paths.evidence)

    _set_state(paths, "generating")
    if use_llm:
        generation = generate_manuscript(
            config,
            evidence,
            api_key=llm_api_key,
            review_rounds=review_rounds,
            candidate_dir=paths.audit / "generation_candidates",
            stage_dir=paths.report / "generation_stages",
        )
    else:
        generation = _deterministic_manuscript(config, evidence)
    save_generation(generation, paths.report)
    claim_ledger = bind_claims(evidence, generation.manuscript)
    write_parquet(paths.evidence / "claim_ledger.parquet", claim_ledger)
    claim_ledger.to_csv(paths.evidence / "claim_ledger.csv", index=False)
    save_evidence(evidence, paths.evidence)
    html_path = _render_html(paths, config, quality, analyses, figures, generation.manuscript)
    word_path = export_word_report(paths.root)
    _set_state(paths, "complete", {"report": str(html_path)})
    _package_manifest(paths)
    return {
        "project_root": str(paths.root),
        "documents": analyses.summary["documents"],
        "figures": len(figures),
        "evidence_items": len(evidence.items),
        "generation_model": generation.model,
        "generation_validation": generation.validation,
        "report": str(html_path),
        "word_report": str(word_path),
        "acquisition_complete": acquisition.manifest.complete,
    }


def _load_canonical(paths: ProjectPaths) -> CanonicalTables:
    names = [
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
    ]
    frames = {name: pd.read_parquet(paths.canonical / f"{name}.parquet") for name in names}
    # Preserve declared schemas for legitimately empty relationship tables.
    if frames["topics"].empty:
        frames["topics"] = frames["topics"].reindex(
            columns=["work_id", "topic_id", "topic", "score", "field"]
        )
    if frames["duplicates"].empty:
        frames["duplicates"] = frames["duplicates"].reindex(
            columns=["kept_work_id", "removed_work_id", "rule"]
        )
    return CanonicalTables(**frames)


def _load_analyses(paths: ProjectPaths) -> AnalysisBundle:
    table_names = [
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
    ]
    tables = {name: pd.read_parquet(paths.analyses / f"{name}.parquet") for name in table_names}
    networks: dict[str, NetworkResult] = {}
    for manifest_path in paths.analyses.glob("network_*_manifest.json"):
        name = manifest_path.name.removeprefix("network_").removesuffix("_manifest.json")
        networks[name] = NetworkResult(
            name=name,
            nodes=pd.read_parquet(paths.analyses / f"network_{name}_nodes.parquet"),
            edges=pd.read_parquet(paths.analyses / f"network_{name}_edges.parquet"),
            metadata=read_json(manifest_path),
        )
    return AnalysisBundle(
        summary=read_json(paths.analyses / "summary.json"),
        networks=networks,
        **tables,
    )


def _load_figures(paths: ProjectPaths) -> list[FigureArtifact]:
    figures = []
    manifest = read_json(paths.figures / "figure_manifest.json")
    records = manifest.get("figures", []) if isinstance(manifest, dict) else manifest
    for item in records:
        png_value = item["png"]
        svg_value = item["svg"]
        figures.append(
            FigureArtifact(
                name=item["name"],
                png=Path(png_value) if Path(png_value).is_absolute() else paths.figures / png_value,
                svg=Path(svg_value) if Path(svg_value).is_absolute() else paths.figures / svg_value,
                caption_facts=item.get("caption_facts", item.get("facts", {})),
                qa=item.get(
                    "qa",
                    {
                        "png_sha256": item.get("png_sha256"),
                        "svg_sha256": item.get("svg_sha256"),
                        "width_px": item.get("width_px"),
                        "height_px": item.get("height_px"),
                    },
                ),
            )
        )
    return order_figures(figures)


def _ensure_report_prerequisites(
    paths: ProjectPaths,
    config: ProjectConfig,
    tables: CanonicalTables,
    manifest: AcquisitionManifest,
) -> None:
    """Materialize the report contract from disk-backed large-scale artifacts."""
    if not (paths.quality / "summary.json").exists():
        _save_quality(paths, build_quality_report(manifest, tables))
    if not (paths.analyses / "summary.json").exists():
        visual = paths.canonical / "visualization"
        if not visual.exists():
            raise FileNotFoundError(
                "analyses/summary.json and canonical/visualization are both missing"
            )
        analyses = build_large_scale_analysis(paths, tables)
        _save_analyses(paths, analyses)
        save_scale_plan(tables, config, paths.audit / "scale_plan.json")


def _load_evidence(paths: ProjectPaths) -> EvidenceBundle:
    items = [
        EvidenceItem.model_validate(item)
        for item in read_json(paths.evidence / "evidence_items.json")
    ]
    graph_data = read_json(paths.evidence / "evidence_graph.json")
    graph = nx.DiGraph()
    for node in graph_data["nodes"]:
        payload = dict(node)
        node_id = payload.pop("id")
        graph.add_node(node_id, **payload)
    for edge in graph_data["edges"]:
        payload = dict(edge)
        source = payload.pop("source")
        target = payload.pop("target")
        graph.add_edge(source, target, **payload)
    return EvidenceBundle(items, graph)


def resume_generation(
    root: Path,
    *,
    use_llm: bool = True,
    llm_api_key: str | None = None,
    review_rounds: int = 1,
) -> dict[str, Any]:
    """Resume a project after all deterministic artifacts have been saved."""
    paths = ProjectPaths(root)
    config = load_config(paths.root / "project.yml")
    tables = _load_canonical(paths)
    manifest = AcquisitionManifest.model_validate(
        read_json(paths.audit / "acquisition_manifest.json")
    )
    _ensure_report_prerequisites(paths, config, tables, manifest)
    analyses = _load_analyses(paths)
    figures = _load_figures(paths)
    quality = QualityReport(
        summary=read_json(paths.quality / "summary.json"),
        field_coverage=pd.read_parquet(paths.quality / "field_coverage.parquet"),
        analysis_readiness=pd.read_parquet(paths.quality / "analysis_readiness.parquet"),
    )
    evidence = build_evidence(manifest, tables, analyses, figures, quality)
    save_evidence(evidence, paths.evidence)
    _set_state(paths, "generating", {"resumed": True})
    if use_llm:
        generation = generate_manuscript(
            config,
            evidence,
            api_key=llm_api_key,
            review_rounds=review_rounds,
            candidate_dir=paths.audit / "generation_candidates",
            stage_dir=paths.report / "generation_stages",
        )
    else:
        generation = _deterministic_manuscript(config, evidence)
    save_generation(generation, paths.report)
    claim_ledger = bind_claims(evidence, generation.manuscript)
    write_parquet(paths.evidence / "claim_ledger.parquet", claim_ledger)
    claim_ledger.to_csv(paths.evidence / "claim_ledger.csv", index=False)
    save_evidence(evidence, paths.evidence)
    html_path = _render_html(paths, config, quality, analyses, figures, generation.manuscript)
    word_path = export_word_report(paths.root)
    _set_state(paths, "complete", {"report": str(html_path), "resumed": True})
    _package_manifest(paths)
    return {
        "project_root": str(paths.root),
        "documents": analyses.summary["documents"],
        "figures": len(figures),
        "evidence_items": len(evidence.items),
        "generation_model": generation.model,
        "generation_validation": generation.validation,
        "report": str(html_path),
        "word_report": str(word_path),
        "acquisition_complete": manifest.complete,
        "resumed": True,
    }


def promote_candidate(
    root: Path,
    candidate: Path,
    *,
    model: str = "deepseek-v4-flash",
) -> dict[str, Any]:
    """Validate and promote a persisted candidate without another model call."""
    paths = ProjectPaths(root)
    config = load_config(paths.root / "project.yml")
    analyses = _load_analyses(paths)
    figures = _load_figures(paths)
    evidence = _load_evidence(paths)
    quality = QualityReport(
        summary=read_json(paths.quality / "summary.json"),
        field_coverage=pd.read_parquet(paths.quality / "field_coverage.parquet"),
        analysis_readiness=pd.read_parquet(paths.quality / "analysis_readiness.parquet"),
    )
    manuscript = _normalize_evidence_tokens(candidate.read_text(encoding="utf-8"))
    validation = validate_manuscript(manuscript, evidence)
    if not validation["valid"]:
        raise QualityGateError(
            "Candidate cannot be promoted: " + json.dumps(validation, ensure_ascii=False)
        )
    result = GenerationResult(
        manuscript=manuscript,
        review=None,
        validation={**validation, "promoted_from": str(candidate)},
        model=model,
        quality=validation["journal_readiness"],
    )
    save_generation(result, paths.report)
    claim_ledger = bind_claims(evidence, result.manuscript)
    write_parquet(paths.evidence / "claim_ledger.parquet", claim_ledger)
    claim_ledger.to_csv(paths.evidence / "claim_ledger.csv", index=False)
    save_evidence(evidence, paths.evidence)
    html_path = _render_html(paths, config, quality, analyses, figures, result.manuscript)
    word_path = export_word_report(paths.root)
    _set_state(
        paths,
        "complete",
        {"report": str(html_path), "promoted_candidate": str(candidate)},
    )
    _package_manifest(paths)
    return {
        "project_root": str(paths.root),
        "report": str(html_path),
        "word_report": str(word_path),
        "generation_validation": result.validation,
        "model": model,
    }


def refine_staged_generation(
    root: Path,
    *,
    llm_api_key: str | None = None,
) -> dict[str, Any]:
    """Evidence-repair reviewed sections without repeating acquisition or drafting."""
    paths = ProjectPaths(root)
    config = load_config(paths.root / "project.yml")
    evidence = _load_evidence(paths)
    analyses = _load_analyses(paths)
    figures = _load_figures(paths)
    quality = QualityReport(
        summary=read_json(paths.quality / "summary.json"),
        field_coverage=pd.read_parquet(paths.quality / "field_coverage.parquet"),
        analysis_readiness=pd.read_parquet(paths.quality / "analysis_readiness.parquet"),
    )
    result = finalize_staged_manuscript(
        config,
        evidence,
        paths.report / "generation_stages",
        api_key=llm_api_key,
    )
    save_generation(result, paths.report)
    claim_ledger = bind_claims(evidence, result.manuscript)
    write_parquet(paths.evidence / "claim_ledger.parquet", claim_ledger)
    claim_ledger.to_csv(paths.evidence / "claim_ledger.csv", index=False)
    save_evidence(evidence, paths.evidence)
    html_path = _render_html(paths, config, quality, analyses, figures, result.manuscript)
    word_path = export_word_report(paths.root)
    _set_state(
        paths,
        "complete",
        {"report": str(html_path), "refined_from_stages": True},
    )
    _package_manifest(paths)
    return {
        "project_root": str(paths.root),
        "report": str(html_path),
        "word_report": str(word_path),
        "generation_validation": result.validation,
        "model": result.model,
        "refined_from_stages": True,
    }


def recompute_downstream(root: Path) -> dict[str, Any]:
    """Rebuild quality, analyses, figures, and evidence without reacquisition."""
    paths = ProjectPaths(root)
    config = load_config(paths.root / "project.yml")
    manifest = AcquisitionManifest.model_validate(
        read_json(paths.audit / "acquisition_manifest.json")
    )
    tables = _load_canonical(paths)
    _save_canonical(paths, tables)
    _set_state(paths, "recomputing_downstream")
    quality = build_quality_report(manifest, tables)
    _save_quality(paths, quality)
    analyses = analyze(
        tables,
        network_candidate_pool=max(config.visualization_max_nodes * 8, 400),
    )
    save_scale_plan(tables, config, paths.audit / "scale_plan.json")
    _save_analyses(paths, analyses)
    export_all(tables, analyses, paths.analyses / "exports")
    figures = render_all(
        tables,
        analyses,
        paths.figures,
        max_nodes=config.visualization_max_nodes,
        label_budget=config.visualization_label_budget,
        seed=config.random_seed,
    )
    write_json(
        paths.figures / "figure_manifest.json",
        [
            {
                "name": figure.name,
                "png": figure.png.name,
                "svg": figure.svg.name,
                "caption_facts": figure.caption_facts,
                "qa": figure.qa,
            }
            for figure in figures
        ],
    )
    evidence = build_evidence(manifest, tables, analyses, figures, quality)
    save_evidence(evidence, paths.evidence)
    word_path = (
        export_word_report(paths.root) if (paths.report / "manuscript.md").exists() else None
    )
    _set_state(
        paths,
        "deterministic_complete",
        {"figures": len(figures), "evidence_items": len(evidence.items)},
    )
    _package_manifest(paths)
    return {
        "project_root": str(paths.root),
        "documents": analyses.summary["documents"],
        "figures": len(figures),
        "evidence_items": len(evidence.items),
        "state": "deterministic_complete",
        "word_report": str(word_path) if word_path else None,
    }
