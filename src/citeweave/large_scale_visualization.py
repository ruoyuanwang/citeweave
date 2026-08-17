from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from .io import load_config, sha256_file, write_json, write_parquet
from .models import ProjectConfig, VisualizationPolicy
from .visual_selection import SelectedNetwork, select_network

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None


PAPER = "#FBFCFE"
INK = "#172033"
MUTED = "#64748B"
GRID = "#DDE4EC"
CLUSTERS = [
    "#2474B5",
    "#E56B39",
    "#2A9D78",
    "#8661C1",
    "#D53E62",
    "#0F9FA8",
    "#C69A1B",
    "#5576D1",
]


def _path_sql(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "svg.fonttype": "none",
        }
    )


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    facts: dict[str, Any],
    *,
    dpi: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    svg = output_dir / f"{name}.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    record = {
        "name": name,
        "png": str(png.resolve()),
        "svg": str(svg.resolve()),
        "png_sha256": sha256_file(png),
        "svg_sha256": sha256_file(svg),
        "width_px": int(fig.get_figwidth() * dpi),
        "height_px": int(fig.get_figheight() * dpi),
        "facts": facts,
    }
    plt.close(fig)
    return record


def _subtitle(ax: plt.Axes, value: str) -> None:
    ax.text(0, 1.012, value, transform=ax.transAxes, color=MUTED, fontsize=8.6, va="bottom")


def _shorten(value: Any, limit: int = 44) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _bar(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    subtitle: str,
    name: str,
    output_dir: Path,
    dpi: int,
    top_n: int = 15,
    x_label: str = "Documents",
) -> dict[str, Any] | None:
    data = frame.dropna(subset=[label]).copy()
    data = data[data[label].astype(str).str.strip().ne("")]
    data = data.nlargest(top_n, value).sort_values(value)
    if data.empty:
        return None
    data[label] = data[label].map(lambda item: _shorten(item, 54))
    fig, ax = plt.subplots(figsize=(11.2, max(5.8, 0.38 * len(data) + 1.9)))
    colors = [CLUSTERS[0]] * len(data)
    colors[-1] = CLUSTERS[1]
    bars = ax.barh(data[label], data[value], color=colors, height=0.68)
    ax.bar_label(bars, labels=[f"{int(value):,}" for value in data[value]], padding=4, fontsize=8)
    ax.set_title(title, loc="left", pad=27)
    _subtitle(ax, subtitle)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(0, max(float(data[value].max()) * 1.16, 1))
    return _save_figure(
        fig,
        output_dir,
        name,
        {
            "displayed": len(data),
            "top_label": str(data.iloc[-1][label]),
            "top_value": int(data.iloc[-1][value]),
        },
        dpi=dpi,
    )


def _descriptive_figures(
    connection: duckdb.DuckDBPyConnection,
    canonical: Path,
    visual: Path,
    output_dir: Path,
    policy: VisualizationPolicy,
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any] | None] = []
    annual = connection.execute(
        f"SELECT * FROM read_parquet('{_path_sql(visual / 'annual_output.parquet')}') ORDER BY year"
    ).df()
    if not annual.empty:
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        ax.plot(
            annual["year"],
            annual["documents"],
            color=CLUSTERS[0],
            lw=2.8,
            marker="o",
            ms=6,
            markerfacecolor=PAPER,
            markeredgewidth=1.8,
        )
        ax.fill_between(annual["year"], annual["documents"], color=CLUSTERS[0], alpha=0.1)
        peak = annual.loc[annual["documents"].idxmax()]
        ax.annotate(
            f"{int(peak['documents']):,}",
            (peak["year"], peak["documents"]),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            color=CLUSTERS[0],
            fontweight="bold",
        )
        ax.set_title("Annual scientific production", loc="left", pad=27)
        _subtitle(ax, "All included records; counts are not sampled")
        ax.set_xlabel("Publication year")
        ax.set_ylabel("Documents")
        ax.spines[["top", "right"]].set_visible(False)
        figures.append(
            _save_figure(
                fig,
                output_dir,
                "annual_publications",
                {
                    "years": [int(value) for value in annual["year"]],
                    "peak_year": int(peak["year"]),
                    "peak_documents": int(peak["documents"]),
                },
                dpi=policy.dpi,
            )
        )
    specifications = [
        (
            "source_productivity.parquet",
            "source_name",
            "documents",
            "documents DESC, citations DESC, source_id",
            "Most productive sources",
            "Full-corpus journal and venue output",
            "top_sources",
        ),
        (
            "author_productivity.parquet",
            "author_name",
            "documents",
            "documents DESC, citations DESC, author_id",
            "Most productive authors",
            "Whole-counted publications; missing author names excluded from ranking",
            "top_authors",
        ),
        (
            "institution_productivity.parquet",
            "institution_name",
            "documents",
            "documents DESC, institution_id",
            "Most productive institutions",
            "Documents with at least one affiliated author",
            "top_institutions",
        ),
    ]
    for filename, label, value, order, title, subtitle, name in specifications:
        frame = connection.execute(
            f"""
            SELECT *
            FROM read_parquet('{_path_sql(visual / filename)}')
            WHERE {label} IS NOT NULL AND trim({label}) <> ''
            ORDER BY {order}
            LIMIT 15
            """
        ).df()
        figures.append(
            _bar(
                frame,
                label=label,
                value=value,
                title=title,
                subtitle=subtitle,
                name=name,
                output_dir=output_dir,
                dpi=policy.dpi,
            )
        )
    types = connection.execute(
        f"SELECT * FROM read_parquet('{_path_sql(visual / 'document_types.parquet')}') "
        "ORDER BY documents DESC"
    ).df()
    if not types.empty:
        head = types.head(7).copy()
        if len(types) > 7:
            head.loc[len(head)] = ["Other", int(types.iloc[7:]["documents"].sum())]
        head = head.sort_values("documents")
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        bars = ax.barh(
            head["document_type"].map(lambda value: _shorten(value, 34)),
            head["documents"],
            color=[CLUSTERS[index % len(CLUSTERS)] for index in range(len(head))],
        )
        ax.bar_label(bars, labels=[f"{int(value):,}" for value in head["documents"]], padding=4)
        ax.set_title("Document type composition", loc="left", pad=27)
        _subtitle(ax, "Small categories are combined as Other")
        ax.set_xlabel("Documents")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(0, float(head["documents"].max()) * 1.15)
        figures.append(
            _save_figure(
                fig,
                output_dir,
                "document_types",
                {
                    "displayed_categories": len(head),
                    "total_documents": int(types["documents"].sum()),
                },
                dpi=policy.dpi,
            )
        )
    works = _path_sql(canonical / "works.parquet")
    citation_sample = connection.execute(
        f"""
        WITH stats AS (
          SELECT quantile_cont(cited_by_count, 0.98) AS cap
          FROM read_parquet('{works}')
        )
        SELECT LEAST(w.cited_by_count, s.cap)::DOUBLE AS shown
        FROM read_parquet('{works}') w, stats s
        WHERE w.cited_by_count IS NOT NULL
        """
    ).df()
    citation_stats = connection.execute(
        f"""
        SELECT count(*) AS documents, avg(cited_by_count) AS mean,
               median(cited_by_count) AS median, max(cited_by_count) AS maximum,
               count(*) FILTER (WHERE cited_by_count = 0) AS zero_cited,
               quantile_cont(cited_by_count, 0.98) AS cap
        FROM read_parquet('{works}')
        """
    ).fetchone()
    if not citation_sample.empty:
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        sns.histplot(citation_sample["shown"], bins=42, color=CLUSTERS[0], edgecolor=PAPER, ax=ax)
        ax.axvline(float(citation_stats[2]), ls="--", lw=2, color=CLUSTERS[1])
        ax.set_title("Citation distribution", loc="left", pad=27)
        _subtitle(
            ax,
            f"Display capped at the 98th percentile ({citation_stats[5]:.0f}); facts use all records",
        )
        ax.set_xlabel("Citations per document")
        ax.set_ylabel("Documents")
        ax.spines[["top", "right"]].set_visible(False)
        figures.append(
            _save_figure(
                fig,
                output_dir,
                "citation_distribution",
                {
                    "documents": int(citation_stats[0]),
                    "mean": float(citation_stats[1]),
                    "median": float(citation_stats[2]),
                    "maximum": int(citation_stats[3]),
                    "zero_cited": int(citation_stats[4]),
                    "display_cap_p98": float(citation_stats[5]),
                },
                dpi=policy.dpi,
            )
        )
    top_cited = connection.execute(
        f"""
        SELECT title, year, cited_by_count
        FROM read_parquet('{works}')
        WHERE title IS NOT NULL AND trim(title) <> ''
        ORDER BY cited_by_count DESC NULLS LAST
        LIMIT 15
        """
    ).df()
    top_cited["label"] = top_cited.apply(
        lambda row: (
            f"{_shorten(row['title'], 58)} ({int(row['year']) if pd.notna(row['year']) else 'n.d.'})"
        ),
        axis=1,
    )
    figures.append(
        _bar(
            top_cited,
            label="label",
            value="cited_by_count",
            title="Most cited documents",
            subtitle="Source-supplied citation counts at retrieval time",
            name="top_cited_documents",
            output_dir=output_dir,
            dpi=policy.dpi,
            x_label="Citations",
        )
    )
    source_frame = connection.execute(
        f"""
        SELECT source_name, documents,
               row_number() OVER (ORDER BY documents DESC, source_name) AS rank,
               sum(documents) OVER (ORDER BY documents DESC, source_name) AS cumulative,
               sum(documents) OVER () AS total
        FROM read_parquet('{_path_sql(visual / "source_productivity.parquet")}')
        WHERE source_name IS NOT NULL AND trim(source_name) <> ''
        ORDER BY documents DESC
        """
    ).df()
    if not source_frame.empty:
        source_frame["zone"] = np.minimum(
            3,
            np.floor(3 * (source_frame["cumulative"] - 1) / source_frame["total"]).astype(int) + 1,
        )
        fig, ax = plt.subplots(figsize=(10.8, 6.3))
        for zone, group in source_frame.groupby("zone"):
            ax.plot(
                group["rank"],
                group["documents"],
                color=CLUSTERS[int(zone) - 1],
                lw=2,
                label=f"Zone {int(zone)} · {len(group):,} sources",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Bradford source concentration", loc="left", pad=27)
        _subtitle(ax, "Sources ranked by productivity; zones contain about one-third of output")
        ax.set_xlabel("Source rank (log scale)")
        ax.set_ylabel("Documents (log scale)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        figures.append(
            _save_figure(
                fig,
                output_dir,
                "bradford_sources",
                {
                    "sources": len(source_frame),
                    "zone_1_sources": int((source_frame["zone"] == 1).sum()),
                },
                dpi=policy.dpi,
            )
        )
    keyword_trends = connection.execute(
        f"""
        WITH top_terms AS (
          SELECT keyword, count(DISTINCT work_id) AS documents
          FROM read_parquet('{_path_sql(canonical / "keywords.parquet")}')
          GROUP BY keyword ORDER BY documents DESC LIMIT 15
        )
        SELECT k.keyword, w.year, count(DISTINCT k.work_id) AS documents
        FROM read_parquet('{_path_sql(canonical / "keywords.parquet")}') k
        JOIN top_terms l USING(keyword)
        JOIN read_parquet('{works}') w USING(work_id)
        WHERE w.year IS NOT NULL
        GROUP BY k.keyword, w.year
        ORDER BY k.keyword, w.year
        """
    ).df()
    if not keyword_trends.empty:
        pivot = keyword_trends.pivot(index="keyword", columns="year", values="documents").fillna(0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(11.8, 7.7))
        sns.heatmap(
            pivot.astype(int),
            cmap=sns.light_palette(CLUSTERS[0], as_cmap=True),
            annot=True,
            fmt="d",
            linewidths=0.55,
            linecolor=PAPER,
            cbar_kws={"label": "Documents"},
            ax=ax,
        )
        ax.set_title("Keyword evolution", loc="left", pad=25)
        _subtitle(ax, "Top terms by unique-document frequency; all years shown")
        ax.set_xlabel("Publication year")
        ax.set_ylabel("")
        figures.append(
            _save_figure(
                fig,
                output_dir,
                "keyword_trends",
                {"keywords": len(pivot), "years": [int(value) for value in pivot.columns]},
                dpi=policy.dpi,
            )
        )
    return [figure for figure in figures if figure is not None]


def _linked_node_table(
    connection: duckdb.DuckDBPyConnection,
    canonical: Path,
    visual: Path,
    *,
    network: str,
    edge_path: Path,
) -> pd.DataFrame:
    """Load only nodes incident to the already-bounded sparse edge table.

    ``select_network`` immediately discards every unlinked node, so applying the
    same semi-join in DuckDB preserves the exact selected network while keeping
    large author, work, and cited-reference dimensions out of pandas memory.
    """
    linked = f"""
        WITH linked_ids AS (
          SELECT source_id AS id FROM read_parquet('{_path_sql(edge_path)}')
          UNION
          SELECT target_id AS id FROM read_parquet('{_path_sql(edge_path)}')
        )
    """
    if network == "coauthorship":
        return connection.execute(
            f"""
            {linked}
            SELECT author_id AS id, coalesce(author_name, author_id) AS label,
                   documents AS occurrences, citations
            FROM read_parquet(
              '{_path_sql(visual / "author_productivity.parquet")}',
              file_row_number = true
            )
            JOIN linked_ids ON linked_ids.id = author_id
            WHERE author_name IS NOT NULL AND trim(author_name) <> ''
            ORDER BY file_row_number
            """
        ).df()
    if network == "institution_collaboration":
        return connection.execute(
            f"""
            {linked}
            SELECT institution_id AS id, institution_name AS label,
                   documents AS occurrences, country_code
            FROM read_parquet(
              '{_path_sql(visual / "institution_productivity.parquet")}',
              file_row_number = true
            )
            JOIN linked_ids ON linked_ids.id = institution_id
            WHERE institution_name IS NOT NULL AND trim(institution_name) <> ''
            ORDER BY file_row_number
            """
        ).df()
    if network == "keyword_cooccurrence":
        return connection.execute(
            f"""
            {linked},
            keyword_year AS (
              SELECT k.keyword AS id, avg(w.year) AS average_year
              FROM read_parquet('{_path_sql(canonical / "keywords.parquet")}') k
              JOIN linked_ids ON linked_ids.id = k.keyword
              JOIN read_parquet('{_path_sql(canonical / "works.parquet")}') w USING(work_id)
              WHERE w.year IS NOT NULL
              GROUP BY k.keyword
            )
            SELECT keyword AS id, keyword AS label, occurrences, keyword_type
                   , keyword_year.average_year
            FROM read_parquet(
              '{_path_sql(visual / "keyword_occurrences.parquet")}',
              file_row_number = true
            )
            JOIN linked_ids ON linked_ids.id = keyword
            LEFT JOIN keyword_year ON keyword_year.id = keyword
            WHERE keyword IS NOT NULL AND trim(keyword) <> ''
            ORDER BY file_row_number
            """
        ).df()
    if network == "cocitation":
        return connection.execute(
            f"""
            {linked}
            SELECT cited_work_id AS id,
                   coalesce(
                     CASE WHEN cited_author IS NOT NULL AND trim(cited_author) <> ''
                          THEN cited_author || coalesce(' (' || cited_year::VARCHAR || ')', '')
                     END,
                     nullif(cited_doi, ''),
                     nullif(trim(cited_title), ''),
                     cited_work_id
                   ) AS label,
                   local_citations AS occurrences, cited_year AS year
            FROM read_parquet(
              '{_path_sql(visual / "reference_impact.parquet")}',
              file_row_number = true
            )
            JOIN linked_ids ON linked_ids.id = cited_work_id
            WHERE cited_work_id NOT LIKE 'reference:%'
            ORDER BY file_row_number
            """
        ).df()
    if network == "citation":
        return connection.execute(
            f"""
            {linked}
            SELECT work_id AS id, title AS label,
                   greatest(coalesce(cited_by_count, 0), 1) AS occurrences, year
            FROM read_parquet(
              '{_path_sql(canonical / "works.parquet")}',
              file_row_number = true
            )
            JOIN linked_ids ON linked_ids.id = work_id
            WHERE title IS NOT NULL AND trim(title) <> ''
            ORDER BY file_row_number
            """
        ).df()
    raise ValueError(f"Unsupported network node table: {network}")


def _bibliographic_coupling(
    connection: duckdb.DuckDBPyConnection,
    canonical: Path,
    policy: VisualizationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = min(policy.max_candidate_nodes, policy.max_display_nodes * policy.candidate_multiplier)
    works = _path_sql(canonical / "works.parquet")
    references = _path_sql(canonical / "references.parquet")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE coupling_candidates AS
        SELECT work_id AS id, title AS label, greatest(reference_count, 1) AS occurrences,
               year, cited_by_count
        FROM read_parquet('{works}')
        WHERE title IS NOT NULL AND trim(title) <> '' AND reference_count >= 5
        ORDER BY (ln(1 + greatest(cited_by_count, 0)) + 0.35 * ln(1 + reference_count)) DESC,
                 work_id
        LIMIT {int(target)}
        """
    )
    candidate_count = connection.execute("SELECT count(*) FROM coupling_candidates").fetchone()[0]
    max_reference_frequency = max(10, math.ceil(candidate_count * 0.20))
    edges = connection.execute(
        f"""
        WITH candidate_refs AS (
          SELECT DISTINCT r.citing_work_id, r.cited_work_id
          FROM read_parquet('{references}') r
          JOIN coupling_candidates c ON c.id = r.citing_work_id
          WHERE r.cited_work_id IS NOT NULL
            AND r.cited_work_id NOT LIKE 'reference:%'
        ),
        informative_refs AS (
          SELECT cited_work_id
          FROM candidate_refs
          GROUP BY cited_work_id
          HAVING count(*) BETWEEN 2 AND {max_reference_frequency}
        )
        SELECT a.citing_work_id AS source_id, b.citing_work_id AS target_id,
               count(*) AS weight
        FROM candidate_refs a
        JOIN candidate_refs b
          ON a.cited_work_id = b.cited_work_id
         AND a.citing_work_id < b.citing_work_id
        JOIN informative_refs i ON i.cited_work_id = a.cited_work_id
        GROUP BY a.citing_work_id, b.citing_work_id
        HAVING count(*) >= 2
        ORDER BY weight DESC, source_id, target_id
        LIMIT 200000
        """
    ).df()
    nodes = connection.execute("SELECT * FROM coupling_candidates").df()
    return (
        nodes,
        edges,
        {
            "candidate_documents": int(candidate_count),
            "minimum_references_per_document": 5,
            "minimum_shared_references": 2,
            "maximum_reference_document_frequency": int(max_reference_frequency),
            "matrix_materialized": False,
            "construction": "candidate documents first, then sparse shared-reference self-join",
        },
    )


def _label_nodes(network: SelectedNetwork, budget: int) -> list[str]:
    nodes = network.nodes.copy()
    nodes = nodes[nodes["label"].astype(str).str.strip().ne("")]
    cluster_count = max(nodes["cluster"].nunique(), 1)
    quota = max(1, budget // cluster_count)
    selected: list[str] = []
    for _, group in nodes.groupby("cluster"):
        selected.extend(group.nlargest(quota, "importance")["id"].astype(str))
    for node in nodes.nlargest(budget, "importance")["id"].astype(str):
        if len(selected) >= budget:
            break
        if node not in selected:
            selected.append(node)
    return selected[:budget]


def _label_overlap(texts: list[matplotlib.text.Text], fig: plt.Figure) -> tuple[int, float]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer).expanded(1.02, 1.08) for text in texts]
    overlap = 0
    area = 0.0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if left.overlaps(right):
                overlap += 1
                x = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
                y = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
                area += x * y
    return overlap, area


def _network_plot(
    network: SelectedNetwork,
    output_dir: Path,
    policy: VisualizationPolicy,
) -> dict[str, Any]:
    nodes = network.nodes.set_index("id")
    graph = nx.Graph()
    for row in network.edges.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(row.weight),
            strength=float(row.association_strength),
        )
    positions = network.positions
    occurrence = nodes.loc[list(graph), "occurrences"].astype(float).clip(lower=1)
    lo, hi = occurrence.quantile([0.05, 0.95]).tolist()
    scaled = np.sqrt((occurrence.clip(lo, max(hi, lo + 1)) - lo) / max(hi - lo, 1))
    sizes = 55 + 650 * scaled
    weights = np.asarray([data["weight"] for _, _, data in graph.edges(data=True)], dtype=float)
    low, high = np.quantile(weights, [0.05, 0.95])
    edge_widths = 0.25 + 2.2 * np.sqrt(
        (np.clip(weights, low, max(high, low + 1)) - low) / max(high - low, 1)
    )
    fig, ax = plt.subplots(figsize=(13.4, 8.8))
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color="#7D8A9B",
        alpha=0.22,
        ax=ax,
    )
    cluster_values = nodes.loc[list(graph), "cluster"].astype(int)
    colors = [CLUSTERS[(value - 1) % len(CLUSTERS)] for value in cluster_values]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=sizes.loc[list(graph)].tolist(),
        node_color=colors,
        edgecolors=PAPER,
        linewidths=1.0,
        alpha=0.94,
        ax=ax,
    )
    long_labels = network.name in {"cocitation", "citation", "bibliographic_coupling"}
    if long_labels:
        label_budget = min(policy.label_budget, 12)
    elif network.name == "coauthorship":
        label_budget = min(policy.label_budget, 16)
    elif network.name == "institution_collaboration":
        label_budget = min(policy.label_budget, 18)
    else:
        label_budget = min(policy.label_budget, 20)
    label_ids = _label_nodes(network, label_budget)
    texts = []
    importance = nodes["importance"]
    for node in label_ids:
        x, y = positions[node]
        relative = float(importance.loc[node] / max(float(importance.max()), 1e-9))
        label = _shorten(nodes.loc[node, "label"], 25 if long_labels else 26)
        texts.append(
            ax.text(
                x,
                y,
                label,
                fontsize=7.1 + 2.2 * relative,
                ha="center",
                va="center",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": PAPER,
                    "edgecolor": "none",
                    "alpha": 0.84,
                },
            )
        )
    if adjust_text and texts:
        adjust_text(
            texts,
            x=[positions[node][0] for node in graph],
            y=[positions[node][1] for node in graph],
            ax=ax,
            expand=(1.14, 1.25),
            force_text=(0.65, 0.85),
            force_points=(0.20, 0.30),
            ensure_inside_axes=True,
            max_move=(22, 28),
            arrowprops={"arrowstyle": "-", "color": "#8391A3", "lw": 0.45, "alpha": 0.6},
        )
    title_map = {
        "coauthorship": "Author collaboration network",
        "institution_collaboration": "Institution collaboration network",
        "keyword_cooccurrence": "Keyword co-occurrence network",
        "citation": "Within-corpus citation network",
        "cocitation": "Reference co-citation network",
        "bibliographic_coupling": "Document bibliographic coupling",
    }
    ax.set_title(title_map[network.name], loc="left", pad=27)
    _subtitle(
        ax,
        f"{len(graph)} nodes · {len(network.edges)} links · "
        f"minimum occurrence {network.disclosure['minimum_occurrence']} · "
        "association-strength normalized",
    )
    ax.axis("off")
    all_points = np.asarray([positions[node] for node in graph])
    x_pad = max(np.ptp(all_points[:, 0]) * 0.075, 0.12)
    y_pad = max(np.ptp(all_points[:, 1]) * 0.10, 0.12)
    ax.set_xlim(all_points[:, 0].min() - x_pad, all_points[:, 0].max() + x_pad)
    ax.set_ylim(all_points[:, 1].min() - y_pad, all_points[:, 1].max() + y_pad)
    overlap_count, overlap_area = _label_overlap(texts, fig)
    facts = {
        **network.disclosure,
        **network.layout_qa,
        "labels": len(texts),
        "label_overlap_pairs": overlap_count,
        "label_overlap_area_px2": round(overlap_area, 2),
        "size_encoding": "winsorized square-root occurrence",
        "color_encoding": "Louvain community",
    }
    return _save_figure(fig, output_dir, f"network_{network.name}", facts, dpi=policy.dpi)


def _overlay_plot(
    network: SelectedNetwork,
    output_dir: Path,
    policy: VisualizationPolicy,
) -> dict[str, Any] | None:
    if "average_year" not in network.nodes or network.nodes["average_year"].notna().sum() < 4:
        return None
    nodes = network.nodes.set_index("id")
    ids = [node for node in network.positions if pd.notna(nodes.loc[node, "average_year"])]
    years = nodes.loc[ids, "average_year"].astype(float)
    sizes = nodes.loc[ids, "occurrences"].astype(float)
    sizes = 70 + 700 * np.sqrt(sizes / max(float(sizes.max()), 1))
    fig, ax = plt.subplots(figsize=(13.4, 8.8))
    for row in network.edges.itertuples(index=False):
        left, right = network.positions[str(row.source)], network.positions[str(row.target)]
        ax.plot(*zip(left, right), color="#8995A5", alpha=0.14, lw=0.55, zorder=1)
    points = np.asarray([network.positions[node] for node in ids])
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        s=sizes,
        c=years,
        cmap="viridis",
        edgecolor=PAPER,
        linewidth=0.8,
        alpha=0.94,
        zorder=2,
    )
    texts = []
    for node in _label_nodes(network, min(policy.label_budget, 20)):
        if node not in ids:
            continue
        x, y = network.positions[node]
        texts.append(
            ax.text(
                x,
                y,
                _shorten(nodes.loc[node, "label"], 28),
                fontsize=7.5,
                ha="center",
                va="center",
                bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
            )
        )
    if adjust_text and texts:
        adjust_text(texts, ax=ax, expand=(1.12, 1.22), ensure_inside_axes=True, max_move=(20, 25))
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.02)
    colorbar.set_label("Average publication year", fontsize=9.5, labelpad=8)
    colorbar.ax.tick_params(labelsize=8.5)
    ax.set_title("Keyword co-occurrence overlay", loc="left", pad=27)
    _subtitle(ax, "Node color shows the average publication year of documents using each term")
    ax.axis("off")
    x_pad = max(np.ptp(points[:, 0]) * 0.08, 0.12)
    y_pad = max(np.ptp(points[:, 1]) * 0.10, 0.12)
    ax.set_xlim(points[:, 0].min() - x_pad, points[:, 0].max() + x_pad)
    ax.set_ylim(points[:, 1].min() - y_pad, points[:, 1].max() + y_pad)
    overlap, area = _label_overlap(texts, fig)
    return _save_figure(
        fig,
        output_dir,
        "network_keyword_overlay",
        {
            **network.disclosure,
            "year_min": float(years.min()),
            "year_max": float(years.max()),
            "labels": len(texts),
            "label_overlap_pairs": overlap,
            "label_overlap_area_px2": round(area, 2),
        },
        dpi=policy.dpi,
    )


def _density_plot(
    network: SelectedNetwork,
    output_dir: Path,
    policy: VisualizationPolicy,
) -> dict[str, Any]:
    nodes = network.nodes.set_index("id")
    ids = list(network.positions)
    points = np.asarray([network.positions[node] for node in ids])
    weights = np.log1p(nodes.loc[ids, "occurrences"].astype(float).to_numpy())
    x_edges = np.linspace(points[:, 0].min() - 0.2, points[:, 0].max() + 0.2, 260)
    y_edges = np.linspace(points[:, 1].min() - 0.2, points[:, 1].max() + 0.2, 190)
    heat, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=(x_edges, y_edges), weights=weights
    )
    heat = gaussian_filter(heat, sigma=8)
    heat = heat / max(float(heat.max()), 1e-9)
    heat = np.power(heat, 0.62)
    cmap = LinearSegmentedColormap.from_list(
        "vos_density", ["#16233F", "#235E8B", "#20A386", "#D7D83F", "#FFF4A6"]
    )
    fig, ax = plt.subplots(figsize=(13.4, 8.8))
    ax.imshow(
        heat.T,
        origin="lower",
        extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()],
        cmap=cmap,
        interpolation="bilinear",
        aspect="auto",
    )
    ax.scatter(points[:, 0], points[:, 1], s=10, color="#F8FAFC", alpha=0.45, linewidth=0)
    texts = []
    for node in _label_nodes(network, min(policy.label_budget, 18)):
        x, y = network.positions[node]
        texts.append(
            ax.text(
                x,
                y,
                _shorten(nodes.loc[node, "label"], 28),
                color="#FFFFFF",
                fontsize=7.8,
                ha="center",
                va="center",
                bbox={"facecolor": "#142038", "edgecolor": "none", "alpha": 0.52, "pad": 1.4},
            )
        )
    if adjust_text and texts:
        adjust_text(texts, ax=ax, expand=(1.12, 1.20), ensure_inside_axes=True, max_move=(18, 24))
    ax.set_title("Keyword density visualization", loc="left", pad=27, color=INK)
    ax.text(
        0,
        1.012,
        "Hotspots combine local node occurrence and spatial proximity",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.6,
        va="bottom",
    )
    ax.axis("off")
    overlap, area = _label_overlap(texts, fig)
    return _save_figure(
        fig,
        output_dir,
        "network_keyword_density",
        {
            **network.disclosure,
            "density_grid": [260, 190],
            "gaussian_sigma": 8,
            "density_weight": "log(1 + occurrence), gamma 0.62",
            "labels": len(texts),
            "label_overlap_pairs": overlap,
            "label_overlap_area_px2": round(area, 2),
        },
        dpi=policy.dpi,
    )


def _write_network_data(project: Path, network: SelectedNetwork) -> None:
    output = project / "analyses" / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    nodes = network.nodes.copy()
    nodes["x"] = nodes["id"].astype(str).map(lambda node: float(network.positions[node][0]))
    nodes["y"] = nodes["id"].astype(str).map(lambda node: float(network.positions[node][1]))
    write_parquet(output / f"{network.name}_nodes.parquet", nodes)
    write_parquet(output / f"{network.name}_edges.parquet", network.edges)
    write_json(
        output / f"{network.name}_method.json",
        {**network.disclosure, "layout_qa": network.layout_qa},
    )


def render_large_project(
    project: Path,
    *,
    policy_override: VisualizationPolicy | None = None,
) -> dict[str, Any]:
    """Render a processed project from disk-backed aggregates and sparse networks."""
    started = time.perf_counter()
    project = project.resolve()
    config: ProjectConfig = load_config(project / "project.yml")
    policy = policy_override or config.visualization
    canonical = project / "canonical"
    visual = canonical / "visualization"
    if not visual.exists():
        raise FileNotFoundError("canonical/visualization is missing; run `citeweave process` first")
    _theme()
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config.processing.duckdb_memory_limit}'")
    connection.execute("SET threads=4")
    figures = _descriptive_figures(connection, canonical, visual, project / "figures", policy)
    edge_files = {
        "coauthorship": "coauthor_edges.parquet",
        "institution_collaboration": "institution_collaboration_edges.parquet",
        "keyword_cooccurrence": "keyword_cooccurrence_edges.parquet",
        "cocitation": "cocitation_edges.parquet",
        "citation": "direct_citation_edges.parquet",
    }
    network_records: dict[str, Any] = {}
    selected_networks: dict[str, SelectedNetwork] = {}
    for index, (name, filename) in enumerate(edge_files.items()):
        edge_path = visual / filename
        if not edge_path.exists():
            continue
        edges = connection.execute(f"SELECT * FROM read_parquet('{_path_sql(edge_path)}')").df()
        nodes = _linked_node_table(
            connection,
            canonical,
            visual,
            network=name,
            edge_path=edge_path,
        )
        selected = select_network(
            name,
            nodes,
            edges,
            policy,
            seed=config.random_seed + 97 * index,
        )
        if selected is None:
            network_records[name] = {"status": "skipped", "reason": "no eligible linked items"}
            continue
        selected_networks[name] = selected
        _write_network_data(project, selected)
        figure = _network_plot(selected, project / "figures", policy)
        figures.append(figure)
        network_records[name] = {
            "status": "rendered",
            **selected.disclosure,
            **selected.layout_qa,
        }
    coupling_nodes, coupling_edges, coupling_method = _bibliographic_coupling(
        connection, canonical, policy
    )
    coupling = select_network(
        "bibliographic_coupling",
        coupling_nodes,
        coupling_edges,
        policy,
        seed=config.random_seed + 607,
    )
    if coupling is not None:
        coupling.disclosure.update({"sparse_construction": coupling_method})
        selected_networks["bibliographic_coupling"] = coupling
        _write_network_data(project, coupling)
        figures.append(_network_plot(coupling, project / "figures", policy))
        network_records["bibliographic_coupling"] = {
            "status": "rendered",
            **coupling.disclosure,
            **coupling.layout_qa,
        }
    else:
        network_records["bibliographic_coupling"] = {
            "status": "skipped",
            "reason": "insufficient shared references among candidates",
            **coupling_method,
        }
    keyword = selected_networks.get("keyword_cooccurrence")
    if keyword is not None:
        overlay = _overlay_plot(keyword, project / "figures", policy)
        if overlay:
            figures.append(overlay)
        figures.append(_density_plot(keyword, project / "figures", policy))
    connection.close()
    manifest = {
        "version": 1,
        "project": str(project),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "policy": policy.model_dump(mode="json"),
        "scalability": {
            "canonical_relations_loaded_as_full_pandas_tables": False,
            "full_adjacency_matrix_materialized": False,
            "descriptive_statistics": "DuckDB scans full Parquet corpus",
            "network_maps": "candidate-first sparse edges; displayed graph is bounded",
        },
        "networks": network_records,
        "figures": figures,
        "figure_count": len(figures),
    }
    write_json(project / "figures" / "figure_manifest.json", manifest)
    return manifest
