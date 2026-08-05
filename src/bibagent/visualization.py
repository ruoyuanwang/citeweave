from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .analytics import AnalysisBundle, NetworkResult
from .io import sha256_file
from .transform import CanonicalTables

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None


PALETTE = [
    "#2563EB",
    "#F97316",
    "#10B981",
    "#A855F7",
    "#E11D48",
    "#0891B2",
    "#CA8A04",
    "#4F46E5",
    "#059669",
    "#DB2777",
]
INK = "#172033"
MUTED = "#64748B"
GRID = "#DCE3ED"
PAPER = "#FAFBFD"


@dataclass
class FigureArtifact:
    name: str
    png: Path
    svg: Path
    caption_facts: dict[str, Any]
    qa: dict[str, Any]


def set_publication_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#C8D1DC",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.16,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str, facts: dict[str, Any]) -> FigureArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    width, height = fig.get_size_inches() * 220
    qa = {
        "png_exists": png.exists(),
        "svg_exists": svg.exists(),
        "png_sha256": sha256_file(png),
        "svg_sha256": sha256_file(svg),
        "render_width_px": int(width),
        "render_height_px": int(height),
        "minimum_dimension_pass": min(width, height) >= 900,
        "axes_count": len(fig.axes),
    }
    plt.close(fig)
    return FigureArtifact(name, png, svg, facts, qa)


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(
        0,
        1.01,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )


def _horizontal_bar(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    subtitle: str,
    out_dir: Path,
    name: str,
    top_n: int = 15,
    x_label: str = "Publications",
) -> FigureArtifact | None:
    data = frame.dropna(subset=[label]).head(top_n).sort_values(value)
    if data.empty:
        return None
    height = max(5.5, 0.36 * len(data) + 2)
    fig, ax = plt.subplots(figsize=(10.5, height))
    colors = [PALETTE[0]] * len(data)
    colors[-1] = PALETTE[1]
    bars = ax.barh(data[label].astype(str), data[value], color=colors, height=0.68)
    ax.set_title(title, loc="left", pad=26)
    _subtitle(ax, subtitle)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, color=INK)
    max_value = float(data[value].max())
    ax.set_xlim(0, max_value * 1.14 if max_value else 1)
    facts = {
        "top_label": str(data.iloc[-1][label]),
        "top_value": int(data.iloc[-1][value]),
        "displayed": len(data),
    }
    return _save(fig, out_dir, name, facts)


def plot_annual(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.annual
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.plot(
        data["year"],
        data["publications"],
        color=PALETTE[0],
        linewidth=2.8,
        marker="o",
        markersize=6,
        markerfacecolor=PAPER,
        markeredgewidth=2,
    )
    ax.fill_between(data["year"], data["publications"], color=PALETTE[0], alpha=0.10)
    peak = data.loc[data["publications"].idxmax()]
    ax.annotate(
        f"Peak: {int(peak['publications'])}",
        (peak["year"], peak["publications"]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        color=PALETTE[0],
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": PALETTE[0]},
    )
    ax.set_title("Annual scientific production", loc="left", pad=26)
    growth = bundle.summary.get("annual_growth_rate")
    subtitle = (
        f"Corpus output by publication year · CAGR {growth:.1f}%"
        if growth is not None
        else "Corpus output by publication year"
    )
    _subtitle(ax, subtitle)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Documents")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=12))
    facts = {
        "start_year": int(data["year"].min()),
        "end_year": int(data["year"].max()),
        "peak_year": int(peak["year"]),
        "peak_publications": int(peak["publications"]),
        "annual_growth_rate": growth,
    }
    return _save(fig, out_dir, "annual_publications", facts)


def plot_document_types(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.document_types.head(10).sort_values("publications", ascending=False)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    wedges, _ = ax.pie(
        data["publications"],
        startangle=90,
        colors=PALETTE,
        wedgeprops={"width": 0.40, "edgecolor": PAPER, "linewidth": 2},
    )
    total = int(data["publications"].sum())
    ax.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0, -0.13, "documents", ha="center", va="center", fontsize=9, color=MUTED)
    legend_labels = [
        f"{row.document_type or 'Unknown'}  {int(row.publications):,}"
        for row in data.itertuples(index=False)
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9,
    )
    ax.set_title("Document type composition", loc="left", pad=22)
    facts = {
        "dominant_type": str(data.iloc[0]["document_type"]),
        "dominant_count": int(data.iloc[0]["publications"]),
        "displayed_types": len(data),
    }
    return _save(fig, out_dir, "document_types", facts)


def plot_citation_distribution(
    tables: CanonicalTables, bundle: AnalysisBundle, out_dir: Path
) -> FigureArtifact | None:
    values = pd.to_numeric(tables.works["cited_by_count"], errors="coerce").dropna()
    if values.empty:
        return None
    cap = max(float(values.quantile(0.98)), 1)
    shown = values.clip(upper=cap)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bins = min(40, max(8, int(math.sqrt(len(shown)))))
    sns.histplot(shown, bins=bins, color=PALETTE[0], edgecolor=PAPER, ax=ax)
    median = float(values.median())
    ax.axvline(median, color=PALETTE[1], linestyle="--", linewidth=2)
    ax.text(
        median + max(cap * 0.015, 0.15),
        ax.get_ylim()[1] * 0.92,
        f"Median {median:.1f}",
        color=PALETTE[1],
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_title("Citation distribution", loc="left", pad=26)
    _subtitle(
        ax, f"Display winsorized at the 98th percentile ({cap:.0f}); statistics use full data"
    )
    ax.set_xlabel("Citations per document")
    ax.set_ylabel("Documents")
    ax.spines[["top", "right"]].set_visible(False)
    facts = {
        "mean": float(values.mean()),
        "median": median,
        "maximum": float(values.max()),
        "display_cap_p98": cap,
        "zero_cited": int((values == 0).sum()),
    }
    return _save(fig, out_dir, "citation_distribution", facts)


def _select_network(
    network: NetworkResult, max_nodes: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    nodes = network.nodes.copy()
    edges = network.edges.copy()
    if nodes.empty or edges.empty:
        return nodes, edges, {"selected_nodes": len(nodes), "selected_edges": len(edges)}
    nodes["selection_score"] = (
        nodes["weighted_degree"].fillna(0)
        + np.sqrt(nodes["occurrences"].fillna(0).clip(lower=0)) * 0.25
    )
    selected = nodes.nlargest(max_nodes, "selection_score")
    selected_ids = set(selected["id"].astype(str))
    selected_edges = edges[
        edges["source"].astype(str).isin(selected_ids)
        & edges["target"].astype(str).isin(selected_ids)
    ].copy()
    if selected_edges.empty:
        return (
            selected,
            selected_edges,
            {
                "selected_nodes": len(selected),
                "selected_edges": 0,
            },
        )
    selected_edges["_rank_weight"] = selected_edges["association_strength"].fillna(0) * np.log1p(
        selected_edges["weight"].fillna(0)
    )
    edge_budget = max(2 * len(selected), min(3 * len(selected), 280))
    top_edges = selected_edges.nlargest(edge_budget, "_rank_weight")
    graph = nx.Graph()
    for row in selected_edges[["source", "target", "_rank_weight"]].to_dict("records"):
        graph.add_edge(str(row["source"]), str(row["target"]), weight=float(row["_rank_weight"]))
    backbone: set[tuple[str, str]] = set()
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        tree = nx.maximum_spanning_tree(subgraph, weight="weight")
        backbone.update(tuple(sorted(edge)) for edge in tree.edges())
    selected_edges["_key"] = selected_edges.apply(
        lambda row: tuple(sorted((str(row["source"]), str(row["target"])))),
        axis=1,
    )
    kept = pd.concat(
        [top_edges, selected_edges[selected_edges["_key"].isin(backbone)]],
        ignore_index=True,
    ).drop_duplicates(["source", "target"])
    threshold = float(top_edges["_rank_weight"].min()) if len(top_edges) else 0.0
    kept = kept.drop(columns=["_key", "_rank_weight"], errors="ignore")
    return (
        selected,
        kept,
        {
            "selected_nodes": len(selected),
            "selected_edges": len(kept),
            "edge_budget": edge_budget,
            "selection": "weighted degree + occurrence; maximum-spanning backbone + top edges",
            "edge_score_threshold": threshold,
            "preselection_nodes": len(nodes),
            "preselection_edges": len(edges),
        },
    )


def plot_network(
    network: NetworkResult,
    out_dir: Path,
    *,
    max_nodes: int,
    label_budget: int,
    seed: int,
) -> FigureArtifact | None:
    nodes, edges, selection = _select_network(network, max_nodes)
    if nodes.empty or edges.empty:
        return None
    graph = nx.Graph()
    node_lookup = nodes.set_index(nodes["id"].astype(str)).to_dict("index")
    for node_id in nodes["id"].astype(str):
        graph.add_node(node_id)
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            weight=float(row.weight),
            normalized=float(row.association_strength),
        )
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    if not graph:
        return None
    # Lay out communities as macro-nodes, then arrange members locally. This is
    # substantially more readable than one global force layout for dense maps.
    clusters_map = {node: int(node_lookup[node].get("cluster") or 0) for node in graph}
    cluster_ids = sorted(set(clusters_map.values()))
    quotient = nx.Graph()
    quotient.add_nodes_from(cluster_ids)
    for left, right, data in graph.edges(data=True):
        left_cluster = clusters_map[left]
        right_cluster = clusters_map[right]
        if left_cluster == right_cluster:
            continue
        current = quotient.get_edge_data(left_cluster, right_cluster, {}).get("weight", 0)
        quotient.add_edge(left_cluster, right_cluster, weight=current + data["weight"])
    if len(quotient) == 1:
        cluster_centers = {cluster_ids[0]: np.array([0.0, 0.0])}
    else:
        cluster_centers = nx.spring_layout(
            quotient, seed=seed, weight="weight", scale=3.2, iterations=250
        )
    positions: dict[str, np.ndarray] = {}
    for cluster_id in cluster_ids:
        members = [node for node in graph if clusters_map[node] == cluster_id]
        subgraph = graph.subgraph(members)
        if len(members) == 1:
            local = {members[0]: np.array([0.0, 0.0])}
        else:
            local = nx.spring_layout(
                subgraph,
                seed=seed + cluster_id,
                weight="normalized",
                k=1.25 / math.sqrt(len(members)),
                iterations=300,
                scale=0.55 + min(math.sqrt(len(members)) * 0.10, 1.0),
            )
        center = np.asarray(cluster_centers[cluster_id])
        for node, point in local.items():
            positions[node] = np.asarray(point) + center
    fig, ax = plt.subplots(figsize=(14.5, 10.0))
    edge_weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    edge_widths = 0.35 + 2.2 * np.sqrt(edge_weights / max(edge_weights.max(), 1))
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=edge_widths,
        edge_color="#94A3B8",
        alpha=0.28,
    )
    max_occ = max(float(node_lookup[node].get("occurrences") or 0) for node in graph)
    sizes = [
        70 + 720 * math.sqrt(float(node_lookup[node].get("occurrences") or 0) / max(max_occ, 1))
        for node in graph
    ]
    clusters = [int(node_lookup[node].get("cluster") or 0) for node in graph]
    colors = [PALETTE[(cluster - 1) % len(PALETTE)] for cluster in clusters]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=sizes,
        node_color=colors,
        edgecolors=PAPER,
        linewidths=1.2,
        alpha=0.92,
        ax=ax,
    )
    long_label_network = network.name in {"cocitation", "bibliographic_coupling", "citation"}
    effective_label_budget = min(
        label_budget,
        14 if long_label_network else max(12, len(graph) // 4),
    )
    labelable_nodes = [
        node
        for node in graph
        if str(node_lookup[node].get("label") or "").strip().casefold()
        not in {"", "nan", "none", "unresolved reference"}
    ]
    ranked_nodes = sorted(
        labelable_nodes,
        key=lambda node: (
            float(node_lookup[node].get("weighted_degree") or 0),
            float(node_lookup[node].get("occurrences") or 0),
        ),
        reverse=True,
    )
    # Dense maps otherwise spend nearly every label on one dominant component.
    # Reserve a fair share for each visible community, then fill remaining slots
    # globally by network importance.
    active_clusters = [
        cluster for cluster in cluster_ids if any(clusters_map[node] == cluster for node in graph)
    ]
    quota = max(1, effective_label_budget // max(len(active_clusters), 1))
    label_nodes: list[str] = []
    for cluster in active_clusters:
        cluster_ranked = [node for node in ranked_nodes if clusters_map[node] == cluster]
        label_nodes.extend(cluster_ranked[:quota])
    for node in ranked_nodes:
        if len(label_nodes) >= effective_label_budget:
            break
        if node not in label_nodes:
            label_nodes.append(node)
    label_nodes = label_nodes[:effective_label_budget]
    texts = []
    for node in label_nodes:
        x, y = positions[node]
        label = str(node_lookup[node].get("label") or node)
        label_limit = 31 if long_label_network else 39
        label = label if len(label) <= label_limit else label[: label_limit - 1] + "…"
        peers = [member for member in graph if clusters_map[member] == clusters_map[node]]
        center = np.mean([positions[member] for member in peers], axis=0)
        direction = np.asarray([x, y]) - center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            angle = (sum(ord(character) for character in node) % 360) * math.pi / 180
            direction = np.asarray([math.cos(angle), math.sin(angle)])
        else:
            direction /= norm
        text_x, text_y = np.asarray([x, y]) + direction * (0.12 if long_label_network else 0.09)
        texts.append(
            ax.text(
                text_x,
                text_y,
                label,
                fontsize=7.6 if long_label_network else 8.0,
                fontweight="bold",
                ha="left" if direction[0] >= 0 else "right",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": PAPER,
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                zorder=5,
            )
        )
    if adjust_text and len(texts) <= 40:
        adjust_text(
            texts,
            x=[positions[node][0] for node in graph],
            y=[positions[node][1] for node in graph],
            ax=ax,
            expand=(1.18, 1.30),
            force_text=(0.75, 0.95),
            force_points=(0.35, 0.45),
            max_move=(25, 35),
            ensure_inside_axes=True,
            arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.55, "alpha": 0.7},
        )
    title_map = {
        "coauthorship": "Author collaboration network",
        "institution_collaboration": "Institution collaboration network",
        "keyword_cooccurrence": "Keyword co-occurrence network",
        "citation": "Within-corpus citation network",
        "cocitation": "Reference co-citation network",
        "bibliographic_coupling": "Document bibliographic coupling",
    }
    ax.set_title(
        title_map.get(network.name, network.name.replace("_", " ").title()), loc="left", pad=28
    )
    _subtitle(
        ax,
        f"{len(graph)} nodes · {graph.number_of_edges()} displayed edges · "
        f"{nodes['cluster'].nunique()} communities · size = occurrence",
    )
    ax.axis("off")
    ax.margins(0.16)
    cluster_values = sorted(set(clusters))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE[(cluster - 1) % len(PALETTE)],
            markeredgecolor="none",
            label=f"Cluster {cluster}",
            markersize=7,
        )
        for cluster in cluster_values[:10]
    ]
    if len(cluster_values) <= 10:
        ax.legend(handles=handles, loc="lower left", frameon=False, ncol=min(5, len(handles)))
    facts = {
        **selection,
        "displayed_nodes": len(graph),
        "displayed_edges": graph.number_of_edges(),
        "clusters": int(nodes["cluster"].nunique()),
        "label_count": len(label_nodes),
        "layout": "weighted spring layout",
        "seed": seed,
    }
    return _save(fig, out_dir, f"network_{network.name}", facts)


def plot_top_cited(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.top_cited_documents.head(15).copy()
    if data.empty:
        return None
    data["label"] = data.apply(
        lambda row: (
            (str(row["title"])[:62] + "…" if len(str(row["title"])) > 65 else str(row["title"]))
            + (f" ({int(row['year'])})" if pd.notna(row["year"]) else "")
        ),
        axis=1,
    )
    return _horizontal_bar(
        data,
        label="label",
        value="cited_by_count",
        title="Most cited documents",
        subtitle="Source-supplied citation counts at retrieval time",
        out_dir=out_dir,
        name="top_cited_documents",
        top_n=15,
        x_label="Citations",
    )


def plot_bradford(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.bradford_sources
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    for zone, group in data.groupby("zone"):
        ax.plot(
            group["rank"],
            group["publications"],
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=PALETTE[(int(zone) - 1) % len(PALETTE)],
            label=f"Zone {int(zone)} · {len(group)} sources",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Bradford source concentration", loc="left", pad=26)
    _subtitle(
        ax, "Sources ranked by output; zones each contain approximately one-third of documents"
    )
    ax.set_xlabel("Source rank (log scale)")
    ax.set_ylabel("Documents (log scale)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    zone_one = data[data["zone"] == 1]
    facts = {
        "zone_1_sources": len(zone_one),
        "zone_1_documents": int(zone_one["publications"].sum()),
        "all_sources": len(data),
    }
    return _save(fig, out_dir, "bradford_sources", facts)


def plot_keyword_trends(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.keyword_trends
    if data.empty:
        return None
    pivot = data.pivot(index="keyword", columns="year", values="documents").fillna(0).astype(int)
    order = (
        data[["keyword", "global_documents"]]
        .drop_duplicates()
        .sort_values("global_documents", ascending=False)["keyword"]
    )
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    sns.heatmap(
        pivot,
        cmap=sns.light_palette(PALETTE[0], as_cmap=True),
        annot=True,
        fmt="d",
        linewidths=0.7,
        linecolor=PAPER,
        cbar_kws={"label": "Documents"},
        ax=ax,
    )
    ax.set_title("Keyword evolution by publication year", loc="left", pad=24)
    _subtitle(
        ax,
        "Top keywords by unique-document frequency; source and derived terms are disclosed in data",
    )
    ax.set_xlabel("Publication year")
    ax.set_ylabel("")
    facts = {
        "keywords": len(pivot),
        "years": [int(year) for year in pivot.columns],
        "top_keyword": str(pivot.sum(axis=1).idxmax()),
    }
    return _save(fig, out_dir, "keyword_trends", facts)


def plot_three_field(bundle: AnalysisBundle, out_dir: Path) -> FigureArtifact | None:
    data = bundle.three_field
    if data.empty:
        return None
    author_source = data.groupby(["author", "source"])["documents"].sum().reset_index()
    source_keyword = data.groupby(["source", "keyword"])["documents"].sum().reset_index()
    authors = author_source.groupby("author")["documents"].sum().nlargest(8).index.tolist()
    sources = (
        pd.concat(
            [
                author_source.groupby("source")["documents"].sum(),
                source_keyword.groupby("source")["documents"].sum(),
            ],
            axis=1,
        )
        .fillna(0)
        .sum(axis=1)
        .nlargest(8)
        .index.tolist()
    )
    keywords = source_keyword.groupby("keyword")["documents"].sum().nlargest(10).index.tolist()
    author_source = author_source[
        author_source["author"].isin(authors) & author_source["source"].isin(sources)
    ]
    source_keyword = source_keyword[
        source_keyword["source"].isin(sources) & source_keyword["keyword"].isin(keywords)
    ]
    if author_source.empty or source_keyword.empty:
        return None
    fig, ax = plt.subplots(figsize=(13.2, 8.2))
    x_positions = {"author": 0.04, "source": 0.5, "keyword": 0.96}
    y_author = {value: y for value, y in zip(authors, np.linspace(0.9, 0.1, len(authors)))}
    y_source = {value: y for value, y in zip(sources, np.linspace(0.9, 0.1, len(sources)))}
    y_keyword = {value: y for value, y in zip(keywords, np.linspace(0.93, 0.07, len(keywords)))}
    max_weight = max(author_source["documents"].max(), source_keyword["documents"].max())
    for row in author_source.itertuples(index=False):
        ax.plot(
            [x_positions["author"], x_positions["source"]],
            [y_author[row.author], y_source[row.source]],
            color=PALETTE[0],
            alpha=0.12 + 0.35 * row.documents / max_weight,
            linewidth=0.6 + 3.0 * row.documents / max_weight,
            zorder=1,
        )
    for row in source_keyword.itertuples(index=False):
        ax.plot(
            [x_positions["source"], x_positions["keyword"]],
            [y_source[row.source], y_keyword[row.keyword]],
            color=PALETTE[1],
            alpha=0.12 + 0.35 * row.documents / max_weight,
            linewidth=0.6 + 3.0 * row.documents / max_weight,
            zorder=1,
        )
    for label, y in y_author.items():
        ax.text(0.02, y, label, ha="left", va="center", fontsize=8.5, fontweight="bold")
        ax.scatter(0.04, y, s=55, color=PALETTE[0], zorder=3)
    for label, y in y_source.items():
        display = label if len(label) <= 32 else label[:29] + "…"
        ax.text(
            0.5,
            y,
            display,
            ha="center",
            va="center",
            fontsize=8.1,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": PAPER, "edgecolor": GRID},
        )
    for label, y in y_keyword.items():
        display = label if len(label) <= 34 else label[:31] + "…"
        ax.text(0.98, y, display, ha="right", va="center", fontsize=8.3, fontweight="bold")
        ax.scatter(0.96, y, s=55, color=PALETTE[1], zorder=3)
    ax.text(0.04, 0.99, "AUTHORS", transform=ax.transAxes, ha="left", color=MUTED, fontsize=9)
    ax.text(0.5, 0.99, "SOURCES", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    ax.text(0.96, 0.99, "KEYWORDS", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
    ax.set_title("Three-field relationship map", loc="left", pad=26)
    _subtitle(ax, "Leading authors connected to publication sources and recurring keywords")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    facts = {
        "authors": len(authors),
        "sources": len(sources),
        "keywords": len(keywords),
        "author_source_links": len(author_source),
        "source_keyword_links": len(source_keyword),
    }
    return _save(fig, out_dir, "three_field_map", facts)


def plot_thematic_map(network: NetworkResult, out_dir: Path) -> FigureArtifact | None:
    nodes = network.nodes
    edges = network.edges
    if nodes.empty or edges.empty or nodes["cluster"].nunique() < 2:
        return None
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=float(row.weight))
    records = []
    for cluster, group in nodes.groupby("cluster"):
        ids = set(group["id"].astype(str))
        internal = sum(
            data["weight"]
            for left, right, data in graph.edges(data=True)
            if left in ids and right in ids
        )
        external = sum(
            data["weight"]
            for left, right, data in graph.edges(data=True)
            if (left in ids) ^ (right in ids)
        )
        size = max(len(ids), 1)
        density = internal / (size * (size - 1) / 2) if size > 1 else 0
        centrality = external / size
        representative = (
            group.sort_values(["weighted_degree", "occurrences"], ascending=False)
            .head(3)["label"]
            .astype(str)
            .tolist()
        )
        records.append(
            {
                "cluster": int(cluster),
                "density": float(density),
                "centrality": float(centrality),
                "size": int(size),
                "label": " · ".join(representative),
            }
        )
    data = pd.DataFrame(records)
    median_x = data["centrality"].median()
    median_y = data["density"].median()
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    ax.axvline(median_x, color="#CBD5E1", linewidth=1.2)
    ax.axhline(median_y, color="#CBD5E1", linewidth=1.2)
    for row in data.itertuples(index=False):
        ax.scatter(
            row.centrality,
            row.density,
            s=220 + 70 * row.size,
            color=PALETTE[(row.cluster - 1) % len(PALETTE)],
            alpha=0.78,
            edgecolor=PAPER,
            linewidth=1.5,
        )
        label = row.label if len(row.label) <= 55 else row.label[:52] + "…"
        ax.annotate(
            label,
            (row.centrality, row.density),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.text(0.98, 0.96, "Motor themes", transform=ax.transAxes, ha="right", va="top", color=MUTED)
    ax.text(0.02, 0.96, "Niche themes", transform=ax.transAxes, ha="left", va="top", color=MUTED)
    ax.text(
        0.98, 0.04, "Basic themes", transform=ax.transAxes, ha="right", va="bottom", color=MUTED
    )
    ax.text(
        0.02,
        0.04,
        "Emerging / declining",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
    )
    ax.set_title("Thematic map", loc="left", pad=26)
    _subtitle(ax, "Keyword communities positioned by external centrality and internal density")
    ax.set_xlabel("Centrality (connection to other themes)")
    ax.set_ylabel("Density (within-theme cohesion)")
    ax.spines[["top", "right"]].set_visible(False)
    facts = {
        "themes": len(data),
        "centrality_median": float(median_x),
        "density_median": float(median_y),
        "largest_theme": data.loc[data["size"].idxmax(), "label"],
    }
    return _save(fig, out_dir, "thematic_map", facts)


def render_all(
    tables: CanonicalTables,
    bundle: AnalysisBundle,
    out_dir: Path,
    *,
    max_nodes: int = 80,
    label_budget: int = 24,
    seed: int = 42,
) -> list[FigureArtifact]:
    set_publication_theme()
    figures: list[FigureArtifact | None] = [
        plot_annual(bundle, out_dir),
        _horizontal_bar(
            bundle.top_sources,
            label="name",
            value="publications",
            title="Most productive sources",
            subtitle="Documents per journal or publication venue",
            out_dir=out_dir,
            name="top_sources",
        ),
        _horizontal_bar(
            bundle.top_authors,
            label="name",
            value="publications",
            title="Most productive authors",
            subtitle="Whole-counted publications in the included corpus",
            out_dir=out_dir,
            name="top_authors",
        ),
        _horizontal_bar(
            bundle.top_institutions,
            label="name",
            value="publications",
            title="Most productive institutions",
            subtitle="At least one affiliated author per document",
            out_dir=out_dir,
            name="top_institutions",
        ),
        plot_document_types(bundle, out_dir),
        plot_citation_distribution(tables, bundle, out_dir),
        plot_top_cited(bundle, out_dir),
        plot_bradford(bundle, out_dir),
        plot_keyword_trends(bundle, out_dir),
        plot_three_field(bundle, out_dir),
    ]
    for network in bundle.networks.values():
        figures.append(
            plot_network(
                network,
                out_dir,
                max_nodes=max_nodes,
                label_budget=label_budget,
                seed=seed,
            )
        )
    figures.append(plot_thematic_map(bundle.networks["keyword_cooccurrence"], out_dir))
    return [figure for figure in figures if figure is not None]
