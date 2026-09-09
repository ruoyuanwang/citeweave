from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .graph_discovery import _bm25_top_rows, _communities, _load_graph
from .io import read_json, sha256_file, write_json

ARTICLE_TASK_TYPES = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)


def _hash_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def _normalized_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value), set())
    return set()


def _representative_sources(
    workspace: Path,
    tasks: list[dict[str, Any]],
    *,
    per_phenomenon: int,
    topic_hint: str = "",
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    works = pd.read_parquet(workspace / "canonical" / "works.parquet")
    keywords = pd.read_parquet(workspace / "canonical" / "keywords.parquet")
    works["work_id"] = works["work_id"].astype(str)
    keywords["work_id"] = keywords["work_id"].astype(str)
    works["_has_abstract"] = works["abstract"].fillna("").astype(str).str.strip().ne("")
    works = works[works["_has_abstract"]].copy()
    keyword_documents = (
        keywords.groupby("work_id", sort=False)["keyword"]
        .apply(lambda values: " ".join(sorted(set(map(str, values)))))
        .rename("_keyword_document")
    )
    works = works.merge(keyword_documents, on="work_id", how="left")
    works["_keyword_document"] = works["_keyword_document"].fillna("")
    work_rows = works.to_dict("records")
    documents = [
        " ".join(
            (
                str(row.get("title") or ""),
                str(row.get("abstract") or ""),
                str(row.get("_keyword_document") or ""),
            )
        )
        for row in work_rows
    ]
    source_by_id: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, list[str]] = {}
    globally_assigned: set[str] = set()
    globally_assigned_titles: set[str] = set()
    for task in tasks:
        labels = {value.casefold() for value in _strings(task["answer"]) if value.strip()}
        readable_topic = " ".join(
            part for part in topic_hint.replace("_", " ").split() if not part.isdigit()
        )
        answer_terms = " ".join(sorted(labels))
        retrieval_query = " ".join(
            value
            for value in (
                readable_topic,
                str(task.get("question") or ""),
                answer_terms,
                str(task.get("task_type") or ""),
            )
            if value
        )
        ranked = _bm25_top_rows(
            retrieval_query,
            documents,
            [
                {**row, "evidence_id": str(row["work_id"])}
                for row in work_rows
            ],
            limit=len(work_rows),
            tie_key=f"article-source:{topic_hint}:{task['item_id']}",
        )
        topic_terms = {
            part.casefold()
            for part in topic_hint.replace("_", " ").split()
            if len(part) >= 4 and not part.isdigit()
        }
        document_by_work = {
            str(row["work_id"]): document.casefold()
            for row, document in zip(work_rows, documents, strict=True)
        }
        bm25_rank = {
            str(row["work_id"]): rank for rank, row in enumerate(ranked, start=1)
        }
        topic_match_count = {
            str(row["work_id"]): sum(
                term in document_by_work[str(row["work_id"])]
                for term in topic_terms
            )
            for row in ranked
        }
        ranked.sort(
            key=lambda row: (
                -topic_match_count[str(row["work_id"])],
                bm25_rank[str(row["work_id"])],
            )
        )
        candidates: list[tuple[dict[str, Any], str, int]] = []
        candidate_work_ids: set[str] = set()
        candidate_titles: set[str] = set()
        for prefer_new in (True, False):
            for retrieval_rank, row in enumerate(ranked, start=1):
                work_id = str(row["work_id"])
                normalized_title = _normalized_title(row.get("title") or "")
                if work_id in candidate_work_ids:
                    continue
                if (
                    not normalized_title
                    or normalized_title in candidate_titles
                    or normalized_title in globally_assigned_titles
                ):
                    continue
                if prefer_new != (work_id not in globally_assigned):
                    continue
                work_keywords = {
                    value.casefold()
                    for value in str(row.get("_keyword_document") or "").split()
                }
                selection_basis = (
                    "topic_task_bm25_with_answer_keyword"
                    if labels & work_keywords
                    else "topic_task_bm25"
                )
                candidates.append((row, selection_basis, retrieval_rank))
                candidate_work_ids.add(work_id)
                candidate_titles.add(normalized_title)
                if len(candidates) >= per_phenomenon:
                    break
            if len(candidates) >= per_phenomenon:
                break

        selected: list[str] = []
        for row, selection_basis, retrieval_rank in candidates:
            work_id = str(row["work_id"])
            reference_id = _hash_id("REF", work_id)
            abstract = str(row.get("abstract") or "").strip()
            source_by_id.setdefault(
                reference_id,
                {
                    "reference_id": reference_id,
                    "work_id": work_id,
                    "title": str(row.get("title") or "Untitled"),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "doi": str(row.get("doi") or "") or None,
                    "cited_by_count": int(row.get("cited_by_count") or 0),
                    "matched_keyword": next(
                        (
                            value
                            for value in sorted(labels)
                            if value
                            in str(row.get("_keyword_document") or "").casefold()
                        ),
                        None,
                    ),
                    "selection_basis": selection_basis,
                    "retrieval_query": retrieval_query,
                    "retrieval_rank": retrieval_rank,
                    "bm25_rank": bm25_rank[work_id],
                    "topic_terms": sorted(topic_terms),
                    "topic_term_matches": topic_match_count[work_id],
                    "abstract_excerpt": abstract[:1800],
                },
            )
            selected.append(reference_id)
            globally_assigned.add(work_id)
            globally_assigned_titles.add(_normalized_title(row.get("title") or ""))
        dependencies[task["item_id"]] = selected
    return sorted(source_by_id.values(), key=lambda row: row["reference_id"]), dependencies


def _anchor_nodes(tasks: list[dict[str, Any]], graph: nx.Graph) -> set[str]:
    anchors = set()
    for task in tasks:
        anchors.update(value for value in _strings(task["answer"]) if value in graph)
        for step in task["operator_trace"]:
            for key in ("node", "selected_node"):
                value = step.get(key)
                if isinstance(value, str) and value in graph:
                    anchors.add(value)
            for key in ("path", "selected_edge"):
                values = step.get(key)
                if isinstance(values, list):
                    anchors.update(str(value) for value in values if str(value) in graph)
            selected = step.get("selected")
            if isinstance(selected, dict) and str(selected.get("node")) in graph:
                anchors.add(str(selected["node"]))
    return anchors


def _draw_graph_overview(
    *,
    workspace: Path,
    tasks: list[dict[str, Any]],
    output: Path,
    max_nodes: int = 90,
    max_edges: int = 260,
) -> dict[str, Any]:
    graph, summary = _load_graph(workspace, "keyword_cooccurrence", "large")
    communities = _communities(graph)
    anchors = _anchor_nodes(tasks, graph)
    ranked = sorted(
        graph.nodes,
        key=lambda node: (-graph.nodes[node]["importance"], graph.nodes[node]["label"]),
    )
    selected = list(dict.fromkeys([*sorted(anchors), *ranked]))[:max_nodes]
    selected_set = set(selected)
    edges = sorted(
        (
            (source, target, float(attrs["weight"]))
            for source, target, attrs in graph.edges(data=True)
            if source in selected_set and target in selected_set
        ),
        key=lambda row: (-row[2], row[0], row[1]),
    )[:max_edges]
    groups: dict[int, list[str]] = {}
    for node in selected:
        groups.setdefault(communities[node], []).append(node)
    group_ids = sorted(groups)
    positions = {}
    for group_index, community in enumerate(group_ids):
        center_angle = 2 * math.pi * group_index / max(1, len(group_ids))
        center = (2.8 * math.cos(center_angle), 2.8 * math.sin(center_angle))
        members = sorted(
            groups[community],
            key=lambda node: (-graph.nodes[node]["importance"], node),
        )
        for member_index, node in enumerate(members):
            angle = 2 * math.pi * member_index / max(1, len(members))
            radius = 0.25 + 0.09 * math.sqrt(member_index + 1)
            positions[node] = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(4.6, 1.4))
    axis = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    fig.suptitle("Large keyword graph: community structure and registered task anchors")
    palette = plt.get_cmap("tab20")
    for source, target, weight in edges:
        left = positions[source]
        right = positions[target]
        axis.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            color="#7a7a7a",
            alpha=0.12 + 0.22 * min(1.0, math.log1p(weight) / 10),
            linewidth=0.35 + 1.2 * min(1.0, math.log1p(weight) / 10),
            zorder=1,
        )
    importances = [float(graph.nodes[node]["importance"]) for node in selected]
    maximum = max(importances) if importances else 1.0
    for community in group_ids:
        members = groups[community]
        axis.scatter(
            [positions[node][0] for node in members],
            [positions[node][1] for node in members],
            s=[28 + 220 * math.sqrt(float(graph.nodes[node]["importance"]) / maximum) for node in members],
            c=[palette(community % 20)],
            edgecolors=["#151515" if node in anchors else "none" for node in members],
            linewidths=[1.2 if node in anchors else 0 for node in members],
            alpha=0.82,
            label=f"Community {community}",
            zorder=2,
        )
    label_nodes = set(ranked[:6]) | anchors
    ordered_labels = sorted(
        label_nodes & selected_set,
        key=lambda node: (
            node not in anchors,
            -float(graph.nodes[node]["importance"]),
            graph.nodes[node]["label"],
        ),
    )
    for label_index, node in enumerate(ordered_labels, start=1):
        x, y = positions[node]
        callout_angle = 2 * math.pi * (label_index - 1) / max(1, len(ordered_labels))
        callout_radius = 24 + 8 * (label_index % 2)
        axis.annotate(
            str(label_index),
            xy=(x, y),
            xytext=(
                callout_radius * math.cos(callout_angle),
                callout_radius * math.sin(callout_angle),
            ),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#111111",
            bbox={
                "boxstyle": "circle,pad=0.22",
                "facecolor": "#ffffff",
                "edgecolor": "#333333",
                "linewidth": 0.55,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.55,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=3,
        )
    axis.set_aspect("equal")
    axis.axis("off")
    legend_axis.axis("off")
    legend_axis.set_title("Indexed concepts", fontsize=10, loc="left")
    legend_axis.text(
        0,
        0.98,
        "\n".join(
            f"{index}. {graph.nodes[node]['label']}"
            f"{'  [task anchor]' if node in anchors else ''}"
            for index, node in enumerate(ordered_labels, start=1)
        ),
        transform=legend_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.35,
        wrap=True,
    )
    fig.text(
        0.015,
        0.012,
        "Node size = corpus occurrence; color = Louvain community; dark outline = registered task anchor. "
        "Layout is schematic.",
        fontsize=8,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "full_graph_nodes": summary.nodes,
        "full_graph_edges": summary.edges,
        "displayed_nodes": len(selected),
        "displayed_edges": len(edges),
        "anchor_nodes": sorted(anchors),
        "layout_note": "Deterministic community-circle layout; geometry has no scientific meaning.",
    }


def build_article_evidence_pack(
    *,
    benchmark_path: Path,
    workspace: Path,
    output_dir: Path,
    word_target: int = 3000,
) -> dict[str, Any]:
    benchmark = read_json(benchmark_path)
    dataset_id = benchmark["dataset_id"]
    tasks = [
        task
        for task in benchmark["tasks"]
        if task["network"] == "keyword_cooccurrence"
        and task["scale"] == "large"
        and task["task_type"] in ARTICLE_TASK_TYPES
    ]
    if [task["task_type"] for task in tasks] != list(ARTICLE_TASK_TYPES):
        raise ValueError(f"Writer pack requires five ordered Large graph tasks: {dataset_id}")
    sources, dependencies = _representative_sources(
        workspace, tasks, per_phenomenon=3, topic_hint=str(dataset_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = _draw_graph_overview(
        workspace=workspace,
        tasks=tasks,
        output=output_dir / "graph_overview.png",
    )
    phenomena = []
    for task in tasks:
        phenomena.append(
            {
                "phenomenon_id": _hash_id("PH", task["item_id"]),
                "source_item_id": task["item_id"],
                "task_type": task["task_type"],
                "question": task["question"],
                "verified_answer": task["answer"],
                "operator_trace": task["operator_trace"],
                "interpretation_contract": task["interpretation_contract"],
                "graph_evidence_ids": task["evidence_ids"],
                "reference_ids": dependencies[task["item_id"]],
            }
        )
    pack = {
        "schema_version": 1,
        "status": "same_evidence_pack_frozen_before_articles",
        "source_selection_version": "topic-task-bm25-v4-title-deduplicated",
        "dataset_id": dataset_id,
        "source_benchmark": str(benchmark_path.resolve()),
        "source_benchmark_sha256": sha256_file(benchmark_path),
        "writing_brief": {
            "article_type": "structured scientific field-analysis article",
            "word_target": word_target,
            "permitted_range": [int(word_target * 0.9), int(word_target * 1.1)],
            "required_sections": [
                "Abstract",
                "Introduction",
                "Methods",
                "Results",
                "Discussion",
                "Limitations",
                "Conclusion",
            ],
            "required_graph_coverage": "Use all five registered phenomena across Results and Discussion; do not isolate graph evidence in one caption paragraph.",
            "citation_rule": "Cite only supplied REF identifiers; every numeric or graph-derived claim must cite a PH identifier and relevant REF identifiers.",
            "reasoning_rule": "Separate structural observation, possible scientific interpretation, alternative explanation, and limitation.",
            "prohibited": [
                "new external sources",
                "causal claims from structural association",
                "invented mechanisms",
                "community labels treated as scientific topics without literature support",
            ],
        },
        "condition_contract": {
            "shared_inputs": [
                "writing_brief",
                "graph_phenomena",
                "representative_sources",
                "graph_overview_figure",
            ],
            "condition_specific_instructions_excluded": True,
            "article_conditions": [
                "citeweave_graph_review",
                "one_shot_llm",
                "human_same_evidence",
            ],
        },
        "graph_phenomena": phenomena,
        "representative_sources": sources,
        "figures": [figure],
        "quality_gates": {
            "five_registered_phenomena": len(phenomena) == 5,
            "minimum_eight_unique_sources": len(sources) >= 8,
            "all_source_titles_unique": len(
                {_normalized_title(row["title"]) for row in sources}
            )
            == len(sources),
            "minimum_eight_abstract_excerpts": sum(
                bool(row["abstract_excerpt"]) for row in sources
            )
            >= 8,
            "every_phenomenon_has_two_sources": all(
                len(row["reference_ids"]) >= 2 for row in phenomena
            ),
            "figure_hash_bound": bool(figure["sha256"]),
        },
    }
    pack["passed"] = all(pack["quality_gates"].values())
    write_json(output_dir / "evidence_pack.json", pack)
    return pack
