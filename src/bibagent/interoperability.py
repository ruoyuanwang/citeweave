from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .analytics import AnalysisBundle, NetworkResult
from .io import sha256_file, write_json
from .transform import CanonicalTables


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).replace("\t", " ").replace("\r", " ").splitlines())


def export_vosviewer(network: NetworkResult, output_dir: Path) -> dict[str, Any]:
    """Export a candidate network in VOSviewer map/network text formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / f"vosviewer_{network.name}_map.txt"
    network_path = output_dir / f"vosviewer_{network.name}_network.txt"
    nodes = network.nodes.copy()
    edges = network.edges.copy()
    if nodes.empty:
        map_path.write_text(
            "id\tlabel\tweight<Occurrences>\tcluster\tdescription\n", encoding="utf-8"
        )
        network_path.write_text("id1\tid2\tstrength\n", encoding="utf-8")
        return {
            "name": network.name,
            "map": map_path.name,
            "network": network_path.name,
            "nodes": 0,
            "edges": 0,
        }
    ordered = nodes.sort_values(["occurrences", "id"], ascending=[False, True]).reset_index(
        drop=True
    )
    id_map = {str(value): index + 1 for index, value in enumerate(ordered["id"])}
    with map_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("id\tlabel\tweight<Occurrences>\tcluster\tdescription\n")
        for row in ordered.itertuples(index=False):
            handle.write(
                "\t".join(
                    [
                        str(id_map[str(row.id)]),
                        _clean(row.label),
                        str(max(float(row.occurrences), 0)),
                        str(int(getattr(row, "cluster", 0) or 0)),
                        _clean(row.id),
                    ]
                )
                + "\n"
            )
    kept_edges = 0
    with network_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("id1\tid2\tstrength\n")
        for row in edges.itertuples(index=False):
            left, right = id_map.get(str(row.source)), id_map.get(str(row.target))
            if left is None or right is None or left == right:
                continue
            handle.write(f"{left}\t{right}\t{float(row.weight):.12g}\n")
            kept_edges += 1
    return {
        "name": network.name,
        "map": map_path.name,
        "network": network_path.name,
        "nodes": len(ordered),
        "edges": kept_edges,
        "map_sha256": sha256_file(map_path),
        "network_sha256": sha256_file(network_path),
        "scope": "full candidate network; display filtering is applied only to rendered figures",
    }


def export_bibliometrix(tables: CanonicalTables, output_dir: Path) -> Path:
    """Create a bibliometrix-compatible, semicolon-delimited metadata table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    works = tables.works.copy()

    def grouped(frame: pd.DataFrame, key: str, value: str, separator: str = "; ") -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=str)
        return (
            frame.dropna(subset=[value])
            .drop_duplicates([key, value])
            .groupby(key)[value]
            .agg(lambda values: separator.join(_clean(value) for value in values if _clean(value)))
        )

    authorship = tables.authorships.merge(
        tables.authors[["author_id", "name"]], on="author_id", how="left"
    )
    affiliations = authorship.merge(
        tables.institutions[["institution_id", "name"]].rename(
            columns={"name": "institution_name"}
        ),
        on="institution_id",
        how="left",
    )
    references = tables.references.copy()
    if not references.empty:
        references["reference_text"] = references.apply(
            lambda row: ", ".join(
                _clean(value)
                for value in (
                    row.get("cited_author"),
                    row.get("cited_year"),
                    row.get("cited_title"),
                    row.get("cited_doi"),
                )
                if _clean(value)
            ),
            axis=1,
        )
    sources = tables.sources[["source_id", "name", "issn"]].rename(
        columns={"name": "SO", "issn": "SN"}
    )
    frame = works.merge(sources, on="source_id", how="left")
    maps = {
        "AU": grouped(authorship, "work_id", "name"),
        "AF": grouped(authorship, "work_id", "name"),
        "C1": grouped(affiliations, "work_id", "institution_name"),
        "DE": grouped(tables.keywords, "work_id", "keyword"),
        "ID": grouped(tables.topics, "work_id", "topic"),
        "CR": grouped(references, "citing_work_id", "reference_text"),
    }
    for column, mapping in maps.items():
        frame[column] = frame["work_id"].map(mapping).fillna("")
    exported = pd.DataFrame(
        {
            "AU": frame["AU"],
            "AF": frame["AF"],
            "TI": frame["title"],
            "SO": frame["SO"],
            "PY": frame["year"],
            "DI": frame["doi"],
            "DE": frame["DE"],
            "ID": frame["ID"],
            "AB": frame["abstract"],
            "C1": frame["C1"],
            "CR": frame["CR"],
            "TC": frame["cited_by_count"],
            "DT": frame["document_type"],
            "LA": frame["language"],
            "SN": frame["SN"],
            "PU": frame["publisher"],
            "VL": frame["volume"],
            "IS": frame["issue"],
            "BP": frame["pages"],
            "UT": frame["work_id"],
        }
    )
    path = output_dir / "bibliometrix_data.csv"
    exported.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_all(
    tables: CanonicalTables, analyses: AnalysisBundle, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    network_exports = [
        export_vosviewer(network, output_dir) for network in analyses.networks.values()
    ]
    bibliometrix = export_bibliometrix(tables, output_dir)
    manifest = {
        "format_version": 1,
        "bibliometrix": {
            "path": bibliometrix.name,
            "rows": len(tables.works),
            "sha256": sha256_file(bibliometrix),
        },
        "vosviewer": network_exports,
    }
    write_json(output_dir / "export_manifest.json", manifest)
    return manifest
