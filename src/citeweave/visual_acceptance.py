from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageChops
from pyarrow import parquet as pq

from .io import load_config, read_json, sha256_file, write_json


def _content_occupancy(path: Path) -> float:
    image = Image.open(path).convert("RGB")
    corner = image.getpixel((0, 0))
    background = Image.new("RGB", image.size, corner)
    difference = ImageChops.difference(image, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 12 else 0)
    box = difference.getbbox()
    if not box:
        return 0.0
    width = max(box[2] - box[0], 0)
    height = max(box[3] - box[1], 0)
    return width * height / max(image.width * image.height, 1)


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows) if path.exists() else 0


def verify_visualization(project: Path) -> dict[str, Any]:
    """Independently verify scalable-map provenance and static rendering contracts."""
    project = project.resolve()
    manifest_path = project / "figures" / "figure_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "checks": {"manifest_exists": False}}
    config = load_config(project / "project.yml")
    manifest = read_json(manifest_path)
    figures = manifest.get("figures", [])
    networks = manifest.get("networks", {})
    image_checks = []
    for figure in figures:
        png = Path(figure["png"])
        svg = Path(figure["svg"])
        image_checks.append(
            {
                "name": figure["name"],
                "png_exists": png.exists(),
                "svg_exists": svg.exists(),
                "png_hash_matches": png.exists() and sha256_file(png) == figure.get("png_sha256"),
                "svg_hash_matches": svg.exists() and sha256_file(svg) == figure.get("svg_sha256"),
                "minimum_dimensions": min(
                    int(figure.get("width_px", 0)), int(figure.get("height_px", 0))
                )
                >= 1_200,
                "content_occupancy": round(_content_occupancy(png), 4) if png.exists() else 0,
            }
        )
    rendered_networks = {
        name: record for name, record in networks.items() if record.get("status") == "rendered"
    }
    node_cap = int(manifest.get("policy", {}).get("max_display_nodes", 0))
    network_checks = {}
    for name, record in rendered_networks.items():
        method = project / "analyses" / "visualization" / f"{name}_method.json"
        nodes = project / "analyses" / "visualization" / f"{name}_nodes.parquet"
        edges = project / "analyses" / "visualization" / f"{name}_edges.parquet"
        network_checks[name] = {
            "method_disclosed": method.exists(),
            "data_exported": nodes.exists() and edges.exists(),
            "node_cap_respected": int(record.get("displayed_nodes", 0))
            <= (node_cap or config.visualization_max_nodes),
            "matrix_not_materialized": record.get("matrix_materialized") is False,
            "normalization_disclosed": bool(record.get("normalization")),
            "layout_disclosed": bool(record.get("layout")),
            "layout_overlap_ratio": float(record.get("overlap_ratio", 1)),
        }
    edge_inputs = {
        "coauthorship": "coauthor_edges.parquet",
        "institution_collaboration": "institution_collaboration_edges.parquet",
        "keyword_cooccurrence": "keyword_cooccurrence_edges.parquet",
        "cocitation": "cocitation_edges.parquet",
        "citation": "direct_citation_edges.parquet",
    }
    visual_dir = project / "canonical" / "visualization"
    expected_networks = {
        name for name, filename in edge_inputs.items() if _parquet_rows(visual_dir / filename) > 0
    }
    if networks.get("bibliographic_coupling", {}).get("status") == "rendered":
        expected_networks.add("bibliographic_coupling")
    checks = {
        "manifest_exists": True,
        "minimum_figure_count": len(figures) >= 14,
        "png_svg_pairs_valid": all(
            item["png_exists"]
            and item["svg_exists"]
            and item["png_hash_matches"]
            and item["svg_hash_matches"]
            and item["minimum_dimensions"]
            for item in image_checks
        ),
        "no_blank_figures": all(item["content_occupancy"] >= 0.16 for item in image_checks),
        "expected_networks_rendered": expected_networks.issubset(rendered_networks),
        "all_networks_bounded": all(item["node_cap_respected"] for item in network_checks.values()),
        "all_networks_sparse": all(
            item["matrix_not_materialized"] for item in network_checks.values()
        ),
        "network_methods_exported": all(
            item["method_disclosed"] and item["data_exported"] for item in network_checks.values()
        ),
        "layout_overlap_bounded": all(
            item["layout_overlap_ratio"] <= 0.03 for item in network_checks.values()
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "figure_count": len(figures),
        "rendered_networks": sorted(rendered_networks),
        "expected_networks": sorted(expected_networks),
        "images": image_checks,
        "networks": network_checks,
        "note": "VLM review is deliberately external to this automated acceptance stage.",
    }
    write_json(project / "audit" / "visualization_acceptance.json", result)
    return result
