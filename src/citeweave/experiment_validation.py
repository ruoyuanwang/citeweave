from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .io import read_json, sha256_file, write_json


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _network_checks(root: Path) -> tuple[bool, dict[str, Any]]:
    results: dict[str, Any] = {}
    passed = True
    for manifest_path in sorted((root / "analyses").glob("network_*_manifest.json")):
        name = manifest_path.name.removeprefix("network_").removesuffix("_manifest.json")
        nodes = pd.read_parquet(root / "analyses" / f"network_{name}_nodes.parquet")
        edges = pd.read_parquet(root / "analyses" / f"network_{name}_edges.parquet")
        node_ids = set(nodes["id"].astype(str)) if "id" in nodes else set()
        endpoint_ok = bool(
            edges.empty
            or (
                set(edges["source"].astype(str)).issubset(node_ids)
                and set(edges["target"].astype(str)).issubset(node_ids)
            )
        )
        weight_ok = bool(edges.empty or (pd.to_numeric(edges["weight"]) >= 0).all())
        current = {
            "nodes": len(nodes),
            "edges": len(edges),
            "endpoints_exist": endpoint_ok,
            "weights_nonnegative": weight_ok,
        }
        results[name] = current
        passed &= endpoint_ok and weight_ok
    return passed and len(results) >= 5, results


def validate_experiment_project(
    root: Path,
    *,
    year_from: int,
    year_to: int,
    minimum_figures: int = 12,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest = read_json(root / "audit" / "acquisition_manifest.json")
    acquisition_ok = bool(
        manifest.get("failed_pages") == 0
        and (manifest.get("complete") or manifest.get("truncated"))
        and manifest.get("unique_records", 0) > 0
    )
    checks.append(_check("acquisition_integrity", acquisition_ok, manifest))

    canonical_dir = root / "canonical"
    tables = {
        path.stem: pd.read_parquet(path)
        for path in canonical_dir.glob("*.parquet")
    }
    required = {
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
    }
    schema_ok = required.issubset(tables) and not tables["works"].empty
    checks.append(
        _check(
            "canonical_tables_present",
            schema_ok,
            {"present": sorted(tables), "works": len(tables.get("works", []))},
        )
    )

    works = tables["works"]
    years = pd.to_numeric(works["year"], errors="coerce").dropna()
    outside = works.loc[
        works["year"].notna()
        & ~pd.to_numeric(works["year"], errors="coerce").between(year_from, year_to),
        ["work_id", "year"],
    ]
    checks.append(
        _check(
            "registered_year_scope",
            outside.empty,
            {
                "minimum_observed": int(years.min()) if not years.empty else None,
                "maximum_observed": int(years.max()) if not years.empty else None,
                "outside_scope": outside.to_dict("records"),
            },
        )
    )

    work_ids = set(works["work_id"].astype(str))
    author_ids = set(tables["authors"]["author_id"].astype(str))
    institution_ids = set(tables["institutions"]["institution_id"].astype(str))
    authorships = tables["authorships"]
    fk_detail = {
        "authorship_work_orphans": int(
            (~authorships["work_id"].astype(str).isin(work_ids)).sum()
        ),
        "authorship_author_orphans": int(
            (~authorships["author_id"].astype(str).isin(author_ids)).sum()
        ),
        "authorship_institution_orphans": int(
            (
                authorships["institution_id"].notna()
                & ~authorships["institution_id"].astype(str).isin(institution_ids)
            ).sum()
        ),
        "keyword_work_orphans": int(
            (~tables["keywords"]["work_id"].astype(str).isin(work_ids)).sum()
        ),
        "topic_work_orphans": int(
            (~tables["topics"]["work_id"].astype(str).isin(work_ids)).sum()
        ),
        "reference_citing_orphans": int(
            (~tables["references"]["citing_work_id"].astype(str).isin(work_ids)).sum()
        ),
    }
    checks.append(
        _check(
            "canonical_foreign_keys",
            sum(fk_detail.values()) == 0,
            fk_detail,
        )
    )

    network_ok, network_detail = _network_checks(root)
    checks.append(_check("network_artifact_validity", network_ok, network_detail))

    figures = read_json(root / "figures" / "figure_manifest.json")
    figure_errors: list[str] = []
    for figure in figures:
        for field in ("png", "svg"):
            path = root / "figures" / figure[field]
            expected_hash = figure.get("qa", {}).get(f"{field}_sha256")
            if not path.is_file() or (expected_hash and sha256_file(path) != expected_hash):
                figure_errors.append(f"{figure['name']}:{field}")
    checks.append(
        _check(
            "figure_artifact_validity",
            len(figures) >= minimum_figures and not figure_errors,
            {
                "figures": len(figures),
                "minimum": minimum_figures,
                "missing_or_hash_mismatch": figure_errors,
            },
        )
    )

    generation = read_json(root / "report" / "generation_manifest.json")
    manuscript = (root / "report" / "manuscript.md").read_text(encoding="utf-8")
    required_headings = [
        "## Abstract",
        "## 1 Introduction",
        "## 2 Data and methods",
        "## 3 Results",
        "## 4 Discussion and limitations",
        "## 5 Conclusion",
    ]
    generation_detail = {
        "validator_valid": generation.get("validation", {}).get("valid"),
        "missing_english_headings": [
            heading for heading in required_headings if heading not in manuscript
        ],
        "characters": len(manuscript),
        "model": generation.get("model"),
    }
    checks.append(
        _check(
            "english_report_structure",
            generation_detail["validator_valid"]
            and not generation_detail["missing_english_headings"],
            generation_detail,
        )
    )

    graph_data = read_json(root / "evidence" / "evidence_graph.json")
    graph = nx.DiGraph()
    graph.add_nodes_from(node["id"] for node in graph_data["nodes"])
    graph.add_edges_from((edge["source"], edge["target"]) for edge in graph_data["edges"])
    used = generation.get("validation", {}).get("used_evidence_ids", [])
    missing_paths = [
        evidence_id
        for evidence_id in used
        if evidence_id not in graph
        or "raw_snapshot" not in graph
        or not nx.has_path(graph, "raw_snapshot", evidence_id)
    ]
    checks.append(
        _check(
            "evidence_traceability",
            bool(used) and not missing_paths,
            {"used_evidence_items": len(used), "missing_paths": missing_paths},
        )
    )

    kg = read_json(root / "evidence" / "bibliometric_kg.json")
    qa = read_json(root / "evidence" / "graph_qa_benchmark.json")
    graph_grounding_detail = {
        "kg_nodes": len(kg["nodes"]),
        "kg_edges": len(kg["edges"]),
        "qa_items": len(qa),
        "answerable": sum(bool(item["answerable"]) for item in qa),
        "unanswerable": sum(not item["answerable"] for item in qa),
    }
    checks.append(
        _check(
            "graph_grounding_artifacts",
            graph_grounding_detail["kg_nodes"] > 0
            and graph_grounding_detail["kg_edges"] > 0
            and graph_grounding_detail["answerable"] > 0
            and graph_grounding_detail["unanswerable"] > 0,
            graph_grounding_detail,
        )
    )

    package = read_json(root / "audit" / "package_manifest.json")
    mismatches = []
    for item in package["files"]:
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            mismatches.append(item["path"])
    checks.append(
        _check(
            "package_hash_integrity",
            not mismatches,
            {"declared_files": len(package["files"]), "mismatches": mismatches},
        )
    )

    return {
        "benchmark_version": 1,
        "checked_at": datetime.now(UTC),
        "project_root": str(root),
        "passed": all(check["passed"] for check in checks),
        "passed_checks": sum(check["passed"] for check in checks),
        "total_checks": len(checks),
        "checks": checks,
    }


def write_validation_report(result: dict[str, Any], output: Path) -> None:
    write_json(output, result)
    markdown = output.with_suffix(".md")
    lines = [
        "# Pipeline Validity Report",
        "",
        f"- Overall: {'PASS' if result['passed'] else 'FAIL'}",
        f"- Checks passed: {result['passed_checks']}/{result['total_checks']}",
        f"- Project: `{result['project_root']}`",
        "",
        "| Check | Result |",
        "|---|---|",
        *[
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} |"
            for check in result["checks"]
        ],
        "",
        "Machine-readable details are stored in the adjacent JSON file.",
    ]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
