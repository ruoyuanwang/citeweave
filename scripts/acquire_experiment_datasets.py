from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from citeweave.io import read_json, sha256_file, write_json
from citeweave.models import AcquisitionPolicy, ProjectConfig, SearchProtocol, SourceName
from citeweave.workflow import resume_generation, run_project

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "experiments" / "datasets.yml"
DEFAULT_WORKSPACES = REPOSITORY_ROOT / "experiments" / "workspaces"
DEFAULT_RUNS = REPOSITORY_ROOT / "experiments" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire registered English CiteWeave experiment datasets."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", action="append", help="Dataset id; repeat to select several.")
    parser.add_argument("--all", action="store_true", help="Acquire every registered dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow a fresh run only after the caller has removed the existing workspace.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume deterministic generation and delivery from an existing frozen workspace.",
    )
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="Refresh hashes and artifact counts in a completed run record.",
    )
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            text=True,
            encoding="utf-8",
        ).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "status": command("status", "--short"),
    }


def _english_title_coverage(works: pd.DataFrame) -> float:
    titles = works.get("title", pd.Series(dtype=str)).dropna().astype(str)
    if titles.empty:
        return 0.0

    def likely_english(value: str) -> bool:
        letters = [character for character in value if character.isalpha()]
        if not letters:
            return False
        latin = sum("a" <= character.casefold() <= "z" for character in letters)
        return latin / len(letters) >= 0.80

    return float(titles.map(likely_english).mean())


def _topic_relevance_summary(
    works: pd.DataFrame,
    relevance_terms: list[str],
) -> dict[str, Any]:
    text = (
        works.get("title", pd.Series(index=works.index, dtype=str)).fillna("").astype(str)
        + " "
        + works.get("abstract", pd.Series(index=works.index, dtype=str))
        .fillna("")
        .astype(str)
    ).str.casefold()
    term_coverage = {
        term: float(text.str.contains(term.casefold(), regex=False).mean())
        for term in relevance_terms
    }
    all_terms = pd.Series(True, index=works.index, dtype=bool)
    for term in relevance_terms:
        all_terms &= text.str.contains(term.casefold(), regex=False)
    return {
        "terms": relevance_terms,
        "term_coverage": term_coverage,
        "all_terms_count": int(all_terms.sum()),
        "all_terms_rate": float(all_terms.mean()) if len(works) else 0.0,
    }


def _snapshot_summary(workspace: Path, entry: dict[str, Any]) -> dict[str, Any]:
    manifest_path = workspace / "audit" / "acquisition_manifest.json"
    works_path = workspace / "canonical" / "works.parquet"
    manifest = read_json(manifest_path)
    works = pd.read_parquet(works_path)
    raw_files = sorted((workspace / "raw").glob("*"))
    return {
        "acquisition_manifest": manifest,
        "works": len(works),
        "english_title_coverage": _english_title_coverage(works),
        "abstract_coverage": float(works["abstract"].notna().mean()) if len(works) else 0.0,
        "doi_coverage": float(works["doi"].notna().mean()) if len(works) else 0.0,
        "topic_relevance": _topic_relevance_summary(
            works,
            entry.get("relevance_terms", []),
        ),
        "raw_files": [
            {
                "path": path.relative_to(workspace).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in raw_files
            if path.is_file()
        ],
        "canonical_works_sha256": sha256_file(works_path),
        "graph_qa_sha256": sha256_file(
            workspace / "evidence" / "graph_qa_benchmark.json"
        ),
    }


def _config(entry: dict[str, Any]) -> ProjectConfig:
    input_file = entry.get("input_file")
    if input_file:
        input_file = (REPOSITORY_ROOT / input_file).resolve()
    imported_curated_pool = entry["source"] == SourceName.import_file.value
    protocol_keywords = (
        entry.get("relevance_terms", entry["keywords"])
        if imported_curated_pool
        else entry["keywords"]
    )
    query_mode = "all" if imported_curated_pool else entry["query_mode"]
    return ProjectConfig(
        project_id=entry["id"],
        protocol=SearchProtocol(
            title=entry["title"],
            keywords=protocol_keywords,
            query_mode=query_mode,
            year_from=entry["year_from"],
            year_to=entry["year_to"],
            source=SourceName(entry["source"]),
            document_types=entry.get("document_types", []),
            max_records=entry.get("max_records"),
            include_references=entry.get("include_references", True),
            language="en",
            input_file=input_file,
            input_format=entry.get("input_format", "auto"),
            notes=(
                f"Registered CiteWeave experiment dataset; role={entry['role']}. "
                "Capped acquisitions are benchmark samples, not complete source censuses."
            ),
        ),
        acquisition=AcquisitionPolicy(mode="standard"),
    )


def run_dataset(
    entry: dict[str, Any],
    *,
    force: bool = False,
    resume_existing: bool = False,
    refresh_snapshot: bool = False,
) -> dict[str, Any]:
    dataset_id = entry["id"]
    workspace = DEFAULT_WORKSPACES / dataset_id
    run_dir = DEFAULT_RUNS / dataset_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "baseline.json"
    if (workspace / "project.yml").exists():
        if force:
            raise RuntimeError(
                f"{workspace} already exists. Remove that exact workspace before a forced rerun."
            )
        if output.exists() and read_json(output).get("status") == "complete":
            record = read_json(output)
            if refresh_snapshot:
                record["snapshot"] = _snapshot_summary(workspace, entry)
                graph_manifest = workspace / "evidence" / "graph_grounding_manifest.json"
                if graph_manifest.exists():
                    record["result"]["graph_grounding"] = read_json(graph_manifest)
                write_json(output, record)
            return record
        if not resume_existing:
            raise RuntimeError(
                f"Existing workspace is incomplete; pass --resume-existing: {workspace}"
            )

    started = datetime.now(UTC)
    record: dict[str, Any] = {
        "experiment_protocol_version": "0.1",
        "dataset_id": dataset_id,
        "role": entry["role"],
        "started_at": started,
        "status": "running",
        "git": _git_state(),
        "registry_entry": entry,
        "workspace": str(workspace),
    }
    write_json(output, record)
    try:
        result = (
            resume_generation(workspace, use_llm=False)
            if resume_existing and (workspace / "project.yml").exists()
            else run_project(
                workspace,
                _config(entry),
                use_llm=False,
                allow_truncated=True,
            )
        )
        record.update(
            {
                "status": "complete",
                "finished_at": datetime.now(UTC),
                "result": result,
                "snapshot": _snapshot_summary(workspace, entry),
            }
        )
    except BaseException as exc:
        record.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(output, record)
        raise
    write_json(output, record)
    return record


def main() -> None:
    args = parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    entries = registry["datasets"]
    selected = {entry["id"] for entry in entries} if args.all else set(args.dataset or [])
    if not selected:
        raise SystemExit("Select at least one --dataset or pass --all.")
    missing = selected - {entry["id"] for entry in entries}
    if missing:
        raise SystemExit(f"Unknown dataset ids: {sorted(missing)}")

    results = [
        run_dataset(
            entry,
            force=args.force,
            resume_existing=args.resume_existing,
            refresh_snapshot=args.refresh_snapshot,
        )
        for entry in entries
        if entry["id"] in selected
    ]
    json.dump(results, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
