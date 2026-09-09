"""Rebuild in new workspaces; never refinalize or mutate the frozen originals."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from citeweave.io import load_config, read_json, sha256_file, write_json
from citeweave.processing_acceptance import verify_large_processing
from citeweave.workflow import process_project

CODE_FILES = [
    "src/citeweave/transform.py",
    "src/citeweave/bulk_processing.py",
    "src/citeweave/processing_acceptance.py",
    "scripts/reprocess_author_identity_workspaces.py",
]


def code_hashes() -> dict[str, str]:
    return {p: sha256_file(Path(p)) for p in CODE_FILES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.execute:
        raise SystemExit("Choose exactly one of --freeze-only or --execute")
    identity = {
        "amendment_sha256": sha256_file(args.amendment),
        "code_sha256": code_hashes(),
        "output_root": str(args.output_root.resolve()),
    }
    if args.freeze_only:
        if args.freeze.exists():
            raise SystemExit("Refusing to overwrite amendment freeze")
        result_files = [
            str(p)
            for name in (
                "formal_v3_runs",
                "formal_v3_replication_runs",
                "formal_v3_complexity_extension_runs",
            )
            for p in (Path("experiments/graph_discovery_v2") / name).rglob("*")
            if p.is_file()
        ]
        if result_files:
            raise SystemExit("Formal result directories are not empty")
        write_json(
            args.freeze,
            {
                **identity,
                "frozen_at_utc": datetime.now(UTC).isoformat(),
                "formal_result_files": result_files,
            },
        )
        print("Frozen author-identity correction before any formal result file", flush=True)
        return
    freeze = read_json(args.freeze)
    if any(freeze.get(k) != v for k, v in identity.items()):
        raise SystemExit("Correction freeze/code/output identity mismatch")
    results = []
    roots = {
        "primary": Path("experiments/formal_v3_workspaces"),
        "replication": Path("experiments/formal_v3_replication_workspaces"),
    }
    for group, root in roots.items():
        for source in sorted(root.iterdir()):
            if not (source / "project.yml").is_file():
                continue
            if code_hashes() != freeze["code_sha256"]:
                raise SystemExit("Processing code changed during the frozen correction")
            destination = args.output_root / group / source.name
            if destination.resolve().is_relative_to(
                source.resolve()
            ) or source.resolve().is_relative_to(destination.resolve()):
                raise SystemExit("Correction and original source paths must be disjoint")
            lineage = {
                "source_workspace": str(source.resolve()),
                "amendment_sha256": identity["amendment_sha256"],
                "staged_sha256": sha256_file(source / "staged" / "source_records.jsonl.gz"),
                "project_sha256": sha256_file(source / "project.yml"),
            }
            if not destination.exists():
                destination.mkdir(parents=True)
                shutil.copy2(source / "project.yml", destination / "project.yml")
                for directory in ("raw", "staged"):
                    shutil.copytree(source / directory, destination / directory)
                (destination / "audit").mkdir()
                for name in (
                    "harvest_manifest.json",
                    "acquisition_manifest.json",
                    "harvest_metadata_profile.json",
                ):
                    if (source / "audit" / name).is_file():
                        shutil.copy2(source / "audit" / name, destination / "audit" / name)
                write_json(destination / "identity_correction_lineage.json", lineage)
            elif (
                not (destination / "identity_correction_lineage.json").is_file()
                or read_json(destination / "identity_correction_lineage.json") != lineage
            ):
                raise SystemExit(
                    f"Existing correction workspace has different lineage: {destination}"
                )
            if (
                sha256_file(destination / "staged" / "source_records.jsonl.gz")
                != lineage["staged_sha256"]
            ):
                raise SystemExit("Copied staged corpus hash mismatch")
            print(
                json.dumps(
                    {"topic": source.name, "stage": "processing", "destination": str(destination)}
                ),
                flush=True,
            )
            process_project(destination, load_config(source / "project.yml"), resume=True)
            acceptance = verify_large_processing(destination)
            if not acceptance["passed"]:
                raise SystemExit(f"Corrected processing acceptance failed: {destination}")
            c = duckdb.connect()
            c.execute("SET threads=1")
            comparisons = {}
            try:
                for old in sorted((source / "canonical").glob("*.parquet")):
                    if old.stem in {"authors", "authorships"}:
                        continue
                    new = destination / "canonical" / old.name
                    differences = c.execute(
                        "SELECT count(*) FROM ((SELECT * FROM read_parquet(?) EXCEPT ALL SELECT * FROM read_parquet(?)) UNION ALL (SELECT * FROM read_parquet(?) EXCEPT ALL SELECT * FROM read_parquet(?)))",
                        [str(old), str(new), str(new), str(old)],
                    ).fetchone()[0]
                    comparisons[old.name] = {
                        "row_multiset_differences": differences,
                        "old_sha256": sha256_file(old),
                        "new_sha256": sha256_file(new),
                    }
            finally:
                c.close()
            receipt = {
                "dataset_id": source.name,
                "group": group,
                "lineage": lineage,
                "processing_acceptance": acceptance,
                "unaffected_tables": comparisons,
                "status": "reprocessed_not_promoted"
                if all(r["row_multiset_differences"] == 0 for r in comparisons.values())
                else "unexpected_non_author_changes",
            }
            write_json(destination / "identity_correction_verification.json", receipt)
            if receipt["status"] != "reprocessed_not_promoted":
                raise SystemExit("Unexpected non-author canonical changes require investigation")
            results.append(receipt)
            write_json(
                args.output_root / "progress.json",
                {"datasets_completed": len(results), "datasets": results},
            )
            print(json.dumps({"topic": source.name, "stage": receipt["status"]}), flush=True)


if __name__ == "__main__":
    main()
