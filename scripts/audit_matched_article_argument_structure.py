from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from citeweave.article_argument_diagnostics import (
    diagnose_argument_structure,
)
from citeweave.article_quality_diagnostics import matched_section_view, summarize_diagnostics
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT
    / "experiments"
    / "article_quality_v3"
    / "deterministic_argument_structure_diagnostic_v2_protocol.yml"
)
DEFAULT_FREEZE = DEFAULT_PROTOCOL.with_name(
    "deterministic_argument_structure_diagnostic_v2_protocol_freeze.json"
)
DEFAULT_OUTPUT = DEFAULT_PROTOCOL.with_name(
    "deterministic_argument_structure_diagnostic_v2.json"
)


def _rows(protocol: dict[str, Any], cohort: str) -> list[dict[str, Any]]:
    rows = []
    for record in protocol["inputs"][cohort]:
        path = ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Input identity mismatch: {path}")
        view = matched_section_view(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "article_id": record["article_id"],
                "condition": record["condition"],
                "path": record["path"],
                "source_sha256": record["sha256"],
                **diagnose_argument_structure(view),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite deterministic diagnostic: {args.output}")

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    freeze = read_json(args.freeze)
    if freeze.get("sha256") != sha256_file(args.protocol):
        raise RuntimeError("Protocol differs from its freeze artifact")
    implementation = ROOT / protocol["implementation"]["path"]
    if sha256_file(implementation) != protocol["implementation"]["sha256"]:
        raise RuntimeError("Diagnostic implementation differs from frozen identity")

    machine = _rows(protocol, "machine")
    human = _rows(protocol, "published_human")
    by_condition = {
        condition: summarize_diagnostics(
            [row for row in machine if row["condition"] == condition]
        )
        for condition in sorted({row["condition"] for row in machine})
    }
    payload = {
        "schema_version": 2,
        "status": "exploratory_deterministic_argument_diagnostic_complete",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "implementation_sha256": sha256_file(implementation),
        "comparison_scope": list(protocol["comparison_scope"]),
        "interpretation_limits": protocol["interpretation_limits"],
        "machine": {"rows": machine, "summary": summarize_diagnostics(machine)},
        "machine_by_condition": by_condition,
        "published_human": {
            "rows": human,
            "summary": summarize_diagnostics(human),
        },
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
