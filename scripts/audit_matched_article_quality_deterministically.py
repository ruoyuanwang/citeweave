from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from citeweave.article_quality_diagnostics import (
    diagnose_article_text,
    matched_section_view,
    summarize_diagnostics,
)
from citeweave.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MACHINE_ROOT = (
    ROOT
    / "experiments"
    / "article_quality_v3"
    / "matched_article_compiler_claim_ready_development_v1"
)
DEFAULT_HUMAN_ROOT = ROOT / "experiments" / "human_outputs"
DEFAULT_REGISTRY = ROOT / "experiments" / "human_references.yml"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "article_quality_v3"
    / "deterministic_machine_human_quality_diagnostic_v1.json"
)


def _machine_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/*/draft.md")):
        article = path.read_text(encoding="utf-8")
        view = matched_section_view(article)
        condition = path.parent.name
        row = {
            "article_id": f"{path.parent.parent.name}__{condition}",
            "condition": condition,
            "path": str(path.relative_to(ROOT)),
            "source_sha256": sha256_file(path),
            **diagnose_article_text(view),
        }
        rows.append(row)
    if len(rows) != 4:
        raise RuntimeError(f"Expected four matched development drafts, found {len(rows)}")
    return rows


def _human_rows(root: Path, registry_path: Path) -> list[dict[str, Any]]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    references = registry.get("references") or []
    rows = []
    for reference in references:
        path = root / reference["id"] / "reference_report.md"
        if not path.is_file():
            raise RuntimeError(f"Missing cached human reference report: {path}")
        text = path.read_text(encoding="utf-8")
        rows.append(
            {
                "article_id": reference["id"],
                "condition": "published_human_reference",
                "path": str(path.relative_to(ROOT)),
                "source_sha256": sha256_file(path),
                **diagnose_article_text(text),
            }
        )
    if len(rows) != 8:
        raise RuntimeError(f"Expected eight human references, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-root", type=Path, default=DEFAULT_MACHINE_ROOT)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite deterministic diagnostic: {args.output}")

    machine = _machine_rows(args.machine_root)
    human = _human_rows(args.human_root, args.registry)
    by_condition = {
        condition: summarize_diagnostics(
            [row for row in machine if row["condition"] == condition]
        )
        for condition in sorted({row["condition"] for row in machine})
    }
    payload = {
        "schema_version": 1,
        "status": "descriptive_non_llm_diagnostic_complete",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_scope": "Abstract_Results_Discussion_Conclusion",
        "machine_development_only": True,
        "same_topic_or_same_evidence_comparison": False,
        "inferential_test_permitted": False,
        "human_evaluation": False,
        "interpretation_limits": [
            "The machine and human articles cover different topics and evidence.",
            "The cached human reports contain selected article sections, not a uniform full-text genre.",
            "Metrics are deterministic surface diagnostics, not factuality or research-utility judgments.",
            "Condition means contain only two machine articles and cannot support effectiveness claims.",
        ],
        "machine": {"rows": machine, "summary": summarize_diagnostics(machine)},
        "machine_by_condition": by_condition,
        "published_human": {"rows": human, "summary": summarize_diagnostics(human)},
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
