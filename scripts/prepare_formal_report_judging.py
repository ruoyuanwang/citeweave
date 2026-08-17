from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "human_references.yml"
DEFAULT_REPORTS = ROOT / "experiments" / "formal_reports"
DEFAULT_WORKSPACES = ROOT / "experiments" / "formal_workspaces"
DEFAULT_HUMAN = ROOT / "experiments" / "human_outputs"
DEFAULT_OUTPUT = ROOT / "experiments" / "formal_judging" / "report_inputs"
CONDITIONS = ("structured_one_shot", "citeweave_full")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_completed_report(report_root: Path, dataset_id: str, condition: str) -> tuple[str, str]:
    directory = report_root / dataset_id / condition
    report_path = directory / "report.md"
    completion_path = directory / "completion.json"
    if not report_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"Incomplete report condition: {directory}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    report_bytes = report_path.read_bytes()
    if completion.get("report_sha256") != _sha_bytes(report_bytes):
        raise ValueError(f"Report hash mismatch: {report_path}")
    return report_bytes.decode("utf-8"), str(completion["evidence_sha256"])


def _anonymize(markdown: str) -> str:
    value = re.sub(
        r"(?im)^#\s+.*$",
        "# Anonymous Bibliometric Report",
        markdown,
        count=1,
    )
    value = re.sub(r"(?i)\bCiteWeave\b", "the reporting system", value)
    value = re.sub(
        r"(?i)\b(structured[_ -]?one[_ -]?shot|published human reference)\b",
        "the report",
        value,
    )
    return value.strip()


def _matched_length(left: str, right: str, *, maximum_words: int) -> tuple[str, str, int]:
    left_words = left.split()
    right_words = right.split()
    budget = min(len(left_words), len(right_words), maximum_words)
    if budget < 100:
        raise ValueError("Matched report comparison has fewer than 100 words.")
    return " ".join(left_words[:budget]), " ".join(right_words[:budget]), budget


def _human_evidence_items(markdown: str) -> list[dict[str, Any]]:
    """Turn the visible human-reference evidence into citable frozen chunks."""
    items: list[dict[str, Any]] = []
    section = "Unsectioned evidence"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        statement = "\n\n".join(buffer).strip()
        buffer.clear()
        if not statement:
            return
        items.append(
            {
                "evidence_id": f"H{len(items) + 1:03d}",
                "source_type": "reference_article",
                "section": section,
                "statement": statement,
            }
        )

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or "Untitled section"
        elif line:
            buffer.append(line)
        else:
            flush()
    flush()
    if not items:
        raise ValueError("Published human-reference evidence contains no citable text.")
    return items


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report Judge input: {path}")
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def prepare_inputs(
    *,
    references_path: Path,
    reports_root: Path,
    workspaces_root: Path,
    human_root: Path,
    output_root: Path,
    maximum_words: int,
) -> dict[str, Any]:
    registry = yaml.safe_load(references_path.read_text(encoding="utf-8"))
    shared_rows = []
    full_human_rows = []
    one_human_rows = []
    lengths = {}
    for reference in registry["references"]:
        dataset_id = reference["id"]
        one_shot, one_evidence_hash = _read_completed_report(
            reports_root, dataset_id, "structured_one_shot"
        )
        full, full_evidence_hash = _read_completed_report(
            reports_root, dataset_id, "citeweave_full"
        )
        if one_evidence_hash != full_evidence_hash:
            raise ValueError(f"{dataset_id}: report conditions used different evidence.")
        evidence_path = workspaces_root / dataset_id / "evidence" / "evidence_items.json"
        if not evidence_path.is_file() or _sha_bytes(evidence_path.read_bytes()) != full_evidence_hash:
            raise ValueError(f"{dataset_id}: frozen Evidence Bundle hash mismatch.")
        system_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        human_directory = human_root / dataset_id
        human_report_path = human_directory / "reference_report.md"
        human_evidence_path = human_directory / "reference_evidence.md"
        if not human_report_path.is_file() or not human_evidence_path.is_file():
            raise FileNotFoundError(f"{dataset_id}: human report/evidence is missing.")
        human_evidence = _human_evidence_items(
            human_evidence_path.read_text(encoding="utf-8")
        )
        candidates = {
            "structured_one_shot": _anonymize(one_shot),
            "citeweave_full": _anonymize(full),
            "published_human_reference": _anonymize(
                human_report_path.read_text(encoding="utf-8")
            ),
        }
        question = (
            "Which anonymous report provides the stronger evidence-bounded bibliometric "
            "synthesis in coverage, explanation, structure, caution, and overall usefulness?"
        )
        shared_rows.append(
            {
                "sample_id": dataset_id,
                "question": question,
                "canonical_evidence": system_evidence,
                "candidates": {
                    "structured_one_shot": candidates["structured_one_shot"],
                    "citeweave_full": candidates["citeweave_full"],
                },
            }
        )
        for system_condition, target in (
            ("citeweave_full", full_human_rows),
            ("structured_one_shot", one_human_rows),
        ):
            system_text, human_text, words = _matched_length(
                candidates[system_condition],
                candidates["published_human_reference"],
                maximum_words=maximum_words,
            )
            lengths[f"{dataset_id}:{system_condition}"] = words
            target.append(
                {
                    "sample_id": dataset_id,
                    "question": question,
                    "candidates": {
                        system_condition: system_text,
                        "published_human_reference": human_text,
                    },
                    "evidence_by_condition": {
                        system_condition: system_evidence,
                        "published_human_reference": human_evidence,
                    },
                }
            )
    _jsonl(output_root / "one_shot_vs_full.jsonl", shared_rows)
    _jsonl(output_root / "full_vs_human.jsonl", full_human_rows)
    _jsonl(output_root / "one_shot_vs_human.jsonl", one_human_rows)
    manifest = {
        "schema_version": 1,
        "topics": len(shared_rows),
        "maximum_words": maximum_words,
        "matched_word_counts": lengths,
        "comparisons": {
            "one_shot_vs_full": {"evidence_mode": "shared"},
            "full_vs_human": {"evidence_mode": "paired"},
            "one_shot_vs_human": {"evidence_mode": "paired"},
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--workspaces-root", type=Path, default=DEFAULT_WORKSPACES)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-words", type=int, default=4000)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"Refusing existing output directory: {args.output_root}")
    manifest = prepare_inputs(
        references_path=args.references,
        reports_root=args.reports_root,
        workspaces_root=args.workspaces_root,
        human_root=args.human_root,
        output_root=args.output_root,
        maximum_words=args.maximum_words,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
