from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_formal_report_judging.py"
SPEC = importlib.util.spec_from_file_location("prepare_formal_report_judging", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_condition(root: Path, dataset: str, condition: str, report: str, evidence_hash: str):
    directory = root / dataset / condition
    directory.mkdir(parents=True)
    report_bytes = report.encode("utf-8")
    (directory / "report.md").write_bytes(report_bytes)
    (directory / "completion.json").write_text(
        json.dumps(
            {
                "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "evidence_sha256": evidence_hash,
            }
        ),
        encoding="utf-8",
    )


def test_prepares_shared_and_paired_report_judge_inputs(tmp_path: Path):
    dataset = "topic"
    references = tmp_path / "references.yml"
    references.write_text(
        yaml.safe_dump({"references": [{"id": dataset}]}),
        encoding="utf-8",
    )
    workspaces = tmp_path / "workspaces"
    evidence_path = workspaces / dataset / "evidence" / "evidence_items.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps([{"evidence_id": "E001", "statement": "A fact."}]),
        encoding="utf-8",
    )
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    reports = tmp_path / "reports"
    words = " ".join(f"word{i}" for i in range(180))
    _write_condition(
        reports,
        dataset,
        "structured_one_shot",
        f"# Structured One-shot\nCiteWeave {words}",
        evidence_hash,
    )
    _write_condition(
        reports,
        dataset,
        "citeweave_full",
        f"# CiteWeave Full\n{words}",
        evidence_hash,
    )
    human = tmp_path / "human" / dataset
    human.mkdir(parents=True)
    (human / "reference_report.md").write_text(
        f"# Published Human Reference\n{words}", encoding="utf-8"
    )
    (human / "reference_evidence.md").write_text(
        "# Methods\nSearch and tables.", encoding="utf-8"
    )
    output = tmp_path / "judging"

    manifest = MODULE.prepare_inputs(
        references_path=references,
        reports_root=reports,
        workspaces_root=workspaces,
        human_root=tmp_path / "human",
        output_root=output,
        maximum_words=150,
    )

    shared = json.loads((output / "one_shot_vs_full.jsonl").read_text(encoding="utf-8"))
    paired = json.loads((output / "full_vs_human.jsonl").read_text(encoding="utf-8"))
    assert shared["canonical_evidence"][0]["evidence_id"] == "E001"
    assert "evidence_by_condition" in paired
    human_evidence = paired["evidence_by_condition"]["published_human_reference"]
    assert human_evidence == [
        {
            "evidence_id": "H001",
            "source_type": "reference_article",
            "section": "Methods",
            "statement": "Search and tables.",
        }
    ]
    assert "CiteWeave" not in json.dumps(paired)
    assert "Published Human Reference" not in json.dumps(paired)
    assert manifest["matched_word_counts"]["topic:citeweave_full"] == 150
