from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_human_reference_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_human_reference_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

XML = b"""<article><front><article-meta>
<title-group><article-title>Reference study</article-title></title-group>
<abstract><p>Scope statement.</p></abstract>
</article-meta></front><body>
<sec sec-type="methods"><title>Data and methods</title><p>We searched a database.</p></sec>
<sec><title>Results</title><p>The result narrative must not be copied into methods.</p>
<table-wrap><label>Table 1</label><caption><p>Annual output.</p></caption>
<table><tr><td>2020</td><td>42</td></tr></table></table-wrap></sec>
<fig><label>Figure 1</label><caption><p>Collaboration network.</p></caption></fig>
</body></article>"""


def test_extracts_methods_tables_and_figures_without_results_narrative():
    evidence, metadata = MODULE.extract_reference_evidence(XML)

    assert "We searched a database." in evidence
    assert "Annual output." in evidence
    assert "2020 | 42" in evidence
    assert "Collaboration network." in evidence
    assert "result narrative must not be copied" not in evidence
    assert metadata["methods_sections"] == 1
    assert metadata["tables_and_figures"] == 2


def test_evidence_output_is_hash_checked_and_resumable(tmp_path: Path):
    directory = tmp_path / "reference"
    directory.mkdir()
    (directory / "source.xml").write_bytes(XML)

    assert MODULE.prepare_directory(directory) == "created"
    assert MODULE.prepare_directory(directory) == "skipped"
    manifest = json.loads(
        (directory / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_role"].endswith("not system numeric Gold")

    (directory / "reference_evidence.md").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Hash-mismatched"):
        MODULE.prepare_directory(directory)
