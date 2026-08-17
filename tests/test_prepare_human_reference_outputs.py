from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_human_reference_outputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_human_reference_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>A <italic>Human</italic> Report</article-title></title-group>
      <abstract><p>The original abstract sentence.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Methods</title>
      <p>This must not be extracted.</p>
    </sec>
    <sec sec-type="results">
      <title>Results</title>
      <p>The result was 42.</p>
      <sec><title>Network structure</title><p>Cluster A was central.</p></sec>
    </sec>
    <sec>
      <title>Discussion and Conclusion</title>
      <p>The authors discussed the result.</p>
    </sec>
  </body>
</article>
"""


def test_extract_reference_report_keeps_only_requested_sections() -> None:
    report, sections = MODULE.extract_reference_report(SAMPLE_XML)

    assert "# A Human Report" in report
    assert "The original abstract sentence." in report
    assert "The result was 42." in report
    assert "Cluster A was central." in report
    assert "The authors discussed the result." in report
    assert "This must not be extracted." not in report
    assert [item["title"] for item in sections] == [
        "Article Title",
        "Abstract",
        "Results",
        "Discussion and Conclusion",
    ]


def test_prepare_reference_is_resumable_and_rejects_changed_output(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=SAMPLE_XML, request=request)

    reference = {"id": "example", "pmcid": "PMC123"}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            MODULE.prepare_reference(
                reference,
                output_root=tmp_path,
                client=client,
                retries=0,
            )
            == "created"
        )
        assert (
            MODULE.prepare_reference(
                reference,
                output_root=tmp_path,
                client=client,
                retries=0,
            )
            == "skipped"
        )

    manifest = json.loads((tmp_path / "example" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_numeric_results_are_gold"] is False
    assert manifest["extracted_sections"][2]["title"] == "Results"

    (tmp_path / "example" / "reference_report.md").write_text("changed", encoding="utf-8")
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="hash-mismatched"),
    ):
        MODULE.prepare_reference(
            reference,
            output_root=tmp_path,
            client=client,
            retries=0,
        )
