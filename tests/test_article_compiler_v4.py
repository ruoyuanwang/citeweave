from __future__ import annotations

import json
from pathlib import Path

from citeweave.article_compiler_v4 import (
    PROMPT_VERSION,
    build_compiler_section_request,
)


def test_compact_request_preserves_evidence_but_reduces_real_writer_input() -> None:
    writer_path = Path(
        "experiments/article_quality_v2/same_evidence_text_writer_inputs_v1/"
        "explainable_ai_medical_imaging_2016_2025/writer_input.json"
    )
    writer = json.loads(writer_path.read_text(encoding="utf-8"))
    request = build_compiler_section_request(
        writer,
        condition="graph_dependency_compiler",
        section="Methods",
        compiled_sections={},
    )
    content = request["messages"][1]["content"]
    assert PROMPT_VERSION in content
    assert '"abstract_excerpt":' not in content
    assert all(row["phenomenon_id"] in content for row in writer["graph_phenomena"])
    assert all(
        row["reference_id"] in content for row in writer["representative_sources"]
    )
