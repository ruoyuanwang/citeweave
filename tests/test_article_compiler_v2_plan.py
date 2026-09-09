from __future__ import annotations

from pathlib import Path

from citeweave.io import write_json
from scripts.prepare_article_compiler_v2_pilot import build_plan
from scripts.run_article_compiler_v2_pilot import CALL_SEQUENCE


def test_matched_plan_has_two_conditions_and_nine_calls(tmp_path: Path) -> None:
    writer_root = tmp_path / "writer"
    for dataset in ("a", "b"):
        write_json(writer_root / dataset / "writer_input.json", {"dataset_id": dataset})
    plan = build_plan(
        writer_input_root=writer_root,
        dataset_ids=["a", "b"],
        output_root=tmp_path / "output",
        protocol_sha256="abc",
    )
    assert len(CALL_SEQUENCE) == 9
    assert len(plan["articles"]) == 4
    assert plan["maximum_provider_calls"] == 36
    assert {row["condition"] for row in plan["articles"]} == {
        "flat_article_compiler",
        "graph_dependency_compiler",
    }
