from __future__ import annotations

from pathlib import Path

from citeweave.article_generation import (
    build_article_generation_request,
    build_graph_article_blueprint,
    build_machine_article_generation_plan,
)
from citeweave.io import read_json, sha256_file, write_json

TASK_TYPES = (
    "multi_hop_connector",
    "bridge_counterfactual",
    "community_role_contrast",
    "hub_removal_resilience",
    "temporal_structural_shift",
)


def _writer_input(topic: str) -> dict[str, object]:
    sources = [
        {"reference_id": f"REF-{index:016d}", "title": f"Source {index}"}
        for index in range(10)
    ]
    phenomena = []
    for index, task_type in enumerate(TASK_TYPES):
        phenomena.append(
            {
                "phenomenon_id": f"PH-{index:016d}",
                "task_type": task_type,
                "question": f"Question {index}",
                "verified_answer": {"value": index},
                "operator_trace": [{"operator": f"operator_{index}"}],
                "graph_evidence_ids": [f"network:node:{index}"],
                "reference_ids": [
                    sources[index]["reference_id"],
                    sources[index + 5]["reference_id"],
                ],
                "interpretation_contract": {
                    "allowed": "structural description",
                    "forbidden": "causality",
                    "required_limitation": "corpus dependent",
                },
            }
        )
    return {
        "dataset_id": topic,
        "writing_brief": {
            "permitted_range": [2700, 3300],
        },
        "graph_phenomena": phenomena,
        "representative_sources": sources,
        "nonvisual_figure_metadata": {"displayed_nodes": 90},
        "posthoc_figure_commitment": {"figure_sha256": "f" * 64},
    }


def test_blueprint_has_cross_phenomenon_dependencies() -> None:
    blueprint = build_graph_article_blueprint(_writer_input("topic"))
    assert len(blueprint["phenomenon_cards"]) == 5
    assert len(blueprint["cross_phenomenon_synthesis"]) == 3
    assert all(
        len(row["phenomenon_ids"]) == 2
        for row in blueprint["cross_phenomenon_synthesis"]
    )
    assert blueprint["blueprint_sha256"]


def test_one_shot_does_not_receive_citeweave_blueprint() -> None:
    writer_input = _writer_input("topic")
    one_shot = build_article_generation_request(
        writer_input, condition="one_shot_llm"
    )
    citeweave = build_article_generation_request(
        writer_input, condition="citeweave_graph_review"
    )
    assert "GRAPH_BLUEPRINT_JSON" not in one_shot["messages"][1]["content"]
    assert "GRAPH_BLUEPRINT_JSON" in citeweave["messages"][1]["content"]
    assert "graph.png" not in str(one_shot)
    assert one_shot["temperature"] == 0
    assert citeweave["temperature"] == 0


def test_builds_frozen_sixteen_cell_plan(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        topic = f"topic_{index}"
        path = tmp_path / "inputs" / topic / "writer_input.json"
        write_json(path, _writer_input(topic))
        figure = tmp_path / "figures" / f"{topic}.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"figure")
        records.append(
            {
                "dataset_id": topic,
                "pack": str(path.resolve()),
                "pack_sha256": sha256_file(path),
                "evaluation_figure": str(figure.resolve()),
                "evaluation_figure_sha256": sha256_file(figure),
            }
        )
    manifest_path = tmp_path / "inputs" / "manifest.json"
    write_json(
        manifest_path,
        {
            "status": "text_only_same_evidence_writer_inputs_ready",
            "records": records,
        },
    )
    plan = build_machine_article_generation_plan(
        manifest_path,
        output_dir=tmp_path / "generation",
    )
    assert len(plan["cells"]) == 16
    assert {row["condition"] for row in plan["cells"]} == {
        "one_shot_llm",
        "citeweave_graph_review",
    }
    for cell in plan["cells"]:
        assert sha256_file(Path(cell["request"])) == cell["request_sha256"]
        request = read_json(Path(cell["request"]))
        assert request["stream"] is False
