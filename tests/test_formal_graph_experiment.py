from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from citeweave.formal_graph_experiment import (
    GraphAtom,
    JsonlCheckpoint,
    ProviderProfile,
    assert_semantically_equivalent,
    build_formal_graph_grounding,
    comparison_design,
    formal_run_directory,
    provider_for_condition,
    render_flat_context,
    render_graph_context,
    select_condition_items,
    validate_provider_profile,
    verify_formal_graph_grounding,
)
from citeweave.io import read_json, save_config, write_json, write_parquet
from citeweave.models import ProjectConfig, SearchProtocol


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "formal_workspaces" / "topic"
    visual = workspace / "analyses" / "visualization"
    figures = workspace / "figures"
    evidence = workspace / "evidence"
    visual.mkdir(parents=True)
    figures.mkdir()
    evidence.mkdir()
    save_config(
        workspace / "project.yml",
        ProjectConfig(
            project_id="topic",
            protocol=SearchProtocol(
                title="Fixture topic",
                keywords=["fixture"],
                year_from=2020,
                year_to=2021,
            ),
        ),
    )
    nodes = pd.DataFrame(
        [
            {
                "id": "n1",
                "label": "Alpha node",
                "occurrences": 12,
                "cluster": 1,
                "importance": 1.0,
                "x": 0.0,
                "y": 0.0,
            },
            {
                "id": "n2",
                "label": "Beta node",
                "occurrences": 8,
                "cluster": 1,
                "importance": 0.8,
                "x": 1.0,
                "y": 0.0,
            },
            {
                "id": "n3",
                "label": "Gamma node",
                "occurrences": 6,
                "cluster": 2,
                "importance": 0.7,
                "x": 0.5,
                "y": 1.0,
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {"source": "n1", "target": "n2", "weight": 5.0, "association_strength": 0.1},
            {"source": "n2", "target": "n3", "weight": 2.0, "association_strength": 0.04},
        ]
    )
    write_parquet(visual / "keyword_cooccurrence_nodes.parquet", nodes)
    write_parquet(visual / "keyword_cooccurrence_edges.parquet", edges)
    figure = figures / "network_keyword_cooccurrence.png"
    figure.write_bytes(b"fixture png bytes")
    write_json(
        figures / "figure_manifest.json",
        {
            "figures": [
                {
                    "name": "network_keyword_cooccurrence",
                    "png": str(figure.resolve()),
                }
            ]
        },
    )
    write_json(evidence / "graph_qa_benchmark.json", [{"item_id": "bounded"}])
    write_json(evidence / "graph_facts.json", [{"fact_id": "G001"}])
    return workspace


def test_flat_and_graph_renderers_preserve_exact_semantic_atoms() -> None:
    atoms = [
        GraphAtom("a1", "citation", "node:a", "display_label", "Alpha", "label"),
        GraphAtom("a2", "citation", "node:a", "connected_to", "node:b", "entity"),
    ]
    flat = render_flat_context(atoms)
    graph = render_graph_context(atoms)

    assert_semantically_equivalent(flat, graph)
    assert flat["semantic_atom_sha256"] == graph["semantic_atom_sha256"]
    assert set(flat["atom_ids"]) == set(graph["atom_ids"]) == {"a1", "a2"}
    assert graph["paths"] == [
        {"path_id": "path:a2", "nodes": ["node:a", "node:b"], "edges": ["a2"]}
    ]


def test_semantic_equivalence_rejects_context_drift() -> None:
    flat = render_flat_context(
        [GraphAtom("a1", "citation", "node:a", "occurrences", 3, "integer")]
    )
    graph = render_graph_context(
        [GraphAtom("a1", "citation", "node:a", "occurrences", 4, "integer")]
    )
    with pytest.raises(ValueError, match="semantic atom hash"):
        assert_semantically_equivalent(flat, graph)


def test_formal_grounding_uses_displayed_selected_network_and_visible_label_contract(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    manifest = build_formal_graph_grounding(workspace)

    output = workspace / "evidence" / "formal_graph_experiment"
    questions = read_json(output / "benchmark.json")
    size = next(item for item in questions if item["task_type"] == "network_size")
    top = next(item for item in questions if item["task_type"] == "highest_occurrence")
    strongest = next(
        item for item in questions if item["task_type"] == "strongest_connection"
    )
    assert size["gold_answer"] == {"nodes": 3, "links": 2}
    assert size["figure_eligible"] is True
    assert top["gold_answer"] == {"label": "Alpha node"}
    assert top["answer_contract"] == {"label": "visible string"}
    assert strongest["gold_answer"] == {
        "source_label": "Alpha node",
        "target_label": "Beta node",
        "weight": 5.0,
    }
    assert "canonical" not in json.dumps(
        [item["answer_contract"] for item in questions]
    ).casefold()
    assert manifest["figure_vlm_comparison_role"] == "cross_model_extension"
    assert manifest["bounded_graph_qa_sha256"]
    acceptance = verify_formal_graph_grounding(workspace)
    assert acceptance["passed"]
    assert acceptance["contexts_verified"] == len(questions)
    for record in manifest["contexts"]:
        flat = read_json(workspace / record["flat_path"])
        graph = read_json(workspace / record["graph_path"])
        assert_semantically_equivalent(flat, graph)


def test_formal_grounding_acceptance_rejects_context_tampering(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    manifest = build_formal_graph_grounding(workspace)
    flat_path = workspace / manifest["contexts"][0]["flat_path"]
    payload = read_json(flat_path)
    payload["rows"][0]["object"] = 999
    write_json(flat_path, payload)

    acceptance = verify_formal_graph_grounding(workspace)
    assert not acceptance["passed"]
    assert any("hash mismatch" in error for error in acceptance["errors"])


def _profile(tmp_path: Path, *, vision: bool) -> ProviderProfile:
    snapshot = tmp_path / ("vision.json" if vision else "text.json")
    model = "deepseek-series-vlm" if vision else "deepseek-v4-pro"
    write_json(
        snapshot,
        {
            "base_url": "https://provider.example/v1",
            "models": [model],
            "vlm_probe": (
                {"model": model, "accepted_image_input": True} if vision else None
            ),
        },
    )
    return ProviderProfile(
        profile_id="vision" if vision else "text",
        base_url="https://provider.example/v1",
        model=model,
        api_key_env="TEST_KEY",
        modality="vision" if vision else "text",
        capability_snapshot=snapshot,
        supports_data_uri=vision,
    )


def test_provider_profiles_fail_closed_without_verified_vision(tmp_path: Path) -> None:
    text = _profile(tmp_path, vision=False)
    vision = _profile(tmp_path, vision=True)
    assert validate_provider_profile(text, condition="graph_rag").passed
    assert validate_provider_profile(vision, condition="figure_vlm").passed

    wrong = vision.model_copy(update={"supports_data_uri": False})
    rejected = validate_provider_profile(wrong, condition="figure_vlm")
    assert not rejected.passed
    assert any("data-URI" in reason for reason in rejected.reasons)
    design = comparison_design(text, vision)
    assert design["primary_text_panel"]["strict_within_model_comparison"] is True
    assert design["figure_panel"]["role"] == "cross_model_extension"
    assert provider_for_condition(text, vision, "graph_rag") is text
    assert provider_for_condition(text, vision, "figure_vlm") is vision
    with pytest.raises(ValueError, match="vision provider"):
        provider_for_condition(text, None, "figure_vlm")

    unverified_schema = text.model_copy(update={"response_format": "json_schema"})
    schema_rejected = validate_provider_profile(
        unverified_schema, condition="graph_rag"
    )
    assert not schema_rejected.passed
    assert any("json_schema" in reason for reason in schema_rejected.reasons)


def test_same_model_panel_uses_the_common_figure_eligible_item_set(
    tmp_path: Path,
) -> None:
    vision = _profile(tmp_path, vision=True)
    design = comparison_design(vision, vision)
    items = [
        {"item_id": "visible", "figure_eligible": True},
        {"item_id": "hidden", "figure_eligible": False},
    ]
    for condition in ("no_rag", "flat_structured", "graph_rag", "figure_vlm"):
        selected = select_condition_items(items, condition=condition, design=design)
        assert [item["item_id"] for item in selected] == ["visible"]


def test_condition_scoped_run_directories_cannot_overwrite_each_other(
    tmp_path: Path,
) -> None:
    graph = formal_run_directory(
        tmp_path,
        dataset_id="topic",
        run_id="run",
        condition="graph_rag",
    )
    figure = formal_run_directory(
        tmp_path,
        dataset_id="topic",
        run_id="run",
        condition="figure_vlm",
    )
    assert graph != figure
    assert graph.parent == figure.parent


def test_jsonl_checkpoint_resumes_completed_items_and_retries_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "items.jsonl"
    checkpoint = JsonlCheckpoint(path)
    checkpoint.append(
        run_id="run",
        condition="graph_rag",
        item_id="i1",
        request={"model": "m"},
        response=None,
        status="failed",
        elapsed_seconds=0.1,
        error="temporary",
    )
    assert not checkpoint.completed("graph_rag", "i1")
    checkpoint.append(
        run_id="run",
        condition="graph_rag",
        item_id="i1",
        request={"model": "m"},
        response={"model": "m"},
        status="complete",
        elapsed_seconds=0.2,
    )
    resumed = JsonlCheckpoint(path)
    assert resumed.completed("graph_rag", "i1")
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert [record["attempt"] for record in records] == [1, 2]
    with pytest.raises(ValueError, match="already completed"):
        resumed.append(
            run_id="run",
            condition="graph_rag",
            item_id="i1",
            request={"model": "m"},
            response={"model": "m"},
            status="complete",
            elapsed_seconds=0.1,
        )
