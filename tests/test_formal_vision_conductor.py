from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from citeweave.formal_vision_conductor import (
    GENERATOR_ROLE,
    VisionConductorError,
    import_returns,
    load_frozen_topics,
    prepare_assignments,
)
from citeweave.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "formal_datasets_openalex_title_abstract.yml"


def _packet_hash(packet: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            packet,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _output(dataset_id: str, packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packet_sha256": packet["packet_sha256"],
        "dataset_id": dataset_id,
        "generator_role": GENERATOR_ROLE,
        "visible_only": True,
        "results": [
            {
                "item_id": item["item_id"],
                "abstain": False,
                "answer": {"nodes": index + 2, "links": index + 5},
                "explanation": "The two values are visibly printed in the figure.",
            }
            for index, item in enumerate(packet["items"])
        ],
    }


def _formal_fixture(tmp_path: Path) -> dict[str, Any]:
    packet_root = tmp_path / "packets"
    workspace_root = tmp_path / "workspaces"
    packets: dict[str, dict[str, Any]] = {}
    for topic in load_frozen_topics(REGISTRY):
        dataset_id = topic["dataset_id"]
        workspace = workspace_root / dataset_id
        figures = workspace / "figures"
        artifacts = []
        benchmark = []
        items = []
        for index in range(2):
            figure = (figures / f"network_{index}.png").resolve()
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(f"png:{dataset_id}:{index}".encode())
            artifacts.append(
                {
                    "name": f"network_{index}",
                    "png": str(figure),
                    "png_sha256": sha256_file(figure),
                }
            )
            item = {
                "item_id": f"{dataset_id}:network_{index}:network_size",
                "task_type": "network_size",
                "question": f"What size is network {index}?",
                "answer_contract": {"nodes": "integer", "links": "integer"},
                "figure_path": str(figure),
            }
            items.append(item)
            benchmark.append(
                {
                    **item,
                    "figure_eligible": True,
                    "gold_answer": {"SECRET_GOLD": True},
                }
            )
        _write_json(figures / "figure_manifest.json", {"figures": artifacts})
        benchmark_path = (
            workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
        )
        _write_json(benchmark_path, benchmark)
        packet = {
            "schema_version": 1,
            "packet_role": "cross_model_figure_vlm_generation",
            "dataset_id": dataset_id,
            "generator": "independent_codex_visual_subagent",
            "main_inference": False,
            "visible_only": True,
            "prohibited_inputs": [
                "gold_answer",
                "graph JSON",
                "flat structured context",
                "human reference output",
                "other condition output",
            ],
            "instructions": "Visible figures only.",
            "benchmark_sha256": sha256_file(benchmark_path),
            "items": items,
            "output_schema": {
                "schema_version": 1,
                "packet_sha256": "<copy packet_sha256>",
                "dataset_id": dataset_id,
                "generator_role": GENERATOR_ROLE,
                "visible_only": True,
                "results": [
                    {
                        "item_id": "<exact item_id>",
                        "abstain": False,
                        "answer": {"nodes": 0, "links": 0},
                        "explanation": "<brief visual basis>",
                    }
                ],
            },
        }
        packet["packet_sha256"] = _packet_hash(packet)
        _write_json(packet_root / f"{dataset_id}.json", packet)
        packets[dataset_id] = packet
    return {
        "packet_root": packet_root,
        "workspace_root": workspace_root,
        "packets": packets,
        "output_root": tmp_path / "outputs",
        "exchange_root": tmp_path / "exchange",
        "returns_root": tmp_path / "returns",
    }


def _kwargs(paths: dict[str, Any]) -> dict[str, Path]:
    return {
        "registry_path": REGISTRY,
        "packet_root": paths["packet_root"],
        "workspace_root": paths["workspace_root"],
        "output_root": paths["output_root"],
        "exchange_root": paths["exchange_root"],
        "returns_root": paths["returns_root"],
    }


def test_prepare_creates_independent_visible_only_assignments_for_missing_outputs(
    tmp_path: Path,
) -> None:
    paths = _formal_fixture(tmp_path)
    first_id = load_frozen_topics(REGISTRY)[0]["dataset_id"]
    _write_json(
        paths["output_root"] / f"{first_id}.json",
        _output(first_id, paths["packets"][first_id]),
    )

    result = prepare_assignments(**_kwargs(paths))

    assert result["formal_topic_count"] == 8
    assert result["complete_output_count"] == 1
    assert result["missing_output_count"] == 7
    assert "--include-vision" in result["score_command"]
    assert len(result["assignments"]) == 7
    for row in result["assignments"]:
        assignment = json.loads(
            Path(row["assignment_path"]).read_text(encoding="utf-8")
        )
        assert assignment["visible_only"] is True
        assert assignment["controller_metadata_in_scope"] is False
        assert assignment["may_read_unlisted_workspace_files"] is False
        assert assignment["may_modify_pipeline_or_source_artifacts"] is False
        assert assignment["return_path"] == row["return_path"]
        assert all("figure_sha256" in item for item in assignment["items"])
        assert "SECRET_GOLD" not in json.dumps(assignment)


def test_import_is_batch_validated_atomic_and_byte_idempotent(tmp_path: Path) -> None:
    paths = _formal_fixture(tmp_path)
    topics = load_frozen_topics(REGISTRY)
    for topic in topics:
        dataset_id = topic["dataset_id"]
        _write_json(
            paths["returns_root"] / f"{dataset_id}.json",
            _output(dataset_id, paths["packets"][dataset_id]),
        )
    last_id = topics[-1]["dataset_id"]
    bad_path = paths["returns_root"] / f"{last_id}.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["results"].reverse()
    _write_json(bad_path, bad)

    with pytest.raises(VisionConductorError, match="IDs/order/coverage"):
        import_returns(
            **{key: value for key, value in _kwargs(paths).items() if key != "exchange_root"},
            run_score=False,
        )
    assert not paths["output_root"].exists()

    _write_json(bad_path, _output(last_id, paths["packets"][last_id]))
    result = import_returns(
        **{key: value for key, value in _kwargs(paths).items() if key != "exchange_root"},
        run_score=False,
    )
    before = {
        path.name: path.read_bytes() for path in paths["output_root"].glob("*.json")
    }
    again = import_returns(
        **{key: value for key, value in _kwargs(paths).items() if key != "exchange_root"},
        run_score=False,
    )

    assert result["newly_imported_count"] == 8
    assert result["mechanical_score_status"] == "prepared"
    assert again["newly_imported_count"] == 0
    assert before == {
        path.name: path.read_bytes() for path in paths["output_root"].glob("*.json")
    }


def test_prepare_fails_closed_on_figure_hash_tampering(tmp_path: Path) -> None:
    paths = _formal_fixture(tmp_path)
    dataset_id = load_frozen_topics(REGISTRY)[0]["dataset_id"]
    figure = Path(paths["packets"][dataset_id]["items"][0]["figure_path"])
    figure.write_bytes(b"tampered")

    with pytest.raises(VisionConductorError, match="figure SHA-256 differs"):
        prepare_assignments(**_kwargs(paths))
    assert not paths["exchange_root"].exists()


def test_import_rejects_role_or_schema_expansion(tmp_path: Path) -> None:
    paths = _formal_fixture(tmp_path)
    topics = load_frozen_topics(REGISTRY)
    for topic in topics:
        dataset_id = topic["dataset_id"]
        output = _output(dataset_id, paths["packets"][dataset_id])
        _write_json(paths["returns_root"] / f"{dataset_id}.json", output)
    first_id = topics[0]["dataset_id"]
    bad_path = paths["returns_root"] / f"{first_id}.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["generator_role"] = "pipeline_editor"
    bad["hidden_context"] = True
    _write_json(bad_path, bad)

    with pytest.raises(VisionConductorError, match="top-level keys differ"):
        import_returns(
            **{key: value for key, value in _kwargs(paths).items() if key != "exchange_root"},
            run_score=False,
        )
    assert not paths["output_root"].exists()
