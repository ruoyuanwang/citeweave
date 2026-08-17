from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "split_resolved_judgments_by_topic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "split_resolved_judgments_by_topic",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    development = ["dev_a", "dev_b"]
    locked = [f"locked_{index}" for index in range(1, 7)]
    references = tmp_path / "human_references.yml"
    references.write_text(
        yaml.safe_dump(
            {
                "references": [
                    *[
                        {"id": topic, "role": "development"}
                        for topic in development
                    ],
                    *[{"id": topic, "role": "locked"} for topic in locked],
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "packet_id": f"JP{index:020x}",
            "sample_id": f"{topic}:network:item",
            "source": "dual_consensus",
            "conflicts": [],
            "conditions": {"graph_rag": {}, "no_rag": {}},
            "preference": "graph_rag",
        }
        for index, topic in enumerate(locked, 1)
    ]
    input_path = tmp_path / "resolved.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return references, input_path, tmp_path / "split", locked


def test_splits_exactly_six_locked_topics_and_is_idempotent(tmp_path: Path):
    references, input_path, output_root, locked = _fixture(tmp_path)
    counts = MODULE.split_resolved(
        input_path=input_path,
        references_path=references,
        output_root=output_root,
    )
    assert counts == {topic: 1 for topic in locked}
    assert {path.name for path in output_root.iterdir()} == set(locked)
    assert MODULE.split_resolved(
        input_path=input_path,
        references_path=references,
        output_root=output_root,
    ) == counts


def test_rejects_development_or_missing_locked_topic(tmp_path: Path):
    references, input_path, output_root, _ = _fixture(tmp_path)
    rows = input_path.read_text(encoding="utf-8").splitlines()
    value = json.loads(rows[-1])
    value["sample_id"] = "dev_a:network:item"
    rows[-1] = json.dumps(value)
    input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one locked topic"):
        MODULE.split_resolved(
            input_path=input_path,
            references_path=references,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_refuses_to_overwrite_changed_split(tmp_path: Path):
    references, input_path, output_root, locked = _fixture(tmp_path)
    MODULE.split_resolved(
        input_path=input_path,
        references_path=references,
        output_root=output_root,
    )
    path = output_root / locked[0] / "resolved_judgments.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.split_resolved(
            input_path=input_path,
            references_path=references,
            output_root=output_root,
        )
