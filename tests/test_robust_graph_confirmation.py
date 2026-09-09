from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_robust_graph_confirmation_candidates import _sample_packet
from scripts.prepare_robust_graph_confirmation_workspaces import prepare


def _write_protocol(path: Path) -> None:
    candidates = [
        {
            "priority": index,
            "id": f"topic_{index}",
            "title": f"Topic {index}",
            "domain": f"domain_{index}",
            "keywords": [f"term {index}", "anchor"],
            "year_from": 2010,
            "year_to": 2025,
        }
        for index in range(1, 13)
    ]
    path.write_text(
        "status: frozen_before_any_candidate_acquisition\n"
        + "candidate_topics:\n"
        + "".join(
            "  - priority: {priority}\n"
            "    id: {id}\n"
            "    title: {title}\n"
            "    domain: {domain}\n"
            "    keywords: ['{keyword}', anchor]\n"
            "    year_from: 2010\n"
            "    year_to: 2025\n".format(
                priority=row["priority"],
                id=row["id"],
                title=row["title"],
                domain=row["domain"],
                keyword=row["keywords"][0],
            )
            for row in candidates
        ),
        encoding="utf-8",
    )


def test_prepares_twelve_hash_bound_workspaces(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yml"
    _write_protocol(protocol)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    output = tmp_path / "workspaces"
    manifest_path = tmp_path / "manifest.json"
    result = prepare(protocol, freeze, output, manifest_path)
    assert result["candidate_count"] == 12
    assert result["confirmatory"] is True
    assert all((output / f"topic_{index}" / "project.yml").is_file() for index in range(1, 13))
    with pytest.raises(ValueError, match="overwrite"):
        prepare(protocol, freeze, output, manifest_path)


def test_rejects_noncontiguous_priority(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yml"
    _write_protocol(protocol)
    protocol.write_text(
        protocol.read_text(encoding="utf-8").replace("priority: 12", "priority: 13"),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="priority"):
        prepare(protocol, freeze, tmp_path / "workspaces", tmp_path / "manifest.json")


def test_query_packet_is_deterministic_blind_and_capped_at_one_hundred() -> None:
    works = pd.DataFrame(
        [
            {
                "work_id": f"W{index}",
                "title": f"Title {index}",
                "abstract": f"Abstract {index}",
                "year": 2020,
            }
            for index in range(150)
        ]
    )
    dataset = {
        "id": "private_topic_id",
        "priority": 1,
        "keywords": ["concept a", "concept b"],
    }
    first = _sample_packet(
        protocol_sha256="frozen", dataset=dataset, works=works
    )
    second = _sample_packet(
        protocol_sha256="frozen", dataset=dataset, works=works.sample(frac=1, random_state=4)
    )
    assert first == second
    assert len(first["items"]) == 100
    assert "private_topic_id" not in json.dumps(first)
    assert "priority" not in first
