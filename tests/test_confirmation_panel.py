from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.confirmation_panel import (
    _freeze_request_identities,
    build_confirmation_panel,
)
from citeweave.io import read_json, sha256_file, write_json
from scripts.run_robust_graph_confirmation_panel import _audit_progress


class _TokenizerStub:
    @staticmethod
    def count(text: str) -> int:
        return max(1, len(text) // 100)

    @staticmethod
    def count_messages(messages: list[dict[str, str]]) -> int:
        return sum(max(1, len(row["content"]) // 100) for row in messages)


def test_confirmation_builder_refuses_to_create_outputs_before_human_selection(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.yml"
    protocol.write_text(
        "selection:\n"
        "  target_topics: 1\n"
        "benchmark:\n"
        "  conditions: [flat_bm25, graph_hierarchical_retrieval_v2, flat_program, graph_program, operator_only]\n"
        "  record_budget: 20\n"
        "  context_token_budget: 8192\n",
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    write_json(freeze, {"sha256": sha256_file(protocol)})
    selection = tmp_path / "selection.json"
    write_json(selection, {"status": "awaiting_blind_query_adjudication"})
    output = tmp_path / "forbidden_output"
    with pytest.raises(ValueError, match="selection is not complete"):
        build_confirmation_panel(
            protocol,
            freeze,
            selection,
            tmp_path / "collection.json",
            tmp_path / "audit.json",
            tmp_path / "initialization.json",
            tmp_path / "tokenizer.json",
            output_root=output,
        )
    assert not output.exists()


def test_request_identity_freeze_covers_every_task_condition_once(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.json"
    task = {
        "item_id": "ITEM-1",
        "task_type": "path_stability_profile",
        "complexity": 6,
        "question": "Assess registered stability.",
        "answer": {"stability_rate": 0.5},
        "contexts": {
            "flat_bm25": {"rows": [{"evidence_id": "E1", "value": 1}]},
            "graph_program": {
                "evidence_rows": [{"evidence_id": "E1", "value": 1}],
                "operator_trace": [{"operator": "aggregate"}],
            },
        },
    }
    write_json(benchmark, {"dataset_id": "topic", "tasks": [task]})
    construction = {
        "planned_calls": 2,
        "records": [{"dataset_id": "topic", "benchmark": str(benchmark)}],
    }
    rows, summary = _freeze_request_identities(
        construction,
        conditions=("flat_bm25", "graph_program"),
        model="deepseek-v4-pro",
        token_budget=8192,
        tokenizer=_TokenizerStub(),  # type: ignore[arg-type]
    )
    assert len(rows) == 2
    assert {(row["item_id"], row["condition"]) for row in rows} == {
        ("ITEM-1", "flat_bm25"),
        ("ITEM-1", "graph_program"),
    }
    assert len({row["request_payload_sha256"] for row in rows}) == 2
    assert set(summary) == {"flat_bm25", "graph_program"}


def test_execution_audit_enforces_frozen_task_and_message_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runs"
    result_path = output / "topic" / "results.json"
    write_json(
        result_path,
        {
            "records": [
                {
                    "item_id": "ITEM-1",
                    "condition": "graph_program",
                    "status": "complete",
                    "task_payload_sha256": "task-hash",
                    "request_messages_sha256": "message-hash",
                }
            ],
            "attempt_log": [],
        },
    )
    construction = {"records": [{"dataset_id": "topic"}]}
    requests = {
        "records": [
            {
                "dataset_id": "topic",
                "item_id": "ITEM-1",
                "condition": "graph_program",
                "task_payload_sha256": "task-hash",
                "request_messages_sha256": "message-hash",
            }
        ]
    }
    audit = _audit_progress(construction, requests, output)
    assert audit["complete_cells"] == 1
    assert audit["all_observed_identities_match"] is True
    payload = read_json(result_path)
    payload["records"][0]["request_messages_sha256"] = "drift"
    write_json(result_path, payload)
    with pytest.raises(ValueError, match="message identity drift"):
        _audit_progress(construction, requests, output)
