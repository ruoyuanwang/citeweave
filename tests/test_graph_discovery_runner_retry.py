from __future__ import annotations

from scripts.run_graph_discovery_experiment import _prepare_retry, _task_payload_sha256


def test_failed_terminal_record_moves_to_attempt_log_before_retry() -> None:
    records = [
        {"item_id": "a", "condition": "graph_program", "status": "failed_parse"},
        {"item_id": "b", "condition": "graph_program", "status": "complete"},
    ]
    attempt_log: list[dict[str, object]] = []

    retained, attempt = _prepare_retry(
        records,
        attempt_log,
        item_id="a",
        condition="graph_program",
    )

    assert [row["item_id"] for row in retained] == ["b"]
    assert attempt == 2
    assert attempt_log == [
        {
            "item_id": "a",
            "condition": "graph_program",
            "status": "failed_parse",
            "archived_reason": "permitted_identical_retry",
        }
    ]


def test_task_payload_hash_excludes_retrieval_contexts() -> None:
    first = {
        "item_id": "a",
        "question": "q",
        "contexts": {"flat": {"rows": [1]}},
        "context_hashes": {"flat": "x"},
    }
    second = {
        **first,
        "contexts": {"flat": {"rows": [2]}},
        "context_hashes": {"flat": "y"},
    }
    assert _task_payload_sha256(first) == _task_payload_sha256(second)
