from __future__ import annotations

from scripts.merge_formal_v3_mechanism_supplement_results import verify_cell_identity


def test_mechanism_merge_identity_requires_task_context_and_request() -> None:
    expected = {
        "task_payload_sha256": "task",
        "context_sha256": "context",
        "request_messages_sha256": "request",
    }
    assert verify_cell_identity(dict(expected), expected)["passed"]
    corrupted = {**expected, "request_messages_sha256": "different"}
    audit = verify_cell_identity(corrupted, expected)
    assert not audit["passed"]
    assert audit["mismatches"] == ["request_messages_sha256"]
