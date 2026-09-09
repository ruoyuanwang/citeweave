from __future__ import annotations

from scripts.merge_formal_v3_complexity_extension_results import verify_reuse_identity


def test_reuse_identity_requires_task_context_and_request_hashes() -> None:
    expected = {
        "task_payload_sha256": "task",
        "context_sha256": "context",
        "request_messages_sha256": "request",
    }
    assert verify_reuse_identity(dict(expected), expected)["passed"]
    corrupted = {**expected, "context_sha256": "different"}
    audit = verify_reuse_identity(corrupted, expected)
    assert not audit["passed"]
    assert audit["mismatches"] == ["context_sha256"]
