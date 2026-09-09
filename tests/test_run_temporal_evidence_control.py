import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_temporal_control", Path(__file__).parents[1] / "scripts/run_temporal_evidence_control.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_completed_response_is_terminal_regardless_of_quality():
    assert module.terminal({"status": "complete", "score": {"answer_exact": False}})
    assert not module.terminal({"status": "failed_parse", "attempt": 2})
    assert module.terminal({"status": "failed_parse", "attempt": 3})


def test_control_parser_accepts_fences_but_not_nonobject_json():
    def raw(content):
        return {"choices": [{"message": {"content": content}}]}

    assert module.parse_response(raw('```json\n{"answer": {}}\n```')) == {"answer": {}}
    with pytest.raises(TypeError):
        module.parse_response(raw("[]"))


def test_abstention_is_failure_and_bad_citations_do_not_crash_or_resample():
    task = {"item_id": "i", "answer": {"n": 1}, "evidence_ids": ["e"]}
    parsed = {"item_id": "i", "answer": {"n": 1}, "abstain": True}
    assert not module.score_control_response(task, parsed)["answer_exact"]
    parsed = {**parsed, "abstain": False, "evidence_ids": [{"id": "e"}]}
    score = module.score_control_response(task, parsed)
    assert score["answer_exact"]
    assert score["malformed_evidence_ids"]


def test_saved_control_identities_reject_duplicate_or_changed_cells():
    cell = {"item_id": "i", "condition": "flat_hybrid", "messages": []}
    protocol = {"model": "test", "temperature": 0, "max_tokens": 2200}
    record = {
        "item_id": "i",
        "condition": "flat_hybrid",
        "request_sha256": module.request_identity(cell, protocol),
    }
    module.validate_saved({"records": [record]}, [cell], protocol)
    with pytest.raises(ValueError):
        module.validate_saved({"records": [record, record]}, [cell], protocol)
    with pytest.raises(ValueError):
        module.validate_saved(
            {"records": [{**record, "request_sha256": "drift"}]}, [cell], protocol
        )
