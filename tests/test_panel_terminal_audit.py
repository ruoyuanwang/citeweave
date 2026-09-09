from __future__ import annotations

from citeweave.io import sha256_file, write_json
from citeweave.panel_terminal_audit import audit_panel_terminal


def _fixture(tmp_path, *, status: str, attempt: int = 1):
    benchmark = tmp_path / "benchmark.json"
    write_json(
        benchmark,
        {
            "tasks": [
                {"item_id": "i1", "task_type": "multi_hop_connector"}
            ]
        },
    )
    output = tmp_path / "output"
    write_json(
        output / "results.json",
        {
            "manifest": {
                "benchmark_sha256": sha256_file(benchmark),
                "conditions": ["graph_program"],
            },
            "records": [
                {
                    "item_id": "i1",
                    "condition": "graph_program",
                    "status": status,
                    "attempt": attempt,
                }
            ],
        },
    )
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "conditions": ["graph_program"],
            "datasets": [
                {
                    "dataset_id": "d1",
                    "benchmark": str(benchmark),
                    "output": str(output),
                }
            ],
        },
    )
    return plan


def test_terminal_audit_requests_identical_parse_retry(tmp_path) -> None:
    plan = _fixture(tmp_path, status="failed_parse", attempt=2)
    audit = audit_panel_terminal(plan, maximum_parse_attempts=3)
    assert audit["status"] == "needs_retry"
    assert audit["retry_cells"][0]["reason"] == "parse_retry_permitted"


def test_terminal_audit_keeps_third_parse_failure_as_terminal(tmp_path) -> None:
    plan = _fixture(tmp_path, status="failed_parse", attempt=3)
    audit = audit_panel_terminal(plan, maximum_parse_attempts=3)
    assert audit["status"] == "terminal"
    assert audit["terminal_parse_failure_cells"] == 1


def test_terminal_audit_accepts_complete_cell(tmp_path) -> None:
    plan = _fixture(tmp_path, status="complete")
    audit = audit_panel_terminal(plan)
    assert audit["status"] == "terminal"
    assert audit["complete_cells"] == 1
