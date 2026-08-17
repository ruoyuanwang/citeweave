from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from citeweave.report_conditions import (
    CompletedRunError,
    IncompleteRunError,
    ReportConditionRunner,
    ReportRunConfig,
)


class FakeTransport:
    def __init__(self, *, fail_at: int | None = None):
        self.requests: list[dict[str, Any]] = []
        self.fail_at = fail_at

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if self.fail_at == len(self.requests):
            raise RuntimeError("simulated interruption")
        return {
            "id": f"fake-{len(self.requests)}",
            "choices": [
                {
                    "message": {
                        "content": f"# English report stage {len(self.requests)}\n\nClaim [E001]."
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _evidence(path: Path, *, statement: str = "There are 10 works.") -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "evidence_id": "E001",
                    "claim_type": "corpus_size",
                    "statement": statement,
                    "value": 10,
                    "method": "deterministic count",
                    "caveat": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _runner(
    tmp_path: Path,
    evidence: Path,
    condition: str,
    transport: FakeTransport,
    *,
    resume: bool = False,
) -> ReportConditionRunner:
    return ReportConditionRunner(
        evidence_path=evidence,
        output_root=tmp_path / "runs",
        config=ReportRunConfig(dataset_id="topic-a", condition=condition),  # type: ignore[arg-type]
        transport=transport,
        resume=resume,
    )


def test_one_shot_is_exactly_one_traced_call_and_is_immutable(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    transport = FakeTransport()
    result = _runner(tmp_path, evidence, "structured_one_shot", transport).run()

    condition = tmp_path / "runs" / "topic-a" / "structured_one_shot"
    assert result["call_count"] == 1
    assert len(transport.requests) == 1
    trace = json.loads(
        (condition / "calls" / "01_structured_one_shot" / "call.json").read_text("utf-8")
    )
    assert trace["raw_request"] == transport.requests[0]
    assert trace["raw_response"]["usage"]["total_tokens"] == 15
    assert json.loads(
        (condition / "calls" / "01_structured_one_shot" / "response.json").read_text("utf-8")
    ) == trace["raw_response"]
    assert trace["prompt_sha256"]
    assert trace["latency_ms"] >= 0
    assert trace["model"] == "deepseek-v4-pro"
    assert trace["temperature"] == 0.1
    assert trace["seed"] == 42
    assert trace["raw_request"]["seed"] == 42
    assert (condition / "report.md").read_text("utf-8").startswith("# English report")
    with pytest.raises(CompletedRunError):
        _runner(tmp_path, evidence, "structured_one_shot", FakeTransport(), resume=True).run()


def test_full_has_four_calls_and_marks_internal_review_as_revision_only(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    transport = FakeTransport()
    result = _runner(tmp_path, evidence, "citeweave_full", transport).run()

    condition = tmp_path / "runs" / "topic-a" / "citeweave_full"
    assert result["call_count"] == 4
    assert len(transport.requests) == 4
    review = json.loads(
        (condition / "calls" / "03_internal_review" / "call.json").read_text("utf-8")
    )
    assert review["role"] == "internal_revision_review_not_experimental_evaluation"
    manifest = json.loads((condition / "run_manifest.json").read_text("utf-8"))
    assert "must not be reused" in manifest["internal_review_policy"]
    assert result["internal_review_used_as_experimental_evaluation"] is False


def test_resume_reuses_completed_calls_after_interruption(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    first = FakeTransport(fail_at=3)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _runner(tmp_path, evidence, "citeweave_full", first).run()
    assert len(first.requests) == 3

    second = FakeTransport()
    result = _runner(tmp_path, evidence, "citeweave_full", second, resume=True).run()
    assert result["call_count"] == 4
    assert len(second.requests) == 2


def test_incomplete_run_requires_explicit_resume(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path / "evidence.json")
    runner = _runner(tmp_path, evidence, "citeweave_full", FakeTransport(fail_at=1))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run()
    with pytest.raises(IncompleteRunError):
        _runner(tmp_path, evidence, "citeweave_full", FakeTransport()).run()


def test_conditions_reject_different_evidence_hashes(tmp_path: Path) -> None:
    first_evidence = _evidence(tmp_path / "first.json")
    _runner(tmp_path, first_evidence, "structured_one_shot", FakeTransport()).run()
    other_evidence = _evidence(tmp_path / "other.json", statement="Changed evidence.")
    with pytest.raises(RuntimeError, match="same frozen Evidence Bundle"):
        _runner(tmp_path, other_evidence, "citeweave_full", FakeTransport()).run()
