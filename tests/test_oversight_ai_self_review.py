import json
from pathlib import Path

import pytest

from citeweave.io import write_json
from citeweave.oversight_ai_self_review import (
    OversightAISelfReviewError,
    build_ai_review_request,
    build_ai_self_review_plan,
    parse_ai_review_content,
)


class _FakeTokenizer:
    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return len(json.dumps(messages)) // 4


def _packet(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "candidate_response": {"answer": "A"},
        "operator_trace": [{"operator": "test"}],
        "evidence_set_a": [{"evidence_id": "E1", "value": "A"}],
        "evidence_set_b": [{"evidence_id": "E2", "value": "B"}],
    }


def test_build_plan_freezes_both_packet_arms_without_private_paths(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(96):
        case_id = f"C{index:03d}"
        standard = tmp_path / "standard" / f"{case_id}.json"
        adversarial = tmp_path / "adversarial" / f"{case_id}.json"
        write_json(standard, _packet(case_id))
        write_json(adversarial, _packet(case_id))
        from citeweave.io import sha256_file

        records.append(
            {
                "case_id": case_id,
                "dataset_id": f"D{index // 12}",
                "standard_public_path": str(standard.relative_to(tmp_path)),
                "standard_public_sha256": sha256_file(standard),
                "adversarial_public_path": str(adversarial.relative_to(tmp_path)),
                "adversarial_public_sha256": sha256_file(adversarial),
                "internal_path": f"internal/{case_id}.json",
            }
        )
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "status": "prospective_factorial_packets_frozen_before_review",
            "study_role": "fixed_outcome_balanced_real_candidate_benchmark",
            "human_outcomes_inspected": False,
            "records": records,
        },
    )
    tokenizer_manifest = tmp_path / "tokenizer.json"
    write_json(tokenizer_manifest, {"model": "deepseek-v4-pro", "passed": True})
    result = build_ai_self_review_plan(
        packet_manifest_path=manifest,
        tokenizer_manifest_path=tokenizer_manifest,
        output_path=tmp_path / "plan.json",
        tokenizer=_FakeTokenizer(),
    )
    assert result["planned_calls"] == 192
    assert len({row["cell_id"] for row in result["cells"]}) == 192
    assert all("internal" not in row["packet_path"] for row in result["cells"])


def test_response_parser_enforces_human_schema_and_visible_evidence() -> None:
    packet = _packet("C1")
    response = {
        "verdict": "qualify",
        "decisive_evidence_ids": ["E1"],
        "invalid_operator_steps": [],
        "failure_mode": "The answer needs a limitation.",
        "minimal_rewrite": "A, within the supplied scope.",
        "rationale": "E1 supports the core answer but not unrestricted scope.",
    }
    assert parse_ai_review_content(json.dumps(response), packet) == response
    response["decisive_evidence_ids"] = ["HIDDEN"]
    with pytest.raises(OversightAISelfReviewError, match="Unknown decisive"):
        parse_ai_review_content(json.dumps(response), packet)


def test_request_is_deterministic_and_json_only() -> None:
    request = build_ai_review_request(_packet("C1"))
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    assert request == build_ai_review_request(_packet("C1"))
