from __future__ import annotations

import hashlib
import json
from pathlib import Path

from citeweave.article_compiler_v3 import (
    PROMPT_VERSION,
    REPAIR_MAX_TOKENS,
    SECTION_MAX_TOKENS,
    build_compiler_repair_request,
    build_compiler_section_request,
)
from citeweave.io import write_json
from scripts.run_article_compiler_v3_pilot import archive_failed_transport_attempts


def _writer_input() -> dict:
    task_types = (
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    return {
        "dataset_id": "dataset",
        "graph_phenomena": [
            {
                "phenomenon_id": f"PH-{index}",
                "task_type": task_type,
                "question": "question",
                "verified_answer": {"value": index},
                "operator_trace": [{"operator": "operator"}],
                "graph_evidence_ids": [f"GE-{index}"],
                "reference_ids": [f"REF-{2 * index}", f"REF-{2 * index + 1}"],
                "interpretation_contract": {
                    "allowed": "bounded",
                    "forbidden": "causal",
                    "required_limitation": "limited",
                },
            }
            for index, task_type in enumerate(task_types)
        ],
        "representative_sources": [
            {"reference_id": f"REF-{index}"} for index in range(10)
        ],
    }


def test_token_safe_limits_change_only_the_response_ceiling_and_version() -> None:
    writer = _writer_input()
    available = {
        "Abstract": "abstract",
        "Introduction": "introduction",
        "Methods": "methods",
        "Results": "results",
        "Discussion": "discussion",
        "Limitations": "limitations",
        "Conclusion": "conclusion",
    }
    for section, expected in SECTION_MAX_TOKENS.items():
        request = build_compiler_section_request(
            writer,
            condition="flat_article_compiler",
            section=section,
            compiled_sections=available,
        )
        assert request["max_tokens"] == expected
        assert PROMPT_VERSION in request["messages"][1]["content"]
    repair = build_compiler_repair_request(
        writer,
        condition="graph_dependency_compiler",
        section="Results",
        original_body="PH-0 REF-0.",
        compiled_sections={},
    )
    assert repair["max_tokens"] == REPAIR_MAX_TOKENS["Results"]
    assert PROMPT_VERSION in repair["messages"][1]["content"]


def test_failed_transport_attempt_is_archived_before_identical_retry(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "article"
    call_dir = output_dir / "calls" / "01_methods_base"
    request = {"model": "test"}
    write_json(call_dir / "request.json", request)
    request_hash = hashlib.sha256(
        json.dumps(request, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    ).hexdigest()
    write_json(
        call_dir / "execution_record.json",
        {"response_received": False, "request_sha256": request_hash},
    )
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {"articles": [{"output_dir": str(output_dir)}]})
    assert archive_failed_transport_attempts(plan_path) == 1
    assert not (call_dir / "execution_record.json").exists()
    assert (call_dir / "transport_failure_attempt_001.json").is_file()
