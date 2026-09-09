from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.run_graph_discovery_experiment import _messages
from src.citeweave.token_budget import (
    CommandTokenizer,
    canonical_context_json,
    truncate_context_to_token_budget,
    verify_api_usage_probe_artifact,
)


def _words(text: str) -> int:
    return len(text.replace("{", " ").replace("}", " ").split())


def test_truncates_ranked_records_under_exact_counter() -> None:
    context = {
        "representation": "test",
        "summaries": [{"rank": 0}, {"rank": 1}],
        "nodes": [{"rank": index, "text": "evidence evidence"} for index in range(12)],
        "operator_trace": [{"must": "remain intact"}],
    }
    full_tokens = _words(canonical_context_json(context))
    result = truncate_context_to_token_budget(
        context,
        count_tokens=_words,
        token_budget=full_tokens - 8,
    )
    assert result.budgeted_tokens <= result.token_budget
    assert result.retained_records["nodes"] < 12
    assert result.context["operator_trace"] == context["operator_trace"]
    assert result.context["nodes"] == context["nodes"][: result.retained_records["nodes"]]
    assert result.retained_records["summaries"] >= 1


def test_rejects_budget_smaller_than_fixed_operator_context() -> None:
    with pytest.raises(ValueError, match="metadata/operator trace"):
        truncate_context_to_token_budget(
            {"operator_trace": [{"large": "one two three four five"}], "rows": []},
            count_tokens=_words,
            token_budget=1,
        )


def test_trims_only_scaling_community_metrics_and_keeps_selected_rows() -> None:
    metrics = [
        {
            "community": index,
            "nodes": 100 - index,
            "importance": 500 - index,
            "external_share": index / 100,
            "representative": f"entity-{index}-" + "x" * 40,
        }
        for index in range(80)
    ]
    context = {
        "representation": "operator_trace_without_raw_graph",
        "operator_trace": [
            {"operator": "louvain", "seed": 42, "communities": len(metrics)},
            {"operator": "community_aggregate", "metrics": metrics},
            {"operator": "role_contrast", "dominant": 0, "outward": 73},
        ],
        "interpretation_contract": {"forbidden": ["causality"]},
    }
    result = truncate_context_to_token_budget(
        context,
        count_tokens=len,
        token_budget=1500,
    )
    retained = result.context["operator_trace"][1]["metrics"]
    retained_ids = {row["community"] for row in retained}
    assert result.budgeted_tokens <= result.token_budget
    assert len(retained) < len(metrics)
    assert {0, 73} <= retained_ids
    assert result.context["operator_trace"][2] == context["operator_trace"][2]
    assert result.retained_records["operator_trace[1].metrics"] == len(retained)


def test_under_budget_operator_trace_is_byte_identical() -> None:
    context = {
        "operator_trace": [
            {
                "operator": "community_aggregate",
                "metrics": [{"community": 0}, {"community": 1}],
            },
            {"operator": "role_contrast", "dominant": 0, "outward": 1},
        ]
    }
    rendered = canonical_context_json(context)
    result = truncate_context_to_token_budget(
        context,
        count_tokens=len,
        token_budget=len(rendered),
    )
    assert canonical_context_json(result.context) == rendered
    assert result.context_sha256 == hashlib.sha256(rendered.encode()).hexdigest()


def test_nested_fallback_does_not_change_feasible_legacy_allocation() -> None:
    metrics = [{"community": index, "score": index / 10} for index in range(8)]
    context = {
        "rows": [{"rank": index, "text": "x" * 120} for index in range(20)],
        "operator_trace": [
            {"operator": "community_aggregate", "metrics": metrics},
            {"operator": "role_contrast", "dominant": 0, "outward": 7},
        ],
    }
    fixed_tokens = len(
        canonical_context_json({**context, "rows": []})
    )
    result = truncate_context_to_token_budget(
        context,
        count_tokens=len,
        token_budget=fixed_tokens + 400,
    )
    assert result.budgeted_tokens <= result.token_budget
    assert result.context["operator_trace"][0]["metrics"] == metrics
    assert "operator_trace[0].metrics" not in result.retained_records
    assert result.retained_records["rows"] < len(context["rows"])


def test_command_tokenizer_executes_frozen_local_counter(tmp_path: Path) -> None:
    script = tmp_path / "counter.py"
    script.write_text(
        "import json,sys\np=json.load(sys.stdin)\n"
        "print(json.dumps({'tokens': len(p['text'].split())}))\n",
        encoding="utf-8",
    )
    tokenizer = CommandTokenizer(
        {"model": "test-model", "counting_command": [sys.executable, str(script)]},
        manifest_dir=tmp_path,
    )
    assert tokenizer.count("one two three") == 3
    verification = tokenizer.verify(
        [{"probe_id": "p1", "text": "one two", "expected_tokens": 2}]
    )
    assert verification["passed"] is True


def test_command_tokenizer_uses_utf8_for_unicode_payloads(tmp_path: Path) -> None:
    script = tmp_path / "unicode_counter.py"
    script.write_text(
        "import json,sys\n"
        "assert sys.stdin.encoding.lower().replace('-', '') == 'utf8'\n"
        "p=json.load(sys.stdin)\n"
        "print(json.dumps({'tokens': len(p['text'])}, ensure_ascii=False))\n",
        encoding="utf-8",
    )
    tokenizer = CommandTokenizer(
        {"model": "test-model", "counting_command": [sys.executable, str(script)]},
        manifest_dir=tmp_path,
    )
    text = "Göttingen – 癌症"
    assert tokenizer.count(text) == len(text)


def test_command_tokenizer_verifies_full_message_probes(tmp_path: Path) -> None:
    script = tmp_path / "message_counter.py"
    script.write_text(
        "import json,sys\np=json.load(sys.stdin)\n"
        "v=p.get('messages')\n"
        "tokens=sum(len(row['content'].split())+2 for row in v) if v is not None "
        "else len(p['text'].split())\n"
        "print(json.dumps({'tokens': tokens}))\n",
        encoding="utf-8",
    )
    tokenizer = CommandTokenizer(
        {
            "model": "test-model",
            "counting_command": [sys.executable, str(script)],
            "message_counting_command": [sys.executable, str(script)],
        },
        manifest_dir=tmp_path,
    )
    verification = tokenizer.verify_message_probes(
        [
            {
                "probe_id": "messages",
                "messages": [
                    {"role": "system", "content": "one two"},
                    {"role": "user", "content": "three"},
                ],
                "prompt_tokens": 7,
            }
        ]
    )
    assert verification["passed"] is True

    artifact = tmp_path / "api_probes.json"
    artifact.write_text(
        json.dumps(
            {
                "passed": True,
                "requested_model": "test-model",
                "records": [
                    {
                        "probe_id": "messages",
                        "messages": [
                            {"role": "system", "content": "one two"},
                            {"role": "user", "content": "three"},
                        ],
                        "prompt_tokens": 7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload, artifact_verification, reasons = verify_api_usage_probe_artifact(
        {
            "model": "test-model",
            "api_usage_probe_artifact": {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
        },
        manifest_dir=tmp_path,
        tokenizer=tokenizer,
    )
    assert payload["passed"] is True
    assert artifact_verification["passed"] is True
    assert reasons == []


def test_formal_message_assembly_uses_budgeted_context() -> None:
    class FakeTokenizer:
        @staticmethod
        def count(text: str) -> int:
            return len(text)

    task = {
        "item_id": "i1",
        "task_type": "multi_hop_connector",
        "complexity": 3,
        "question": "Find the weighted path.",
        "answer": {"hops": 2},
        "contexts": {
            "flat_bm25": {
                "representation": "flat",
                "rows": [
                    {"evidence_id": f"e{index}", "text": "x" * 50}
                    for index in range(20)
                ],
                "operator_trace": [{"must": "remain"}],
            }
        },
    }
    messages, audit = _messages(
        task,
        "flat_bm25",
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        token_budget=500,
    )
    assert audit["budgeted_context_tokens"] <= 500
    assert audit["retained_records"]["rows"] < 20
    assert '"must":"remain"' in messages[1]["content"]
