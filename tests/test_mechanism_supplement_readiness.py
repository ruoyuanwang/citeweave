from __future__ import annotations

from pathlib import Path

import yaml

from citeweave.io import sha256_file, write_json
from scripts.audit_formal_v3_mechanism_supplement_readiness import (
    audit_mechanism_supplement_readiness,
)


class _Tokenizer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def verify(self, _probes):
        return {"passed": True, "probes": []}

    def count_messages(self, _messages):
        return 1

    def count(self, _text):
        return 1

    def truncate_text_for_messages(self, text, **_kwargs):
        return text, {"truncated": False, "tokens": 1}


def test_readiness_materializes_cells_but_blocks_before_sources(
    tmp_path: Path, monkeypatch
) -> None:
    extension = tmp_path / "extension.yml"
    extension.write_text("schema_version: 1\n", encoding="utf-8")
    supplement = tmp_path / "supplement.yml"
    supplement.write_text(
        yaml.safe_dump(
            {
                "source_bindings": {
                    "complexity_extension_protocol_sha256": sha256_file(extension),
                    "complexity_execution_amendment_sha256": "a" * 64,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    write_json(freeze, {"sha256": sha256_file(supplement)})
    benchmark_root = tmp_path / "benchmarks"
    records = []
    task_types = (
        "direct_edge_lookup",
        "node_attribute_lookup",
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    )
    for dataset_index in range(8):
        dataset_id = f"d{dataset_index}"
        tasks = []
        for scale in ("small", "medium", "large"):
            for task_type in task_types:
                tasks.append(
                    {
                        "item_id": f"{dataset_id}:{scale}:{task_type}",
                        "network": "keyword_cooccurrence",
                        "scale": scale,
                        "task_type": task_type,
                        "complexity": 1,
                        "question": "q",
                        "answer": {"value": 1},
                        "interpretation_contract": {
                            "required_limitation": "limited"
                        },
                            "evidence_ids": ["e1"],
                            "contexts": {
                                "flat_program": {"rows": [{"value": 1}]},
                                "operator_only": {"rows": [{"value": 1}]},
                            },
                    }
                )
        benchmark = benchmark_root / dataset_id / "benchmark.json"
        write_json(
            benchmark,
            {
                "dataset_id": dataset_id,
                "formal_v3_complexity_extension": {
                    "protocol_sha256": sha256_file(extension)
                },
                "tasks": tasks,
            },
        )
        records.append(
            {
                "dataset_id": dataset_id,
                "benchmark_sha256": sha256_file(benchmark),
                "source_panel": "primary" if dataset_index < 4 else "replication",
            }
        )
    write_json(
        benchmark_root / "construction_manifest.json",
        {
            "status": "constructed_not_executed",
            "protocol_sha256": sha256_file(extension),
            "datasets": 8,
            "tasks": 168,
            "records": records,
        },
    )
    tokenizer = tmp_path / "tokenizer.json"
    write_json(
        tokenizer,
        {
            "model": "deepseek-v4-pro",
            "passed": True,
            "verified_against_api_usage": True,
            "context_token_budget": 8192,
            "verification_probes": [],
        },
    )
    monkeypatch.setattr(
        "scripts.audit_formal_v3_mechanism_supplement_readiness.CommandTokenizer",
        _Tokenizer,
    )
    monkeypatch.setattr(
        "scripts.audit_formal_v3_mechanism_supplement_readiness.verify_api_usage_probe_artifact",
        lambda *_args, **_kwargs: ({}, {"passed": True, "probes": []}, []),
    )
    audits = {}
    for name, count in {"primary": 800, "replication": 120, "extension": 312}.items():
        path = tmp_path / f"{name}_terminal.json"
        write_json(path, {"status": "needs_retry", "expected_cells": count})
        audits[name] = path
    result = audit_mechanism_supplement_readiness(
        supplement_protocol_path=supplement,
        supplement_freeze_path=freeze,
        extension_protocol_path=extension,
        benchmark_root=benchmark_root,
        tokenizer_manifest_path=tokenizer,
        source_terminal_audits=audits,
        execution_root=tmp_path / "execution",
    )
    assert result["status"] == "blocked"
    assert len(result["cell_identity_manifest"]) == 336
    assert result["protocol_sha256"] == sha256_file(extension)
    assert result["supplement_protocol_sha256"] == sha256_file(supplement)
    assert "primary_source_not_terminal" in result["blocking_reasons"]
