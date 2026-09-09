from __future__ import annotations

from pathlib import Path

import pytest

from citeweave.formal_request import task_payload_sha256
from citeweave.io import read_json, sha256_file, write_json
from citeweave.oversight_heldout_packets import (
    OversightHeldoutPacketError,
    freeze_heldout_case_selection,
    materialize_heldout_factorial_packets,
)


def _task(dataset: str, task_type: str, scale: str) -> dict:
    item_id = f"{dataset}:keyword:{scale}:{task_type}"
    evidence_id = f"EDGE-{scale}-{task_type}"
    return {
        "item_id": item_id,
        "dataset_id": dataset,
        "network": "keyword",
        "scale": scale,
        "task_type": task_type,
        "complexity": 3,
        "question": f"Audit {task_type} at {scale} scale.",
        "answer": {"value": 1},
        "answer_alternatives": [],
        "evidence_ids": [evidence_id],
        "operator_trace": [{"operator": task_type, "value": 1}],
        "interpretation_contract": {
            "allowed": "descriptive graph statement",
            "forbidden": ["causality"],
            "required_limitation": "Depends on the corpus and graph construction.",
        },
        "contexts": {
            "graph_program": {
                "nodes": [],
                "edges": [{"evidence_id": evidence_id, "weight": 1}],
                "operator_trace": [{"operator": task_type, "value": 1}],
                "interpretation_contract": {},
            }
        },
        "context_hashes": {"graph_program": "context"},
    }


def _panel(root: Path, name: str, dataset_offset: int) -> tuple[Path, Path, Path]:
    benchmark_root = root / f"{name}_benchmarks"
    results_root = root / f"{name}_results"
    construction_records = []
    task_types = [
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    ]
    for relative_index in range(4):
        dataset = f"domain_{dataset_offset + relative_index}"
        tasks = [
            _task(dataset, task_type, scale)
            for task_type in task_types
            for scale in ("small", "medium", "large")
        ]
        benchmark_path = benchmark_root / dataset / "benchmark.json"
        write_json(
            benchmark_path,
            {"dataset_id": dataset, "tasks": tasks},
        )
        construction_records.append(
            {
                "dataset_id": dataset,
                "tasks": 15,
                "benchmark_sha256": sha256_file(benchmark_path),
            }
        )
        result_rows = []
        for task in tasks:
            response = {
                "answer": {"value": 1},
                "evidence_ids": list(task["evidence_ids"]),
                "limitation": "Depends on the corpus.",
                "abstain": False,
            }
            result_rows.append(
                {
                    "item_id": task["item_id"],
                    "condition": "graph_program",
                    "status": "complete",
                    "task_payload_sha256": task_payload_sha256(task),
                    "response": response,
                    "score": {
                        "answer_exact": True,
                        "evidence_precision": 1.0,
                        "evidence_recall": 1.0,
                        "evidence_f1": 1.0,
                        "has_required_limitation": True,
                        "abstain": False,
                    },
                }
            )
        write_json(
            results_root / dataset / "results.json",
            {
                "manifest": {"benchmark_sha256": sha256_file(benchmark_path)},
                "records": result_rows,
            },
        )
    construction = root / f"{name}_construction.json"
    write_json(
        construction,
        {
            "status": "constructed_not_executed",
            "records": construction_records,
        },
    )
    return construction, benchmark_root, results_root


def _terminal_audits(root: Path) -> list[Path]:
    paths = []
    for name in ("primary", "replication", "extension"):
        path = root / f"{name}_audit.json"
        write_json(
            path,
            {
                "status": "terminal",
                "expected_cells": 1,
                "complete_cells": 1,
                "terminal_parse_failure_cells": 0,
                "retry_cells": [],
                "integrity_reasons": [],
            },
        )
        paths.append(path)
    return paths


def test_selection_is_result_blind_and_materialization_waits_for_terminal_panels(
    tmp_path: Path,
) -> None:
    primary, primary_benchmarks, primary_results = _panel(tmp_path, "primary", 0)
    replication, replication_benchmarks, replication_results = _panel(
        tmp_path, "replication", 4
    )
    selection_path = tmp_path / "selection.json"
    selection = freeze_heldout_case_selection(
        primary_construction_manifest_path=primary,
        replication_construction_manifest_path=replication,
        primary_benchmark_root=primary_benchmarks,
        replication_benchmark_root=replication_benchmarks,
        primary_results_root=primary_results,
        replication_results_root=replication_results,
        output_path=selection_path,
    )
    assert selection["cases"] == 96
    assert selection["selection_reads_model_results"] is False
    assert all("score" not in row and "response" not in row for row in selection["records"])
    audits = _terminal_audits(tmp_path)
    output = tmp_path / "packets"
    manifest = materialize_heldout_factorial_packets(
        selection_path=selection_path,
        terminal_audit_paths=audits,
        output_root=output,
    )
    assert manifest["cases"] == 96
    assert manifest["same_evidence_across_packet_arms"] is True
    public = read_json(output / manifest["records"][0]["adversarial_public_path"])
    assert "gold_verdict" not in public
    assert "deterministic_score" not in public
    assert "presentation" not in public
    assert public["generation_condition_hidden"] is True


def test_materialization_rejects_nonterminal_audit(tmp_path: Path) -> None:
    primary, primary_benchmarks, primary_results = _panel(tmp_path, "primary", 0)
    replication, replication_benchmarks, replication_results = _panel(
        tmp_path, "replication", 4
    )
    selection_path = tmp_path / "selection.json"
    freeze_heldout_case_selection(
        primary_construction_manifest_path=primary,
        replication_construction_manifest_path=replication,
        primary_benchmark_root=primary_benchmarks,
        replication_benchmark_root=replication_benchmarks,
        primary_results_root=primary_results,
        replication_results_root=replication_results,
        output_path=selection_path,
    )
    audits = _terminal_audits(tmp_path)
    audit = read_json(audits[2])
    audit["status"] = "needs_retry"
    audit["retry_cells"] = [{"item_id": "missing"}]
    write_json(audits[2], audit)
    with pytest.raises(OversightHeldoutPacketError, match="not cleanly terminal"):
        materialize_heldout_factorial_packets(
            selection_path=selection_path,
            terminal_audit_paths=audits,
            output_root=tmp_path / "packets",
        )
