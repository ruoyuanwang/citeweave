from __future__ import annotations

import json
from pathlib import Path

import pytest

from citeweave.io import read_json, write_json
from citeweave.robust_graph_statistics import (
    RobustGraphStatisticsError,
    analyze_robust_graph_synthesis,
)
from citeweave.robust_graph_synthesis import (
    TASK_TYPES,
    answer_field_accuracy,
    build_benchmark_panel,
    build_topic_benchmark,
)
from scripts.audit_robust_graph_synthesis_readiness import audit


def _case(dataset_id: str, variant_id: str, family: str, *, unstable: bool = False) -> dict:
    return {
        "dataset_id": dataset_id,
        "variant": {
            "variant_id": variant_id,
            "family": family,
            "minimum_weight": 1,
            "drop_fraction": 0.05 if family == "edge_dropout" else 0.0,
            "drop_seed": 11,
            "community_seed": 42,
            "resolution": 1.0,
            "window_years": 2 if family == "temporal_window" else 3,
        },
        "graph": {
            "nodes": 2000,
            "edges": 10000,
            "components": 1,
            "largest_component_fraction": 0.9 if unstable else 1.0,
            "communities": 5,
        },
        "measurements": {
            "multi_hop_connector": {"status": "measured", "hops": 3, "path": ["A", "B", "C", "D"], "total_inverse_weight_distance": 1.2, "cross_community": True, "original_path_still_optimal": not unstable},
            "bridge_counterfactual": {"status": "measured", "component_increase": 0, "alternate_hops_after_deletion": 2, "redundant_connector": True, "cross_community": True, "highest_betweenness_selection_retested": False},
            "community_role_contrast": {"status": "measured", "disconnected_communities": [], "dominant": {"community": 1, "representative": "A", "external_share": 0.2, "importance": 5}, "outward": {"community": 2, "representative": "B", "external_share": 0.8, "importance": 3}},
            "hub_removal_resilience": {"status": "measured", "frozen_hub_rank": 2 if unstable else 1, "frozen_hub_still_argmax": not unstable, "replacement_hub": "B", "reselected_hub": "C" if unstable else "A", "components_after": 2, "largest_component_fraction_of_original_nodes": 0.8 if unstable else 0.99},
            "temporal_structural_shift": {"status": "measured", "emerging_label": "E2" if unstable else "E", "established_label": "S", "same_community": not unstable, "emerging_growth_ratio": 2.0, "established_weighted_degree": 4.0, "early_years": [2018, 2019], "recent_years": [2024, 2025]},
        },
        "comparison": {
            "multi_hop_connector": {"comparable": True, "distance_change": 0.2 if unstable else 0.0, "distance_equal": not unstable, "cross_community_equal": True, "original_path_still_optimal": not unstable},
            "bridge_counterfactual": {"comparable": True, "redundancy_equal": not unstable, "alternate_hops_equal": not unstable, "cross_community_equal": True},
            "community_role_contrast": {"comparable": True, "dominant_membership_jaccard": 0.5 if unstable else 1.0, "dominant_representative_equal": not unstable, "outward_membership_jaccard": 0.2 if unstable else 1.0, "outward_representative_equal": not unstable, "outward_external_share_change": 0.1 if unstable else 0.0},
            "hub_removal_resilience": {"comparable": True, "frozen_hub_still_argmax": not unstable, "replacement_hub_equal": not unstable, "largest_component_fraction_change": -0.19 if unstable else 0.0, "components_after_change": 1 if unstable else 0},
            "temporal_structural_shift": {"comparable": True, "emerging_label_equal": not unstable, "established_label_equal": True, "same_community_equal": not unstable, "growth_ratio_change": 0.3 if unstable else 0.0},
        },
    }


def _write_topic(path: Path, dataset_id: str) -> None:
    path.mkdir(parents=True)
    variants = [("baseline", "baseline")]
    variants += [(f"weight_{i}", "minimum_weight") for i in (2, 3, 5)]
    variants += [(f"seed_{i}", "community_seed") for i in (0, 1, 17)]
    variants += [(f"resolution_{i}", "resolution") for i in (0.5, 1.5, 2.0)]
    variants += [(f"drop_{fraction}_{seed}", "edge_dropout") for fraction in (0.01, 0.05) for seed in (11, 29, 47)]
    variants += [(f"window_{i}", "temporal_window") for i in (2, 4)]
    for index, (variant_id, family) in enumerate(variants):
        payload = _case(dataset_id, variant_id, family)
        if index in {3}:
            payload["measurements"]["multi_hop_connector"]["original_path_still_optimal"] = False
            payload["comparison"]["multi_hop_connector"].update(
                {"distance_change": 0.2, "distance_equal": False, "original_path_still_optimal": False}
            )
        if index in {3, 8, 12, 17}:
            payload["comparison"]["community_role_contrast"].update(
                {"dominant_membership_jaccard": 0.5, "dominant_representative_equal": False, "outward_membership_jaccard": 0.2, "outward_representative_equal": False, "outward_external_share_change": 0.1}
            )
        if index in {8, 12}:
            payload["graph"]["largest_component_fraction"] = 0.9
            payload["measurements"]["hub_removal_resilience"].update(
                {"frozen_hub_rank": 2, "frozen_hub_still_argmax": False, "reselected_hub": "C", "largest_component_fraction_of_original_nodes": 0.8}
            )
            payload["comparison"]["hub_removal_resilience"].update(
                {"frozen_hub_still_argmax": False, "replacement_hub_equal": False, "largest_component_fraction_change": -0.19, "components_after_change": 1}
            )
        if index in {17}:
            payload["measurements"]["temporal_structural_shift"].update(
                {"emerging_label": "E2", "same_community": False}
            )
            payload["comparison"]["temporal_structural_shift"].update(
                {"emerging_label_equal": False, "same_community_equal": False, "growth_ratio_change": 0.3}
            )
        (path / f"{variant_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_topic_benchmark_has_complex_synthesis_and_matched_mechanisms(tmp_path: Path) -> None:
    topic = tmp_path / "topic_a"
    _write_topic(topic, "topic_a")
    benchmark = build_topic_benchmark(topic)
    assert [task["task_type"] for task in benchmark["tasks"]] == list(TASK_TYPES)
    assert all(task["complexity"] >= 6 for task in benchmark["tasks"])
    for task in benchmark["tasks"]:
        contexts = task["contexts"]
        assert set(contexts) == {"flat_bm25", "graph_hierarchical_retrieval_v2", "flat_program", "graph_program", "operator_only"}
        assert contexts["flat_program"]["derived_rows"] == contexts["graph_program"]["operator_trace"]
        assert contexts["flat_program"]["rows"] == contexts["graph_program"]["evidence_rows"]
        assert len(contexts["flat_program"]["rows"]) <= 20
        assert len(task["evidence_ids"]) <= 8
    triage = benchmark["tasks"][-1]
    assert 0 < triage["answer"]["least_stable_rate"] < 1
    assert triage["answer"]["joint_robustness_label"] == "heterogeneous"


def test_panel_requires_eight_topics_and_reports_interior_difficulty(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(8):
        _write_topic(source / f"topic_{index}", f"topic_{index}")
    output = tmp_path / "output"
    manifest = build_benchmark_panel(source, output)
    assert manifest["topics"] == 8
    assert manifest["tasks"] == 40
    assert manifest["planned_calls"] == 200
    assert manifest["difficulty_diagnostic"]["interior_rate_fields"] > 0
    assert manifest["difficulty_diagnostic"]["distinct_rates"] > 2
    assert (output / "construction_manifest.json").is_file()


def test_panel_fails_closed_for_missing_topic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_topic(source / "topic_0", "topic_0")
    with pytest.raises(ValueError, match="Expected eight"):
        build_benchmark_panel(source, tmp_path / "output")


def test_answer_field_accuracy_is_continuous_and_strict() -> None:
    gold = {"rate": 0.75, "label": "sensitive", "count": 3}
    assert answer_field_accuracy(gold, gold) == (3, 3)
    assert answer_field_accuracy(gold, {"rate": 0.75, "label": "stable", "count": 3}) == (2, 3)
    assert answer_field_accuracy(gold, None) == (0, 3)


def test_readiness_audit_checks_information_parity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(8):
        _write_topic(source / f"topic_{index}", f"topic_{index}")
    output = tmp_path / "output"
    build_benchmark_panel(source, output)
    manifest_path = output / "construction_manifest.json"
    protocol = tmp_path / "protocol.yml"
    protocol.write_text(
        "\n".join(
            [
                "status: frozen_after_deterministic_task_construction_before_any_model_outcomes",
                "source_bindings:",
                "  construction_manifest:",
                f"    sha256: {__import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()}",
                "panel:",
                "  context_record_budget: 20",
                "  conditions:",
                "    - flat_bm25",
                "    - graph_hierarchical_retrieval_v2",
                "    - flat_program",
                "    - graph_program",
                "    - operator_only",
            ]
        ),
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"sha256": __import__("hashlib").sha256(protocol.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    result = audit(protocol, freeze, output)
    assert result["status"] == "ready_for_development_execution"
    assert all(result["parity_checks"].values())

    benchmark_path = output / "topic_0" / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["tasks"][0]["contexts"]["flat_program"]["rows"] = []
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["benchmark_sha256"] = __import__("hashlib").sha256(
        benchmark_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol_text = protocol.read_text(encoding="utf-8")
    old_hash = result["construction_manifest_sha256"]
    new_hash = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    protocol.write_text(protocol_text.replace(old_hash, new_hash), encoding="utf-8")
    freeze.write_text(
        json.dumps({"sha256": __import__("hashlib").sha256(protocol.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    failed = audit(protocol, freeze, output)
    assert failed["status"] == "blocked"
    assert any(reason.startswith("program_raw_evidence_parity_failure") for reason in failed["blocking_reasons"])


def test_development_statistics_require_complete_cells_and_use_field_accuracy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for index in range(8):
        _write_topic(source / f"topic_{index}", f"topic_{index}")
    panel = tmp_path / "panel"
    build_benchmark_panel(source, panel)
    construction_path = panel / "construction_manifest.json"
    construction = read_json(construction_path)
    run_root = tmp_path / "runs"
    conditions = [
        "flat_bm25",
        "graph_hierarchical_retrieval_v2",
        "flat_program",
        "graph_program",
        "operator_only",
    ]
    for panel_record in construction["records"]:
        benchmark_path = Path(panel_record["benchmark"])
        benchmark = read_json(benchmark_path)
        records = []
        for task in benchmark["tasks"]:
            for condition in conditions:
                answer = dict(task["answer"])
                evidence_ids = list(task["evidence_ids"])
                if condition == "flat_bm25":
                    answer = {}
                    evidence_ids = []
                elif condition == "graph_hierarchical_retrieval_v2":
                    answer[next(iter(answer))] = "deliberately_wrong"
                elif condition == "operator_only":
                    evidence_ids = []
                records.append(
                    {
                        "item_id": task["item_id"],
                        "condition": condition,
                        "status": "complete",
                        "elapsed_seconds": 1.0,
                        "usage": {"total_tokens": 100},
                        "response": {
                            "item_id": task["item_id"],
                            "abstain": False,
                            "answer": answer,
                            "phenomenon": "test",
                            "evidence_ids": evidence_ids,
                            "alternative_explanation": "test",
                            "limitation": "registered grid only",
                        },
                    }
                )
        output = run_root / panel_record["dataset_id"] / "results.json"
        write_json(
            output,
            {
                "manifest": {
                    "benchmark_sha256": panel_record["benchmark_sha256"],
                    "conditions": conditions,
                },
                "records": records,
            },
        )
    result = analyze_robust_graph_synthesis(
        construction_path, run_root, bootstrap_samples=200
    )
    assert result["completed_cells"] == 200
    assert result["condition_summary"]["graph_program"]["field_accuracy"] == 1
    assert result["condition_summary"]["flat_bm25"]["field_accuracy"] == 0
    assert result["contrasts"]["D3_representation_only"]["equivalent"] is True
    assert result["promotion_gate"]["passed"] is True

    first = construction["records"][0]["dataset_id"]
    result_path = run_root / first / "results.json"
    incomplete = read_json(result_path)
    incomplete["records"].pop()
    write_json(result_path, incomplete)
    with pytest.raises(RobustGraphStatisticsError, match="incomplete"):
        analyze_robust_graph_synthesis(
            construction_path, run_root, bootstrap_samples=200
        )
