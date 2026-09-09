"""Freeze a 96-cell equal-temporal-information control, without any provider calls.

Neural contexts are attached later from the already-running frozen graph index.
This preparatory artifact cannot authorize execution by itself.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from citeweave.formal_request import canonical_sha256
from citeweave.graph_robustness import load_verified_trends
from citeweave.io import read_json, sha256_file, write_json
from citeweave.temporal_evidence_control import build_temporal_annex, control_item_id
from citeweave.token_budget import CommandTokenizer, canonical_context_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/graph_discovery_v2"
OUTPUT = BASE / "temporal_shared_evidence_control_v1"
CONDITIONS = (
    "flat_hybrid",
    "flat_neural_dense",
    "graph_hierarchical_retrieval_v2",
    "graph_program",
)


def main():
    freeze = OUTPUT / "protocol_freeze.json"
    if freeze.exists():
        raise SystemExit("Control already frozen; inspect it rather than overwrite")
    tokenizer_path = BASE / "formal_v3_execution_prerequisites/deepseek_v4_tokenizer_manifest.json"
    tokenizer_payload = read_json(tokenizer_path)
    tokenizer = CommandTokenizer(tokenizer_payload, manifest_dir=tokenizer_path.parent)
    if not tokenizer.verify(tokenizer_payload["verification_probes"])["passed"]:
        raise ValueError("Tokenizer probes failed")
    construction_path = (
        BASE / "formal_v3_complexity_extension_benchmarks/construction_manifest.json"
    )
    construction = read_json(construction_path)
    protocol = {
        "schema_version": 1,
        "status": "prospective_shared_information_control",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "role": "Supplemental diagnostic; original formal tasks/results/statistical decisions remain intact.",
        "motivation": "Twenty-four temporal tasks admit different answers under identical static-context requests. Original gaps confound information access and task-definition visibility with computation.",
        "topics": 8,
        "source_tasks": 24,
        "conditions": list(CONDITIONS),
        "planned_provider_calls": 96,
        "model": "deepseek-v4-pro",
        "temperature": 0,
        "thinking": "disabled",
        "max_tokens": 2200,
        "total_context_token_budget": 8192,
        "maximum_shared_annex_tokens": 3072,
        "graph_retrieval_record_budget": 160,
        "shared_raw_time_records_maximum": 15,
        "budget_rule": "Annex is fixed metadata, retained in full in every method and counted inside the same 8192-token context budget. Existing graph records use the same deterministic truncator; there is no extra token allowance.",
        "task_rule": "Preserve questions, answer-field names, gold answers, and graph retrieval outputs. Give all methods the same raw candidate time table and explicit selection/smoothing/tie rules. Use separate item IDs and result roots.",
        "score_rule": "Primary score is the original exact answer with tolerance 1e-4; parse failure/abstention count as failure. Temporal-table citations are valid additional evidence, so cross-panel evidence-F1 is not comparable.",
        "source_results_use": "Compare same-task original and shared-information accuracies descriptively; do not call the change a pure time-data effect because operational definitions are also made explicit.",
        "analysis": {
            "unit": "eight topics, not 24 independent tasks",
            "primary_display": "per-topic/per-scale paired accuracy and original-minus-control method gaps",
            "uncertainty": "topic-cluster bootstrap 10000 resamples, seed 20260827; descriptive intervals, no new confirmatory significance claims",
            "main_results": "Report original frozen analyses unchanged; additionally report temporal and non-temporal strata separately, without deleting unfavorable cells.",
        },
        "execution_gates": [
            "Original corrected execution qualification and explicit promotion remain necessary.",
            "All four contexts per task, including new neural sidecars, are hash-bound before the first control response.",
            "All 96 token-budget and annex-equality checks pass before execution.",
            "No modification to original graph indexes, writer inputs, or human-review packets.",
            "Maximum three identical parse attempts; no resampling completed responses for better quality.",
        ],
        "automatic_api_resume": False,
    }
    records, bindings = [], {}
    for row in construction["records"]:
        topic, panel = row["dataset_id"], row["source_panel"]
        benchmark_path = construction_path.parent / topic / "benchmark.json"
        if sha256_file(benchmark_path) != row["benchmark_sha256"]:
            raise ValueError("Source benchmark drift")
        workspace = ROOT / "experiments/formal_v3_identity_corrected_workspaces" / panel / topic
        original = (
            ROOT
            / "experiments"
            / ("formal_v3_workspaces" if panel == "primary" else "formal_v3_replication_workspaces")
            / topic
        )
        trends_path = original / "analyses/keyword_trends.parquet"
        annex = build_temporal_annex(topic, load_verified_trends(workspace, trends_path))
        tokens = tokenizer.count(canonical_context_json(annex))
        if tokens > protocol["maximum_shared_annex_tokens"]:
            raise ValueError(f"Annex exceeds prospective maximum: {topic}: {tokens}")
        annex_path = OUTPUT / topic / "shared_temporal_annex.json"
        if annex_path.exists():
            raise ValueError("Partial artifacts exist; inspect before attempting reconstruction")
        write_json(annex_path, annex)
        tasks = [
            t
            for t in read_json(benchmark_path)["tasks"]
            if t["task_type"] == "temporal_structural_shift"
        ]
        if len(tasks) != 3:
            raise ValueError("Three registered scales required")
        records.append(
            {
                "dataset_id": topic,
                "source_benchmark": str(benchmark_path),
                "source_benchmark_sha256": sha256_file(benchmark_path),
                "annex_path": str(annex_path),
                "annex_sha256": sha256_file(annex_path),
                "annex_tokens": tokens,
                "tasks": [
                    {
                        "item_id": control_item_id(t["item_id"]),
                        "source_item_id": t["item_id"],
                        "scale": t["scale"],
                        "source_task_sha256": canonical_sha256(t),
                    }
                    for t in tasks
                ],
            }
        )
        for path in (
            benchmark_path,
            trends_path,
            annex_path,
            workspace / "canonical/keywords.parquet",
            workspace / "canonical/works.parquet",
            workspace / "canonical/visualization/keyword_occurrences.parquet",
        ):
            bindings[str(path)] = sha256_file(path)
        print(json.dumps({"topic": topic, "annex_tokens": tokens}), flush=True)
    if len(records) != 8:
        raise ValueError("Eight topics required")
    plan = {
        "schema_version": 1,
        "status": "awaiting_neural_contexts_and_full_message_preflight",
        "records": records,
        "calls": 96,
        "automatic_api_resume": False,
    }
    protocol_path = OUTPUT / "protocol.json"
    plan_path = OUTPUT / "source_plan.json"
    write_json(protocol_path, protocol)
    write_json(plan_path, plan)
    for path in (
        protocol_path,
        plan_path,
        construction_path,
        tokenizer_path,
        Path(__file__).resolve(),
        ROOT / "src/citeweave/temporal_evidence_control.py",
        ROOT / "src/citeweave/graph_robustness.py",
    ):
        bindings[str(path)] = sha256_file(path)
    write_json(
        freeze,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": sha256_file(protocol_path),
            "bindings": bindings,
            "formal_and_control_provider_outcomes": 0,
            "note": "Only the scientific design and annex inputs are frozen; execution code and all message identities still require a separate pre-response freeze.",
        },
    )


if __name__ == "__main__":
    main()
