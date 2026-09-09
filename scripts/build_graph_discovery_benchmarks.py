from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.graph_discovery import NETWORK_SPECS, build_discovery_benchmark
from citeweave.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT / "experiments" / "formal_workspaces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "benchmarks",
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--network", action="append", choices=sorted(NETWORK_SPECS))
    parser.add_argument(
        "--scale",
        action="append",
        choices=("small", "medium", "large"),
    )
    parser.add_argument(
        "--task-type",
        action="append",
        choices=(
            "multi_hop_connector",
            "bridge_counterfactual",
            "community_role_contrast",
            "hub_removal_resilience",
            "temporal_structural_shift",
            "productivity_centrality_divergence",
        ),
    )
    parser.add_argument("--record-budget", type=int, default=160)
    args = parser.parse_args()

    datasets = args.dataset or sorted(
        path.name for path in args.workspace_root.iterdir() if path.is_dir()
    )
    networks = tuple(args.network or NETWORK_SPECS)
    scales = tuple(args.scale or ("small", "medium", "large"))
    records = []
    for dataset in datasets:
        workspace = args.workspace_root / dataset
        output = args.output_root / dataset
        try:
            benchmark = build_discovery_benchmark(
                workspace,
                output,
                networks=networks,
                scales=scales,
                task_types=tuple(args.task_type) if args.task_type else None,
                record_budget=args.record_budget,
            )
            record = {
                "dataset_id": dataset,
                "status": "complete",
                "tasks": len(benchmark["tasks"]),
                "task_types": sorted({task["task_type"] for task in benchmark["tasks"]}),
                "scales": benchmark["scales"],
                "tasks_by_network_scale": {
                    f"{network}:{scale}": sum(
                        task["network"] == network and task["scale"] == scale
                        for task in benchmark["tasks"]
                    )
                    for network in networks
                    for scale in scales
                },
            }
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            record = {
                "dataset_id": dataset,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        print(dataset, record["status"], record.get("tasks", record.get("error")))
        write_json(
            args.output_root / "construction_summary.json",
            {"schema_version": 1, "record_budget": args.record_budget, "records": records},
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
