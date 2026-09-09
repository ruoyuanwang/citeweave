from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "climate_change_risks_1990_2021",
    "digital_twins_healthcare_2012_2024",
    "gene_editing_als_2004_2024",
    "global_microplastics_2004_2019",
    "plant_heat_drought_2008_2021",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "benchmarks",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "graph_discovery_v2" / "cross_topic_panel_20260820",
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--scale",
        action="append",
        choices=("small", "medium", "large"),
        help="Repeat to run multiple scales; defaults to large.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=[
            "no_reference",
            "flat_retrieval",
            "flat_tfidf",
            "flat_bm25",
            "flat_lsa",
            "flat_neural_dense",
            "flat_hybrid",
            "graph_retrieval",
            "flat_program",
            "graph_query_retrieval",
            "graph_hierarchical_retrieval",
            "graph_hierarchical_retrieval_v2",
            "graph_program",
            "operator_only",
        ],
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    scales = args.scale or ["large"]
    task_types = args.task_type or ["multi_hop_connector", "temporal_structural_shift"]
    runner = ROOT / "scripts" / "run_graph_discovery_experiment.py"
    for dataset in datasets:
        command = [
            sys.executable,
            str(runner),
            "--benchmark",
            str(args.benchmark_root / dataset / "benchmark.json"),
            "--output",
            str(args.output_root / dataset),
        ]
        for scale in scales:
            command.extend(["--scale", scale])
        for task_type in task_types:
            command.extend(["--task-type", task_type])
        if args.conditions:
            command.extend(["--conditions", *args.conditions])
        if args.execute:
            command.append("--execute")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
