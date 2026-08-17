from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path("experiments/formal_workspaces"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/vision_packets"))
    args = parser.parse_args()

    workspace = (args.workspace_root / args.dataset).resolve()
    benchmark_path = (
        workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
    )
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    eligible = [item for item in benchmark if item.get("figure_eligible")]
    if not eligible:
        raise SystemExit(f"{args.dataset}: no figure-eligible items")
    packet = {
        "schema_version": 1,
        "packet_role": "cross_model_figure_vlm_generation",
        "dataset_id": args.dataset,
        "generator": "independent_codex_visual_subagent",
        "main_inference": False,
        "visible_only": True,
        "prohibited_inputs": [
            "gold_answer",
            "graph JSON",
            "flat structured context",
            "human reference output",
            "other condition output",
        ],
        "instructions": (
            "Inspect each referenced figure visually and answer only its paired question. "
            "Use no external tools or data other than opening the listed images. Return one "
            "result per item with item_id, abstain, answer, and explanation. If the visible "
            "figure does not establish the requested values, abstain."
        ),
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "items": [
            {
                "item_id": item["item_id"],
                "task_type": item["task_type"],
                "question": item["question"],
                "answer_contract": item["answer_contract"],
                "figure_path": item["figure_path"],
            }
            for item in eligible
        ],
        "output_schema": {
            "schema_version": 1,
            "packet_sha256": "<copy packet_sha256>",
            "dataset_id": args.dataset,
            "generator_role": "codex_visual_subagent",
            "visible_only": True,
            "results": [
                {
                    "item_id": "<exact item_id>",
                    "abstain": False,
                    "answer": {"nodes": 0, "links": 0},
                    "explanation": "<brief visual basis>",
                }
            ],
        },
    }
    canonical = json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    packet["packet_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = args.output_root / f"{args.dataset}.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite packet: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
