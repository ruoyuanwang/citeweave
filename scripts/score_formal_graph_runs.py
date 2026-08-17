from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from citeweave.experiment_benchmark import score_graph_predictions
from citeweave.io import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("no_rag", "flat_structured", "graph_rag")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_idempotent(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite a different formal mechanical score: {path}"
            )
        return
    atomic_write_bytes(path, payload)


def _parse_content(value: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("Completion content must be one JSON object")
    return parsed


def _checkpoint_predictions(path: Path) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            continue
        try:
            content = row["response"]["choices"][0]["message"]["content"]
            prediction = _parse_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            prediction = {
                "item_id": row.get("item_id"),
                "abstain": False,
                "answer": None,
                "parse_error": f"{type(exc).__name__} at line {line_number}",
            }
        prediction["item_id"] = row["item_id"]
        completed[row["item_id"]] = prediction
    return list(completed.values())


def _packet_hash(packet: dict[str, Any]) -> str:
    without_hash = {key: value for key, value in packet.items() if key != "packet_sha256"}
    encoded = json.dumps(
        without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_text(dataset: str, run_id: str, benchmark: list[dict[str, Any]]) -> None:
    for condition in CONDITIONS:
        run_dir = ROOT / "experiments" / "formal_runs" / dataset / run_id / condition
        checkpoint = run_dir / "items.jsonl"
        manifest = run_dir / "run_manifest.json"
        if not checkpoint.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Missing completed run inputs: {run_dir}")
        predictions = _checkpoint_predictions(checkpoint)
        score = score_graph_predictions(benchmark, predictions)
        score["formal_metadata"] = {
            "dataset_id": dataset,
            "condition": condition,
            "run_id": run_id,
            "run_manifest_sha256": _sha(manifest),
            "checkpoint_sha256": _sha(checkpoint),
            "primary_mechanical_metrics": [
                "exact_answer_accuracy",
                "structured_unsupported_answer_rate",
                "abstention_f1",
            ],
        }
        _write_json_idempotent(run_dir / "score.json", score)


def _score_vision(
    dataset: str,
    run_id: str,
    benchmark: list[dict[str, Any]],
    vision_root: Path,
) -> None:
    packet_path = ROOT / "experiments" / "vision_packets" / f"{dataset}.json"
    output_path = vision_root / f"{dataset}.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    output = json.loads(output_path.read_text(encoding="utf-8"))
    if packet.get("packet_sha256") != _packet_hash(packet):
        raise ValueError(f"{dataset}: invalid Figure/VLM packet hash")
    if output.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError(f"{dataset}: Figure/VLM output packet hash differs")
    if output.get("visible_only") is not True:
        raise ValueError(f"{dataset}: Figure/VLM output is not declared visible-only")
    expected_ids = [item["item_id"] for item in packet["items"]]
    predictions = output.get("results")
    if not isinstance(predictions, list):
        raise TypeError(f"{dataset}: Figure/VLM results must be a list")
    if [item.get("item_id") for item in predictions] != expected_ids:
        raise ValueError(f"{dataset}: Figure/VLM result IDs/order differ from packet")
    eligible = [item for item in benchmark if item.get("figure_eligible")]
    if [item["item_id"] for item in eligible] != expected_ids:
        raise ValueError(f"{dataset}: packet differs from eligible benchmark panel")
    score = score_graph_predictions(eligible, predictions)
    run_dir = ROOT / "experiments" / "formal_runs" / dataset / run_id / "figure_vlm"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "dataset_id": dataset,
        "condition": "figure_vlm",
        "role": "cross_model_extension",
        "generator": "independent_codex_visual_subagent",
        "main_inference": False,
        "visible_only": True,
        "packet_sha256": _sha(packet_path),
        "output_sha256": _sha(output_path),
        "items": len(eligible),
        "item_ids": expected_ids,
    }
    _write_json_idempotent(run_dir / "run_manifest.json", manifest)
    score["formal_metadata"] = {
        **manifest,
        "primary_mechanical_metrics": [
            "exact_answer_accuracy",
            "structured_unsupported_answer_rate",
        ],
    }
    _write_json_idempotent(run_dir / "score.json", score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--run-id", default="formal_v2_nonthinking_20260806")
    parser.add_argument("--vision-root", type=Path, default=Path("experiments/vision_outputs"))
    parser.add_argument("--include-vision", action="store_true")
    args = parser.parse_args()

    for dataset in args.dataset:
        benchmark_path = (
            ROOT
            / "experiments"
            / "formal_workspaces"
            / dataset
            / "evidence"
            / "formal_graph_experiment"
            / "benchmark.json"
        )
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        _score_text(dataset, args.run_id, benchmark)
        if args.include_vision:
            _score_vision(
                dataset,
                args.run_id,
                benchmark,
                args.vision_root.resolve(),
            )
        print(dataset)


if __name__ == "__main__":
    main()
