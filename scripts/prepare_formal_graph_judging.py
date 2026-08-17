from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from citeweave.judge_protocol import canonical_json

ROOT = Path(__file__).resolve().parents[1]


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


def _text_candidates(path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "complete":
            continue
        content = str(row["response"]["choices"][0]["message"]["content"])
        try:
            candidate = canonical_json(_parse_content(content))
        except (json.JSONDecodeError, TypeError):
            candidate = content
        completed[row["item_id"]] = candidate
    return completed


def _vision_candidates(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["item_id"]: canonical_json(item) for item in payload["results"]
    }


def _context(workspace: Path, item_id: str) -> dict[str, Any]:
    manifest = json.loads(
        (
            workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    record = next(row for row in manifest["contexts"] if row["item_id"] == item_id)
    return json.loads((workspace / record["graph_path"]).read_text(encoding="utf-8"))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--run-id", default="formal_v2_nonthinking_20260806")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/formal_graph_judging_inputs"),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"Refusing existing output directory: {args.output_root}")

    graph_vs_no = []
    graph_vs_flat = []
    graph_vs_figure = []
    for dataset in args.dataset:
        workspace = ROOT / "experiments" / "formal_workspaces" / dataset
        benchmark = json.loads(
            (
                workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
            ).read_text(encoding="utf-8")
        )
        run_root = ROOT / "experiments" / "formal_runs" / dataset / args.run_id
        candidates = {
            condition: _text_candidates(run_root / condition / "items.jsonl")
            for condition in ("no_rag", "flat_structured", "graph_rag")
        }
        figure = _vision_candidates(
            ROOT / "experiments" / "vision_outputs" / f"{dataset}.json"
        )
        for item in benchmark:
            item_id = item["item_id"]
            evidence = {
                "item_id": item_id,
                "answerable": item["answerable"],
                "answer_contract": item["answer_contract"],
                "gold_answer": item["gold_answer"],
                "graph_evidence": _context(workspace, item_id),
            }
            base = {
                "sample_id": item_id,
                "question": item["question"],
                "canonical_evidence": evidence,
            }
            graph_vs_no.append(
                {
                    **base,
                    "candidates": {
                        "graph_rag": candidates["graph_rag"][item_id],
                        "no_rag": candidates["no_rag"][item_id],
                    },
                }
            )
            graph_vs_flat.append(
                {
                    **base,
                    "candidates": {
                        "graph_rag": candidates["graph_rag"][item_id],
                        "flat_structured": candidates["flat_structured"][item_id],
                    },
                }
            )
            if item.get("figure_eligible"):
                graph_vs_figure.append(
                    {
                        **base,
                        "candidates": {
                            "graph_rag": candidates["graph_rag"][item_id],
                            "figure_vlm": figure[item_id],
                        },
                    }
                )

    _write(args.output_root / "graph_vs_no.jsonl", graph_vs_no)
    _write(args.output_root / "graph_vs_flat.jsonl", graph_vs_flat)
    _write(args.output_root / "graph_vs_figure.jsonl", graph_vs_figure)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "datasets": args.dataset,
        "comparisons": {
            "graph_vs_no": len(graph_vs_no),
            "graph_vs_flat": len(graph_vs_flat),
            "graph_vs_figure": len(graph_vs_figure),
        },
        "figure_vlm_role": "cross_model_extension",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
