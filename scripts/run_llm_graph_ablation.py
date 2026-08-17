from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from citeweave.experiment_benchmark import (
    load_json_records,
    save_benchmark_result,
    score_graph_predictions,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "graph-qa-v3"


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise TypeError("Model output must be one JSON object")
    return value


def _graph_context(item: dict[str, Any], facts: list[dict[str, Any]]) -> str:
    candidates = [fact for fact in facts if fact["network"] == item["network"]]
    context: dict[str, Any] = {"retrieved_graph_facts": candidates}
    if not item["answerable"]:
        context["query_resolution"] = {
            "target_label": item["evidence_operation"]["target_label"],
            "node_found": False,
            "network_node": item["evidence_operation"]["network_node"],
        }
    return json.dumps(context, ensure_ascii=False)


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "abstain": {"type": "boolean"},
            "answer": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "evidence_nodes": {"type": "array", "items": {"type": "string"}},
            "evidence_edges": {"type": "array", "items": {"type": "string"}},
            "statement": {"type": "string"},
        },
        "required": [
            "item_id",
            "abstain",
            "answer",
            "evidence_nodes",
            "evidence_edges",
            "statement",
        ],
        "additionalProperties": False,
    }


def _messages(
    item: dict[str, Any],
    *,
    condition: str,
    facts: list[dict[str, Any]],
    figure_path: Path | None,
) -> list[dict[str, Any]]:
    instruction = (
        "Answer one bibliometric graph question. Return JSON only with keys: "
        "item_id (string), abstain (boolean), answer (object or null), "
        "evidence_nodes (array of strings), evidence_edges (array of strings), "
        "and statement (string). Never invent evidence IDs. If the supplied "
        "context cannot establish the answer, set abstain=true, answer=null, "
        "and use empty evidence arrays. When graph context establishes an answer, "
        "copy the matching fact value and evidence IDs character-for-character. "
        "Use only IDs present in the matching fact's evidence_nodes and "
        "evidence_edges arrays. Never shorten 'fact:G001' to 'G001', never add "
        "a bare DOI, and never add an uncited identifier."
    )
    answer_schemas = {
        "network_size": '{"nodes": integer, "edges": integer}',
        "highest_weighted_degree": (
            '{"id": string, "weighted_degree": number}; extra copied fields are allowed'
        ),
        "cluster_count": '{"clusters": integer}; extra copied fields are allowed',
        "strongest_edge": (
            '{"source": string, "target": string, "weight": number}; '
            "extra copied fields are allowed"
        ),
        "unanswerable_false_premise": "null with abstain=true",
    }
    prompt = (
        f"ITEM_ID: {item['item_id']}\n"
        f"TASK_TYPE: {item['task_type']}\n"
        f"ANSWER_OBJECT_SCHEMA: {answer_schemas[item['task_type']]}\n"
        f"QUESTION: {item['question']}"
    )
    if condition == "graph_rag":
        prompt += "\nGRAPH_CONTEXT:\n" + _graph_context(item, facts)
    if condition != "figure_vlm":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]

    if figure_path is None or not figure_path.is_file():
        raise FileNotFoundError(f"Missing network figure: {figure_path}")
    encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-context, figure-VLM, or Graph-RAG LLM graph QA."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["no_graph", "figure_vlm", "graph_rag"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object"],
        default="json_schema",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing {args.api_key_env}; no model requests were attempted."
        )
    workspace = REPOSITORY_ROOT / "experiments" / "workspaces" / args.dataset
    run_dir = REPOSITORY_ROOT / "experiments" / "runs" / args.dataset
    output = run_dir / f"llm_{args.condition}_{args.model.replace('/', '_')}.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite completed output: {output}")

    items = load_json_records(workspace / "evidence" / "graph_qa_benchmark.json")
    if args.limit:
        items = items[: args.limit]
    facts = load_json_records(workspace / "evidence" / "graph_facts.json")
    predictions_by_id: dict[str, dict[str, Any]] = {}
    raw_records_by_id: dict[str, dict[str, Any]] = {}
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    started = time.perf_counter()

    with httpx.Client(timeout=120) as client:
        def run_item(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            figure_path = (
                workspace / "figures" / f"network_{item['network']}.png"
                if args.condition == "figure_vlm"
                else None
            )
            messages = _messages(
                item,
                condition=args.condition,
                facts=facts,
                figure_path=figure_path,
            )
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "response_format": (
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bibliometric_graph_answer",
                            "strict": True,
                            "schema": _response_schema(),
                        },
                    }
                    if args.response_format == "json_schema"
                    else {"type": "json_object"}
                ),
            }
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            response_body = response.json()
            content = response_body["choices"][0]["message"]["content"]
            try:
                prediction = _parse_json_object(content)
            except (json.JSONDecodeError, TypeError):
                prediction = {
                    "item_id": item["item_id"],
                    "abstain": False,
                    "answer": None,
                    "evidence_nodes": [],
                    "evidence_edges": [],
                    "statement": content,
                }
            prediction["item_id"] = item["item_id"]
            raw_record = {
                "item_id": item["item_id"],
                "condition": args.condition,
                "request": {
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                    "response_format": args.response_format,
                    "messages": messages,
                },
                "response": response_body,
                "parsed_prediction": prediction,
            }
            return prediction, raw_record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_item, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                prediction, raw_record = future.result()
                predictions_by_id[item["item_id"]] = prediction
                raw_records_by_id[item["item_id"]] = raw_record

    predictions = [predictions_by_id[item["item_id"]] for item in items]
    raw_records = [raw_records_by_id[item["item_id"]] for item in items]
    elapsed_seconds = time.perf_counter() - started
    score = score_graph_predictions(items, predictions)
    save_benchmark_result(
        score,
        output,
        condition=args.condition,
        metadata={
            "dataset_id": args.dataset,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "workers": args.workers,
            "response_format": args.response_format,
            "items": len(items),
            "elapsed_seconds": elapsed_seconds,
            "raw_records": raw_records,
        },
    )
    print(
        f"{args.dataset} {args.condition}: "
        f"accuracy={score['metrics']['exact_answer_accuracy']:.3f}, "
        f"UCR={score['metrics']['unsupported_claim_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
