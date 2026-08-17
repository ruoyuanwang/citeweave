from __future__ import annotations

import argparse
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

from citeweave.formal_graph_experiment import (
    JsonlCheckpoint,
    ProviderProfile,
    comparison_design,
    formal_run_directory,
    provider_for_condition,
    select_condition_items,
    validate_provider_profile,
)
from citeweave.io import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
FORMAL_WORKSPACES = ROOT / "experiments" / "formal_workspaces"
CONDITION_PROMPT_VERSION = "formal-graph-qa-v2-nonthinking"


def _profile(path: Path) -> ProviderProfile:
    if path.suffix.casefold() in {".yml", ".yaml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    profile = ProviderProfile.model_validate(payload)
    if not profile.capability_snapshot.is_absolute():
        profile = profile.model_copy(
            update={"capability_snapshot": (path.parent / profile.capability_snapshot).resolve()}
        )
    return profile


def _base_prompt(item: dict[str, Any]) -> str:
    return (
        f"ITEM_ID: {item['item_id']}\n"
        f"TASK_TYPE: {item['task_type']}\n"
        f"ANSWER_CONTRACT: {json.dumps(item['answer_contract'], ensure_ascii=False)}\n"
        f"QUESTION: {item['question']}"
    )


def _messages(
    workspace: Path,
    item: dict[str, Any],
    context_record: dict[str, Any],
    condition: str,
) -> list[dict[str, Any]]:
    instruction = (
        "Answer the bibliometric network question using only the supplied material. "
        "Return one JSON object with item_id, abstain, answer, and explanation. "
        "Use visible labels and numeric values; never invent canonical identifiers. "
        "If the material does not establish the answer, abstain. Keep the explanation "
        "to two to four sentences."
    )
    prompt = _base_prompt(item)
    if condition == "flat_structured":
        context = read_json(workspace / context_record["flat_path"])
        prompt += "\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False)
    elif condition == "graph_rag":
        context = read_json(workspace / context_record["graph_path"])
        prompt += "\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False)
    if condition != "figure_vlm":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
    figure_path = Path(item["figure_path"])
    if not figure_path.is_file():
        raise FileNotFoundError(f"Figure input is missing: {figure_path}")
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


def _request(profile: ProviderProfile, messages: list[dict[str, Any]]) -> dict[str, Any]:
    response_format: dict[str, Any]
    if profile.response_format == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "formal_bibliometric_graph_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "abstain": {"type": "boolean"},
                        "answer": {
                            "anyOf": [{"type": "object"}, {"type": "null"}]
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["item_id", "abstain", "answer", "explanation"],
                    "additionalProperties": False,
                },
            },
        }
    else:
        response_format = {"type": "json_object"}
    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
        "thinking": {"type": "disabled"},
        "response_format": response_format,
        "stream": False,
    }
    if profile.supports_seed:
        payload["seed"] = 42
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["no_rag", "flat_structured", "graph_rag", "figure_vlm"],
    )
    parser.add_argument("--text-profile", type=Path, required=True)
    parser.add_argument("--vision-profile", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    workspace = (FORMAL_WORKSPACES / args.dataset).resolve()
    evidence = workspace / "evidence" / "formal_graph_experiment"
    manifest_path = evidence / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            "Formal graph grounding is missing; run the formal visualize/grounding stage first."
        )
    text_profile = _profile(args.text_profile)
    vision_profile = _profile(args.vision_profile) if args.vision_profile else None
    try:
        profile = provider_for_condition(
            text_profile,
            vision_profile,
            args.condition,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validation = validate_provider_profile(profile, condition=args.condition)
    if not validation.passed:
        raise SystemExit(
            "Provider capability validation failed: " + "; ".join(validation.reasons)
        )
    design = comparison_design(text_profile, vision_profile)
    benchmark = read_json(evidence / "benchmark.json")
    manifest = read_json(manifest_path)
    contexts = {item["item_id"]: item for item in manifest["contexts"]}
    items = select_condition_items(
        benchmark,
        condition=args.condition,
        design=design,
    )
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No eligible questions for condition {args.condition}")
    run_id = args.run_id or uuid.uuid4().hex
    run_dir = formal_run_directory(
        ROOT,
        dataset_id=args.dataset,
        run_id=run_id,
        condition=args.condition,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": run_id,
        "dataset_id": args.dataset,
        "condition": args.condition,
        "prompt_version": CONDITION_PROMPT_VERSION,
        "profile": profile.model_dump(mode="json"),
        "capability_validation": validation.model_dump(mode="json"),
        "comparison_design": design,
        "grounding_manifest_sha256": sha256_file(manifest_path),
        "benchmark_sha256": sha256_file(evidence / "benchmark.json"),
        "items": len(items),
        "item_ids": [item["item_id"] for item in items],
    }
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        if read_json(run_manifest_path) != run_manifest:
            raise SystemExit(
                f"Run manifest exists with a different contract: {run_manifest_path}"
            )
    else:
        write_json(run_manifest_path, run_manifest)
    if not args.execute:
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str))
        return

    api_key = os.getenv(profile.api_key_env)
    if not api_key:
        raise SystemExit(f"{profile.api_key_env} is not visible to this process.")
    checkpoint = JsonlCheckpoint(run_dir / "items.jsonl")
    endpoint = profile.base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        for item in items:
            if checkpoint.completed(args.condition, item["item_id"]):
                continue
            messages = _messages(
                workspace,
                item,
                contexts[item["item_id"]],
                args.condition,
            )
            request = _request(profile, messages)
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                )
                response.raise_for_status()
                body = response.json()
                returned_model = body.get("model")
                if returned_model and returned_model != profile.model:
                    raise RuntimeError(
                        f"Provider returned model {returned_model!r}, expected {profile.model!r}"
                    )
                checkpoint.append(
                    run_id=run_id,
                    condition=args.condition,
                    item_id=item["item_id"],
                    request=request,
                    response=body,
                    status="complete",
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                checkpoint.append(
                    run_id=run_id,
                    condition=args.condition,
                    item_id=item["item_id"],
                    request=request,
                    response=None,
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise


if __name__ == "__main__":
    main()
