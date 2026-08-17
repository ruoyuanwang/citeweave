from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
SYSTEM_PROMPT = """You revise rejected OpenAlex bibliometric Boolean queries.
Return strict JSON only and revise only supplied rejected ids. OpenAlex supports uppercase
AND/OR/NOT, parentheses, and exact phrases in double quotes; search covers title, abstract, and
full text. Construct a reproducible expression that is inclusive of standard synonymous wording
but still requires every essential topic dimension. Avoid application-specific restrictions.
Do not infer or copy any human reference paper's search string, database, corpus size, or results.
Every returned expression must be below 1000 characters and contain no year restriction."""


class BooleanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    search_expression: str = Field(min_length=3, max_length=1000)
    rationale: str


class BooleanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[BooleanProposal]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(stripped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "experiments" / "query_generation",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite registry: {args.output}")
    registry_bytes = args.registry.read_bytes()
    judgment_bytes = args.judgment.read_bytes()
    registry = yaml.safe_load(registry_bytes)
    judgment = json.loads(judgment_bytes)
    decisions = {item["id"]: item for item in judgment["decisions"]}
    datasets = {item["id"]: item for item in registry["datasets"]}
    if set(decisions) != set(datasets):
        raise SystemExit("Judgment IDs differ from registry IDs.")
    rejected = [
        {
            "id": dataset_id,
            "topic": datasets[dataset_id]["topic"],
            "current_keywords": datasets[dataset_id]["keywords"],
            "current_search_expression": datasets[dataset_id].get("search_expression"),
            "judge_reason": decision["reason"],
        }
        for dataset_id, decision in decisions.items()
        if decision["decision"] == "reject"
    ]
    if not rejected:
        raise SystemExit("No rejected queries require revision.")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not visible.")
    request_body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Revise each rejected OpenAlex query.",
                        "rejected_queries": rejected,
                        "output_schema": {
                            "queries": [
                                {
                                    "id": "exact rejected id",
                                    "search_expression": "OpenAlex Boolean expression",
                                    "rationale": "one sentence",
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 3000,
        "thinking": {"type": "disabled"},
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    started = datetime.now(UTC)
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_body,
        )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    proposals = BooleanResponse.model_validate(_extract_json(content))
    revised = {item.id: item for item in proposals.queries}
    expected = {item["id"] for item in rejected}
    if set(revised) != expected:
        raise SystemExit("Model revision IDs do not exactly match rejected IDs.")
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    response_hash = _sha(content)
    archive = args.archive_dir / f"{response_hash}.txt"
    archive_bytes = (content.rstrip() + "\n").encode("utf-8")
    if archive.exists() and archive.read_bytes() != archive_bytes:
        raise SystemExit(f"Refusing altered archive: {archive}")
    if not archive.exists():
        archive.write_bytes(archive_bytes)
    for dataset_id, proposal in revised.items():
        datasets[dataset_id]["search_expression"] = proposal.search_expression
        datasets[dataset_id]["query_rationale"] = proposal.rationale
    for dataset in registry["datasets"]:
        dataset["query_status"] = "query_judge_pending"
    history = list(registry.get("boolean_revision_history") or [])
    history.append(
        {
            "round": len(history) + 1,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "base_url": BASE_URL,
            "input_registry_sha256": _sha(registry_bytes),
            "judgment_sha256": _sha(judgment_bytes),
            "revised_ids": sorted(expected),
            "prompt_sha256": _sha(SYSTEM_PROMPT),
            "request_sha256": _sha(_canonical(request_body)),
            "response_sha256": response_hash,
            "response_archive": archive.relative_to(ROOT).as_posix(),
            "usage": payload.get("usage", {}),
            "response_id": payload.get("id"),
        }
    )
    registry["boolean_revision_history"] = history
    registry["generated_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
