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
DEFAULT_REFERENCES = ROOT / "experiments" / "human_references.yml"
DEFAULT_OUTPUT = ROOT / "experiments" / "formal_datasets.yml"
DEFAULT_ARCHIVE = ROOT / "experiments" / "query_generation"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

SYSTEM_PROMPT = """You design reproducible bibliometric search protocols.
Return strict JSON only. Each query must be independently designed from the topic description;
the human reference paper is not a query source and its database, search string, corpus size,
or numerical results must not be copied. The target source is Crossref query.bibliographic.
Use two to five concise English concept phrases. Avoid Boolean syntax inside a phrase.
The final query is formed by joining phrases with AND. Prefer precision over a very broad query,
but include standard domain terminology. Do not add year words."""


class QueryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    keywords: list[str] = Field(min_length=2, max_length=5)
    rationale: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[QueryProposal]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(stripped)


def _request(
    topics: list[dict[str, Any]],
) -> tuple[QueryResponse, dict[str, Any], str]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not visible to this process. Set it in the environment that "
            "launches Codex, then restart Codex so the process inherits it."
        )
    user_payload = {
        "task": "Generate one independent Crossref bibliographic query per topic.",
        "topics": [
            {
                "id": item["id"],
                "topic": item["topic"],
                "year_from": item.get("formal_year_from", item["reference_year_from"]),
                "year_to": item.get("formal_year_to", item["reference_year_to"]),
            }
            for item in topics
        ],
        "output_schema": {
            "queries": [
                {
                    "id": "exact topic id",
                    "keywords": ["concept phrase 1", "concept phrase 2"],
                    "rationale": "one sentence",
                }
            ]
        },
    }
    request_body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 4000,
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
    parsed = QueryResponse.model_validate(_extract_json(content))
    audit = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "base_url": BASE_URL,
        "prompt_sha256": _sha256(SYSTEM_PROMPT),
        "request_sha256": _sha256(json.dumps(request_body, sort_keys=True, ensure_ascii=False)),
        "response_sha256": _sha256(content),
        "usage": payload.get("usage", {}),
        "response_id": payload.get("id"),
    }
    return parsed, audit, content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite frozen registry: {args.output}")
    reference_bytes = args.references.read_bytes()
    reference_registry = yaml.safe_load(reference_bytes)
    topics = reference_registry["references"]
    response, audit, raw_response = _request(topics)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    response_archive = args.archive_dir / f"{audit['response_sha256']}.txt"
    response_bytes = (raw_response.rstrip() + "\n").encode("utf-8")
    if response_archive.exists() and response_archive.read_bytes() != response_bytes:
        raise RuntimeError(f"Refusing altered query-generation archive: {response_archive}")
    if not response_archive.exists():
        response_archive.write_bytes(response_bytes)
    proposals = {item.id: item for item in response.queries}
    expected_ids = {item["id"] for item in topics}
    if set(proposals) != expected_ids:
        raise RuntimeError(
            f"Model returned mismatched ids: missing={sorted(expected_ids - set(proposals))}, "
            f"extra={sorted(set(proposals) - expected_ids)}"
        )

    registry = {
        "schema_version": "1.0",
        "status": "query_judge_pending",
        "generated_at": datetime.now(UTC).isoformat(),
        "human_reference_registry_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "generator": {
            **audit,
            "response_archive": response_archive.relative_to(ROOT).as_posix(),
        },
        "full_data_definition": (
            "All raw records returned by the frozen Crossref query.bibliographic protocol over "
            "the complete natural-year interval are exhaustively paginated; all unique source "
            "identities are retained in the deduplicated analysis corpus."
        ),
        "datasets": [
            {
                "id": item["id"],
                "role": item["role"],
                "topic": item["topic"],
                "human_reference_id": item["id"],
                "source": "crossref",
                "title": f"Independent full-year Crossref census: {item['topic']}",
                "keywords": proposals[item["id"]].keywords,
                "query_mode": "all",
                "year_from": item.get("formal_year_from", item["reference_year_from"]),
                "year_to": item.get("formal_year_to", item["reference_year_to"]),
                "document_types": ["journal-article"],
                # Crossref query.bibliographic has no server-side language filter.
                # Language coverage is profiled after exhaustive raw acquisition.
                "language": None,
                "max_records": None,
                "include_references": True,
                "query_rationale": proposals[item["id"]].rationale,
                "query_status": "query_judge_pending",
            }
            for item in topics
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
