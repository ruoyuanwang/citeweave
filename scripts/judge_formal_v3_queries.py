from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import yaml

from citeweave.formal_protocol_amendment import effective_query_datasets
from citeweave.io import sha256_file, write_json


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, candidate = line.partition("=")
        if separator and key.strip().casefold() == "deepseek" and candidate.strip():
            return candidate.strip()
    raise RuntimeError("DeepSeek API key is unavailable")


def _parse(content: str) -> dict[str, Any]:
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(content)
    if not isinstance(payload.get("judgments"), list):
        raise TypeError("Judge response lacks judgments")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path("apikey.md"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a frozen query judgment")
    raw_protocol = args.protocol.read_bytes()
    protocol = yaml.safe_load(raw_protocol)
    amendment = None
    if args.amendment:
        amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
        if amendment.get("base_protocol_sha256") != hashlib.sha256(raw_protocol).hexdigest():
            raise ValueError("Query amendment does not target the supplied protocol")
        if amendment.get("acquisition_started") is not False:
            raise ValueError("Query amendment must be frozen before acquisition")
    datasets = effective_query_datasets(protocol, amendment=amendment)
    system = (
        "Act as an independent information-retrieval protocol reviewer. Evaluate frozen "
        "bibliometric dataset queries before any data acquisition or model outcome is seen. "
        "For each query, judge construct validity, likely precision/recall under Boolean AND "
        "search over title/abstract/full text, date-window adequacy, and domain distinctness. "
        "Do not optimize for a desired experimental result. Return one JSON object only with "
        "keys overall_approved and judgments. Each judgment must contain id, approved, "
        "severity (none|minor|major), issue, and rationale. Reject only material defects that "
        "require a pre-acquisition amendment. Minor caveats may remain approved."
    )
    user = json.dumps(
        {
            "full_data_definition": (
                "Exhaustive OpenAlex cursor pagination of articles matching all keyword "
                "phrases over complete natural years; no record cap."
            ),
            "datasets": datasets,
            "excluded_existing_topics": protocol["contamination_policy"][
                "excluded_existing_topics"
            ],
        },
        ensure_ascii=False,
    )
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request_bytes = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_api_key(args.api_key_file, args.api_key_env)}"},
        json=request,
        timeout=180,
    )
    response.raise_for_status()
    raw_response = response.json()
    content = raw_response["choices"][0]["message"]["content"]
    judgment = _parse(content)
    expected_ids = {row["id"] for row in datasets}
    observed_ids = {row.get("id") for row in judgment["judgments"]}
    if observed_ids != expected_ids:
        raise ValueError("Judge response dataset IDs do not match the frozen protocol")
    artifact = {
        "schema_version": 1,
        "judge_role": "independent_pre_acquisition_query_reviewer",
        "protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
        "query_amendment_sha256": (
            sha256_file(args.amendment) if args.amendment else None
        ),
        "model": args.model,
        "temperature": 0,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "usage": raw_response.get("usage"),
        "judgment": judgment,
        "outcome_visibility": "no acquisition counts, graph structures, or model outcomes supplied",
    }
    write_json(args.output, artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
