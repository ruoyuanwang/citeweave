from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from citeweave.formal_request import canonical_sha256
from citeweave.io import write_json

PROBES = (
    {
        "probe_id": "empty_user",
        "messages": [{"role": "user", "content": ""}],
    },
    {
        "probe_id": "ascii_user",
        "messages": [{"role": "user", "content": "A frozen tokenizer probe."}],
    },
    {
        "probe_id": "chinese_user",
        "messages": [{"role": "user", "content": "这是固定的分词验证探针。"}],
    },
    {
        "probe_id": "graph_json_user",
        "messages": [
            {
                "role": "user",
                "content": '{"edge":["节点甲","node_b"],"weight":12.5}',
            }
        ],
    },
    {
        "probe_id": "system_user_boundary",
        "messages": [
            {"role": "system", "content": "Return one short answer."},
            {"role": "user", "content": "固定边界探针 / frozen boundary probe"},
        ],
    },
    {
        "probe_id": "multiturn_boundary",
        "messages": [
            {"role": "system", "content": "Tokenization audit only."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second: 图结构"},
        ],
    },
)


def _api_key(path: Path | None, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    if path and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, candidate = line.partition("=")
            if separator and key.strip().casefold() == "deepseek":
                value = candidate.strip()
                if value:
                    return value
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=240, follow_redirects=True) as client:
        for probe in PROBES:
            request = {
                "model": args.model,
                "messages": probe["messages"],
                "temperature": 0,
                "max_tokens": 1,
                "thinking": {"type": "disabled"},
                "stream": False,
            }
            response = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=request,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
                raise RuntimeError(
                    f"API did not return a positive prompt_tokens count: {probe['probe_id']}"
                )
            records.append(
                {
                    "probe_id": probe["probe_id"],
                    "messages": probe["messages"],
                    "messages_sha256": canonical_sha256(probe["messages"]),
                    "request_without_secret": request,
                    "request_sha256": canonical_sha256(request),
                    "prompt_tokens": prompt_tokens,
                    "usage": usage,
                    "response_model": payload.get("model"),
                    "response_id": payload.get("id"),
                    "response_created": payload.get("created"),
                    "finish_reason": (payload.get("choices") or [{}])[0].get(
                        "finish_reason"
                    ),
                }
            )
            print(probe["probe_id"], prompt_tokens)
    output = {
        "schema_version": 1,
        "status": "api_usage_probes_complete",
        "purpose": "Tokenizer message-encoding verification only; contains no benchmark task or outcome.",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "requested_model": args.model,
        "passed": len(records) == len(PROBES),
        "records": records,
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
