from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "api_capabilities" / "deepseek_models.json"
# A 1x1 transparent PNG used only to verify that a configured model accepts image input.
PROBE_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
    )
).decode("ascii")


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--text-model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--vlm-model", default=os.getenv("DEEPSEEK_VLM_MODEL"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite capability snapshot: {args.output}")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not visible to this process.")
    base_url = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        models_response = client.get(f"{base_url}/models", headers=headers)
        models_response.raise_for_status()
        models_payload = models_response.json()
        ids = sorted(
            item["id"]
            for item in models_payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
        vlm_probe = None
        if args.vlm_model:
            request = {
                "model": args.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Return only the word transparent."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{PROBE_PNG}",
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            }
            response = client.post(
                f"{base_url}/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json=request,
            )
            vlm_probe = {
                "model": args.vlm_model,
                "request_sha256": _sha(request),
                "status_code": response.status_code,
                "accepted_image_input": response.is_success,
                "response": response.json() if response.is_success else response.text[:1000],
            }

    snapshot = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "models": ids,
        "required_text_model": args.text_model,
        "text_model_available": args.text_model in ids,
        "configured_vlm_model": args.vlm_model,
        "vlm_probe": vlm_probe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
