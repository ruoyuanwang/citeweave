from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from citeweave.io import write_json
from citeweave.phenomenon_cards import build_phenomenon_cards

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "phenomenon-article-v1-20260820"


def _key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    return path.read_text(encoding="utf-8").strip()


def _messages(cards: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are writing a rigorous bibliometric research article. Use only the verified "
        "phenomenon cards and representative works supplied. Do not invent sources, causal "
        "mechanisms, study results, or domain facts. Every graph-derived sentence must cite "
        "one or more [CARD:...] identifiers; literature-linked sentences must cite [WORK:...]. "
        "Treat graph structure as descriptive evidence. Integrate phenomena across sections "
        "instead of writing one isolated figure-caption paragraph per card."
    )
    user = (
        "Write a 2200-3200 word English Markdown article with Title, Abstract, Introduction, "
        "Methods, Results, Discussion, Limitations, and Conclusion. Results must combine at "
        "least four cards and explicitly include one counterfactual, one global community-role "
        "contrast, and one temporal-structural contrast. Discussion must give at least two "
        "alternative explanations and distinguish corpus structure from real-world causality. "
        "Use representative works to add domain specificity, but only summarize information "
        "present in their titles/abstract excerpts. End with a Provenance Map listing each "
        "major claim and its CARD/WORK dependencies.\n\nPHENOMENON_CARDS:\n"
        + json.dumps(cards, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cards = build_phenomenon_cards(
        benchmark_path=args.benchmark,
        results_path=args.results,
        workspace=args.workspace.resolve(),
        output_path=args.output / "phenomenon_cards.json",
    )
    if not cards["passed"]:
        raise SystemExit(f"Phenomenon-card quality gates failed: {cards['quality_gates']}")
    messages = _messages(cards)
    manifest = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "temperature": 0,
        "thinking": "disabled",
        "cards_sha256": hashlib.sha256(
            (args.output / "phenomenon_cards.json").read_bytes()
        ).hexdigest(),
        "card_count": len(cards["cards"]),
        "quality_gates": cards["quality_gates"],
    }
    write_json(args.output / "generation_manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return
    request = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8000,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=360, follow_redirects=True) as client:
        response = client.post(
            args.base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {_key(args.api_key_file, args.api_key_env)}",
                "Content-Type": "application/json",
            },
            json=request,
        )
        response.raise_for_status()
        raw = response.json()
    article = raw["choices"][0]["message"]["content"].strip() + "\n"
    article_path = args.output / "article.md"
    article_path.write_text(article, encoding="utf-8")
    manifest.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "usage": raw.get("usage"),
            "finish_reason": raw["choices"][0].get("finish_reason"),
            "article_sha256": hashlib.sha256(article_path.read_bytes()).hexdigest(),
            "article_words": len(article.split()),
        }
    )
    write_json(args.output / "generation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
