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

ROOT = Path(__file__).resolve().parents[1]


def _api_key(path: Path) -> str:
    value = os.getenv("DEEPSEEK_API_KEY")
    if value:
        return value.strip()
    raw = path.read_text(encoding="utf-8").strip()
    for line in raw.splitlines():
        key, separator, candidate = line.partition("=")
        if separator and key.strip().casefold() == "deepseek" and candidate.strip():
            return candidate.strip()
    return raw


def _bounded(text: str, limit: int = 40_000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit * 2 // 3] + "\n[...middle omitted...]\n" + text[-limit // 3 :]


def _messages(topic: str, left: str, right: str) -> list[dict[str, str]]:
    rubric = {
        "epistemic_calibration": "distinguishes description, interpretation, and causality",
        "phenomenon_complexity": "uses multi-hop, global, temporal, or counterfactual phenomena rather than lookup facts",
        "cross_evidence_synthesis": "integrates multiple analyses into a defensible argument",
        "graph_derived_insight": "extracts relational/global phenomena rather than restating rankings",
        "argumentative_coherence": "builds a clear argument rather than a catalogue of metrics",
        "domain_specificity": "connects patterns to concrete domain literature without invention",
        "methodological_transparency": "makes corpus, operators, evidence, uncertainty, and limitations auditable",
        "scholarly_style": "is precise, readable, proportionate, and non-repetitive",
        "research_value": "produces useful hypotheses or conclusions beyond descriptive rankings",
    }
    system = (
        "Act as a blinded senior reviewer of bibliometric articles. Do not assume the longer "
        "article is better. Do not reward provenance identifiers merely for existing; reward "
        "them only when they make substantive claims checkable. The texts may use different "
        "corpus boundaries, so judge writing and reasoning rather than numeric agreement. "
        "Score each dimension 1-5. Return JSON only with scores, preferred, preference_reason, "
        "strongest_gap_in_A, strongest_gap_in_B, and overclaim_examples. scores maps every "
        "dimension to {A: integer, B: integer}; preferred is A, B, or tie."
    )
    user = (
        f"TOPIC: {topic}\nRUBRIC: {json.dumps(rubric, ensure_ascii=False)}\n\n"
        f"ARTICLE A:\n{_bounded(left)}\n\nARTICLE B:\n{_bounded(right)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse(raw: dict[str, Any]) -> dict[str, Any]:
    content = raw["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--comparator-label", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    candidate = args.candidate.read_text(encoding="utf-8")
    comparator = args.comparator.read_text(encoding="utf-8")
    candidate_is_a = int(
        hashlib.sha256(
            f"{args.topic}|{args.candidate_label}|{args.comparator_label}".encode()
        ).hexdigest()[:8],
        16,
    ) % 2 == 0
    left, right = (candidate, comparator) if candidate_is_a else (comparator, candidate)
    manifest = {
        "schema_version": 1,
        "topic": args.topic,
        "candidate_label": args.candidate_label,
        "comparator_label": args.comparator_label,
        "candidate_blind_label": "A" if candidate_is_a else "B",
        "model": args.model,
        "temperature": 0,
        "calls": 1,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        return
    request = {
        "model": args.model,
        "messages": _messages(args.topic, left, right),
        "temperature": 0,
        "max_tokens": 2600,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=360, follow_redirects=True) as client:
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key(args.api_key_file)}",
                "Content-Type": "application/json",
            },
            json=request,
        )
        response.raise_for_status()
        raw = response.json()
    verdict = _parse(raw)
    candidate_key = "A" if candidate_is_a else "B"
    comparator_key = "B" if candidate_is_a else "A"
    result = {
        "manifest": manifest,
        "preference": (
            args.candidate_label
            if verdict["preferred"] == candidate_key
            else args.comparator_label
            if verdict["preferred"] == comparator_key
            else "tie"
        ),
        "scores": {
            dimension: {
                args.candidate_label: values[candidate_key],
                args.comparator_label: values[comparator_key],
            }
            for dimension, values in verdict["scores"].items()
        },
        "verdict": verdict,
        "usage": raw.get("usage"),
        "elapsed_seconds": time.perf_counter() - started,
        "caveat": "Blinded LLM pre-audit; independent domain-expert review is still required.",
    }
    write_json(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
