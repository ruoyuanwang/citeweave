from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from citeweave.io import write_json

ROOT = Path(__file__).resolve().parents[1]
RISK_TERMS = re.compile(
    r"\b(?:indicates?|suggests?|reveals?|demonstrates?|proves?|driven by|dominant|"
    r"hot topic|mature|nascent|phase transition|influential|impactful)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|\b)")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _api_key(path: Path) -> str:
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().casefold() == "deepseek" and value.strip():
            return value.strip()
    raise RuntimeError("DeepSeek key is unavailable")


def _diagnostics(text: str) -> dict[str, Any]:
    tokens = TOKEN.findall(text)
    sentences = [item.strip() for item in SENTENCE.split(text) if item.strip()]
    headings = re.findall(
        r"^#{1,4}\s+.+$|^\*\*\d+(?:\.\d+)*[^*]+\*\*$",
        text,
        flags=re.MULTILINE,
    )
    return {
        "characters": len(text),
        "words": len(tokens),
        "sentences": len(sentences),
        "mean_sentence_words": len(tokens) / max(1, len(sentences)),
        "type_token_ratio": len({token.casefold() for token in tokens}) / max(1, len(tokens)),
        "headings": len(headings),
        "numeric_mentions_per_1000_words": len(NUMBER.findall(text)) * 1000 / max(1, len(tokens)),
        "interpretive_risk_markers_per_1000_words": len(RISK_TERMS.findall(text))
        * 1000
        / max(1, len(tokens)),
        "evidence_citations": len(re.findall(r"\[E\d+", text)),
        "reference_citations": len(re.findall(r"\[\d+(?:[–,-]\d+)*\]", text)),
    }


def _bounded(text: str, limit: int = 36000) -> str:
    if len(text) <= limit:
        return text
    # Preserve both evidence-heavy results and the paper's final synthesis.
    head = text[: limit * 2 // 3]
    tail = text[-limit // 3 :]
    return head + "\n\n[... middle omitted under the registered character cap ...]\n\n" + tail


def _parse(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


def _messages(topic: str, a: str, b: str) -> list[dict[str, str]]:
    rubric = {
        "epistemic_calibration": "distinguishes description, association, interpretation, and causality",
        "cross_evidence_synthesis": "integrates multiple analyses into a defensible higher-level finding",
        "graph_derived_insight": "extracts relational/global phenomena rather than restating top nodes",
        "argumentative_coherence": "builds a clear argument rather than a catalogue of metrics",
        "domain_specificity": "connects bibliometric patterns to concrete domain literature",
        "methodological_transparency": "makes corpus, parameters, uncertainty, and limitations auditable",
        "scholarly_style": "precise, readable, non-repetitive academic prose",
        "research_value": "produces useful hypotheses or conclusions beyond descriptive rankings",
    }
    system = (
        "Act as a blinded senior reviewer of bibliometric articles. The two texts may use "
        "different corpora, so never compare their numeric findings for agreement and never "
        "reward a text merely for being longer. Judge writing and scientific reasoning only. "
        "Score each rubric dimension from 1 (poor) to 5 (excellent). Return JSON only with "
        "keys scores, preferred, preference_reason, strongest_gap_in_A, strongest_gap_in_B, "
        "and overclaim_examples. scores must map each rubric name to {A: integer, B: integer}. "
        "preferred must be A, B, or tie. Quote at most eight words per example."
    )
    user = (
        f"TOPIC: {topic}\nRUBRIC: {json.dumps(rubric, ensure_ascii=False)}\n\n"
        f"ARTICLE A:\n{_bounded(a)}\n\nARTICLE B:\n{_bounded(b)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=ROOT / "experiments" / "formal_reports")
    parser.add_argument("--human-root", type=Path, default=ROOT / "experiments" / "human_outputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "apikey.md")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    topics = sorted(
        path.name
        for path in args.human_root.iterdir()
        if (path / "reference_report.md").is_file()
        and (args.generated_root / path.name / "citeweave_full" / "report.md").is_file()
    )
    if args.limit:
        topics = topics[: args.limit]
    plan = {"topics": topics, "model": args.model, "blinded": True, "calls": len(topics)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "manifest.json", plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    key = _api_key(args.api_key_file)
    results_path = args.output / "results.json"
    records = []
    if results_path.is_file():
        records = list(json.loads(results_path.read_text(encoding="utf-8")).get("records") or [])
    completed = {record["topic"] for record in records}
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for topic in topics:
            if topic in completed:
                continue
            generated = (args.generated_root / topic / "citeweave_full" / "report.md").read_text(
                encoding="utf-8"
            )
            human = (args.human_root / topic / "reference_report.md").read_text(encoding="utf-8")
            generated_is_a = int(hashlib.sha256(topic.encode()).hexdigest()[:8], 16) % 2 == 0
            article_a, article_b = (generated, human) if generated_is_a else (human, generated)
            request = {
                "model": args.model,
                "messages": _messages(topic, article_a, article_b),
                "temperature": 0,
                "max_tokens": 2200,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            started = time.perf_counter()
            response = client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=request,
            )
            response.raise_for_status()
            raw = response.json()
            verdict = _parse(raw)
            preferred = verdict.get("preferred")
            generated_preference = (
                "generated"
                if (preferred == "A") == generated_is_a and preferred in {"A", "B"}
                else "human"
                if preferred in {"A", "B"}
                else "tie"
            )
            generated_key = "A" if generated_is_a else "B"
            human_key = "B" if generated_is_a else "A"
            records.append(
                {
                    "topic": topic,
                    "generated_blind_label": generated_key,
                    "generated_preference": generated_preference,
                    "diagnostics": {
                        "generated": _diagnostics(generated),
                        "human": _diagnostics(human),
                    },
                    "scores": {
                        dimension: {
                            "generated": values[generated_key],
                            "human": values[human_key],
                        }
                        for dimension, values in verdict["scores"].items()
                    },
                    "verdict": verdict,
                    "usage": raw.get("usage"),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            write_json(results_path, {"manifest": plan, "records": records})
            print(topic, generated_preference)

    dimensions = sorted(records[0]["scores"]) if records else []
    summary = {
        "topics": len(records),
        "preferences": {
            label: sum(record["generated_preference"] == label for record in records)
            for label in ("generated", "human", "tie")
        },
        "mean_scores": {
            dimension: {
                group: statistics.mean(record["scores"][dimension][group] for record in records)
                for group in ("generated", "human")
            }
            for dimension in dimensions
        },
        "mean_diagnostics": {
            metric: {
                group: statistics.mean(record["diagnostics"][group][metric] for record in records)
                for group in ("generated", "human")
            }
            for metric in records[0]["diagnostics"]["generated"]
        }
        if records
        else {},
        "caveat": (
            "This is a blinded LLM rubric audit of writing and reasoning, not human evaluation "
            "and not a factual comparison of corpora. Independent expert review remains required."
        ),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
