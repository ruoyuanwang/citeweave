from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.formal_protocol import (
    canonical_json,
    query_payload,
    query_set_sha256,
    sha256_json,
)
from citeweave.io import atomic_write_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets.yml"
RUBRIC_VERSION = "query-judge-v5"
QUERY_JUDGE_PROMPT = """Act only as an independent Query Judge. You did not generate these
queries. For each topic, assess whether the proposed concept phrases and source contract are
unambiguous, topically valid, sufficiently inclusive without becoming an unrelated generic query,
and safe from obvious homonyms. For source=openalex, a non-null search_expression is the exact
frozen Boolean query; otherwise query_mode=all joins every quoted concept phrase with AND.
The query_contract search_scope is binding: title_abstract means OpenAlex applies the Boolean
expression only to bibliographic title and abstract metadata; title means title only; fulltext means
the broader default search over title, abstract, and full text. Judge the explicit
year_from/year_to fields only;
an id can retain the publication year of a linked human
reference while the formal protocol intentionally ends at the preceding complete natural year.
For a field-wide or global mapping topic, a single unambiguous field term is acceptable and
'global' means no geographic filter rather than a required phrase. Do not attempt to reproduce or
infer any published reference paper's search string, database, corpus size, or numeric results.
Return strict JSON with judge_role='codex_query_judge', rubric_version, prompt_sha256,
packet_sha256, and one decision per id. Each decision is accept or reject and requires a concise
reason. Reject rather than silently revising a query."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite Query Judge packet: {args.output}")
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    if registry.get("status") != "query_judge_pending":
        raise SystemExit("Registry must be query_judge_pending.")
    datasets = registry["datasets"]
    prompt_sha = sha256_json(QUERY_JUDGE_PROMPT)
    packet = {
        "rubric_version": RUBRIC_VERSION,
        "prompt": QUERY_JUDGE_PROMPT,
        "prompt_sha256": prompt_sha,
        "query_set_sha256": query_set_sha256(datasets),
        "items": [
            {
                "id": item["id"],
                "topic": item["topic"],
                "query_contract": query_payload(item),
                "query_rationale": item.get("query_rationale"),
            }
            for item in datasets
        ],
        "output_schema": {
            "judge_role": "codex_query_judge",
            "rubric_version": RUBRIC_VERSION,
            "prompt_sha256": prompt_sha,
            "packet_sha256": "<copy packet_sha256>",
            "decisions": [
                {"id": "<exact id>", "decision": "accept|reject", "reason": "<reason>"}
            ],
        },
    }
    packet["packet_sha256"] = sha256_json(packet)
    payload = (canonical_json(packet) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(args.output, payload)
    print(json.dumps(packet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
