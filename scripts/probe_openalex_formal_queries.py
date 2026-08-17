from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import yaml

from citeweave.bulk_acquisition import BulkSourceAdapter
from citeweave.connectors.openalex import OpenAlexConnector
from citeweave.models import SearchProtocol, SourceName


def _abstract_excerpt(work: dict, *, limit: int = 600) -> str | None:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return None
    positioned = [
        (position, token)
        for token, positions in inverted.items()
        if isinstance(positions, list)
        for position in positions
        if isinstance(position, int)
    ]
    text = " ".join(token for _, token in sorted(positioned))
    return text[:limit] if text else None


def _sample_record(work: dict) -> dict:
    primary_topic = work.get("primary_topic") or {}
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name") or work.get("title"),
        "publication_year": work.get("publication_year"),
        "relevance_score": work.get("relevance_score"),
        "primary_topic": primary_topic.get("display_name"),
        "abstract_excerpt": _abstract_excerpt(work),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--random-sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite empirical probe: {args.output}")
    registry_bytes = args.registry.read_bytes()
    registry = yaml.safe_load(registry_bytes)
    connector = OpenAlexConnector(
        args.output.parent / "openalex_probe_raw",
        api_key=os.getenv("OPENALEX_API_KEY"),
        max_retries=4,
        requests_per_second=2,
    )
    adapter = BulkSourceAdapter(connector)
    results = []
    try:
        for item in registry["datasets"]:
            protocol = SearchProtocol(
                title=item["title"],
                keywords=item["keywords"],
                query_mode=item["query_mode"],
                search_expression=item.get("search_expression"),
                search_scope=item.get("search_scope", "fulltext"),
                year_from=item["year_from"],
                year_to=item["year_to"],
                source=SourceName.openalex,
                document_types=item["document_types"],
                language=item.get("language"),
                max_records=None,
            )
            params = adapter._params(
                protocol,
                date(item["year_from"], 1, 1),
                date(item["year_to"], 12, 31),
                page_size=args.sample_size,
                cursor="*",
            )
            payload = connector._request_json(connector.endpoint, params=params)
            samples = [_sample_record(work) for work in payload.get("results", [])]
            random_params = dict(params)
            random_params.pop("cursor", None)
            random_params["sample"] = args.random_sample_size
            random_params["per_page"] = args.random_sample_size
            random_params["seed"] = args.seed
            random_payload = connector._request_json(
                connector.endpoint,
                params=random_params,
            )
            random_samples = [
                _sample_record(work) for work in random_payload.get("results", [])
            ]
            results.append(
                {
                    "id": item["id"],
                    "search_scope": item.get("search_scope", "fulltext"),
                    "search": params.get("search"),
                    "filter": params["filter"],
                    "count": int(payload.get("meta", {}).get("count", 0)),
                    "top_relevance_samples": samples,
                    "random_samples": random_samples,
                }
            )
    finally:
        connector.close()
    audit = {
        "schema_version": 1,
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "api_key_used": bool(os.getenv("OPENALEX_API_KEY")),
        "sample_size": args.sample_size,
        "random_sample_size": args.random_sample_size,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {item["id"]: item["count"] for item in results},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
