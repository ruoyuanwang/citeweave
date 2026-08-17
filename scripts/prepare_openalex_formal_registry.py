from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejection-audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite registry: {args.output}")
    source_bytes = args.input.read_bytes()
    rejection_bytes = args.rejection_audit.read_bytes()
    registry = yaml.safe_load(source_bytes)
    if registry.get("status") != "frozen":
        raise SystemExit("Input registry must be the previously frozen query set.")
    for dataset in registry["datasets"]:
        dataset["source"] = "openalex"
        dataset["title"] = (
            f"Independent full-year OpenAlex Boolean census: {dataset['topic']}"
        )
        dataset["document_types"] = ["article"]
        dataset["query_status"] = "query_judge_pending"
    registry["status"] = "query_judge_pending"
    registry["generated_at"] = datetime.now(UTC).isoformat()
    registry["full_data_definition"] = (
        "All OpenAlex works matching the frozen Boolean search over title, abstract, "
        "or full text and the complete natural-year interval are exhaustively cursor-paginated; "
        "all unique OpenAlex work identities are retained in the analysis corpus."
    )
    registry["source_transition"] = {
        "from": "crossref_query_bibliographic",
        "to": "openalex_boolean_search",
        "reason": (
            "Empirical root-count validation showed that Crossref query.bibliographic "
            "does not enforce the intended Boolean concept intersection."
        ),
        "input_registry_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rejection_audit_sha256": hashlib.sha256(rejection_bytes).hexdigest(),
        "transitioned_at": datetime.now(UTC).isoformat(),
    }
    registry.pop("freeze_attestation", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(registry, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
