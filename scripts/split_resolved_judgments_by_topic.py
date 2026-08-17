from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from citeweave.judge_protocol import canonical_json


def _locked_topics(references: Path) -> list[str]:
    registry = yaml.safe_load(references.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get("references"), list):
        raise TypeError("Reference registry must contain a references list")
    topics = [
        str(item["id"])
        for item in registry["references"]
        if item.get("role") == "locked"
    ]
    if len(topics) != 6 or len(set(topics)) != 6:
        raise ValueError("Formal split requires exactly six unique locked topics")
    return topics


def split_resolved(
    *,
    input_path: Path,
    references_path: Path,
    output_root: Path,
    filename: str = "resolved_judgments.jsonl",
) -> dict[str, int]:
    topics = _locked_topics(references_path)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Resolved input is empty")
    grouped: dict[str, list[dict]] = {topic: [] for topic in topics}
    packet_ids: set[str] = set()
    for row in rows:
        packet_id = str(row.get("packet_id") or "")
        if not packet_id or packet_id in packet_ids:
            raise ValueError(f"Missing or duplicate packet_id: {packet_id!r}")
        packet_ids.add(packet_id)
        sample_id = str(row.get("sample_id") or "")
        matches = [
            topic
            for topic in topics
            if sample_id == topic or sample_id.startswith(f"{topic}:")
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot map sample_id to one locked topic: {sample_id}")
        grouped[matches[0]].append(row)
    empty = [topic for topic, topic_rows in grouped.items() if not topic_rows]
    if empty:
        raise ValueError(f"No resolved judgments for locked topics: {empty}")

    rendered = {
        topic: "".join(canonical_json(row) + "\n" for row in topic_rows)
        for topic, topic_rows in grouped.items()
    }
    if output_root.exists():
        actual_entries = {path.name for path in output_root.iterdir()}
        if actual_entries != set(topics):
            raise FileExistsError(
                "Refusing a non-exact formal topic tree; "
                f"missing={sorted(set(topics) - actual_entries)}, "
                f"extra={sorted(actual_entries - set(topics))}"
            )
        for topic, content in rendered.items():
            path = output_root / topic / filename
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"Refusing to overwrite different formal split artifact: {path}"
                )
        return {topic: len(topic_rows) for topic, topic_rows in grouped.items()}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for topic, content in rendered.items():
            path = stage / topic / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        stage.replace(output_root)
    finally:
        if stage.exists():
            for path in sorted(stage.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            stage.rmdir()
    return {topic: len(topic_rows) for topic, topic_rows in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("experiments/human_references.yml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--filename", default="resolved_judgments.jsonl")
    args = parser.parse_args()

    counts = split_resolved(
        input_path=args.input,
        references_path=args.references,
        output_root=args.output_root,
        filename=args.filename,
    )
    print(f"{sum(counts.values())} rows split across {len(counts)} locked topics")


if __name__ == "__main__":
    main()
