from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from citeweave.formal_statistics import (
    ADAPTIVE_CONDITIONS,
    BASELINE_ORIGINAL_CONDITION,
    GRAPH_COMPARISONS,
    POST_REVIEW_CONDITIONS,
    REPORT_COMPARISONS,
    FormalStatisticsError,
    _read_jsonl,
    _validate_resolved_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = ROOT / "experiments" / "human_references.yml"
DEFAULT_REPORT_ROOT = ROOT / "experiments" / "formal_judging" / "report_resolved"
DEFAULT_GRAPH_ROOT = ROOT / "experiments" / "formal_judging" / "graph_resolved"
DEFAULT_ADAPTIVE_ROOT = ROOT / "experiments" / "formal_adaptive_topic_counts"
DEFAULT_OUTPUT = ROOT / "experiments" / "formal_statistics_manifest.json"
RESOLVED_FILENAME = "resolved_judgments.jsonl"
COUNT_FIELDS = {
    "items",
    "review_requests",
    "final_quality_passed",
    "auto_accepts",
    "unsafe_auto_accepts",
}


class ManifestBuildError(ValueError):
    """Raised when formal inputs do not form one closed six-topic result set."""


def _read_locked_topics(references_path: Path) -> list[str]:
    try:
        registry = yaml.safe_load(references_path.read_text(encoding="utf-8"))
        references = registry["references"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ManifestBuildError(
            f"Cannot read the human-reference registry: {references_path}"
        ) from exc
    if not isinstance(references, list):
        raise ManifestBuildError("human_references.yml references must be a list")

    locked: list[str] = []
    development: set[str] = set()
    all_ids: set[str] = set()
    for index, reference in enumerate(references, 1):
        if not isinstance(reference, dict):
            raise ManifestBuildError(f"Reference {index} must be an object")
        topic = reference.get("id")
        role = reference.get("role")
        if not isinstance(topic, str) or not topic.strip():
            raise ManifestBuildError(f"Reference {index} has no valid id")
        if topic in all_ids:
            raise ManifestBuildError(f"Duplicate reference id: {topic}")
        all_ids.add(topic)
        if role == "locked":
            locked.append(topic)
        elif role == "development":
            development.add(topic)

    if len(locked) != 6 or len(set(locked)) != 6:
        raise ManifestBuildError(
            f"Formal statistics require exactly six unique role=locked topics; "
            f"found {len(locked)}"
        )
    if set(locked) & development:
        raise ManifestBuildError("A topic cannot be both locked and development")
    return locked


def _require_exact_directories(root: Path, expected: set[str], *, label: str) -> None:
    if not root.is_dir():
        raise ManifestBuildError(f"{label} root does not exist: {root}")
    actual = {entry.name for entry in root.iterdir()}
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ManifestBuildError(
            f"{label} entries differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    non_directories = sorted(name for name in actual if not (root / name).is_dir())
    if non_directories:
        raise ManifestBuildError(
            f"{label} entries must all be directories: {non_directories}"
        )


def _validate_resolved_file(
    path: Path,
    *,
    topic: str,
    comparison: str,
    condition_a: str,
    condition_b: str,
) -> None:
    try:
        rows = _validate_resolved_rows(
            _read_jsonl(path),
            topic=topic,
            comparison=comparison,
            condition_a=condition_a,
            condition_b=condition_b,
        )
    except FormalStatisticsError as exc:
        raise ManifestBuildError(str(exc)) from exc
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id != topic and not sample_id.startswith(f"{topic}:"):
            raise ManifestBuildError(
                f"{path}: sample_id {sample_id!r} belongs to a different topic"
            )


def _comparison_specs(
    root: Path,
    contracts: dict[str, tuple[str, str]],
    topics: list[str],
) -> list[dict[str, Any]]:
    _require_exact_directories(root, set(contracts), label=str(root))
    specs: list[dict[str, Any]] = []
    for comparison, (condition_a, condition_b) in contracts.items():
        comparison_root = root / comparison
        _require_exact_directories(
            comparison_root,
            set(topics),
            label=f"{comparison} topics",
        )
        files: dict[str, Path] = {}
        for topic in topics:
            path = comparison_root / topic / RESOLVED_FILENAME
            if not path.is_file():
                raise ManifestBuildError(f"Missing resolved judgments: {path}")
            if path.stat().st_size == 0:
                raise ManifestBuildError(f"Resolved judgments file is empty: {path}")
            _validate_resolved_file(
                path,
                topic=topic,
                comparison=comparison,
                condition_a=condition_a,
                condition_b=condition_b,
            )
            files[topic] = path.resolve()
        specs.append(
            {
                "name": comparison,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "files": files,
            }
        )
    return specs


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestBuildError(f"{label} must be a non-negative integer")
    return value


def _validate_adaptive_file(path: Path, topic: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ManifestBuildError(f"Missing or empty adaptive result: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Cannot read valid adaptive JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("topic_id") != topic:
        raise ManifestBuildError(f"{path}: topic_id must equal {topic!r}")
    contract = payload.get("comparison_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("topic_role") != "locked"
        or contract.get("formal_results_used") is not True
        or contract.get("post_review_conditions") != list(POST_REVIEW_CONDITIONS)
    ):
        raise ManifestBuildError(
            f"{path}: adaptive comparison contract is not locked formal output"
        )
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(ADAPTIVE_CONDITIONS):
        actual = set(conditions) if isinstance(conditions, dict) else set()
        raise ManifestBuildError(
            f"{path}: adaptive conditions differ: "
            f"missing={sorted(set(ADAPTIVE_CONDITIONS) - actual)}, "
            f"extra={sorted(actual - set(ADAPTIVE_CONDITIONS))}"
        )
    for condition, raw_counts in conditions.items():
        if not isinstance(raw_counts, dict) or not COUNT_FIELDS.issubset(raw_counts):
            actual = set(raw_counts) if isinstance(raw_counts, dict) else set()
            raise ManifestBuildError(
                f"{path}/{condition}: missing count fields "
                f"{sorted(COUNT_FIELDS - actual)}"
            )
        counts = {
            field: _nonnegative_integer(
                raw_counts[field],
                label=f"{path}/{condition}.{field}",
            )
            for field in COUNT_FIELDS
        }
        if counts["items"] == 0:
            raise ManifestBuildError(f"{path}/{condition}: items cannot be zero")
        if counts["review_requests"] + counts["auto_accepts"] != counts["items"]:
            raise ManifestBuildError(
                f"{path}/{condition}: review_requests + auto_accepts must equal items"
            )
        if counts["final_quality_passed"] > counts["items"]:
            raise ManifestBuildError(
                f"{path}/{condition}: final_quality_passed exceeds items"
            )
        if counts["unsafe_auto_accepts"] > counts["auto_accepts"]:
            raise ManifestBuildError(
                f"{path}/{condition}: unsafe_auto_accepts exceeds auto_accepts"
            )
        if condition == BASELINE_ORIGINAL_CONDITION and (
            counts["review_requests"] != 0
            or counts["auto_accepts"] != counts["items"]
            or counts["unsafe_auto_accepts"]
            != counts["items"] - counts["final_quality_passed"]
        ):
            raise ManifestBuildError(
                f"{path}/{condition}: untouched baseline counts do not reconcile"
            )


def _relative_path(path: Path, manifest_parent: Path) -> str:
    return Path(os.path.relpath(path, manifest_parent)).as_posix()


def build_manifest(
    *,
    references_path: Path,
    report_root: Path,
    graph_root: Path,
    adaptive_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    topics = _read_locked_topics(references_path)
    report_specs = _comparison_specs(report_root, REPORT_COMPARISONS, topics)
    graph_specs = _comparison_specs(graph_root, GRAPH_COMPARISONS, topics)

    if not adaptive_root.is_dir():
        raise ManifestBuildError(
            f"Adaptive result root does not exist: {adaptive_root}"
        )
    actual_adaptive = {path.name for path in adaptive_root.iterdir()}
    expected_adaptive = {f"{topic}.json" for topic in topics}
    if actual_adaptive != expected_adaptive:
        raise ManifestBuildError(
            "Adaptive topics differ: "
            f"missing={sorted(expected_adaptive - actual_adaptive)}, "
            f"extra={sorted(actual_adaptive - expected_adaptive)}"
        )
    adaptive_files: dict[str, Path] = {}
    for topic in topics:
        path = (adaptive_root / f"{topic}.json").resolve()
        _validate_adaptive_file(path, topic)
        adaptive_files[topic] = path

    parent = output_path.resolve().parent
    for specs in (report_specs, graph_specs):
        for spec in specs:
            spec["files"] = {
                topic: _relative_path(path, parent)
                for topic, path in spec["files"].items()
            }
    return {
        "version": 1,
        "topics": topics,
        "report_comparisons": report_specs,
        "graph_comparisons": graph_specs,
        "adaptive_results": {
            topic: _relative_path(path, parent)
            for topic, path in adaptive_files.items()
        },
    }


def write_idempotent(path: Path, manifest: dict[str, Any]) -> bool:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestBuildError(
                f"Refusing to overwrite unreadable existing manifest: {path}"
            ) from exc
        if existing != manifest:
            raise ManifestBuildError(
                f"Refusing to overwrite a different existing manifest: {path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fail-closed six-topic input manifest for formal statistics. "
            "Resolved roots use <comparison>/<topic>/resolved_judgments.jsonl."
        )
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--adaptive-root", type=Path, default=DEFAULT_ADAPTIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = build_manifest(
            references_path=args.references,
            report_root=args.report_root,
            graph_root=args.graph_root,
            adaptive_root=args.adaptive_root,
            output_path=args.output,
        )
        created = write_idempotent(args.output, manifest)
    except ManifestBuildError as exc:
        raise SystemExit(f"Formal statistics manifest refused the inputs: {exc}") from exc
    print(
        f"{'Created' if created else 'Verified unchanged'} {args.output.resolve()} "
        f"for {len(manifest['topics'])} locked topics"
    )


if __name__ == "__main__":
    main()
