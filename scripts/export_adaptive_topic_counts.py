from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from citeweave.adaptive_original_evaluation import BASELINE_CONDITION
from citeweave.formal_adaptive_review import FORMAL_CONDITIONS
from citeweave.io import atomic_write_bytes


def _select_topics(references: dict, *, development_calibration: bool) -> list[str]:
    selected_role = "development" if development_calibration else "locked"
    topics = [
        item["id"]
        for item in references["references"]
        if item.get("role") == selected_role
    ]
    expected_topic_count = 2 if development_calibration else 6
    if len(topics) != expected_topic_count or len(set(topics)) != expected_topic_count:
        raise ValueError(
            f"{selected_role} export requires exactly {expected_topic_count} unique topics"
        )
    return topics


def _validate_baseline_contract(
    manifest: dict, *, development_calibration: bool
) -> None:
    if (
        manifest.get("evaluation_target")
        != "untouched_pre_intervention_original_candidate"
        or manifest.get("judge_may_modify_artifacts") is not False
        or manifest.get("evaluation_updates_feedback_memory") is not False
        or manifest.get("formal_results_used") is not (not development_calibration)
    ):
        raise ValueError("Baseline-original manifest violates the evaluation contract")


def _write_immutable_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different topic counts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("experiments/human_references.yml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/formal_adaptive_topic_counts"),
    )
    parser.add_argument(
        "--baseline-original-root",
        type=Path,
        required=True,
        help=(
            "Completed adaptive-original evaluation root. Its per-topic counts are "
            "merged as baseline_original for direct pre/post comparison."
        ),
    )
    parser.add_argument(
        "--development-calibration",
        action="store_true",
        help=(
            "Export only the two development topics with formal_results_used=false. "
            "The default formal mode exports only the six locked topics."
        ),
    )
    args = parser.parse_args()

    references = yaml.safe_load(args.references.read_text(encoding="utf-8"))
    selected_role = "development" if args.development_calibration else "locked"
    topics = _select_topics(
        references,
        development_calibration=args.development_calibration,
    )
    states = {
        condition: json.loads(
            (args.run_root / condition / "state.json").read_text(encoding="utf-8")
        )
        for condition in FORMAL_CONDITIONS
    }
    if not all(state.get("completed") for state in states.values()):
        raise SystemExit("Adaptive run is incomplete")
    baseline_manifest_path = args.baseline_original_root / "manifest.json"
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="utf-8")
    )
    _validate_baseline_contract(
        baseline_manifest,
        development_calibration=args.development_calibration,
    )
    baseline_samples_by_topic = {
        topic: {
            row["sample_id"]
            for row in baseline_manifest.get("items", [])
            if row.get("dataset_id") == topic
        }
        for topic in topics
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        baseline_path = args.baseline_original_root / "topic_counts" / f"{topic}.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing baseline-original topic metrics: {baseline_path}")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (
            baseline.get("topic_id") != topic
            or baseline.get("condition") != BASELINE_CONDITION
        ):
            raise ValueError(f"Invalid baseline-original topic metrics: {baseline_path}")
        baseline_counts = baseline.get("counts", {})
        expected_samples = baseline_samples_by_topic[topic]
        if (
            not expected_samples
            or baseline_counts.get("items") != len(expected_samples)
            or baseline_counts.get("review_requests") != 0
            or baseline_counts.get("auto_accepts") != len(expected_samples)
            or baseline_counts.get("unsafe_auto_accepts")
            != len(expected_samples) - baseline_counts.get("final_quality_passed", -1)
        ):
            raise ValueError(f"Baseline-original counts do not reconcile: {baseline_path}")
        conditions = {BASELINE_CONDITION: baseline_counts}
        for condition, state in states.items():
            rows = [row for row in state["records"] if row["dataset_id"] == topic]
            if not rows:
                raise ValueError(f"{condition}: no rows for {topic}")
            observed_samples = {row["sample_id"] for row in rows}
            if len(observed_samples) != len(rows) or observed_samples != expected_samples:
                raise ValueError(
                    f"{condition}: sample pairing differs from baseline-original for {topic}"
                )
            auto = [row for row in rows if row["auto_accepted"]]
            conditions[condition] = {
                "items": len(rows),
                "review_requests": sum(
                    bool(row["review_requested"]) for row in rows
                ),
                "final_quality_passed": sum(
                    bool(row["quality_passed"]) for row in rows
                ),
                "auto_accepts": len(auto),
                "unsafe_auto_accepts": sum(
                    not bool(row["quality_passed"]) for row in auto
                ),
            }
        _write_immutable_json(
            args.output_root / f"{topic}.json",
            {
                "schema_version": 2,
                "topic_id": topic,
                "conditions": conditions,
                "comparison_contract": {
                    "baseline_original": (
                        "independent quality evaluation of the untouched candidate"
                    ),
                    "post_review_conditions": list(FORMAL_CONDITIONS),
                    "topic_role": selected_role,
                    "formal_results_used": not args.development_calibration,
                },
            },
        )
    print(f"{len(topics)} topic files -> {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
