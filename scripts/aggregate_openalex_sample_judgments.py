from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("relevant", "borderline", "irrelevant")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _datasets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", payload.get("datasets"))
    if not isinstance(rows, list):
        raise TypeError("Judgment file must contain results or datasets")
    return rows


def _fleiss_kappa(ratings: list[list[str]]) -> float | None:
    if not ratings:
        return None
    raters = len(ratings[0])
    if raters < 2 or any(len(row) != raters for row in ratings):
        raise ValueError("Every sample must have the same number of raters")
    agreement = []
    totals = Counter()
    for row in ratings:
        counts = Counter(row)
        totals.update(row)
        agreement.append(
            (sum(value * value for value in counts.values()) - raters)
            / (raters * (raters - 1))
        )
    observed = sum(agreement) / len(agreement)
    denominator = len(ratings) * raters
    expected = sum((totals[label] / denominator) ** 2 for label in LABELS)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite aggregate: {args.output}")
    if len(args.judgment) != 3:
        raise SystemExit("Exactly three independent judgment files are required")

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    probe_ids = {
        row["id"]: [item["id"] for item in row["random_samples"]]
        for row in probe["results"]
    }
    by_judge: list[dict[str, dict[str, str]]] = []
    for path in args.judgment:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("blind_to_outputs") is not True:
            raise ValueError(f"{path}: judge was not declared blind")
        datasets: dict[str, dict[str, str]] = {}
        for dataset in _datasets(payload):
            item_labels = {
                item["work_id"]: item["label"] for item in dataset["judgments"]
            }
            if list(item_labels) != probe_ids[dataset["dataset_id"]]:
                raise ValueError(f"{path}: work IDs/order differ from the probe")
            if set(item_labels.values()) - set(LABELS):
                raise ValueError(f"{path}: unknown relevance label")
            datasets[dataset["dataset_id"]] = item_labels
        if set(datasets) != set(probe_ids):
            raise ValueError(f"{path}: dataset IDs differ from the probe")
        by_judge.append(datasets)

    results = []
    all_ratings: list[list[str]] = []
    for dataset_id, work_ids in probe_ids.items():
        majority = Counter()
        unanimous = 0
        ratings_for_dataset = []
        items = []
        for work_id in work_ids:
            labels = [judge[dataset_id][work_id] for judge in by_judge]
            ratings_for_dataset.append(labels)
            all_ratings.append(labels)
            counts = Counter(labels)
            label = counts.most_common(1)[0][0]
            majority[label] += 1
            unanimous += int(len(counts) == 1)
            items.append(
                {
                    "work_id": work_id,
                    "labels": labels,
                    "majority_label": label,
                    "unanimous": len(counts) == 1,
                }
            )
        total = len(work_ids)
        results.append(
            {
                "dataset_id": dataset_id,
                "sample_size": total,
                "majority_counts": {label: majority[label] for label in LABELS},
                "majority_relevant_rate": majority["relevant"] / total,
                "majority_non_irrelevant_rate": (
                    majority["relevant"] + majority["borderline"]
                )
                / total,
                "unanimous_rate": unanimous / total,
                "fleiss_kappa": _fleiss_kappa(ratings_for_dataset),
                "items": items,
            }
        )

    total = len(all_ratings)
    pooled_majority = Counter(
        Counter(row).most_common(1)[0][0] for row in all_ratings
    )
    output = {
        "schema_version": 1,
        "role": "observational_query_precision_audit",
        "controls_or_modifies_pipeline": False,
        "probe": {"path": str(args.probe), "sha256": _sha(args.probe)},
        "judgments": [
            {"path": str(path), "sha256": _sha(path)} for path in args.judgment
        ],
        "raters": 3,
        "samples": total,
        "pooled": {
            "majority_counts": {label: pooled_majority[label] for label in LABELS},
            "majority_relevant_rate": pooled_majority["relevant"] / total,
            "majority_non_irrelevant_rate": (
                pooled_majority["relevant"] + pooled_majority["borderline"]
            )
            / total,
            "fleiss_kappa": _fleiss_kappa(all_ratings),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["pooled"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
