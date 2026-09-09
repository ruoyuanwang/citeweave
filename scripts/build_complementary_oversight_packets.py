from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from citeweave.complementary_oversight import build_adversarial_review_packet
from citeweave.io import read_json, sha256_file, write_json

_TASK_CHALLENGES = {
    "multi_hop_connector": (
        "The path may reflect shared terminology or indexing conventions rather than "
        "semantic equivalence, transfer, or a scientific mechanism."
    ),
    "bridge_counterfactual": (
        "High betweenness can coexist with short alternate paths and can change under "
        "another graph threshold or construction rule."
    ),
    "community_role_contrast": (
        "Community size and the partition algorithm can create an outward-looking residual "
        "group; a community identifier is not a scientific topic."
    ),
    "hub_removal_resilience": (
        "Node deletion is a graph perturbation rather than a real intervention, and a "
        "replacement hub can preserve the largest component."
    ),
    "temporal_structural_shift": (
        "Growth ratios may be driven by the selected windows or indexing coverage and do "
        "not establish emergence, future impact, or breakthrough status."
    ),
}


def build_packets(*, pack_paths: list[Path], output_root: Path) -> dict[str, Any]:
    public_root = output_root / "public"
    internal_root = output_root / "internal"
    records = []
    for pack_path in sorted(pack_paths):
        pack = read_json(pack_path)
        if pack.get("passed") is not True:
            raise ValueError(f"Source evidence pack is not passed: {pack_path}")
        references = {
            row["reference_id"]: row for row in pack["representative_sources"]
        }
        for phenomenon in pack["graph_phenomena"]:
            case_id = str(phenomenon["phenomenon_id"])
            support = [
                {
                    "evidence_id": evidence_id,
                    "evidence_type": "graph_record",
                }
                for evidence_id in phenomenon["graph_evidence_ids"]
            ]
            for reference_id in phenomenon["reference_ids"]:
                reference = references[reference_id]
                support.append(
                    {
                        "evidence_id": reference_id,
                        "evidence_type": "representative_source",
                        "title": reference["title"],
                        "year": reference["year"],
                        "doi": reference.get("doi"),
                        "abstract_excerpt": str(reference.get("abstract_excerpt") or "")[
                            :600
                        ],
                    }
                )
            contract = phenomenon["interpretation_contract"]
            challenge = [
                {
                    "evidence_id": f"CH-{case_id}-task",
                    "evidence_type": "registered_alternative_explanation",
                    "text": _TASK_CHALLENGES[phenomenon["task_type"]],
                },
                {
                    "evidence_id": f"CH-{case_id}-limit",
                    "evidence_type": "required_limitation",
                    "text": contract["required_limitation"],
                },
            ]
            claim = (
                f"Registered question: {phenomenon['question']}\n"
                "Candidate structural answer: "
                + json.dumps(
                    phenomenon["verified_answer"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            public, internal = build_adversarial_review_packet(
                case_id=case_id,
                dataset_id=str(pack["dataset_id"]),
                domain=str(pack["dataset_id"]),
                claim=claim,
                supporting_evidence=support,
                challenging_evidence=challenge,
                operator_trace=phenomenon["operator_trace"],
                alternative_explanations=[
                    _TASK_CHALLENGES[phenomenon["task_type"]]
                ],
                forbidden_inferences=list(contract["forbidden"]),
            )
            public_path = public_root / f"{case_id}.json"
            internal_path = internal_root / f"{case_id}.json"
            write_json(public_path, public)
            write_json(internal_path, internal)
            records.append(
                {
                    "case_id": case_id,
                    "dataset_id": pack["dataset_id"],
                    "task_type": phenomenon["task_type"],
                    "source_pack": str(pack_path),
                    "source_pack_sha256": sha256_file(pack_path),
                    "public_path": str(public_path.relative_to(output_root)),
                    "public_sha256": sha256_file(public_path),
                    "internal_path": str(internal_path.relative_to(output_root)),
                    "internal_sha256": sha256_file(internal_path),
                }
            )
    datasets = sorted({row["dataset_id"] for row in records})
    task_counts: dict[str, int] = {}
    for row in records:
        task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
    manifest = {
        "schema_version": 1,
        "status": "packets_built_before_real_review",
        "human_outcomes_inspected": False,
        "datasets": len(datasets),
        "dataset_ids": datasets,
        "cases": len(records),
        "task_type_counts": dict(sorted(task_counts.items())),
        "public_role_blinding": True,
        "generation_condition_blinded": True,
        "gold_answer_blinded": True,
        "records": records,
    }
    write_json(output_root / "manifest.json", manifest)
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["canonical_manifest_sha256"] = digest
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-pack-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    pack_paths = sorted(args.writer_pack_root.glob("*/evidence_pack.json"))
    if not pack_paths:
        raise SystemExit("No frozen evidence packs found")
    manifest = build_packets(pack_paths=pack_paths, output_root=args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
