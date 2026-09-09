"""Outcome-blind selection replay and independent pre/post revision evaluation.

Replay measures defect capture, not the unobserved quality of counterfactual articles.
Pre/post effects concern changed evidence-bearing paragraphs, not whole manuscripts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from .article_claim_review import _opaque, _reviewers_for_topic
from .article_review_revision import JUDGMENT_FIELDS, _packet_index, resolve_article_claim_reviews
from .io import read_json, sha256_file, write_json


def exact_topic_sign_flip(effects: list[float]) -> dict[str, Any]:
    if not effects or len(effects) > 20 or any(not math.isfinite(v) for v in effects):
        raise ValueError("Exact sign flips require 1--20 finite topic effects")
    observed = mean(effects)
    null = [
        mean(s * v for s, v in zip(signs, effects, strict=True))
        for signs in product((-1, 1), repeat=len(effects))
    ]
    rng = Random(20260827)
    bootstrap = sorted(mean(rng.choices(effects, k=len(effects))) for _ in range(5000))
    return {
        "topics": len(effects),
        "mean_difference": observed,
        "one_sided_p": sum(v >= observed - 1e-12 for v in null) / len(null),
        "enumerated_sign_patterns": len(null),
        "topic_bootstrap_95_ci_descriptive": [bootstrap[124], bootstrap[4874]],
        "assumption": "paired topic-effect signs are exchangeable under the null",
    }


def build_article_review_routing_plan(
    primary_root: Path,
    *,
    output_path: Path,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    uniform_repetitions: int = 100,
    seed: int = 20260827,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("Routing plan is already frozen")
    if any(read_json(p).get("results") for p in (primary_root / "returns").glob("*.json")):
        raise ValueError("Routing must be frozen before any reviewer outcomes")
    if uniform_repetitions < 1 or any(not 0 < f < 1 for f in fractions):
        raise ValueError("Invalid routing repetitions or budget fractions")
    indexed = _packet_index(primary_root)
    payloads = {
        pid: read_json(primary_root / "packets" / "article" / f"{pid}.json") for pid in indexed
    }
    topic_ids: dict[str, list[str]] = defaultdict(list)
    for pid, row in indexed.items():
        topic_ids[row["dataset_id"]].append(pid)
    features = {}
    for topic, packet_ids in topic_ids.items():
        for pid in packet_ids:
            risk = payloads[pid].get("risk_features", {})
            score = (
                4 * int(risk.get("causal_language", False))
                + 2 * int(risk.get("numeric", False))
                + 2 * int(risk.get("multiple_phenomena", False))
                + int(risk.get("discussion_interpretation", False))
            )
            tokens = set(indexed[pid]["evidence_tokens"])
            impact = sum(
                bool(tokens & set(indexed[other]["evidence_tokens"])) for other in packet_ids
            )
            features[pid] = {
                "dataset_id": topic,
                "risk_score": score,
                "shared_evidence_claims": max(1, impact),
            }
    routes = [
        {
            "policy": "always_review",
            "fraction": 1.0,
            "repeat": 0,
            "selected_packet_ids": sorted(indexed),
        }
    ]
    for fraction in fractions:
        for policy in ("uniform_hash", "risk_priority", "dependency_active"):
            for repeat in range(uniform_repetitions if policy == "uniform_hash" else 1):
                selected = []
                for topic, packet_ids in sorted(topic_ids.items()):

                    def rank(
                        pid: str, policy: str = policy, topic: str = topic, repeat: int = repeat
                    ) -> tuple[int, str]:
                        feature = features[pid]
                        value = 0 if policy == "uniform_hash" else feature["risk_score"]
                        if policy == "dependency_active":
                            value = (value + 1) * feature["shared_evidence_claims"]
                        return -value, _opaque("ORDER", topic, pid, str(repeat), seed=seed)

                    selected.extend(
                        sorted(packet_ids, key=rank)[: math.ceil(len(packet_ids) * fraction)]
                    )
                routes.append(
                    {
                        "policy": policy,
                        "fraction": fraction,
                        "repeat": repeat,
                        "selected_packet_ids": sorted(selected),
                    }
                )
    result = {
        "schema_version": 1,
        "status": "routing_frozen_before_outcomes",
        "primary_root": str(primary_root.resolve()),
        "primary_manifest_sha256": sha256_file(primary_root / "internal_manifest.json"),
        "budget_unit": "number of dual-reviewed claims per topic, not seconds",
        "features": features,
        "routes": routes,
        "uniform_repetitions": uniform_repetitions,
        "interpretation": "Dependency score is a pre-outcome shared-evidence priority heuristic, not a learned capability model.",
    }
    write_json(output_path, result)
    write_json(
        output_path.with_suffix(".freeze.json"),
        {"sha256": sha256_file(output_path), "human_outcomes_present": False},
    )
    return result


def analyze_article_review_routing(
    routing_path: Path,
    resolved_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    routing = read_json(routing_path)
    if read_json(routing_path.with_suffix(".freeze.json"))["sha256"] != sha256_file(routing_path):
        raise ValueError("Routing plan differs from its pre-outcome freeze")
    root = Path(routing["primary_root"])
    if routing["primary_manifest_sha256"] != sha256_file(root / "internal_manifest.json"):
        raise ValueError("Routing packet manifest drift")
    supplied = read_json(resolved_path)
    adjudication = supplied.get("adjudication_root")
    verified = resolve_article_claim_reviews(
        root, adjudication_root=Path(adjudication) if adjudication else None
    )
    if supplied != verified:
        raise ValueError("Replay requires validated frozen real-review returns")
    outcomes = {row["packet_id"]: row for row in verified["resolved"]}
    if set(outcomes) != set(routing["features"]):
        raise ValueError("Routing/outcome packet identities differ")
    actionable = {
        pid for pid, row in outcomes.items() if row["action"] in {"rewrite", "reject_claim"}
    }
    route_results = []
    for route in routing["routes"]:
        selected = set(route["selected_packet_ids"])
        if not selected <= outcomes.keys():
            raise ValueError("Route includes unknown packets")
        captured = selected & actionable
        route_results.append(
            {
                "policy": route["policy"],
                "fraction": route["fraction"],
                "repeat": route["repeat"],
                "selected_claims": len(selected),
                "captured_actionable_claims": len(captured),
                "actionable_capture_rate": len(captured) / len(actionable) if actionable else None,
                "actual_dual_review_plus_adjudication_seconds": sum(
                    outcomes[p]["total_review_seconds"] for p in selected
                ),
                "direct_corrective_paragraphs_surfaced": len(
                    {(outcomes[p]["dataset_id"], outcomes[p]["paragraph_id"]) for p in captured}
                ),
            }
        )
    result = {
        "schema_version": 1,
        "status": "selection_replay_analyzed",
        "routing_sha256": sha256_file(routing_path),
        "resolved_sha256": sha256_file(resolved_path),
        "actionable_claims": len(actionable),
        "routes": route_results,
        "not_estimated": [
            "causal labor savings",
            "quality of ungenerated counterfactual articles",
            "unseen-topic generalization",
        ],
        "interpretation": "Same claim-count budget; real review time is reported, not assumed equal. All labels were collected once; this is offline selection replay.",
    }
    if output_path:
        write_json(output_path, result)
    return result


def build_revision_evaluation_packets(
    primary_root: Path,
    worklist_path: Path,
    replacements_path: Path,
    revision_manifest_path: Path,
    evaluator_roster_path: Path,
    *,
    output_root: Path,
    seed: int = 20260827,
) -> dict[str, Any]:
    if (output_root / "internal_manifest.json").exists():
        raise ValueError("Revision evaluation packets already frozen")
    indexed = _packet_index(primary_root)
    worklist = read_json(worklist_path)
    receipt = read_json(revision_manifest_path)
    if receipt["worklist_sha256"] != sha256_file(worklist_path) or receipt[
        "replacement_manifest_sha256"
    ] != sha256_file(replacements_path):
        raise ValueError("Revision receipt input mismatch")
    for article in receipt["articles"]:
        if sha256_file(Path(article["reviewed_path"])) != article["reviewed_sha256"]:
            raise ValueError("Reviewed article hash drift")
    replacements = {
        (r["dataset_id"], r["paragraph_id"]): r
        for r in read_json(replacements_path)["replacements"]
    }
    roster = read_json(evaluator_roster_path)
    topics = sorted({row["dataset_id"] for row in worklist["articles"]})
    topic_reviewers = {}
    for topic in topics:
        excluded = {
            reviewer
            for row in indexed.values()
            if row["dataset_id"] == topic
            for reviewer in [*row["primary_reviewers"], row["adjudicator"]]
        }
        eligible = {
            "reviewers": [
                row
                for row in roster["reviewers"]
                if row.get("independent_from_article_production") is True
                and row["reviewer_id"] not in excluded
            ]
        }
        topic_reviewers[topic] = _reviewers_for_topic(eligible, topic, seed=seed)
    assignments: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"article": []})
    all_source_packets = {
        pid: read_json(primary_root / "packets" / "article" / f"{pid}.json") for pid in indexed
    }
    internal_packets, packet_records = [], []
    for item in worklist["worklist"]:
        topic, paragraph_id = item["dataset_id"], item["paragraph_id"]
        pair_id = _opaque("PAIR", topic, paragraph_id, item["draft_sha256"], seed=seed)
        primary, adjudicator = topic_reviewers[topic]
        supporting_packets = [
            packet
            for pid, packet in all_source_packets.items()
            if indexed[pid]["dataset_id"] == topic
            and (
                any(
                    row["phenomenon_id"] in item["allowed_evidence_ids"]
                    for row in packet.get("phenomena", [])
                )
                or pid in {row["packet_id"] for row in item["directives"]}
            )
        ]
        phenomena = {
            row["phenomenon_id"]: row
            for packet in supporting_packets
            for row in packet.get("phenomena", [])
        }
        sources = {
            row["reference_id"]: row
            for packet in supporting_packets
            for row in packet.get("sources", [])
        }
        for version, content in (
            ("pre", item["original_text"]),
            ("post", replacements[(topic, paragraph_id)]["replacement_text"].strip()),
        ):
            packet_id = _opaque("RV", pair_id, version, seed=seed)
            packet = {
                "schema_version": 1,
                "packet_id": packet_id,
                "review_layer": "article",
                "claim": content,
                "paragraph_context": content,
                "phenomena": list(phenomena.values()),
                "sources": list(sources.values()),
                "allowed_decisive_evidence_ids": item["allowed_evidence_ids"],
                "question": "Independently audit this evidence-bearing paragraph. Judge all substantive assertions; abstain if supplied evidence is insufficient.",
            }
            path = output_root / "packets" / "article" / f"{packet_id}.json"
            write_json(path, packet)
            packet_records.append({"packet_id": packet_id, "sha256": sha256_file(path)})
            internal_packets.append(
                {
                    "packet_id": packet_id,
                    "dataset_id": topic,
                    "pair_id": pair_id,
                    "version": version,
                    "paragraph_id": paragraph_id,
                    "candidate_id": pair_id,
                    "primary_reviewers": primary,
                    "adjudicator": adjudicator,
                }
            )
            for reviewer in primary:
                assignments[reviewer]["article"].append(packet_id)
    for reviewer, assigned in assignments.items():
        Random(_opaque("SHUFFLE", reviewer, seed=seed)).shuffle(assigned["article"])
    public = {
        "schema_version": 1,
        "status": "independent_revision_evaluation_ready",
        "topics": len(topics),
        "paragraph_pairs": len(worklist["worklist"]),
        "packets": len(packet_records),
        "primary_reviews_required": 2 * len(packet_records),
        "roster_sha256": sha256_file(evaluator_roster_path),
        "worklist_sha256": sha256_file(worklist_path),
        "revision_receipt_sha256": sha256_file(revision_manifest_path),
        "packet_records": packet_records,
        "blinding_limit": "Order and version labels are hidden; similar wording may reveal pairing. This is not guaranteed perfect blinding.",
    }
    write_json(output_root / "manifest.json", public)
    write_json(
        output_root / "internal_manifest.json",
        {
            **public,
            "topic_ids": topics,
            "assignments": dict(assignments),
            "internal_packets": internal_packets,
            "topic_adjudicators": {
                topic: reviewers[1] for topic, reviewers in topic_reviewers.items()
            },
        },
    )
    return public


def analyze_revision_evaluation(
    evaluation_root: Path,
    resolved_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(evaluation_root / "internal_manifest.json")
    indexed = _packet_index(evaluation_root)
    supplied = read_json(resolved_path)
    adjudication = supplied.get("adjudication_root")
    verified = resolve_article_claim_reviews(
        evaluation_root, adjudication_root=Path(adjudication) if adjudication else None
    )
    if supplied != verified:
        raise ValueError("Evaluation results differ from frozen expert returns")
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in verified["resolved"]:
        internal = indexed[row["packet_id"]]
        version = internal["version"]
        if version in pairs[internal["pair_id"]]:
            raise ValueError("Duplicate pre/post paragraph version")
        pairs[internal["pair_id"]][version] = row
    pair_results = []
    for pair_id, versions in sorted(pairs.items()):
        if set(versions) != {"pre", "post"}:
            raise ValueError("Incomplete pre/post pair")
        pre, post = versions["pre"], versions["post"]

        # Abstention is never counted as successful support.
        def strict(row: dict[str, Any]) -> int:
            return int(row["action"] != "abstain" and all(row[f] for f in JUDGMENT_FIELDS))

        pair_results.append(
            {
                "pair_id": pair_id,
                "dataset_id": pre["dataset_id"],
                "pre_strict": strict(pre),
                "post_strict": strict(post),
                "strict_change": strict(post) - strict(pre),
                "check_score_change": sum(post[f] for f in JUDGMENT_FIELDS)
                - sum(pre[f] for f in JUDGMENT_FIELDS),
                "abstention_present": pre["action"] == "abstain" or post["action"] == "abstain",
            }
        )
    topic_effects = []
    for topic in manifest["topic_ids"]:
        rows = [row for row in pair_results if row["dataset_id"] == topic]
        topic_effects.append(
            {
                "dataset_id": topic,
                "changed_paragraphs": len(rows),
                "strict_change": mean(row["strict_change"] for row in rows) if rows else 0.0,
            }
        )
    effects = [row["strict_change"] for row in topic_effects]
    result = {
        "schema_version": 1,
        "status": "independent_revision_effect_analyzed",
        "estimand": "Mean within-topic strict support/calibration change among modified evidence-bearing paragraphs; no-change topics contribute zero",
        "primary_test": exact_topic_sign_flip(effects),
        "topic_effects": topic_effects,
        "pairs": pair_results,
        "strict_harm_pairs": sum(row["strict_change"] < 0 for row in pair_results),
        "abstention_pairs": sum(row["abstention_present"] for row in pair_results),
        "claim_limit": "Not a randomized human-vs-no-human study and not evidence of whole-article superiority.",
    }
    if output_path:
        write_json(output_path, result)
    return result
