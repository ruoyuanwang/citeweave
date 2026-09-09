from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .article_expert_evaluation import ARTICLE_CONDITIONS, HOLISTIC_DIMENSIONS
from .io import read_json, sha256_file, write_json


def _opaque_id(prefix: str, *parts: str, seed: int) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:12].upper()}"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expert_task_contract(
    evaluator_id: str,
    topic_code: str,
    task_type: str,
    *,
    article_code: str | None = None,
    claim_id: str | None = None,
    left_article_code: str | None = None,
    right_article_code: str | None = None,
) -> dict[str, str]:
    identity = {
        "evaluator_id": evaluator_id,
        "topic_code": topic_code,
        "task_type": task_type,
    }
    if task_type == "pairwise":
        if not left_article_code or not right_article_code:
            raise ValueError("Pairwise task requires two article codes")
        identity.update(
            left_article_code=left_article_code,
            right_article_code=right_article_code,
        )
    else:
        if not article_code:
            raise ValueError("Article task requires an article code")
        identity["article_code"] = article_code
        if task_type == "claim":
            if not claim_id:
                raise ValueError("Claim task requires a claim ID")
            identity["claim_id"] = claim_id
    task_id = f"XRT-{_canonical_sha(identity)[:16].upper()}"
    definition = {"task_id": task_id, **identity}
    return {**definition, "task_definition_sha256": _canonical_sha(definition)}


def _public_phenomenon(row: dict[str, Any], token_map: dict[str, str]) -> dict[str, Any]:
    return {
        "evidence_id": token_map[str(row["phenomenon_id"])],
        "evidence_type": "graph_phenomenon",
        "task_type": row.get("task_type"),
        "question": row.get("question"),
        "verified_answer": row.get("verified_answer"),
        "operator_trace": row.get("operator_trace", []),
        "interpretation_contract": row.get("interpretation_contract", {}),
        "graph_evidence_ids": row.get("graph_evidence_ids", []),
        "related_evidence_ids": [
            token_map[value]
            for value in row.get("reference_ids", [])
            if value in token_map
        ],
    }


def _public_source(row: dict[str, Any], token_map: dict[str, str]) -> dict[str, Any]:
    return {
        "evidence_id": token_map[str(row["reference_id"])],
        "evidence_type": "bibliographic_source",
        "title": row.get("title"),
        "year": row.get("year"),
        "doi": row.get("doi"),
        "abstract_excerpt": row.get("abstract_excerpt"),
    }


def build_article_expert_collection(
    intake_path: Path,
    packet_manifest_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Add a condition-blind evidence viewer and collection contract to expert packets."""
    intake = read_json(intake_path)
    packet_manifest = read_json(packet_manifest_path)
    if packet_manifest.get("status") != "expert_packets_ready":
        raise ValueError("Expert packet manifest is not ready")
    if packet_manifest.get("intake_sha256") != sha256_file(intake_path):
        raise ValueError("Expert packet manifest does not match the article intake")
    seed = int(packet_manifest["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_records = {}
    topic_codes_by_evaluator: dict[str, set[str]] = {}
    for record in packet_manifest.get("evaluator_packets", []):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Evaluator packet changed after freeze: {path}")
        packet = read_json(path)
        evaluator_id = str(record["evaluator_id"])
        if packet.get("evaluator_id") != evaluator_id:
            raise ValueError("Evaluator packet identity mismatch")
        packet_records[evaluator_id] = {
            "path": str(path.resolve()),
            "sha256": record["sha256"],
        }
        topic_codes_by_evaluator[evaluator_id] = {
            str(row["topic_code"]) for row in packet.get("topics", [])
        }

    rows_by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in intake.get("articles", []):
        rows_by_topic.setdefault(str(row["topic_id"]), []).append(row)
    viewers = {}
    for topic_id, rows in sorted(rows_by_topic.items()):
        pack_hashes = {str(row.get("writer_pack_sha256")) for row in rows}
        pack_paths = {str(row.get("writer_pack")) for row in rows}
        if len(rows) != len(ARTICLE_CONDITIONS) or len(pack_hashes) != 1 or len(pack_paths) != 1:
            raise ValueError(f"Topic {topic_id} does not share one evidence pack")
        pack_path = Path(next(iter(pack_paths)))
        pack_hash = next(iter(pack_hashes))
        if not pack_path.is_file() or sha256_file(pack_path) != pack_hash:
            raise ValueError(f"Topic evidence pack changed after intake: {topic_id}")
        pack = read_json(pack_path)
        phenomena = list(pack.get("graph_phenomena", []))
        sources = list(pack.get("representative_sources", []))
        raw_tokens = [str(row["phenomenon_id"]) for row in phenomena] + [
            str(row["reference_id"]) for row in sources
        ]
        if len(raw_tokens) != len(set(raw_tokens)):
            raise ValueError(f"Duplicate evidence identity in topic {topic_id}")
        token_map = {
            token: _opaque_id("EV", topic_id, token, seed=seed) for token in raw_tokens
        }
        topic_code = _opaque_id("TOP", topic_id, seed=seed)
        visible_items = [
            *(_public_phenomenon(row, token_map) for row in phenomena),
            *(_public_source(row, token_map) for row in sources),
        ]
        viewer = {
            "schema_version": 1,
            "topic_code": topic_code,
            "blinding": {
                "condition_hidden": True,
                "raw_evidence_ids_hidden": True,
                "automatic_scores_hidden": True,
                "retrieval_ranks_hidden": True,
            },
            "items": visible_items,
        }
        serialized = json.dumps(viewer, ensure_ascii=False, sort_keys=True)
        leaked = [token for token in raw_tokens if token in serialized]
        if leaked:
            raise AssertionError("Raw PH/REF identities leaked into expert evidence viewer")
        path = output_dir / "evidence" / f"{topic_code}.json"
        write_json(path, viewer)
        viewers[topic_code] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "items": len(visible_items),
        }

    expected_topic_codes = {
        code for codes in topic_codes_by_evaluator.values() for code in codes
    }
    if expected_topic_codes != set(viewers):
        raise ValueError("Evaluator topics and evidence viewers do not match")

    claim_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    task_registry: dict[str, list[dict[str, str]]] = {}
    for evaluator_id, record in packet_records.items():
        packet = read_json(Path(record["path"]))
        evaluator_tasks = []
        for topic in packet.get("topics", []):
            topic_code = str(topic["topic_code"])
            articles = topic.get("articles", [])
            task_counts["holistic"] += len(articles)
            task_counts["pairwise"] += len(list(itertools.combinations(articles, 2)))
            for article in articles:
                evaluator_tasks.append(
                    expert_task_contract(
                        evaluator_id,
                        topic_code,
                        "holistic",
                        article_code=str(article["article_code"]),
                    )
                )
                claims = article.get("claims", [])
                if any(not claim.get("evidence_tokens") for claim in claims):
                    raise ValueError(
                        "Every expert claim must expose at least one blinded evidence token"
                    )
                task_counts["claim"] += len(claims)
                for claim in claims:
                    evaluator_tasks.append(
                        expert_task_contract(
                            evaluator_id,
                            topic_code,
                            "claim",
                            article_code=str(article["article_code"]),
                            claim_id=str(claim["claim_id"]),
                        )
                    )
                    claim_counts[f"{article['article_code']}::{claim['claim_id']}"] += 1
            for left, right in itertools.combinations(articles, 2):
                evaluator_tasks.append(
                    expert_task_contract(
                        evaluator_id,
                        topic_code,
                        "pairwise",
                        left_article_code=str(left["article_code"]),
                        right_article_code=str(right["article_code"]),
                    )
                )
        task_registry[evaluator_id] = evaluator_tasks
    if claim_counts and set(claim_counts.values()) != {2}:
        raise ValueError("Every sampled claim must have exactly two primary evaluators")

    manifest = {
        "schema_version": 1,
        "status": "expert_collection_ready",
        "collection_mode": "primary",
        "enabled_task_types": ["holistic", "claim", "pairwise"],
        "condition_blind_collection": True,
        "packet_manifest": str(packet_manifest_path.resolve()),
        "packet_manifest_sha256": sha256_file(packet_manifest_path),
        "intake_sha256": sha256_file(intake_path),
        "seed": seed,
        "evaluator_packets": packet_records,
        "evidence_viewers": viewers,
        "task_counts": dict(sorted(task_counts.items())),
        "task_registry": task_registry,
        "submission_contract": {
            "holistic_dimensions": list(HOLISTIC_DIMENSIONS),
            "holistic_scale": [1, 5],
            "claim_decisive_evidence_required": True,
            "immutable_primary_submission": True,
            "server_visible_heartbeat_timing": True,
            "condition_labels_never_served": True,
        },
    }
    write_json(output_dir / "collection_manifest.json", manifest)
    return manifest


def validate_article_expert_primary_returns(
    collection_manifest_path: Path,
    protocol_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Validate a complete primary panel and restore private condition mappings."""
    collection = read_json(collection_manifest_path)
    if collection.get("status") != "expert_collection_ready":
        raise ValueError("Expert collection is not ready")
    packet_manifest_path = Path(collection["packet_manifest"])
    if (
        not packet_manifest_path.is_file()
        or sha256_file(packet_manifest_path) != collection["packet_manifest_sha256"]
    ):
        raise ValueError("Private expert packet manifest changed after collection freeze")
    packet_manifest = read_json(packet_manifest_path)
    article_rows = packet_manifest.get("article_quality_controls", {}).get(
        "articles", []
    )
    article_map = {
        str(row["article_code"]): {
            "topic_id": str(row["topic_id"]),
            "condition": str(row["condition"]),
        }
        for row in article_rows
    }
    if not article_map:
        raise ValueError("Private packet manifest lacks article condition mapping")
    if any(row["condition"] not in ARTICLE_CONDITIONS for row in article_map.values()):
        raise ValueError("Private packet manifest contains an unknown article condition")

    evaluator_roles = {}
    for row in packet_manifest.get("assignments", []):
        evaluator_id = str(row["evaluator_id"])
        role = str(row["evaluator_role"])
        previous = evaluator_roles.setdefault(evaluator_id, role)
        if previous != role:
            raise ValueError("Evaluator role changes across topic assignments")

    expected_registry = collection.get("task_registry", {})
    expected_counts = Counter()
    raw_return_hashes = {}
    holistic = []
    claims = []
    preferences = []
    for evaluator_id, contracts in expected_registry.items():
        return_path = collection_manifest_path.parent / "returns" / f"{evaluator_id}.json"
        if not return_path.is_file():
            raise ValueError(f"Missing expert return: {evaluator_id}")
        payload = read_json(return_path)
        if payload.get("evaluator_id") != evaluator_id:
            raise ValueError("Expert return identity mismatch")
        results = payload.get("results", [])
        result_ids = [row.get("task_id") for row in results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"Duplicate expert task return: {evaluator_id}")
        contract_by_id = {row["task_id"]: row for row in contracts}
        if set(result_ids) != set(contract_by_id):
            missing = sorted(set(contract_by_id) - set(result_ids))
            extra = sorted(set(result_ids) - set(contract_by_id))
            raise ValueError(
                f"Incomplete or foreign expert returns for {evaluator_id}: "
                f"missing={missing}, extra={extra}"
            )
        raw_return_hashes[evaluator_id] = sha256_file(return_path)
        for result in results:
            contract = contract_by_id[result["task_id"]]
            for field, expected in contract.items():
                if result.get(field) != expected:
                    raise ValueError(
                        f"Server-owned expert task field changed: {evaluator_id}/{field}"
                    )
            if (
                result.get("timing_method")
                != "visibility_heartbeat_server_accounted"
                or not isinstance(result.get("review_seconds"), (int, float))
                or isinstance(result.get("review_seconds"), bool)
                or result["review_seconds"] <= 0
            ):
                raise ValueError("Expert return lacks valid server-accounted timing")
            task_type = str(contract["task_type"])
            expected_counts[task_type] += 1
            if task_type == "pairwise":
                left_code = contract["left_article_code"]
                right_code = contract["right_article_code"]
                preferred_code = result.get("preferred_article_code")
                if preferred_code not in {left_code, right_code}:
                    raise ValueError("Pairwise return selects an unassigned article")
                left = article_map[left_code]
                right = article_map[right_code]
                preferred = article_map[preferred_code]
                if left["topic_id"] != right["topic_id"]:
                    raise ValueError("Pairwise articles do not share a topic")
                preferences.append(
                    {
                        "topic_id": left["topic_id"],
                        "evaluator_id": evaluator_id,
                        "left_condition": left["condition"],
                        "right_condition": right["condition"],
                        "preferred_condition": preferred["condition"],
                        "evaluation_seconds": float(result["review_seconds"]),
                    }
                )
                continue
            article = article_map.get(contract["article_code"])
            if article is None:
                raise ValueError("Expert return references an unknown article code")
            expected_topic_code = _opaque_id(
                "TOP", article["topic_id"], seed=int(collection["seed"])
            )
            if contract["topic_code"] != expected_topic_code:
                raise ValueError("Expert return topic code does not match private topic")
            if task_type == "holistic":
                scores = {}
                for dimension in HOLISTIC_DIMENSIONS:
                    score = result.get(dimension)
                    if (
                        not isinstance(score, int)
                        or isinstance(score, bool)
                        or not 1 <= score <= 5
                    ):
                        raise ValueError("Expert holistic score is outside 1--5")
                    scores[dimension] = score
                role = evaluator_roles.get(evaluator_id)
                if role not in {"domain_expert", "methods_expert"}:
                    raise ValueError("Expert return lacks a registered evaluator role")
                holistic.append(
                    {
                        "topic_id": article["topic_id"],
                        "article_id": contract["article_code"],
                        "condition": article["condition"],
                        "evaluator_id": evaluator_id,
                        "evaluator_role": role,
                        **scores,
                        "evaluation_seconds": float(result["review_seconds"]),
                    }
                )
            elif task_type == "claim":
                cannot_assess = result.get("cannot_assess")
                labels = {
                    field: result.get(field)
                    for field in (
                        "supported",
                        "correct",
                        "overclaim",
                        "evidence_sufficient",
                    )
                }
                if not isinstance(cannot_assess, bool):
                    raise ValueError("Claim return lacks cannot_assess boolean")
                if cannot_assess and any(value is not None for value in labels.values()):
                    raise ValueError("Cannot-assess return includes substantive labels")
                if not cannot_assess and any(
                    not isinstance(value, bool) for value in labels.values()
                ):
                    raise ValueError("Assessable claim return lacks boolean labels")
                claims.append(
                    {
                        "topic_id": article["topic_id"],
                        "article_id": contract["article_code"],
                        "condition": article["condition"],
                        "claim_id": contract["claim_id"],
                        "evaluator_id": evaluator_id,
                        "stratum": result.get("stratum"),
                        **labels,
                        "cannot_assess": cannot_assess,
                        "evaluation_seconds": float(result["review_seconds"]),
                        "adjudication": False,
                    }
                )
            else:
                raise ValueError(f"Unknown expert task type: {task_type}")
    declared_counts = {
        key: int(value)
        for key, value in collection.get("task_counts", {}).items()
        if int(value) > 0
    }
    if dict(sorted(expected_counts.items())) != dict(sorted(declared_counts.items())):
        raise ValueError("Validated return counts differ from collection task counts")

    claim_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in claims:
        claim_groups.setdefault((row["article_id"], row["claim_id"]), []).append(row)
    disagreements = []
    for (article_code, claim_id), rows in sorted(claim_groups.items()):
        if len(rows) != 2 or len({row["evaluator_id"] for row in rows}) != 2:
            raise ValueError("Every claim requires two unique primary expert returns")
        outcomes = {
            (
                row["supported"],
                row["correct"],
                row["overclaim"],
                row["evidence_sufficient"],
                row["cannot_assess"],
            )
            for row in rows
        }
        if len(outcomes) > 1:
            disagreements.append(
                {
                    "article_code": article_code,
                    "claim_id": claim_id,
                    "topic_id": rows[0]["topic_id"],
                    "excluded_primary_evaluators": sorted(
                        row["evaluator_id"] for row in rows
                    ),
                }
            )

    result = {
        "schema_version": 1,
        "status": (
            "awaiting_blind_claim_adjudication"
            if disagreements
            else "analysis_input_ready_no_claim_disagreements"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "collection_manifest_sha256": sha256_file(collection_manifest_path),
        "packet_manifest_sha256": collection["packet_manifest_sha256"],
        "raw_return_sha256": raw_return_hashes,
        "primary_integrity": {
            "evaluators": len(expected_registry),
            "tasks": sum(expected_counts.values()),
            "task_counts": dict(sorted(expected_counts.items())),
            "claim_pairs": len(claim_groups),
            "claim_disagreements": len(disagreements),
            "all_registered_tasks_returned_once": True,
            "all_server_owned_fields_match": True,
            "all_timings_server_accounted": True,
        },
        "holistic_ratings": holistic,
        "claim_ratings": claims,
        "forced_pairwise_preferences": preferences,
        "adjudication_worklist": disagreements,
    }
    write_json(output_path, result)
    return result


def prepare_article_expert_adjudication(
    primary_validation_path: Path,
    primary_collection_manifest_path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Route real claim disagreements to a third, outcome-blind evaluator."""
    primary = read_json(primary_validation_path)
    collection = read_json(primary_collection_manifest_path)
    if primary.get("status") != "awaiting_blind_claim_adjudication":
        raise ValueError("Primary validation has no unresolved claim disagreements")
    if primary.get("collection_manifest_sha256") != sha256_file(
        primary_collection_manifest_path
    ):
        raise ValueError("Primary validation is not bound to this collection")
    packet_manifest_path = Path(collection["packet_manifest"])
    if sha256_file(packet_manifest_path) != collection["packet_manifest_sha256"]:
        raise ValueError("Private packet manifest changed after primary collection")
    packet_manifest = read_json(packet_manifest_path)
    seed = int(collection["seed"])

    eligible_by_topic: dict[str, set[str]] = {}
    for row in packet_manifest.get("assignments", []):
        eligible_by_topic.setdefault(str(row["topic_id"]), set()).add(
            str(row["evaluator_id"])
        )

    article_catalog = {}
    claim_catalog = {}
    topic_assets = {}
    for packet_record in collection["evaluator_packets"].values():
        packet = read_json(Path(packet_record["path"]))
        for topic in packet.get("topics", []):
            topic_code = str(topic["topic_code"])
            assets = {
                "figure_path": topic["figure_path"],
                "figure_sha256": topic["figure_sha256"],
            }
            previous_assets = topic_assets.setdefault(topic_code, assets)
            if previous_assets != assets:
                raise ValueError("Topic assets differ across evaluator packets")
            for article in topic.get("articles", []):
                article_code = str(article["article_code"])
                article_record = {
                    "article_code": article_code,
                    "article_path": article["article_path"],
                    "article_sha256": article["article_sha256"],
                }
                previous_article = article_catalog.setdefault(article_code, article_record)
                if previous_article != article_record:
                    raise ValueError("Blinded article differs across evaluator packets")
                for claim in article.get("claims", []):
                    key = (article_code, str(claim["claim_id"]))
                    previous_claim = claim_catalog.setdefault(key, claim)
                    if _canonical_sha(previous_claim) != _canonical_sha(claim):
                        raise ValueError("Blinded claim differs across evaluator packets")

    loads: Counter[tuple[str, str]] = Counter()
    routed = []
    for row in sorted(
        primary["adjudication_worklist"],
        key=lambda item: (item["topic_id"], item["article_code"], item["claim_id"]),
    ):
        topic_id = str(row["topic_id"])
        excluded = set(row["excluded_primary_evaluators"])
        candidates = sorted(eligible_by_topic.get(topic_id, set()) - excluded)
        if not candidates:
            raise ValueError(
                f"No independent third evaluator is available for {topic_id}/"
                f"{row['claim_id']}"
            )
        adjudicator = min(
            candidates,
            key=lambda value: (
                loads[(topic_id, value)],
                _canonical_sha(
                    {
                        "seed": seed,
                        "topic_id": topic_id,
                        "article_code": row["article_code"],
                        "claim_id": row["claim_id"],
                        "evaluator_id": value,
                    }
                ),
            ),
        )
        loads[(topic_id, adjudicator)] += 1
        article_code = str(row["article_code"])
        claim_id = str(row["claim_id"])
        claim = claim_catalog.get((article_code, claim_id))
        article = article_catalog.get(article_code)
        if claim is None or article is None:
            raise ValueError("Disputed claim is absent from frozen evaluator packets")
        topic_code = _opaque_id("TOP", topic_id, seed=seed)
        task = expert_task_contract(
            adjudicator,
            topic_code,
            "claim",
            article_code=article_code,
            claim_id=claim_id,
        )
        routed.append(
            {
                **task,
                "topic_id": topic_id,
                "evaluator_id": adjudicator,
                "article": article,
                "claim": claim,
                "excluded_primary_evaluators": sorted(excluded),
            }
        )

    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for row in routed:
        grouped.setdefault(row["evaluator_id"], {}).setdefault(
            row["topic_code"], {}
        ).setdefault(row["article"]["article_code"], []).append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_packets = {}
    task_registry = {}
    for evaluator_id, topic_groups in sorted(grouped.items()):
        topics = []
        contracts = []
        for topic_code, article_groups in sorted(topic_groups.items()):
            articles = []
            for article_code, rows in sorted(article_groups.items()):
                rows.sort(key=lambda item: item["claim"]["claim_id"])
                article = rows[0]["article"]
                articles.append(
                    {
                        **article,
                        "claims": [row["claim"] for row in rows],
                    }
                )
                contracts.extend(
                    {
                        key: row[key]
                        for key in (
                            "task_id",
                            "task_definition_sha256",
                            "evaluator_id",
                            "topic_code",
                            "task_type",
                            "article_code",
                            "claim_id",
                        )
                    }
                    for row in rows
                )
            topics.append(
                {
                    "topic_code": topic_code,
                    **topic_assets[topic_code],
                    "articles": articles,
                }
            )
        packet = {
            "schema_version": 1,
            "evaluator_id": evaluator_id,
            "hidden_fields_excluded": [
                "condition",
                "author_identity",
                "automatic_scores",
                "primary_evaluator_identity",
                "primary_answers",
            ],
            "topics": topics,
        }
        path = output_dir / "adjudicators" / f"{evaluator_id}.json"
        write_json(path, packet)
        evaluator_packets[evaluator_id] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        task_registry[evaluator_id] = contracts

    relevant_topic_codes = {row["topic_code"] for row in routed}
    public_manifest = {
        "schema_version": 1,
        "status": "expert_collection_ready",
        "collection_mode": "blind_claim_adjudication",
        "enabled_task_types": ["claim"],
        "condition_blind_collection": True,
        "packet_manifest": collection["packet_manifest"],
        "packet_manifest_sha256": collection["packet_manifest_sha256"],
        "intake_sha256": collection["intake_sha256"],
        "seed": seed,
        "evaluator_packets": evaluator_packets,
        "evidence_viewers": {
            code: collection["evidence_viewers"][code]
            for code in sorted(relevant_topic_codes)
        },
        "task_counts": {"claim": len(routed)},
        "task_registry": task_registry,
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "submission_contract": collection["submission_contract"],
    }
    public_path = output_dir / "collection_manifest.json"
    write_json(public_path, public_manifest)
    private_manifest = {
        "schema_version": 1,
        "status": "blind_claim_adjudication_ready",
        "primary_validation": str(primary_validation_path.resolve()),
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "assignments": [
            {
                key: row[key]
                for key in (
                    "task_id",
                    "task_definition_sha256",
                    "topic_id",
                    "topic_code",
                    "article_code",
                    "claim_id",
                    "evaluator_id",
                    "excluded_primary_evaluators",
                )
            }
            for row in routed
        ],
    }
    private_path = output_dir / "private_adjudication_manifest.json"
    write_json(private_path, private_manifest)
    public_manifest["private_adjudication_manifest"] = str(private_path.resolve())
    public_manifest["private_adjudication_manifest_sha256"] = sha256_file(private_path)
    write_json(public_path, public_manifest)
    return public_manifest


def finalize_article_expert_adjudication(
    primary_validation_path: Path,
    adjudication_collection_manifest_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Merge one independent blind adjudication into every disputed claim."""
    primary = read_json(primary_validation_path)
    collection = read_json(adjudication_collection_manifest_path)
    if primary.get("status") != "awaiting_blind_claim_adjudication":
        raise ValueError("Primary validation does not require adjudication")
    if collection.get("collection_mode") != "blind_claim_adjudication":
        raise ValueError("Collection is not a blind claim-adjudication panel")
    if collection.get("primary_validation_sha256") != sha256_file(
        primary_validation_path
    ):
        raise ValueError("Adjudication collection is not bound to primary validation")
    private_path = Path(collection["private_adjudication_manifest"])
    if sha256_file(private_path) != collection["private_adjudication_manifest_sha256"]:
        raise ValueError("Private adjudication assignments changed after routing")
    private = read_json(private_path)
    assignments = {row["task_id"]: row for row in private["assignments"]}
    if len(assignments) != len(private["assignments"]):
        raise ValueError("Duplicate private adjudication task identity")

    primary_claims = {
        (row["article_id"], row["claim_id"]): row
        for row in primary["claim_ratings"]
    }
    adjudications = []
    return_hashes = {}
    observed_task_ids = set()
    for evaluator_id, contracts in collection["task_registry"].items():
        path = adjudication_collection_manifest_path.parent / "returns" / f"{evaluator_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing adjudicator return: {evaluator_id}")
        payload = read_json(path)
        if payload.get("evaluator_id") != evaluator_id:
            raise ValueError("Adjudicator return identity mismatch")
        contract_by_id = {row["task_id"]: row for row in contracts}
        results = payload.get("results", [])
        if {row.get("task_id") for row in results} != set(contract_by_id):
            raise ValueError("Adjudicator return set is incomplete or foreign")
        return_hashes[evaluator_id] = sha256_file(path)
        for result in results:
            task_id = result["task_id"]
            if task_id in observed_task_ids:
                raise ValueError("Adjudication task was returned more than once")
            observed_task_ids.add(task_id)
            contract = contract_by_id[task_id]
            assignment = assignments.get(task_id)
            if assignment is None:
                raise ValueError("Adjudication result lacks a frozen private assignment")
            for field, expected in contract.items():
                if result.get(field) != expected:
                    raise ValueError("Adjudication server-owned field changed")
            if evaluator_id in assignment["excluded_primary_evaluators"]:
                raise ValueError("A primary evaluator adjudicated their own claim")
            if (
                result.get("timing_method")
                != "visibility_heartbeat_server_accounted"
                or not isinstance(result.get("review_seconds"), (int, float))
                or isinstance(result.get("review_seconds"), bool)
                or result["review_seconds"] <= 0
            ):
                raise ValueError("Adjudication lacks valid server-accounted timing")
            primary_row = primary_claims.get(
                (assignment["article_code"], assignment["claim_id"])
            )
            if primary_row is None:
                raise ValueError("Adjudication references an unknown primary claim")
            cannot_assess = result.get("cannot_assess")
            labels = {
                field: result.get(field)
                for field in (
                    "supported",
                    "correct",
                    "overclaim",
                    "evidence_sufficient",
                )
            }
            if cannot_assess is True and any(
                value is not None for value in labels.values()
            ):
                raise ValueError("Cannot-assess adjudication includes substantive labels")
            if cannot_assess is not True and any(
                not isinstance(value, bool) for value in labels.values()
            ):
                raise ValueError("Assessable adjudication lacks boolean labels")
            adjudications.append(
                {
                    "topic_id": primary_row["topic_id"],
                    "article_id": primary_row["article_id"],
                    "condition": primary_row["condition"],
                    "claim_id": primary_row["claim_id"],
                    "evaluator_id": evaluator_id,
                    "stratum": primary_row["stratum"],
                    **labels,
                    "cannot_assess": bool(cannot_assess),
                    "evaluation_seconds": float(result["review_seconds"]),
                    "adjudication": True,
                }
            )
    if observed_task_ids != set(assignments):
        raise ValueError("Not every frozen dispute received one adjudication")

    result = {
        "schema_version": 1,
        "status": "analysis_input_ready_after_blind_adjudication",
        "protocol_sha256": primary["protocol_sha256"],
        "primary_validation_sha256": sha256_file(primary_validation_path),
        "adjudication_collection_sha256": sha256_file(
            adjudication_collection_manifest_path
        ),
        "adjudicator_return_sha256": return_hashes,
        "holistic_ratings": primary["holistic_ratings"],
        "claim_ratings": [*primary["claim_ratings"], *adjudications],
        "forced_pairwise_preferences": primary["forced_pairwise_preferences"],
        "adjudication_integrity": {
            "disputes": len(assignments),
            "adjudications": len(adjudications),
            "one_independent_adjudicator_per_dispute": True,
            "primary_answers_hidden_from_adjudicator": True,
            "all_timings_server_accounted": True,
        },
    }
    write_json(output_path, result)
    return result
