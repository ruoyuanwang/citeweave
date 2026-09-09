from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from .io import read_json, sha256_file, write_json
from .review_ui import (
    REVIEW_HTML,
    Heartbeat,
    ReviewLayer,
    ReviewStore,
    ReviewSubmission,
)
from .review_voi import SEVERITY_WEIGHT, _claim_prior

ReviewerQuery = Annotated[str, Query()]
LayerQuery = Annotated[ReviewLayer, Query()]
ReviewToken = Annotated[str | None, Header()]
GENESIS_EVENT_SHA256 = "0" * 64


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class LiveSequentialVoiRouter:
    """Outcome-adaptive, reviewer-private routing for development deployments.

    The router is deliberately excluded from confirmatory label collection.  It uses
    only outcomes already submitted by the same reviewer, retains a hash-chained
    selection trace, and separates skipped unreviewed descendants from previously
    reviewed claims that must be reopened after an upstream dependency fails.
    """

    def __init__(self, config: dict[str, Any], manifest: dict[str, Any]) -> None:
        if config.get("status") != "frozen_live_sequential_voi_development":
            raise ValueError("Live VOI config is not in its frozen development state")
        if config.get("development_only") is not True:
            raise ValueError("Live VOI routing must be marked development-only")
        if config.get("formal_outcome_collection") is not False:
            raise ValueError("Live VOI routing cannot collect formal outcomes")
        if config.get("human_outcomes_before_freeze") != 0:
            raise ValueError("Live VOI config was not frozen before human outcomes")
        if config.get("policy") != "sequential_dependency_value_of_information":
            raise ValueError("Unexpected live review policy")
        self.seed = int(config.get("seed", 20260827))
        features = config.get("features") or []
        self.features = {str(row["packet_id"]): row for row in features}
        if not self.features or len(self.features) != len(features):
            raise ValueError("Live VOI features must be nonempty and unique")
        for packet_id, row in self.features.items():
            if not row.get("routing_scope_id") or not row.get("dependency_ids"):
                raise ValueError(f"Incomplete live VOI feature: {packet_id}")
            if row.get("severity") not in SEVERITY_WEIGHT:
                raise ValueError(f"Invalid live VOI severity: {packet_id}")
        self.estimated_seconds = {
            str(key): float(value)
            for key, value in (config.get("estimated_review_seconds") or {}).items()
        }
        if set(self.estimated_seconds) != set(self.features) or any(
            value <= 0 or not math.isfinite(value)
            for value in self.estimated_seconds.values()
        ):
            raise ValueError("Every live VOI packet needs a positive frozen time estimate")
        self.budgets = {
            str(key): float(value)
            for key, value in (config.get("reviewer_budget_seconds") or {}).items()
        }
        article_reviewers = {
            reviewer
            for reviewer, layers in manifest["assignments"].items()
            if layers.get("article")
        }
        if set(self.budgets) != article_reviewers or any(
            value <= 0 or not math.isfinite(value) for value in self.budgets.values()
        ):
            raise ValueError("Every article reviewer needs one positive frozen VOI budget")
        for reviewer in article_reviewers:
            assigned = set(manifest["assignments"][reviewer]["article"])
            missing = assigned - set(self.features)
            if missing:
                raise ValueError(f"Article assignment lacks live VOI features: {sorted(missing)}")

        self.dependency_claims: dict[tuple[str, str], set[str]] = defaultdict(set)
        for packet_id, row in self.features.items():
            scope = str(row["routing_scope_id"])
            for dependency_id in row["dependency_ids"]:
                self.dependency_claims[(scope, str(dependency_id))].add(packet_id)

    def _score(
        self,
        packet_id: str,
        *,
        posterior: dict[tuple[str, str], list[float]],
        unresolved: set[str],
    ) -> tuple[float, float, str]:
        row = self.features[packet_id]
        scope = str(row["routing_scope_id"])
        expected_propagation = 0.0
        for dependency_id in row["dependency_ids"]:
            key = (scope, str(dependency_id))
            alpha, beta = posterior[key]
            new_reach = len(self.dependency_claims[key] & unresolved)
            expected_propagation += alpha / (alpha + beta) * new_reach
        utility = SEVERITY_WEIGHT[row["severity"]] * (
            _claim_prior(row) + expected_propagation
        )
        score = utility / self.estimated_seconds[packet_id]
        tie = hashlib.sha256(f"{self.seed}\x1f{packet_id}".encode()).hexdigest()
        return score, expected_propagation, tie

    def _select(
        self,
        assigned: set[str],
        *,
        reviewed: set[str],
        propagated_unreviewed: set[str],
        predicted_seconds: float,
        budget_seconds: float,
        posterior: dict[tuple[str, str], list[float]],
    ) -> dict[str, Any] | None:
        unresolved = set(self.features) - reviewed - propagated_unreviewed
        candidates = assigned - reviewed - propagated_unreviewed
        affordable = {
            packet_id
            for packet_id in candidates
            if predicted_seconds + self.estimated_seconds[packet_id] <= budget_seconds
        }
        if not affordable:
            return None
        def ordering_key(value: str) -> tuple[float, str]:
            score, _, tie = self._score(
                value, posterior=posterior, unresolved=unresolved
            )
            return score, tie

        packet_id = max(
            affordable,
            key=ordering_key,
        )
        score, expected_propagation, _ = self._score(
            packet_id, posterior=posterior, unresolved=unresolved
        )
        return {
            "packet_id": packet_id,
            "score_before_reveal": score,
            "expected_marginal_downstream_coverage": expected_propagation,
            "estimated_review_seconds": self.estimated_seconds[packet_id],
            "severity": self.features[packet_id]["severity"],
        }

    def snapshot(
        self,
        reviewer: str,
        assigned_packet_ids: list[str],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assigned = set(assigned_packet_ids)
        budget = self.budgets[reviewer]
        posterior = {key: [1.0, 4.0] for key in self.dependency_claims}
        reviewed: set[str] = set()
        propagated_unreviewed: set[str] = set()
        revision_targets: set[str] = set()
        reopened_reviewed: set[str] = set()
        predicted_seconds = 0.0
        actual_seconds = 0.0
        prior_event_sha256 = GENESIS_EVENT_SHA256
        trace = []

        for step, result in enumerate(results, start=1):
            packet_id = str(result["packet_id"])
            selection = self._select(
                assigned,
                reviewed=reviewed,
                propagated_unreviewed=propagated_unreviewed,
                predicted_seconds=predicted_seconds,
                budget_seconds=budget,
                posterior=posterior,
            )
            if selection is None or selection["packet_id"] != packet_id:
                raise ValueError("Submitted article return order violates frozen live VOI routing")
            row = self.features[packet_id]
            scope = str(row["routing_scope_id"])
            allowed = {str(value) for value in row["dependency_ids"]}
            invalid = {str(value) for value in result.get("invalid_dependency_ids") or []}
            if not invalid <= allowed:
                raise ValueError(f"Live VOI outcome cites an invisible dependency: {packet_id}")
            actual = float(result.get("review_seconds", 0))
            if actual <= 0 or not math.isfinite(actual):
                raise ValueError(f"Invalid server review time in live VOI return: {packet_id}")

            reviewed_before = set(reviewed)
            reviewed.add(packet_id)
            newly_propagated: set[str] = set()
            newly_reopened: set[str] = set()
            for dependency_id in allowed:
                key = (scope, dependency_id)
                alpha, beta = posterior[key]
                if dependency_id in invalid:
                    posterior[key] = [alpha + 1, beta]
                    affected = self.dependency_claims[key] - {packet_id}
                    newly_reopened.update(affected & reviewed_before)
                    newly_propagated.update(affected - reviewed)
                    revision_targets.update(affected | {packet_id})
                elif result.get("evidence_sufficient") is True:
                    posterior[key] = [alpha, beta + 1]
            newly_propagated -= propagated_unreviewed
            newly_reopened -= reopened_reviewed
            propagated_unreviewed.update(newly_propagated)
            reopened_reviewed.update(newly_reopened)
            predicted_seconds += self.estimated_seconds[packet_id]
            actual_seconds += actual
            event_core = {
                "step": step,
                **selection,
                "actual_review_seconds": actual,
                "action": result.get("action"),
                "invalid_dependency_ids": sorted(invalid),
                "newly_propagated_unreviewed_claim_ids": sorted(newly_propagated),
                "newly_reopened_reviewed_claim_ids": sorted(newly_reopened),
                "prior_event_sha256": prior_event_sha256,
            }
            event = {**event_core, "event_sha256": _canonical_sha256(event_core)}
            recorded_event = result.get("routing_event")
            if recorded_event != event:
                raise ValueError("Live VOI routing event hash or content mismatch")
            trace.append(event)
            prior_event_sha256 = event["event_sha256"]

        next_selection = self._select(
            assigned,
            reviewed=reviewed,
            propagated_unreviewed=propagated_unreviewed,
            predicted_seconds=predicted_seconds,
            budget_seconds=budget,
            posterior=posterior,
        )
        unreviewed = assigned - reviewed - propagated_unreviewed
        status = (
            "next_packet_ready"
            if next_selection
            else "budget_exhausted"
            if unreviewed
            else "routing_complete"
        )
        return {
            "status": status,
            "reviewer_code": reviewer,
            "budget_seconds": budget,
            "predicted_seconds_scheduled": predicted_seconds,
            "actual_review_seconds": actual_seconds,
            "directly_reviewed_packet_ids": sorted(reviewed),
            "propagated_unreviewed_packet_ids": sorted(propagated_unreviewed & assigned),
            "reopened_reviewed_packet_ids": sorted(reopened_reviewed & assigned),
            "revision_target_packet_ids": sorted(revision_targets),
            "remaining_reviewable_packet_ids": sorted(unreviewed),
            "next_selection": next_selection,
            "last_event_sha256": prior_event_sha256,
            "trace": trace,
        }

    def build_event(
        self,
        snapshot: dict[str, Any],
        *,
        packet_id: str,
        answers: dict[str, Any],
        actual_review_seconds: float,
    ) -> dict[str, Any]:
        selection = snapshot.get("next_selection")
        if not selection or selection["packet_id"] != packet_id:
            raise ValueError("Submission is not the current live VOI selection")
        reviewed = set(snapshot["directly_reviewed_packet_ids"])
        propagated = set(snapshot["propagated_unreviewed_packet_ids"])
        row = self.features[packet_id]
        scope = str(row["routing_scope_id"])
        invalid = {str(value) for value in answers.get("invalid_dependency_ids") or []}
        allowed = {str(value) for value in row["dependency_ids"]}
        if not invalid <= allowed:
            raise ValueError(f"Live VOI outcome cites an invisible dependency: {packet_id}")
        newly_propagated: set[str] = set()
        newly_reopened: set[str] = set()
        for dependency_id in invalid:
            affected = self.dependency_claims[(scope, dependency_id)] - {packet_id}
            newly_reopened.update(affected & reviewed)
            newly_propagated.update(affected - reviewed - propagated)
        event_core = {
            "step": len(snapshot["trace"]) + 1,
            **selection,
            "actual_review_seconds": actual_review_seconds,
            "action": answers.get("action"),
            "invalid_dependency_ids": sorted(invalid),
            "newly_propagated_unreviewed_claim_ids": sorted(newly_propagated),
            "newly_reopened_reviewed_claim_ids": sorted(newly_reopened),
            "prior_event_sha256": snapshot["last_event_sha256"],
        }
        return {**event_core, "event_sha256": _canonical_sha256(event_core)}


class LiveVoiReviewStore(ReviewStore):
    def __init__(self, packet_root: Path, access_path: Path) -> None:
        super().__init__(packet_root, access_path)
        config_path = self.packet_root / "live_voi_config.json"
        freeze_path = self.packet_root / "live_voi_config_freeze.json"
        if not config_path.is_file() or not freeze_path.is_file():
            raise ValueError("Live VOI review requires a config and adjacent freeze")
        freeze = read_json(freeze_path)
        if freeze.get("sha256") != sha256_file(config_path):
            raise ValueError("Live VOI config differs from its pre-outcome freeze")
        self.router = LiveSequentialVoiRouter(read_json(config_path), self.manifest)

    def _article_snapshot(self, reviewer: str) -> dict[str, Any]:
        results = [
            row
            for row in self._load_return(reviewer)["results"]
            if row.get("review_layer") == "article"
        ]
        return self.router.snapshot(
            reviewer,
            self.manifest["assignments"][reviewer]["article"],
            results,
        )

    def progress(self, reviewer: str) -> dict[str, Any]:
        progress = super().progress(reviewer)
        if self.manifest["assignments"][reviewer].get("article"):
            snapshot = self._article_snapshot(reviewer)
            progress["layers"]["article"] = {
                "completed": len(snapshot["directly_reviewed_packet_ids"]),
                "propagated": len(snapshot["propagated_unreviewed_packet_ids"]),
                "reopened": len(snapshot["reopened_reviewed_packet_ids"]),
                "remaining": len(snapshot["remaining_reviewable_packet_ids"]),
                "total": len(self.manifest["assignments"][reviewer]["article"]),
                "routing_status": snapshot["status"],
                "predicted_seconds_scheduled": snapshot["predicted_seconds_scheduled"],
                "budget_seconds": snapshot["budget_seconds"],
            }
        return progress

    def next_packet(self, reviewer: str, layer: ReviewLayer) -> dict[str, Any]:
        if layer != "article":
            return super().next_packet(reviewer, layer)
        with self.lock:
            returned = self._load_return(reviewer)
            completed = {result["packet_id"] for result in returned["results"]}
            state = self._load_state(reviewer)
            active = state["active"].get(layer)
            if active and active["packet_id"] not in completed:
                packet_id = active["packet_id"]
                routing = active["routing"]
            else:
                snapshot = self._article_snapshot(reviewer)
                routing = snapshot["next_selection"]
                if routing is None:
                    return {
                        "complete": True,
                        "completion_reason": snapshot["status"],
                        "revision_target_packet_ids": snapshot["revision_target_packet_ids"],
                        "progress": self.progress(reviewer),
                    }
                packet_id = routing["packet_id"]
                now = time.time()
                active = {
                    "packet_id": packet_id,
                    "session_id": secrets.token_urlsafe(24),
                    "started_at": now,
                    "last_heartbeat": now,
                    "active_seconds": 0.0,
                    "routing": routing,
                }
                state["active"][layer] = active
                write_json(self._state_path(reviewer), state)
            packet = read_json(self.packet_root / "packets" / layer / f"{packet_id}.json")
            packet.setdefault("packet_id", packet_id)
            return {
                "complete": False,
                "layer": layer,
                "session_id": active["session_id"],
                "packet": packet,
                "routing": {
                    "policy": "sequential_dependency_value_of_information",
                    **routing,
                },
                "progress": self.progress(reviewer),
            }

    def submit(
        self,
        reviewer: str,
        layer: ReviewLayer,
        payload: ReviewSubmission,
    ) -> dict[str, Any]:
        if layer != "article":
            return super().submit(reviewer, layer, payload)
        with self.lock:
            state = self._load_state(reviewer)
            active = state["active"].get(layer)
            self._require_active(active, payload.packet_id, payload.session_id)
            assigned = set(self.manifest["assignments"][reviewer][layer])
            if payload.packet_id not in assigned:
                raise ValueError("Packet is not assigned to this reviewer")
            returned = self._load_return(reviewer)
            if any(result["packet_id"] == payload.packet_id for result in returned["results"]):
                raise ValueError("Packet was already submitted and is frozen")
            packet = read_json(
                self.packet_root / "packets" / layer / f"{payload.packet_id}.json"
            )
            if {"routing_event", "review_layer"} & payload.answers.keys():
                raise ValueError("Answers cannot override server-owned live routing fields")
            answers = self._validate_answers(layer, dict(payload.answers), packet)
            snapshot = self._article_snapshot(reviewer)
            self._touch(active)
            review_seconds = round(max(0.001, float(active["active_seconds"])), 3)
            routing_event = self.router.build_event(
                snapshot,
                packet_id=payload.packet_id,
                answers=answers,
                actual_review_seconds=review_seconds,
            )
            result = {
                "packet_id": payload.packet_id,
                "reviewer_code": reviewer,
                "review_layer": "article",
                **answers,
                "review_seconds": review_seconds,
                "server_elapsed_seconds": round(
                    max(0.001, time.time() - float(active["started_at"])), 3
                ),
                "timing_method": "visibility_heartbeat_server_accounted",
                "submitted_at_unix": time.time(),
                "routing_event": routing_event,
            }
            candidate_results = [
                row
                for row in returned["results"]
                if row.get("review_layer") == "article"
            ] + [result]
            self.router.snapshot(reviewer, list(assigned), candidate_results)
            returned["results"].append(result)
            write_json(self._return_path(reviewer), returned)
            del state["active"][layer]
            write_json(self._state_path(reviewer), state)
            return {
                "accepted": True,
                "result": result,
                "progress": self.progress(reviewer),
            }


def create_live_voi_review_app(packet_root: Path, access_path: Path) -> FastAPI:
    store = LiveVoiReviewStore(packet_root, access_path)
    app = FastAPI(title="CiteWeave Live Sequential VOI Review", version="3.0-dev")

    def auth(reviewer: str, token: str | None) -> None:
        try:
            store.authorize(reviewer, token or "")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return REVIEW_HTML

    @app.get("/api/progress")
    def progress(reviewer: ReviewerQuery, x_review_token: ReviewToken = None) -> dict[str, Any]:
        auth(reviewer, x_review_token)
        return store.progress(reviewer)

    @app.get("/api/next")
    def next_packet(
        reviewer: ReviewerQuery,
        layer: LayerQuery,
        x_review_token: ReviewToken = None,
    ) -> dict[str, Any]:
        auth(reviewer, x_review_token)
        return store.next_packet(reviewer, layer)

    @app.post("/api/heartbeat")
    def heartbeat(
        payload: Heartbeat,
        reviewer: ReviewerQuery,
        layer: LayerQuery,
        x_review_token: ReviewToken = None,
    ) -> dict[str, Any]:
        auth(reviewer, x_review_token)
        try:
            return store.heartbeat(reviewer, layer, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/submit")
    def submit(
        payload: ReviewSubmission,
        reviewer: ReviewerQuery,
        layer: LayerQuery,
        x_review_token: ReviewToken = None,
    ) -> dict[str, Any]:
        auth(reviewer, x_review_token)
        try:
            return store.submit(reviewer, layer, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
