from __future__ import annotations

import hashlib
import secrets
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .io import read_json, write_json

ReviewLayer = Literal[
    "factual", "semantic", "adversarial", "source", "calibration", "article"
]
ReviewerQuery = Annotated[str, Query()]
LayerQuery = Annotated[ReviewLayer, Query()]
ReviewToken = Annotated[str | None, Header()]


class ReviewSubmission(BaseModel):
    packet_id: str
    session_id: str
    answers: dict[str, Any]


class Heartbeat(BaseModel):
    packet_id: str
    session_id: str


class ReviewStore:
    """Blind packet delivery with server-accounted active review time."""

    def __init__(self, packet_root: Path, access_path: Path) -> None:
        self.packet_root = packet_root.resolve()
        self.manifest = read_json(self.packet_root / "internal_manifest.json")
        self.access = read_json(access_path.resolve())["reviewer_token_sha256"]
        self.lock = threading.RLock()

    def authorize(self, reviewer: str, token: str) -> None:
        expected = self.access.get(reviewer)
        observed = hashlib.sha256(token.encode()).hexdigest()
        if not expected or not secrets.compare_digest(expected, observed):
            raise PermissionError("Invalid reviewer credential")
        if reviewer not in self.manifest["assignments"]:
            raise PermissionError("Reviewer has no assignment")

    def _return_path(self, reviewer: str) -> Path:
        return self.packet_root / "returns" / f"{reviewer}.json"

    def _state_path(self, reviewer: str) -> Path:
        return self.packet_root / "state" / f"{reviewer}.json"

    def _load_return(self, reviewer: str) -> dict[str, Any]:
        path = self._return_path(reviewer)
        if path.is_file():
            return read_json(path)
        return {"schema_version": 1, "reviewer_code": reviewer, "results": []}

    def _load_state(self, reviewer: str) -> dict[str, Any]:
        path = self._state_path(reviewer)
        if path.is_file():
            return read_json(path)
        return {"schema_version": 1, "reviewer_code": reviewer, "active": {}}

    def progress(self, reviewer: str) -> dict[str, Any]:
        completed = {result["packet_id"] for result in self._load_return(reviewer)["results"]}
        layers = {}
        for layer, assigned in self.manifest["assignments"][reviewer].items():
            done = sum(packet_id in completed for packet_id in assigned)
            layers[layer] = {
                "completed": done,
                "total": len(assigned),
                "remaining": len(assigned) - done,
            }
        return {"reviewer_code": reviewer, "layers": layers}

    def next_packet(self, reviewer: str, layer: ReviewLayer) -> dict[str, Any]:
        with self.lock:
            returned = self._load_return(reviewer)
            completed = {result["packet_id"] for result in returned["results"]}
            assigned = self.manifest["assignments"][reviewer][layer]
            state = self._load_state(reviewer)
            active = state["active"].get(layer)
            if active and active["packet_id"] not in completed:
                packet_id = active["packet_id"]
            else:
                packet_id = next(
                    (value for value in assigned if value not in completed),
                    None,
                )
                if packet_id is None:
                    return {"complete": True, "progress": self.progress(reviewer)}
                now = time.time()
                active = {
                    "packet_id": packet_id,
                    "session_id": secrets.token_urlsafe(24),
                    "started_at": now,
                    "last_heartbeat": now,
                    "active_seconds": 0.0,
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
                "progress": self.progress(reviewer),
            }

    @staticmethod
    def _touch(active: dict[str, Any]) -> None:
        now = time.time()
        elapsed = max(0.0, now - float(active["last_heartbeat"]))
        active["active_seconds"] = float(active["active_seconds"]) + min(elapsed, 10.0)
        active["last_heartbeat"] = now

    def heartbeat(self, reviewer: str, layer: ReviewLayer, payload: Heartbeat) -> dict[str, Any]:
        with self.lock:
            state = self._load_state(reviewer)
            active = state["active"].get(layer)
            self._require_active(active, payload.packet_id, payload.session_id)
            self._touch(active)
            write_json(self._state_path(reviewer), state)
            return {"active_seconds": round(float(active["active_seconds"]), 3)}

    @staticmethod
    def _require_active(active: dict[str, Any] | None, packet_id: str, session_id: str) -> None:
        if (
            not active
            or active.get("packet_id") != packet_id
            or active.get("session_id") != session_id
        ):
            raise ValueError("Packet session is absent, stale, or mismatched")

    @staticmethod
    def _validate_answers(
        layer: ReviewLayer,
        answers: dict[str, Any],
        packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reserved = {
            "packet_id",
            "reviewer_code",
            "session_id",
            "review_seconds",
            "server_elapsed_seconds",
            "timing_method",
            "submitted_at_unix",
        }
        if reserved & answers.keys():
            raise ValueError("Answers cannot override server-owned identity or timing")
        rationale = answers.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("A nonempty rationale is required")
        if layer == "factual":
            for field in ("answer_correct", "evidence_sufficient"):
                if not isinstance(answers.get(field), bool):
                    raise TypeError(f"{field} must be boolean")
            if answers.get("action") not in {
                "accept",
                "rewrite",
                "reject_evidence",
                "abstain",
            }:
                raise ValueError("Invalid factual action")
            if not isinstance(answers.get("unsupported_fields", []), list):
                raise ValueError("unsupported_fields must be a list")
        elif layer == "semantic":
            for field in ("calibrated", "alternative_adequate", "limitation_adequate"):
                if not isinstance(answers.get(field), bool):
                    raise TypeError(f"{field} must be boolean")
            score = answers.get("phenomenon_value")
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
                raise ValueError("phenomenon_value must be an integer from 1 to 5")
            if answers.get("action") not in {
                "accept",
                "rewrite",
                "reject_claim",
                "abstain",
            }:
                raise ValueError("Invalid semantic action")
            if not isinstance(answers.get("guard", {}), dict):
                raise ValueError("guard must be an object")
            if (
                answers.get("action") == "rewrite"
                and not str(answers.get("replacement") or "").strip()
            ):
                raise ValueError("Rewrite requires replacement text")
        elif layer == "adversarial":
            if answers.get("verdict") not in {
                "supported",
                "qualify",
                "reject",
                "abstain",
            }:
                raise ValueError("Invalid adversarial verdict")
            decisive = answers.get("decisive_evidence_ids")
            if not isinstance(decisive, list) or not all(
                isinstance(value, str) and value.strip() for value in decisive
            ):
                raise ValueError("decisive_evidence_ids must be a nonempty-string list")
            if len(decisive) != len(set(decisive)):
                raise ValueError("decisive_evidence_ids must be unique")
            if answers["verdict"] != "abstain" and not decisive:
                raise ValueError("A non-abstain verdict requires decisive evidence")
            steps = answers.get("invalid_operator_steps")
            if (
                not isinstance(steps, list)
                or any(not isinstance(value, int) or isinstance(value, bool) for value in steps)
                or any(value < 0 for value in steps)
                or len(steps) != len(set(steps))
            ):
                raise ValueError("invalid_operator_steps must contain unique nonnegative integers")
            failure_mode = answers.get("failure_mode")
            if not isinstance(failure_mode, str) or not failure_mode.strip():
                raise ValueError("A nonempty failure_mode is required")
            rewrite = answers.get("minimal_rewrite")
            if rewrite is not None and not isinstance(rewrite, str):
                raise TypeError("minimal_rewrite must be a string or null")
            if answers["verdict"] == "qualify" and not str(rewrite or "").strip():
                raise ValueError("A qualify verdict requires a minimal rewrite")
            packet = packet or {}
            valid_evidence = {
                str(row.get("evidence_id"))
                for field in ("evidence_set_a", "evidence_set_b")
                for row in packet.get(field, [])
                if isinstance(row, dict) and row.get("evidence_id")
            }
            unknown = sorted(set(decisive) - valid_evidence)
            if unknown:
                raise ValueError(f"Unknown decisive evidence IDs: {unknown}")
            trace_size = len(packet.get("operator_trace", []))
            if any(value >= trace_size for value in steps):
                raise ValueError("invalid_operator_steps contains an out-of-range index")
        elif layer == "source":
            if answers.get("topic_relevance") not in {
                "direct",
                "contextual",
                "irrelevant",
                "cannot_assess",
            }:
                raise ValueError("Invalid source topic relevance")
            if answers.get("evidence_role") not in {
                "supports_interpretation",
                "context_only",
                "challenges",
                "irrelevant",
                "cannot_assess",
            }:
                raise ValueError("Invalid source evidence role")
            failure_type = answers.get("failure_type")
            if failure_type not in {
                "none",
                "generic_graph_anchor",
                "query_too_broad",
                "off_topic",
                "insufficient_detail",
                "conflicting_evidence",
                "other",
            }:
                raise ValueError("Invalid source failure type")
            span = answers.get("decisive_span")
            if span is not None and not isinstance(span, str):
                raise TypeError("decisive_span must be a string or null")
            if span:
                source_text = " ".join(
                    (
                        str((packet or {}).get("source", {}).get("title") or ""),
                        str((packet or {}).get("source", {}).get("abstract_excerpt") or ""),
                    )
                )
                if span not in source_text:
                    raise ValueError("decisive_span must be an exact visible source span")
            if (
                answers["topic_relevance"] != "cannot_assess"
                and answers["evidence_role"] != "cannot_assess"
                and not str(span or "").strip()
            ):
                raise ValueError("A decisive source span is required")
            terms = answers.get("suggested_query_terms")
            if not isinstance(terms, list) or any(
                not isinstance(value, str) or not value.strip() for value in terms
            ):
                raise ValueError("suggested_query_terms must be a string list")
            if len(terms) != len(set(terms)):
                raise ValueError("suggested_query_terms must be unique")
        elif layer == "calibration":
            if answers.get("verdict") not in {"valid", "invalid", "abstain"}:
                raise ValueError("Invalid calibration verdict")
            error_type = answers.get("error_type")
            if error_type is not None and not isinstance(error_type, str):
                raise TypeError("error_type must be a string or null")
            decisive = answers.get("decisive_evidence_ids")
            if not isinstance(decisive, list) or any(
                not isinstance(value, str) or not value.strip() for value in decisive
            ):
                raise ValueError("decisive_evidence_ids must be a string list")
            if len(decisive) != len(set(decisive)):
                raise ValueError("decisive_evidence_ids must be unique")
        else:
            for field in (
                "factual_supported",
                "interpretation_calibrated",
                "alternative_adequate",
                "evidence_sufficient",
            ):
                if not isinstance(answers.get(field), bool):
                    raise TypeError(f"{field} must be boolean")
            action = answers.get("action")
            if action not in {"accept", "rewrite", "reject_claim", "abstain"}:
                raise ValueError("Invalid article claim action")
            decisive = answers.get("decisive_evidence_ids")
            invalid_dependencies = answers.get("invalid_dependency_ids")
            for field, values in (
                ("decisive_evidence_ids", decisive),
                ("invalid_dependency_ids", invalid_dependencies),
            ):
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or not value.strip() for value in values
                ):
                    raise ValueError(f"{field} must be a nonempty-string list")
                if len(values) != len(set(values)):
                    raise ValueError(f"{field} must be unique")
            if action != "abstain" and not decisive:
                raise ValueError("A non-abstain article judgment requires evidence")
            allowed = set((packet or {}).get("allowed_decisive_evidence_ids", []))
            unknown = sorted((set(decisive) | set(invalid_dependencies)) - allowed)
            if unknown:
                raise ValueError(f"Unknown article evidence IDs: {unknown}")
            if not isinstance(answers.get("guard", {}), dict):
                raise ValueError("guard must be an object")
            judgments = [
                answers[field]
                for field in (
                    "factual_supported",
                    "interpretation_calibrated",
                    "alternative_adequate",
                    "evidence_sufficient",
                )
            ]
            if action == "accept" and (not all(judgments) or invalid_dependencies):
                raise ValueError(
                    "Accept requires all article checks to pass and no invalid dependencies"
                )
            if (
                action in {"rewrite", "reject_claim"}
                and all(judgments)
                and not invalid_dependencies
            ):
                raise ValueError("A corrective article action requires a recorded defect")
            if action == "rewrite" and not str(answers.get("replacement") or "").strip():
                raise ValueError("Rewrite requires replacement text")
        return answers

    def submit(
        self,
        reviewer: str,
        layer: ReviewLayer,
        payload: ReviewSubmission,
    ) -> dict[str, Any]:
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
            packet = read_json(self.packet_root / "packets" / layer / f"{payload.packet_id}.json")
            answers = self._validate_answers(layer, dict(payload.answers), packet)
            self._touch(active)
            result = {
                "packet_id": payload.packet_id,
                "reviewer_code": reviewer,
                **answers,
                "review_seconds": round(max(0.001, float(active["active_seconds"])), 3),
                "server_elapsed_seconds": round(
                    max(0.001, time.time() - float(active["started_at"])), 3
                ),
                "timing_method": "visibility_heartbeat_server_accounted",
                "submitted_at_unix": time.time(),
            }
            returned["results"].append(result)
            write_json(self._return_path(reviewer), returned)
            del state["active"][layer]
            write_json(self._state_path(reviewer), state)
            return {"accepted": True, "result": result, "progress": self.progress(reviewer)}


def create_review_app(packet_root: Path, access_path: Path) -> FastAPI:
    store = ReviewStore(packet_root, access_path)
    app = FastAPI(title="CiteWeave Blinded Human Review", version="2.0")

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


REVIEW_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CiteWeave Blinded Review</title><style>
body{font:15px system-ui;margin:0;background:#f4f6fa;color:#172033}main{max-width:1100px;margin:auto;padding:24px}
.bar,.card{background:white;border:1px solid #dce2ed;border-radius:12px;padding:18px;margin-bottom:16px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input,select,textarea,button{font:inherit;padding:8px}
textarea{width:100%;min-height:84px;box-sizing:border-box}button{background:#3157d5;color:white;border:0;border-radius:7px}
pre{white-space:pre-wrap;max-height:420px;overflow:auto;background:#f7f8fb;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
label{display:block;margin:8px 0}.muted{color:#657087}.error{color:#b42318}@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>CiteWeave Blinded Human Review</h1>
<div class="bar"><input id="reviewer" placeholder="Reviewer code"><input id="token" type="password" placeholder="Access token">
<select id="layer"><option value="factual">Factual</option><option value="semantic">Semantic</option><option value="adversarial">Adversarial graph audit</option><option value="source">Source relevance</option><option value="calibration">Reviewer calibration</option><option value="article">Article claim audit</option></select>
<button onclick="loadNext()">Load next</button><span id="progress" class="muted"></span></div>
<div id="status" class="error"></div><section id="workspace" hidden>
<div class="card"><h2 id="question"></h2><div class="grid"><div><h3>Candidate</h3><pre id="candidate"></pre></div>
<div><h3>Visible evidence</h3><pre id="evidence"></pre></div></div></div><div class="card" id="form"></div></section>
</main><script>
let current=null, heartbeat=null; const $=id=>document.getElementById(id);
function headers(){return {'X-Review-Token':$('token').value,'Content-Type':'application/json'}}
function query(){return `reviewer=${encodeURIComponent($('reviewer').value)}&layer=${$('layer').value}`}
async function loadNext(){clearInterval(heartbeat);$('status').textContent='';let r=await fetch('/api/next?'+query(),{headers:headers()});
if(!r.ok){$('status').textContent=(await r.json()).detail;return} current=await r.json();
if(current.complete){$('workspace').hidden=true;$('progress').textContent='Layer complete';return} render(current);
heartbeat=setInterval(()=>{if(!document.hidden)fetch('/api/heartbeat?'+query(),{method:'POST',headers:headers(),body:JSON.stringify({packet_id:current.packet.packet_id,session_id:current.session_id})})},5000)}
function render(x){$('workspace').hidden=false;let p=x.packet,l=$('layer').value;$('question').textContent=p.question||p.task||(l==='source'?'Source relevance audit':l==='calibration'?'Reviewer calibration':l==='article'?'Article claim audit':'Adversarial claim audit');
$('candidate').textContent=l==='article'?p.claim+'\n\nParagraph context:\n'+(p.paragraph_context||p.claim):l==='adversarial'?p.claim:l==='source'?JSON.stringify(p.phenomenon,null,2):l==='calibration'?JSON.stringify({candidate_answer:p.candidate_answer,candidate_conclusion:p.candidate_conclusion,original_text:p.original_text,proposed_revision:p.proposed_revision},null,2):JSON.stringify(p.candidate||p.verified_structured_answer,null,2);
$('evidence').textContent=l==='adversarial'?JSON.stringify({evidence_set_a:p.evidence_set_a,evidence_set_b:p.evidence_set_b,operator_trace:p.operator_trace,alternative_explanations:p.alternative_explanations,forbidden_inferences:p.forbidden_inferences},null,2):l==='source'?JSON.stringify(p.source,null,2):l==='calibration'?JSON.stringify({operator_trace:p.operator_trace,evidence_set_a:p.evidence_set_a,evidence_set_b:p.evidence_set_b,forbidden_inferences:p.forbidden_inferences,review_instructions:p.review_instructions},null,2):l==='article'?JSON.stringify({phenomena:p.phenomena,sources:p.sources,allowed_decisive_evidence_ids:p.allowed_decisive_evidence_ids,risk_features:p.risk_features},null,2):JSON.stringify(p.visible_evidence||p.interpretation_contract,null,2);
$('progress').textContent=JSON.stringify(x.progress.layers[l]);$('form').innerHTML=l==='factual'?factualForm():l==='semantic'?semanticForm():l==='adversarial'?adversarialForm():l==='source'?sourceForm():l==='calibration'?calibrationForm():articleForm()}
function factualForm(){return `<h3>Factual judgment</h3><label><input id="answer_correct" type="checkbox"> Answer correct</label>
<label><input id="evidence_sufficient" type="checkbox"> Evidence sufficient</label><label>Unsupported fields (comma separated)<input id="unsupported_fields"></label>
<label>Action <select id="action"><option>accept</option><option>rewrite</option><option>reject_evidence</option><option>abstain</option></select></label>
<label>Correction JSON<textarea id="correction"></textarea></label><label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function semanticForm(){return `<h3>Semantic judgment</h3><label><input id="calibrated" type="checkbox"> Calibrated</label>
<label>Phenomenon value (1-5)<input id="phenomenon_value" type="number" min="1" max="5" value="3"></label>
<label><input id="alternative_adequate" type="checkbox"> Alternative explanation adequate</label><label><input id="limitation_adequate" type="checkbox"> Limitation adequate</label>
<label>Action <select id="action"><option>accept</option><option>rewrite</option><option>reject_claim</option><option>abstain</option></select></label>
<label>Target span<textarea id="target_span"></textarea></label><label>Replacement<textarea id="replacement"></textarea></label>
<label>Transfer guard JSON<textarea id="guard">{}</textarea></label><label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function adversarialForm(){return `<h3>Two-sided graph claim audit</h3><label>Verdict <select id="verdict"><option>supported</option><option>qualify</option><option>reject</option><option>abstain</option></select></label>
<label>Decisive evidence IDs (comma separated)<input id="decisive_evidence_ids"></label><label>Invalid operator step indices (comma separated, zero based)<input id="invalid_operator_steps"></label>
<label>Failure mode<textarea id="failure_mode" required></textarea></label><label>Smallest defensible rewrite<textarea id="minimal_rewrite"></textarea></label>
<label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function sourceForm(){return `<h3>Phenomenon-source relevance audit</h3><label>Topic relevance <select id="topic_relevance"><option>direct</option><option>contextual</option><option>irrelevant</option><option>cannot_assess</option></select></label>
<label>Evidence role <select id="evidence_role"><option>supports_interpretation</option><option>context_only</option><option>challenges</option><option>irrelevant</option><option>cannot_assess</option></select></label>
<label>Decisive exact source span<textarea id="decisive_span"></textarea></label><label>Failure type <select id="failure_type"><option>none</option><option>generic_graph_anchor</option><option>query_too_broad</option><option>off_topic</option><option>insufficient_detail</option><option>conflicting_evidence</option><option>other</option></select></label>
<label>Suggested query terms (comma separated)<input id="suggested_query_terms"></label><label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function calibrationForm(){return `<h3>Reviewer capability calibration</h3><label>Verdict <select id="verdict"><option>valid</option><option>invalid</option><option>abstain</option></select></label>
<label>Error type (optional)<input id="error_type"></label><label>Decisive evidence or trace IDs (comma separated)<input id="decisive_evidence_ids"></label>
<label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function articleForm(){return `<h3>Claim-level article audit</h3><label><input id="factual_supported" type="checkbox"> Factually supported</label>
<label><input id="interpretation_calibrated" type="checkbox"> Interpretation calibrated</label><label><input id="alternative_adequate" type="checkbox"> Alternative or failure condition adequate</label><label><input id="evidence_sufficient" type="checkbox"> Evidence dependencies sufficient</label>
<label>Action <select id="action"><option>accept</option><option>rewrite</option><option>reject_claim</option><option>abstain</option></select></label>
<label>Decisive evidence IDs (comma separated)<input id="decisive_evidence_ids"></label><label>Invalid dependency IDs (comma separated)<input id="invalid_dependency_ids"></label>
<label>Replacement claim (required for rewrite)<textarea id="replacement"></textarea></label><label>Transfer guard JSON<textarea id="guard">{}</textarea></label>
<label>Rationale<textarea id="rationale" required></textarea></label><button onclick="submitReview()">Freeze submission</button>`}
function parseJson(id,fallback){let v=$(id).value.trim();return v?JSON.parse(v):fallback}
async function submitReview(){try{let l=$('layer').value;let a=l==='factual'?{answer_correct:$('answer_correct').checked,evidence_sufficient:$('evidence_sufficient').checked,
unsupported_fields:$('unsupported_fields').value.split(',').map(x=>x.trim()).filter(Boolean),action:$('action').value,correction:parseJson('correction',null),rationale:$('rationale').value}:
 l==='semantic'?{calibrated:$('calibrated').checked,phenomenon_value:Number($('phenomenon_value').value),alternative_adequate:$('alternative_adequate').checked,
limitation_adequate:$('limitation_adequate').checked,action:$('action').value,target_span:$('target_span').value||null,replacement:$('replacement').value||null,guard:parseJson('guard',{}),rationale:$('rationale').value}:
 l==='adversarial'?{verdict:$('verdict').value,decisive_evidence_ids:$('decisive_evidence_ids').value.split(',').map(x=>x.trim()).filter(Boolean),invalid_operator_steps:$('invalid_operator_steps').value.split(',').map(x=>x.trim()).filter(Boolean).map(Number),failure_mode:$('failure_mode').value,minimal_rewrite:$('minimal_rewrite').value||null,rationale:$('rationale').value}:
 l==='source'?{topic_relevance:$('topic_relevance').value,evidence_role:$('evidence_role').value,decisive_span:$('decisive_span').value||null,failure_type:$('failure_type').value,suggested_query_terms:$('suggested_query_terms').value.split(',').map(x=>x.trim()).filter(Boolean),rationale:$('rationale').value}:
 l==='calibration'?{verdict:$('verdict').value,error_type:$('error_type').value||null,decisive_evidence_ids:$('decisive_evidence_ids').value.split(',').map(x=>x.trim()).filter(Boolean),rationale:$('rationale').value}:
 {factual_supported:$('factual_supported').checked,interpretation_calibrated:$('interpretation_calibrated').checked,alternative_adequate:$('alternative_adequate').checked,evidence_sufficient:$('evidence_sufficient').checked,action:$('action').value,decisive_evidence_ids:$('decisive_evidence_ids').value.split(',').map(x=>x.trim()).filter(Boolean),invalid_dependency_ids:$('invalid_dependency_ids').value.split(',').map(x=>x.trim()).filter(Boolean),replacement:$('replacement').value||null,guard:parseJson('guard',{}),rationale:$('rationale').value};
let r=await fetch('/api/submit?'+query(),{method:'POST',headers:headers(),body:JSON.stringify({packet_id:current.packet.packet_id,session_id:current.session_id,answers:a})});
if(!r.ok){$('status').textContent=(await r.json()).detail;return}await loadNext()}catch(e){$('status').textContent=e.message}}
</script></body></html>"""
