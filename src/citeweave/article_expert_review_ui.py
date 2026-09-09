from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import secrets
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .article_expert_collection import expert_task_contract
from .article_expert_evaluation import ARTICLE_CONDITIONS, HOLISTIC_DIMENSIONS
from .io import read_json, sha256_file, write_json

TaskType = Literal["holistic", "claim", "pairwise"]
EvaluatorQuery = Annotated[str, Query()]
ExpertToken = Annotated[str | None, Header()]


class ExpertSubmission(BaseModel):
    task_id: str
    session_id: str
    answers: dict[str, Any]


class ExpertHeartbeat(BaseModel):
    task_id: str
    session_id: str


class ArticleExpertStore:
    """Condition-blind article evaluation with immutable, server-timed submissions."""

    def __init__(self, collection_manifest_path: Path, access_path: Path) -> None:
        self.manifest_path = collection_manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("status") != "expert_collection_ready":
            raise ValueError("Article expert collection is not ready")
        self.access = read_json(access_path.resolve())["reviewer_token_sha256"]
        self.enabled_task_types = set(
            self.manifest.get(
                "enabled_task_types", ["holistic", "claim", "pairwise"]
            )
        )
        if not self.enabled_task_types or not self.enabled_task_types.issubset(
            {"holistic", "claim", "pairwise"}
        ):
            raise ValueError("Expert collection has invalid enabled task types")
        self.lock = threading.RLock()
        self.tasks: dict[str, list[dict[str, Any]]] = {}
        self._load_and_validate_tasks()

    def _load_and_validate_tasks(self) -> None:
        viewers = {}
        for topic_code, record in self.manifest["evidence_viewers"].items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Evidence viewer changed after freeze: {topic_code}")
            viewers[topic_code] = read_json(path)
        for evaluator_id, record in self.manifest["evaluator_packets"].items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Evaluator packet changed after freeze: {evaluator_id}")
            packet = read_json(path)
            if packet.get("evaluator_id") != evaluator_id:
                raise ValueError("Evaluator packet identity mismatch")
            tasks = []
            for topic in packet.get("topics", []):
                topic_code = str(topic["topic_code"])
                viewer = viewers.get(topic_code)
                if viewer is None:
                    raise ValueError(f"Missing evidence viewer for {topic_code}")
                figure_path = Path(topic["figure_path"])
                if not figure_path.is_file() or sha256_file(figure_path) != topic["figure_sha256"]:
                    raise ValueError(f"Figure changed after freeze: {topic_code}")
                articles = []
                for article in topic.get("articles", []):
                    article_path = Path(article["article_path"])
                    if not article_path.is_file() or sha256_file(article_path) != article["article_sha256"]:
                        raise ValueError(
                            f"Blinded article changed after freeze: {article['article_code']}"
                        )
                    text = article_path.read_text(encoding="utf-8")
                    if any(label in text for label in ARTICLE_CONDITIONS):
                        raise ValueError("Condition label leaked into blinded article")
                    evidence_ids = {row["evidence_id"] for row in viewer.get("items", [])}
                    for claim in article.get("claims", []):
                        tokens = claim.get("evidence_tokens", [])
                        if not tokens or not set(tokens).issubset(evidence_ids):
                            raise ValueError(
                                "Expert claim lacks a complete blinded evidence mapping"
                            )
                    articles.append({**article, "_text": text})
                    if "holistic" in self.enabled_task_types:
                        tasks.append(
                            self._task(
                                evaluator_id,
                                topic_code,
                                "holistic",
                                article_code=article["article_code"],
                                article=article,
                                article_text=text,
                                viewer=viewer,
                                figure_path=figure_path,
                                figure_sha256=topic["figure_sha256"],
                            )
                        )
                    for claim in article.get("claims", []):
                        if "claim" in self.enabled_task_types:
                            tasks.append(
                                self._task(
                                    evaluator_id,
                                    topic_code,
                                    "claim",
                                    article_code=article["article_code"],
                                    article=article,
                                    article_text=text,
                                    claim=claim,
                                    viewer=viewer,
                                    figure_path=figure_path,
                                    figure_sha256=topic["figure_sha256"],
                                )
                            )
                if "pairwise" in self.enabled_task_types:
                    for left_index, left in enumerate(articles):
                        for right in articles[left_index + 1 :]:
                            tasks.append(
                                self._task(
                                    evaluator_id,
                                    topic_code,
                                    "pairwise",
                                    left_article=left,
                                    right_article=right,
                                    viewer=viewer,
                                    figure_path=figure_path,
                                    figure_sha256=topic["figure_sha256"],
                                )
                            )
            identifiers = [task["task_id"] for task in tasks]
            if len(identifiers) != len(set(identifiers)):
                raise AssertionError("Duplicate expert task identity")
            registered = self.manifest.get("task_registry", {}).get(evaluator_id)
            observed = [
                {
                    key: task[key]
                    for key in task
                    if key
                    in {
                        "task_id",
                        "task_definition_sha256",
                        "evaluator_id",
                        "topic_code",
                        "task_type",
                        "article_code",
                        "claim_id",
                        "left_article_code",
                        "right_article_code",
                    }
                }
                for task in tasks
            ]
            if registered != observed:
                raise ValueError("Evaluator task registry differs from packet-derived tasks")
            self.tasks[evaluator_id] = tasks

    @staticmethod
    def _task(
        evaluator_id: str,
        topic_code: str,
        task_type: TaskType,
        **payload: Any,
    ) -> dict[str, Any]:
        if task_type == "pairwise":
            definition = expert_task_contract(
                evaluator_id,
                topic_code,
                task_type,
                left_article_code=payload["left_article"]["article_code"],
                right_article_code=payload["right_article"]["article_code"],
            )
        else:
            definition = expert_task_contract(
                evaluator_id,
                topic_code,
                task_type,
                article_code=payload["article_code"],
                claim_id=(
                    payload["claim"]["claim_id"] if task_type == "claim" else None
                ),
            )
        return {
            **definition,
            **payload,
        }

    def authorize(self, evaluator_id: str, token: str) -> None:
        expected = self.access.get(evaluator_id)
        observed = hashlib.sha256(token.encode()).hexdigest()
        if not expected or not secrets.compare_digest(expected, observed):
            raise PermissionError("Invalid evaluator credential")
        if evaluator_id not in self.tasks:
            raise PermissionError("Evaluator has no assignment")

    def _return_path(self, evaluator_id: str) -> Path:
        return self.root / "returns" / f"{evaluator_id}.json"

    def _state_path(self, evaluator_id: str) -> Path:
        return self.root / "state" / f"{evaluator_id}.json"

    def _load_return(self, evaluator_id: str) -> dict[str, Any]:
        path = self._return_path(evaluator_id)
        if path.is_file():
            return read_json(path)
        return {"schema_version": 1, "evaluator_id": evaluator_id, "results": []}

    def _load_state(self, evaluator_id: str) -> dict[str, Any]:
        path = self._state_path(evaluator_id)
        if path.is_file():
            return read_json(path)
        return {"schema_version": 1, "evaluator_id": evaluator_id, "active": None}

    def progress(self, evaluator_id: str) -> dict[str, Any]:
        completed = {
            row["task_id"] for row in self._load_return(evaluator_id)["results"]
        }
        totals = Counter(task["task_type"] for task in self.tasks[evaluator_id])
        done = Counter(
            task["task_type"]
            for task in self.tasks[evaluator_id]
            if task["task_id"] in completed
        )
        return {
            "evaluator_id": evaluator_id,
            "completed": len(completed),
            "total": len(self.tasks[evaluator_id]),
            "by_type": {
                key: {
                    "completed": done[key],
                    "total": totals[key],
                    "remaining": totals[key] - done[key],
                }
                for key in sorted(totals)
            },
        }

    @staticmethod
    def _figure_data_url(path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _public_task(self, task: dict[str, Any]) -> dict[str, Any]:
        common = {
            key: task[key]
            for key in (
                "task_id",
                "task_definition_sha256",
                "task_type",
                "topic_code",
            )
        }
        common["figure_data_url"] = self._figure_data_url(task["figure_path"])
        common["figure_sha256"] = task["figure_sha256"]
        common["evidence_viewer"] = task["viewer"]
        if task["task_type"] == "pairwise":
            common["left_article"] = {
                "article_code": task["left_article"]["article_code"],
                "article_sha256": task["left_article"]["article_sha256"],
                "text": task["left_article"]["_text"],
            }
            common["right_article"] = {
                "article_code": task["right_article"]["article_code"],
                "article_sha256": task["right_article"]["article_sha256"],
                "text": task["right_article"]["_text"],
            }
        else:
            common["article"] = {
                "article_code": task["article_code"],
                "article_sha256": task["article"]["article_sha256"],
                "text": task["article_text"],
            }
            if task["task_type"] == "claim":
                common["claim"] = task["claim"]
                common["allowed_decisive_evidence_ids"] = task["claim"][
                    "evidence_tokens"
                ]
        serialized = json.dumps(common, ensure_ascii=False)
        if any(label in serialized for label in ARTICLE_CONDITIONS):
            raise AssertionError("Condition label entered public expert task")
        return common

    def next_task(self, evaluator_id: str) -> dict[str, Any]:
        with self.lock:
            returned = self._load_return(evaluator_id)
            completed = {row["task_id"] for row in returned["results"]}
            state = self._load_state(evaluator_id)
            active = state.get("active")
            if active and active["task_id"] not in completed:
                task_id = active["task_id"]
            else:
                task_id = next(
                    (
                        task["task_id"]
                        for task in self.tasks[evaluator_id]
                        if task["task_id"] not in completed
                    ),
                    None,
                )
                if task_id is None:
                    return {"complete": True, "progress": self.progress(evaluator_id)}
                now = time.time()
                active = {
                    "task_id": task_id,
                    "session_id": secrets.token_urlsafe(24),
                    "started_at": now,
                    "last_heartbeat": now,
                    "active_seconds": 0.0,
                }
                state["active"] = active
                write_json(self._state_path(evaluator_id), state)
            task = next(task for task in self.tasks[evaluator_id] if task["task_id"] == task_id)
            return {
                "complete": False,
                "session_id": active["session_id"],
                "task": self._public_task(task),
                "progress": self.progress(evaluator_id),
            }

    @staticmethod
    def _touch(active: dict[str, Any]) -> None:
        now = time.time()
        elapsed = max(0.0, now - float(active["last_heartbeat"]))
        active["active_seconds"] = float(active["active_seconds"]) + min(elapsed, 10.0)
        active["last_heartbeat"] = now

    @staticmethod
    def _require_active(
        active: dict[str, Any] | None, task_id: str, session_id: str
    ) -> None:
        if (
            not active
            or active.get("task_id") != task_id
            or active.get("session_id") != session_id
        ):
            raise ValueError("Expert task session is absent, stale, or mismatched")

    @staticmethod
    def _validate_answers(task: dict[str, Any], answers: dict[str, Any]) -> None:
        reserved = {
            "task_id",
            "task_type",
            "topic_code",
            "article_code",
            "claim_id",
            "left_article_code",
            "right_article_code",
            "task_definition_sha256",
            "evaluator_id",
            "review_seconds",
            "server_elapsed_seconds",
        }
        if reserved & answers.keys():
            raise ValueError("Answers cannot override server-owned fields")
        rationale = answers.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("A nonempty rationale is required")
        task_type = task["task_type"]
        if task_type == "holistic":
            for dimension in HOLISTIC_DIMENSIONS:
                score = answers.get(dimension)
                if (
                    not isinstance(score, int)
                    or isinstance(score, bool)
                    or not 1 <= score <= 5
                ):
                    raise ValueError(f"{dimension} must be an integer from 1 to 5")
        elif task_type == "claim":
            cannot_assess = answers.get("cannot_assess")
            if not isinstance(cannot_assess, bool):
                raise TypeError("cannot_assess must be boolean")
            labels = ("supported", "correct", "overclaim", "evidence_sufficient")
            if cannot_assess:
                if any(answers.get(field) is not None for field in labels):
                    raise ValueError("Cannot-assess must leave substantive labels null")
            elif any(not isinstance(answers.get(field), bool) for field in labels):
                raise TypeError("Assessable claims require four boolean labels")
            decisive = answers.get("decisive_evidence_ids")
            if not isinstance(decisive, list) or any(
                not isinstance(value, str) or not value.strip() for value in decisive
            ):
                raise ValueError("decisive_evidence_ids must be a string list")
            if len(decisive) != len(set(decisive)):
                raise ValueError("decisive_evidence_ids must be unique")
            allowed = set(task["claim"]["evidence_tokens"])
            unknown = sorted(set(decisive) - allowed)
            if unknown:
                raise ValueError(f"Unknown decisive evidence IDs: {unknown}")
            if not cannot_assess and not decisive:
                raise ValueError("An assessable claim requires decisive evidence")
        else:
            preferred = answers.get("preferred_article_code")
            allowed = {
                task["left_article"]["article_code"],
                task["right_article"]["article_code"],
            }
            if preferred not in allowed:
                raise ValueError("Pairwise preference must select one visible article")

    def heartbeat(
        self, evaluator_id: str, payload: ExpertHeartbeat
    ) -> dict[str, Any]:
        with self.lock:
            state = self._load_state(evaluator_id)
            active = state.get("active")
            self._require_active(active, payload.task_id, payload.session_id)
            self._touch(active)
            write_json(self._state_path(evaluator_id), state)
            return {"active_seconds": round(float(active["active_seconds"]), 3)}

    def submit(
        self, evaluator_id: str, payload: ExpertSubmission
    ) -> dict[str, Any]:
        with self.lock:
            state = self._load_state(evaluator_id)
            active = state.get("active")
            self._require_active(active, payload.task_id, payload.session_id)
            task = next(
                task
                for task in self.tasks[evaluator_id]
                if task["task_id"] == payload.task_id
            )
            returned = self._load_return(evaluator_id)
            if any(row["task_id"] == payload.task_id for row in returned["results"]):
                raise ValueError("Expert task was already submitted and is frozen")
            answers = dict(payload.answers)
            self._validate_answers(task, answers)
            self._touch(active)
            identity_fields = {
                key: task[key]
                for key in (
                    "task_id",
                    "task_definition_sha256",
                    "task_type",
                    "topic_code",
                )
            }
            if task["task_type"] == "pairwise":
                identity_fields.update(
                    left_article_code=task["left_article"]["article_code"],
                    right_article_code=task["right_article"]["article_code"],
                )
            else:
                identity_fields["article_code"] = task["article_code"]
                if task["task_type"] == "claim":
                    identity_fields["claim_id"] = task["claim"]["claim_id"]
                    identity_fields["stratum"] = task["claim"]["stratum"]
            result = {
                **identity_fields,
                "evaluator_id": evaluator_id,
                **answers,
                "review_seconds": round(
                    max(0.001, float(active["active_seconds"])), 3
                ),
                "server_elapsed_seconds": round(
                    max(0.001, time.time() - float(active["started_at"])), 3
                ),
                "timing_method": "visibility_heartbeat_server_accounted",
                "submitted_at_unix": time.time(),
            }
            returned["results"].append(result)
            write_json(self._return_path(evaluator_id), returned)
            state["active"] = None
            write_json(self._state_path(evaluator_id), state)
            return {
                "accepted": True,
                "result": result,
                "progress": self.progress(evaluator_id),
            }


def create_article_expert_app(
    collection_manifest_path: Path, access_path: Path
) -> FastAPI:
    store = ArticleExpertStore(collection_manifest_path, access_path)
    app = FastAPI(title="CiteWeave Blinded Article Expert Evaluation", version="1.0")

    def auth(evaluator: str, token: str | None) -> None:
        try:
            store.authorize(evaluator, token or "")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return EXPERT_HTML

    @app.get("/api/progress")
    def progress(
        evaluator: EvaluatorQuery, x_review_token: ExpertToken = None
    ) -> dict[str, Any]:
        auth(evaluator, x_review_token)
        return store.progress(evaluator)

    @app.get("/api/next")
    def next_task(
        evaluator: EvaluatorQuery, x_review_token: ExpertToken = None
    ) -> dict[str, Any]:
        auth(evaluator, x_review_token)
        return store.next_task(evaluator)

    @app.post("/api/heartbeat")
    def heartbeat(
        payload: ExpertHeartbeat,
        evaluator: EvaluatorQuery,
        x_review_token: ExpertToken = None,
    ) -> dict[str, Any]:
        auth(evaluator, x_review_token)
        try:
            return store.heartbeat(evaluator, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/submit")
    def submit(
        payload: ExpertSubmission,
        evaluator: EvaluatorQuery,
        x_review_token: ExpertToken = None,
    ) -> dict[str, Any]:
        auth(evaluator, x_review_token)
        try:
            return store.submit(evaluator, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


EXPERT_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>CiteWeave Expert Evaluation</title>
<style>body{font:15px system-ui;margin:0;background:#f4f6fa;color:#172033}main{max-width:1250px;margin:auto;padding:22px}.bar,.card{background:#fff;border:1px solid #dce2ed;border-radius:12px;padding:16px;margin-bottom:14px}.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap}input,select,textarea,button{font:inherit;padding:8px}textarea{width:100%;min-height:85px;box-sizing:border-box}button{background:#3157d5;color:white;border:0;border-radius:7px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.article{white-space:pre-wrap;max-height:560px;overflow:auto;background:#f8f9fc;padding:12px}.evidence{white-space:pre-wrap;max-height:360px;overflow:auto;background:#f8f9fc;padding:12px}img{max-width:100%}.error{color:#b42318}.scores{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}@media(max-width:800px){.grid,.scores{grid-template-columns:1fr}}</style></head><body><main>
<h1>Blinded Article Expert Evaluation</h1><div class="bar"><input id="evaluator" placeholder="Evaluator code"><input id="token" type="password" placeholder="Access token"><button onclick="loadNext()">Load next</button><span id="progress"></span></div><div id="status" class="error"></div><section id="workspace" hidden><div class="card"><h2 id="title"></h2><img id="figure"><div id="articles" class="grid"></div></div><div class="card"><h3>Blinded evidence viewer</h3><pre id="evidence" class="evidence"></pre></div><div class="card" id="form"></div></section>
<script>let current=null,hb=null;const $=x=>document.getElementById(x);function headers(){return {'X-Review-Token':$('token').value,'Content-Type':'application/json'}}function query(){return 'evaluator='+encodeURIComponent($('evaluator').value)}
async function loadNext(){clearInterval(hb);$('status').textContent='';let r=await fetch('/api/next?'+query(),{headers:headers()});if(!r.ok){$('status').textContent=(await r.json()).detail;return}current=await r.json();if(current.complete){$('workspace').hidden=true;$('progress').textContent='Complete';return}render(current.task);$('progress').textContent=current.progress.completed+'/'+current.progress.total;hb=setInterval(()=>{if(!document.hidden)fetch('/api/heartbeat?'+query(),{method:'POST',headers:headers(),body:JSON.stringify({task_id:current.task.task_id,session_id:current.session_id})})},5000)}
function render(t){$('workspace').hidden=false;$('title').textContent=t.task_type+' — '+t.topic_code;$('figure').src=t.figure_data_url;$('evidence').textContent=JSON.stringify(t.evidence_viewer.items,null,2);if(t.task_type==='pairwise'){$('articles').innerHTML='<div><h3>'+t.left_article.article_code+'</h3><pre class="article"></pre></div><div><h3>'+t.right_article.article_code+'</h3><pre class="article"></pre></div>';let ps=$('articles').querySelectorAll('pre');ps[0].textContent=t.left_article.text;ps[1].textContent=t.right_article.text;$('form').innerHTML=pairForm(t)}else{$('articles').innerHTML='<div style="grid-column:1/-1"><h3>'+t.article.article_code+'</h3><pre class="article"></pre></div>';$('articles').querySelector('pre').textContent=t.article.text;$('form').innerHTML=t.task_type==='holistic'?holisticForm():claimForm(t)}}
function holisticForm(){let ds=['factual_accuracy','evidence_traceability','phenomenon_depth','alternative_explanations','epistemic_calibration','domain_specificity','argumentative_coherence','research_utility'];return '<h3>Independent holistic scores (1–5)</h3><div class="scores">'+ds.map(d=>'<label>'+d+' <select id="'+d+'"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></label>').join('')+'</div><label>Rationale<textarea id="rationale"></textarea></label><button onclick="submitReview()">Freeze submission</button>'}
function claimForm(t){return '<h3>Claim verification</h3><p><b>'+t.claim.stratum+':</b> '+t.claim.text+'</p><label><input id="cannot_assess" type="checkbox"> Cannot assess</label><label><input id="supported" type="checkbox"> Supported</label><label><input id="correct" type="checkbox"> Correct</label><label><input id="overclaim" type="checkbox"> Overclaim</label><label><input id="evidence_sufficient" type="checkbox"> Evidence sufficient</label><label>Decisive evidence IDs <input id="decisive" value="'+t.allowed_decisive_evidence_ids.join(',')+'"></label><label>Rationale<textarea id="rationale"></textarea></label><button onclick="submitReview()">Freeze submission</button>'}
function pairForm(t){return '<h3>Forced pairwise preference</h3><select id="preferred"><option value="'+t.left_article.article_code+'">'+t.left_article.article_code+'</option><option value="'+t.right_article.article_code+'">'+t.right_article.article_code+'</option></select><label>Rationale<textarea id="rationale"></textarea></label><button onclick="submitReview()">Freeze submission</button>'}
async function submitReview(){let t=current.task,a;if(t.task_type==='holistic'){a={rationale:$('rationale').value};['factual_accuracy','evidence_traceability','phenomenon_depth','alternative_explanations','epistemic_calibration','domain_specificity','argumentative_coherence','research_utility'].forEach(d=>a[d]=Number($(d).value))}else if(t.task_type==='claim'){let c=$('cannot_assess').checked;a={cannot_assess:c,supported:c?null:$('supported').checked,correct:c?null:$('correct').checked,overclaim:c?null:$('overclaim').checked,evidence_sufficient:c?null:$('evidence_sufficient').checked,decisive_evidence_ids:c?[]:$('decisive').value.split(',').map(x=>x.trim()).filter(Boolean),rationale:$('rationale').value}}else{a={preferred_article_code:$('preferred').value,rationale:$('rationale').value}}let r=await fetch('/api/submit?'+query(),{method:'POST',headers:headers(),body:JSON.stringify({task_id:t.task_id,session_id:current.session_id,answers:a})});if(!r.ok){$('status').textContent=(await r.json()).detail;return}await loadNext()}</script></main></body></html>"""
