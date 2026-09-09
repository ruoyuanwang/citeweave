from __future__ import annotations

import hashlib
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .io import read_json, sha256_file, write_json

ReviewerQuery = Annotated[str, Query()]
ReviewToken = Annotated[str | None, Header()]


class QueryReviewSubmission(BaseModel):
    task_id: str
    session_id: str
    relevant: bool | None
    cannot_assess: bool


class QueryReviewHeartbeat(BaseModel):
    task_id: str
    session_id: str


class QueryRelevanceReviewStore:
    """Blind, immutable and server-timed relevance review collection."""

    def __init__(self, manifest_path: Path, access_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = read_json(self.manifest_path)
        status = self.manifest.get("status")
        if status == "blind_primary_query_review_ready":
            self.mode = "primary"
            self.packet_key = "primary_packets"
            self.registry_key = "primary_task_registry"
        elif status == "blind_query_adjudication_ready":
            self.mode = "adjudication"
            self.packet_key = "adjudication_packets"
            self.registry_key = "adjudication_task_registry"
        else:
            raise ValueError("Query relevance review collection is not ready")
        access = read_json(access_path.resolve())
        self.access = access.get("reviewer_token_sha256", {})
        self.lock = threading.RLock()
        self.tasks: dict[str, list[dict[str, Any]]] = {}
        self._load_tasks()

    @staticmethod
    def _contract(task: dict[str, Any]) -> dict[str, str]:
        fields = (
            "task_id",
            "reviewer_id",
            "candidate_code",
            "item_code",
            "role",
            "task_definition_sha256",
        )
        return {field: str(task[field]) for field in fields}

    def _load_tasks(self) -> None:
        for reviewer_id, record in self.manifest[self.packet_key].items():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen query-review packet changed: {reviewer_id}")
            packet = read_json(path)
            if packet.get("reviewer_id") != reviewer_id:
                raise ValueError("Query-review packet identity mismatch")
            tasks = []
            if self.mode == "primary":
                for candidate in packet["candidates"]:
                    for item in candidate["items"]:
                        tasks.append(
                            {
                                **self._contract(item),
                                "concepts": candidate["concepts"],
                                "instructions": candidate["instructions"],
                                "title": item.get("title"),
                                "abstract": item.get("abstract"),
                                "year": item.get("year"),
                            }
                        )
            else:
                for row in packet["tasks"]:
                    tasks.append(
                        {
                            **self._contract(row),
                            "concepts": row["concepts"],
                            "instructions": row["instructions"],
                            "title": row["item"].get("title"),
                            "abstract": row["item"].get("abstract"),
                            "year": row["item"].get("year"),
                        }
                    )
            observed = [self._contract(task) for task in tasks]
            if observed != self.manifest[self.registry_key].get(reviewer_id):
                raise ValueError("Packet-derived tasks differ from frozen task registry")
            if len({task["task_id"] for task in tasks}) != len(tasks):
                raise ValueError("Duplicate query-review task identity")
            self.tasks[reviewer_id] = tasks

    def authorize(self, reviewer_id: str, token: str) -> None:
        expected = self.access.get(reviewer_id)
        observed = hashlib.sha256(token.encode()).hexdigest()
        if not expected or not secrets.compare_digest(expected, observed):
            raise PermissionError("Invalid reviewer credential")
        if reviewer_id not in self.tasks:
            raise PermissionError("Reviewer has no assignment")

    def _return_path(self, reviewer_id: str) -> Path:
        return self.root / "returns" / self.mode / f"{reviewer_id}.json"

    def _state_path(self, reviewer_id: str) -> Path:
        return self.root / "state" / self.mode / f"{reviewer_id}.json"

    def _load_return(self, reviewer_id: str) -> dict[str, Any]:
        path = self._return_path(reviewer_id)
        if path.is_file():
            return read_json(path)
        return {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "blind_attestation": True,
            "independent_completion_attestation": True,
            "submitted_at": None,
            "results": [],
        }

    def _load_state(self, reviewer_id: str) -> dict[str, Any]:
        path = self._state_path(reviewer_id)
        if path.is_file():
            return read_json(path)
        return {
            "schema_version": 1,
            "reviewer_id": reviewer_id,
            "active": None,
        }

    def progress(self, reviewer_id: str) -> dict[str, int | str]:
        completed = {
            row["task_id"] for row in self._load_return(reviewer_id)["results"]
        }
        total = len(self.tasks[reviewer_id])
        return {
            "reviewer_id": reviewer_id,
            "completed": len(completed),
            "remaining": total - len(completed),
            "total": total,
        }

    @staticmethod
    def _advance_visible_time(active: dict[str, Any], now: float) -> None:
        last = float(active["last_heartbeat_at"])
        active["visible_seconds"] = float(active["visible_seconds"]) + min(
            max(0.0, now - last), 45.0
        )
        active["last_heartbeat_at"] = now

    def next_task(self, reviewer_id: str) -> dict[str, Any]:
        with self.lock:
            returned = self._load_return(reviewer_id)
            completed = {row["task_id"] for row in returned["results"]}
            remaining = [
                task for task in self.tasks[reviewer_id] if task["task_id"] not in completed
            ]
            state = self._load_state(reviewer_id)
            if not remaining:
                state["active"] = None
                write_json(self._state_path(reviewer_id), state)
                return {"complete": True, "progress": self.progress(reviewer_id)}
            task = remaining[0]
            active = state.get("active")
            now = time.time()
            if not active or active.get("task_id") != task["task_id"]:
                active = {
                    "task_id": task["task_id"],
                    "session_id": secrets.token_urlsafe(18),
                    "opened_at": now,
                    "last_heartbeat_at": now,
                    "visible_seconds": 0.0,
                }
                state["active"] = active
                write_json(self._state_path(reviewer_id), state)
            return {
                "complete": False,
                "mode": self.mode,
                "task": task,
                "session_id": active["session_id"],
                "progress": self.progress(reviewer_id),
            }

    def heartbeat(
        self, reviewer_id: str, task_id: str, session_id: str
    ) -> dict[str, Any]:
        with self.lock:
            state = self._load_state(reviewer_id)
            active = state.get("active")
            if (
                not active
                or active.get("task_id") != task_id
                or active.get("session_id") != session_id
            ):
                raise ValueError("Heartbeat does not match the active review task")
            self._advance_visible_time(active, time.time())
            write_json(self._state_path(reviewer_id), state)
            return {"accepted": True, "visible_seconds": active["visible_seconds"]}

    def submit(
        self, reviewer_id: str, submission: QueryReviewSubmission
    ) -> dict[str, Any]:
        with self.lock:
            task_catalog = {
                task["task_id"]: task for task in self.tasks[reviewer_id]
            }
            task = task_catalog.get(submission.task_id)
            if task is None:
                raise ValueError("Submission references a foreign task")
            returned = self._load_return(reviewer_id)
            if any(
                row["task_id"] == submission.task_id for row in returned["results"]
            ):
                raise ValueError("Review submissions are immutable")
            state = self._load_state(reviewer_id)
            active = state.get("active")
            if (
                not active
                or active.get("task_id") != submission.task_id
                or active.get("session_id") != submission.session_id
            ):
                raise ValueError("Submission does not match the active review session")
            self._advance_visible_time(active, time.time())
            seconds = max(float(active["visible_seconds"]), 0.001)
            if submission.cannot_assess:
                if self.mode == "adjudication":
                    raise ValueError("Adjudicators must resolve every disputed item")
                if submission.relevant is not None:
                    raise ValueError("Cannot-assess must not include a relevance label")
                rationale = "insufficient_visible_information"
            else:
                if not isinstance(submission.relevant, bool):
                    raise ValueError("Assessable task requires a boolean relevance label")
                rationale = (
                    "both_central"
                    if submission.relevant
                    else "one_or_both_peripheral"
                )
            contract = self._contract(task)
            result = {
                **contract,
                "relevant": submission.relevant,
                "cannot_assess": submission.cannot_assess,
                "rationale_code": rationale,
                "review_seconds": seconds,
            }
            returned["results"].append(result)
            returned["submitted_at"] = datetime.now(UTC).isoformat()
            write_json(self._return_path(reviewer_id), returned)
            state["active"] = None
            write_json(self._state_path(reviewer_id), state)
            return {"accepted": True, "result": result, "progress": self.progress(reviewer_id)}


def _page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>盲法查询相关性审核</title><style>
body{font-family:system-ui,sans-serif;max-width:920px;margin:30px auto;padding:0 18px;color:#172033;background:#f5f7fb}
.card{background:white;border-radius:14px;padding:22px;box-shadow:0 4px 24px #17203318;margin:16px 0}label{display:block;margin:10px 0}
input,button{font:inherit;padding:9px}button{background:#2156d8;color:white;border:0;border-radius:8px;cursor:pointer}.muted{color:#647087}.concept{display:inline-block;background:#e8efff;padding:5px 9px;margin-right:8px;border-radius:99px}
</style></head><body><h1>盲法查询相关性审核</h1>
<div class="card"><label>审核者编号 <input id="reviewer"></label><label>访问令牌 <input id="token" type="password"></label><button onclick="loadTask()">进入审核</button></div>
<div id="work" class="card" hidden><div id="progress" class="muted"></div><div id="concepts"></div><h2 id="title"></h2><div id="year" class="muted"></div><p id="abstract"></p>
<label><input type="radio" name="decision" value="yes"> 两个概念都是本文核心</label>
<label><input type="radio" name="decision" value="no"> 至少一个概念不是核心</label>
<label id="cannotRow"><input type="radio" name="decision" value="cannot"> 可见文本不足以判断</label>
<button onclick="submitTask()">提交并进入下一条</button></div><div id="message"></div>
<script>
let current=null,timer=null; const q=id=>document.getElementById(id); const headers=()=>({'X-Review-Token':q('token').value,'Content-Type':'application/json'});
async function loadTask(){const r=q('reviewer').value;const x=await fetch('/api/next?reviewer='+encodeURIComponent(r),{headers:headers()});const d=await x.json();if(!x.ok){q('message').textContent=d.detail;return}if(d.complete){q('work').hidden=true;q('message').textContent='全部审核完成。';return}current=d;q('work').hidden=false;q('progress').textContent=`${d.progress.completed}/${d.progress.total}`;q('concepts').innerHTML=d.task.concepts.map(x=>`<span class="concept">${x}</span>`).join('');q('title').textContent=d.task.title||'（无标题）';q('year').textContent=d.task.year||'';q('abstract').textContent=d.task.abstract||'（无摘要）';q('cannotRow').hidden=d.mode==='adjudication';document.querySelectorAll('input[name=decision]').forEach(x=>x.checked=false);clearInterval(timer);timer=setInterval(()=>fetch('/api/heartbeat?reviewer='+encodeURIComponent(r),{method:'POST',headers:headers(),body:JSON.stringify({task_id:current.task.task_id,session_id:current.session_id})}),15000)}
async function submitTask(){const d=document.querySelector('input[name=decision]:checked');if(!d){alert('请选择判断');return}const value=d.value;const body={task_id:current.task.task_id,session_id:current.session_id,relevant:value==='cannot'?null:value==='yes',cannot_assess:value==='cannot'};const x=await fetch('/api/submit?reviewer='+encodeURIComponent(q('reviewer').value),{method:'POST',headers:headers(),body:JSON.stringify(body)});const out=await x.json();if(!x.ok){alert(out.detail);return}loadTask()}
</script></body></html>"""


def create_query_relevance_review_app(
    manifest_path: Path, access_path: Path
) -> FastAPI:
    store = QueryRelevanceReviewStore(manifest_path, access_path)
    app = FastAPI(title="Blind Query Relevance Review")

    def authorize(reviewer: str, token: str | None) -> None:
        try:
            store.authorize(reviewer, token or "")
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page()

    @app.get("/api/next")
    def next_task(reviewer: ReviewerQuery, x_review_token: ReviewToken = None) -> dict[str, Any]:
        authorize(reviewer, x_review_token)
        return store.next_task(reviewer)

    @app.post("/api/heartbeat")
    def heartbeat(
        payload: QueryReviewHeartbeat,
        reviewer: ReviewerQuery,
        x_review_token: ReviewToken = None,
    ) -> dict[str, Any]:
        authorize(reviewer, x_review_token)
        try:
            return store.heartbeat(reviewer, payload.task_id, payload.session_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/submit")
    def submit(
        payload: QueryReviewSubmission,
        reviewer: ReviewerQuery,
        x_review_token: ReviewToken = None,
    ) -> dict[str, Any]:
        authorize(reviewer, x_review_token)
        try:
            return store.submit(reviewer, payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
