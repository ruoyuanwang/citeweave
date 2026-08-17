from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from .io import atomic_write_bytes, write_json

Condition = Literal["structured_one_shot", "citeweave_full"]


class CompletedRunError(RuntimeError):
    """Raised when an immutable, completed condition would be overwritten."""


class IncompleteRunError(RuntimeError):
    """Raised when an incomplete condition exists but resume was not requested."""


class CompletionTransport(Protocol):
    def complete(self, request: dict[str, Any]) -> dict[str, Any]: ...


class DeepSeekTransport:
    """Small OpenAI-compatible transport; authentication is never written to disk."""

    def __init__(self, *, api_key: str | None, base_url: str, timeout: float = 300):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required.")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        response: httpx.Response | None = None
        for attempt in range(7):
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request,
            )
            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt == 6:
                break
            retry_after = response.headers.get("retry-after")
            time.sleep(min(float(retry_after) if retry_after else 2**attempt, 45))
        assert response is not None
        if response.status_code >= 400:
            # Do not include headers because they may contain provider-specific secrets.
            safe_body = response.text.replace(self.api_key, "***")
            raise RuntimeError(
                f"DeepSeek API returned HTTP {response.status_code}: {safe_body[:1000]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(  # noqa: TRY004
                "DeepSeek API returned a non-object JSON response."
            )
        return payload


@dataclass(frozen=True)
class ReportRunConfig:
    dataset_id: str
    condition: Condition
    model: str = "deepseek-v4-pro"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.1
    seed: int = 42
    max_tokens: int = 8000


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _extract_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Completion response does not contain choices[0].message.content."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Completion response contains an empty report.")
    return content.strip()


def _evidence_packet(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        value = value["items"]
    if not isinstance(value, list) or not value:
        raise ValueError("Evidence Bundle must be a non-empty JSON list or an object with items.")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("evidence_id"), str):
            raise ValueError(  # noqa: TRY004
                "Every Evidence Bundle item must have a string evidence_id."
            )
        ids.append(item["evidence_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Evidence Bundle evidence_id values must be unique.")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str)


BASE_SYSTEM = """You are an expert bibliometric researcher. Write only in English.
Every empirical statement must be supported by a nearby evidence identifier such as [E001].
Use only the supplied frozen Evidence Bundle. Never invent numbers, entities, citations, causal
claims, statistical significance, or full-text findings. Clearly distinguish observations from
interpretations and state relevant retrieval, metadata, citation-window, and network limitations."""

ONE_SHOT_USER = """Create a complete, publication-style bibliometric report in English in one
response. Include a title, structured abstract, introduction, reproducible methods, results,
discussion, limitations, and conclusion. Explain the major performance, social, conceptual, and
intellectual structures supported by the evidence. Integrate findings instead of listing metrics.
Do not describe these instructions and do not add a reference list.

FROZEN EVIDENCE BUNDLE:
{evidence}
"""

PLAN_USER = """Develop a detailed editorial plan for a publication-style bibliometric report in
English. The plan must cover a title, structured abstract, introduction, reproducible methods,
results, discussion, limitations, and conclusion. Map every planned empirical claim to evidence
identifiers. Return only the plan.

FROZEN EVIDENCE BUNDLE:
{evidence}
"""

DRAFT_USER = """Write the complete publication-style bibliometric report in English by following
the editorial plan. Use nearby [E000] citations for every empirical statement. Include a title,
structured abstract, introduction, reproducible methods, results, discussion, limitations, and
conclusion. Do not add a reference list or facts absent from the frozen Evidence Bundle.

EDITORIAL PLAN:
{plan}

FROZEN EVIDENCE BUNDLE:
{evidence}
"""

REVIEW_SYSTEM = """You are CiteWeave's internal evidence and methods reviewer. This review exists
only to guide revision; it is not an experimental quality rating and must never be reused as the
external LLM-as-Judge evaluation. Identify unsupported, overstated, incomplete, or methodologically
ambiguous claims and prescribe concrete revisions. Write in English."""

REVIEW_USER = """Review the draft against the exact frozen Evidence Bundle. Return a prioritized
revision memo only. Check evidence citations, numerical fidelity, methodological reproducibility,
network interpretation, coverage limitations, synthesis, and completeness.

DRAFT:
{draft}

FROZEN EVIDENCE BUNDLE:
{evidence}
"""

REVISION_USER = """Return the complete revised publication-style bibliometric report in English.
Apply the internal review only where it is supported by the frozen Evidence Bundle. Preserve nearby
[E000] citations, remove unsupported claims, and retain the full report structure. Return only the
report, without an editorial note or reference list.

EDITORIAL PLAN:
{plan}

DRAFT:
{draft}

INTERNAL REVISION MEMO (NOT AN EXPERIMENTAL JUDGE SCORE):
{review}

FROZEN EVIDENCE BUNDLE:
{evidence}
"""


class ReportConditionRunner:
    def __init__(
        self,
        *,
        evidence_path: Path,
        output_root: Path,
        config: ReportRunConfig,
        transport: CompletionTransport,
        resume: bool = False,
    ):
        self.evidence_path = evidence_path.resolve()
        self.output_root = output_root.resolve()
        self.config = config
        self.transport = transport
        self.resume = resume
        self.evidence_bytes = self.evidence_path.read_bytes()
        self.evidence_sha256 = _sha256(self.evidence_bytes)
        self.evidence_value = json.loads(self.evidence_bytes)
        self.evidence = _evidence_packet(self.evidence_value)
        self.dataset_root = self.output_root / config.dataset_id
        self.condition_dir = self.dataset_root / config.condition
        self.calls_dir = self.condition_dir / "calls"
        self._lock_path = self.condition_dir / ".run.lock"
        self._lock_acquired = False

    def _lock_evidence(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.dataset_root / "frozen_evidence.json"
        expected = {
            "dataset_id": self.config.dataset_id,
            "evidence_sha256": self.evidence_sha256,
            "evidence_source": str(self.evidence_path),
        }
        if lock_path.exists():
            actual = json.loads(lock_path.read_text(encoding="utf-8"))
            if actual.get("evidence_sha256") != self.evidence_sha256:
                raise RuntimeError(
                    "Dataset conditions must use the same frozen Evidence Bundle; hash mismatch."
                )
            return
        payload = json.dumps(expected, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            self._lock_evidence()
            return
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)

    def _prepare(self) -> None:
        self._lock_evidence()
        completion = self.condition_dir / "completion.json"
        if completion.exists():
            raise CompletedRunError(f"Completed condition is immutable: {self.condition_dir}")
        if self.condition_dir.exists() and any(self.condition_dir.iterdir()) and not self.resume:
            raise IncompleteRunError(
                f"Incomplete condition exists; pass resume=True: {self.condition_dir}"
            )
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self._lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise RuntimeError(f"Condition is already running: {self.condition_dir}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self._lock_acquired = True
        manifest_path = self.condition_dir / "run_manifest.json"
        manifest = {
            "schema_version": 1,
            "dataset_id": self.config.dataset_id,
            "condition": self.config.condition,
            "evidence_sha256": self.evidence_sha256,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "report_language": "English",
            "internal_review_policy": (
                "Revision-only artifact. It must not be reused as an experimental evaluation."
            ),
            "external_evaluation_policy": (
                "Use a separately instantiated external Judge with independent prompts and traces."
            ),
        }
        if manifest_path.exists():
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _canonical_bytes(actual) != _canonical_bytes(manifest):
                raise RuntimeError("Resume configuration does not match the existing condition.")
        else:
            write_json(manifest_path, manifest)

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "thinking": {"type": "disabled"},
            "stream": False,
        }

    def _complete(self, call_id: str, role: str, *, system: str, user: str) -> str:
        call_dir = self.calls_dir / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        request = self._request(system, user)
        request_bytes = _canonical_bytes(request)
        request_sha = _sha256(request_bytes)
        prompt_sha = _sha256(_canonical_bytes({"system": system, "user": user}))
        call_path = call_dir / "call.json"
        if call_path.exists():
            saved = json.loads(call_path.read_text(encoding="utf-8"))
            if saved.get("request_sha256") != request_sha:
                raise RuntimeError(f"Resume prompt/config mismatch for {call_id}.")
            return _extract_content(saved["raw_response"])

        write_json(call_dir / "request.json", request)
        started = time.perf_counter()
        response = self.transport.complete(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        write_json(call_dir / "response.json", response)
        record = {
            "call_id": call_id,
            "role": role,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "prompt_sha256": prompt_sha,
            "request_sha256": request_sha,
            "latency_ms": latency_ms,
            "usage": response.get("usage"),
            "raw_request": request,
            "raw_response": response,
        }
        write_json(call_path, record)
        return _extract_content(response)

    def _run_one_shot(self) -> str:
        return self._complete(
            "01_structured_one_shot",
            "report_generation",
            system=BASE_SYSTEM,
            user=ONE_SHOT_USER.format(evidence=self.evidence),
        )

    def _run_full(self) -> str:
        plan = self._complete(
            "01_editorial_plan",
            "planning",
            system=BASE_SYSTEM,
            user=PLAN_USER.format(evidence=self.evidence),
        )
        draft = self._complete(
            "02_complete_draft",
            "drafting",
            system=BASE_SYSTEM,
            user=DRAFT_USER.format(plan=plan, evidence=self.evidence),
        )
        review = self._complete(
            "03_internal_review",
            "internal_revision_review_not_experimental_evaluation",
            system=REVIEW_SYSTEM,
            user=REVIEW_USER.format(draft=draft, evidence=self.evidence),
        )
        return self._complete(
            "04_revised_report",
            "revision",
            system=BASE_SYSTEM,
            user=REVISION_USER.format(
                plan=plan,
                draft=draft,
                review=review,
                evidence=self.evidence,
            ),
        )

    def run(self) -> dict[str, Any]:
        try:
            self._prepare()
            report = (
                self._run_one_shot()
                if self.config.condition == "structured_one_shot"
                else self._run_full()
            )
            expected_calls = 1 if self.config.condition == "structured_one_shot" else 4
            call_files = sorted(self.calls_dir.glob("*/call.json"))
            if len(call_files) != expected_calls:
                raise RuntimeError(
                    f"{self.config.condition} requires exactly {expected_calls} completed calls; "
                    f"found {len(call_files)}."
                )
            report_path = self.condition_dir / "report.md"
            atomic_write_bytes(report_path, (report.rstrip() + "\n").encode("utf-8"))
            completion = {
                "status": "complete",
                "dataset_id": self.config.dataset_id,
                "condition": self.config.condition,
                "evidence_sha256": self.evidence_sha256,
                "report_sha256": _sha256(report_path.read_bytes()),
                "call_count": len(call_files),
                "model": self.config.model,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "report_language": "English",
                "internal_review_used_as_experimental_evaluation": False,
            }
            write_json(self.condition_dir / "completion.json", completion)
            return completion
        finally:
            if self._lock_acquired:
                self._lock_path.unlink(missing_ok=True)
                self._lock_acquired = False


def run_report_condition(
    *,
    evidence_path: Path,
    output_root: Path,
    config: ReportRunConfig,
    api_key: str | None = None,
    resume: bool = False,
    transport: CompletionTransport | None = None,
) -> dict[str, Any]:
    selected_transport = transport or DeepSeekTransport(
        api_key=api_key,
        base_url=config.base_url,
    )
    return ReportConditionRunner(
        evidence_path=evidence_path,
        output_root=output_root,
        config=config,
        transport=selected_transport,
        resume=resume,
    ).run()
