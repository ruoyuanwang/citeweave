from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_context_json(context: dict[str, Any]) -> str:
    return json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class TokenBudgetResult:
    context: dict[str, Any]
    full_tokens: int
    budgeted_tokens: int
    token_budget: int
    retained_records: dict[str, int]
    original_records: dict[str, int]
    context_sha256: str


@dataclass(frozen=True)
class _TrimmableList:
    """One deterministically ordered list that may be shortened for transport."""

    label: str
    path: tuple[str | int, ...]
    values: list[Any]
    required_indices: tuple[int, ...] = ()


class CommandTokenizer:
    """Exact model-token counter provided by a frozen local command.

    The command receives one JSON object per invocation on stdin and must emit
    ``{"tokens": <integer>}``. Shell interpretation is never used.
    """

    def __init__(self, manifest: dict[str, Any], *, manifest_dir: Path):
        command = manifest.get("counting_command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError("Tokenizer manifest requires a non-empty counting_command list")
        executable = Path(command[0])
        if not executable.is_absolute():
            executable = (manifest_dir / executable).resolve()
        self.command = [str(executable), *command[1:]]
        message_command = manifest.get("message_counting_command") or command
        if not isinstance(message_command, list) or not message_command or not all(
            isinstance(part, str) and part for part in message_command
        ):
            raise ValueError("message_counting_command must be a non-empty string list")
        message_executable = Path(message_command[0])
        if not message_executable.is_absolute():
            message_executable = (manifest_dir / message_executable).resolve()
        self.message_command = [str(message_executable), *message_command[1:]]
        self.model = str(manifest["model"])
        self.timeout_seconds = float(manifest.get("timeout_seconds", 30))

    def _run(self, command: list[str], payload: dict[str, Any]) -> int:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=self.timeout_seconds,
            check=True,
            shell=False,
            env=child_env,
        )
        payload = json.loads(completed.stdout)
        tokens = payload.get("tokens")
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("Tokenizer command did not return a non-negative integer")
        return tokens

    def count(self, text: str) -> int:
        return self._run(self.command, {"text": text})

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return self._run(self.message_command, {"messages": messages})

    def verify(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        for probe in probes:
            observed = self.count(str(probe["text"]))
            expected = int(probe["expected_tokens"])
            results.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "expected_tokens": expected,
                    "observed_tokens": observed,
                    "passed": observed == expected,
                }
            )
        return {"passed": bool(results) and all(row["passed"] for row in results), "probes": results}

    def verify_message_probes(self, probes: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        for probe in probes:
            messages = probe.get("messages")
            if not isinstance(messages, list):
                raise TypeError("Message probe requires a messages list")
            observed = self.count_messages(messages)
            expected = int(probe["prompt_tokens"])
            results.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "expected_prompt_tokens": expected,
                    "observed_prompt_tokens": observed,
                    "passed": observed == expected,
                }
            )
        return {
            "passed": bool(results) and all(row["passed"] for row in results),
            "probes": results,
        }


def verify_api_usage_probe_artifact(
    manifest: dict[str, Any],
    *,
    manifest_dir: Path,
    tokenizer: CommandTokenizer,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec = manifest.get("api_usage_probe_artifact")
    if not isinstance(spec, dict):
        return {}, {"passed": False, "probes": []}, [
            "tokenizer_api_usage_probe_artifact_missing"
        ]
    path = Path(str(spec.get("path") or ""))
    if not path.is_absolute():
        path = manifest_dir / path
    if not path.is_file():
        return {}, {"passed": False, "probes": []}, [
            f"tokenizer_api_usage_probe_file_missing:{path}"
        ]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
        return {}, {"passed": False, "probes": []}, [
            "tokenizer_api_usage_probe_hash_mismatch"
        ]
    payload = json.loads(raw)
    reasons = []
    if payload.get("passed") is not True:
        reasons.append("tokenizer_api_usage_probes_not_passed")
    if payload.get("requested_model") != manifest.get("model"):
        reasons.append("tokenizer_api_usage_probe_model_mismatch")
    verification = tokenizer.verify_message_probes(payload.get("records") or [])
    if not verification["passed"]:
        reasons.append("tokenizer_message_encoding_probe_mismatch")
    return payload, verification, reasons


def _path_set(root: dict[str, Any], path: tuple[str | int, ...], value: list[Any]) -> None:
    current: Any = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _community_metric_requirements(trace: list[Any]) -> set[Any]:
    """Return community ids consumed by a later registered selection operator.

    The community aggregate can contain hundreds of diagnostic rows, while the
    downstream ``role_contrast`` step explicitly identifies the two rows needed
    to answer the registered task.  This derives requirements from the operator
    program itself, never from a gold answer or provider outcome.
    """

    required: set[Any] = set()
    for step in trace:
        if not isinstance(step, dict) or step.get("operator") != "role_contrast":
            continue
        for key in ("dominant", "outward"):
            if key in step:
                required.add(step[key])
    return required


def _candidate_lists(
    context: dict[str, Any], *, include_operator_metrics: bool = False
) -> list[_TrimmableList]:
    candidates = [
        _TrimmableList(label=key, path=(key,), values=value)
        for key in ("summaries", "nodes", "edges", "rows")
        if isinstance((value := context.get(key)), list)
    ]
    if not include_operator_metrics:
        return candidates
    # Operator programs are normally compact fixed metadata.  Community-role
    # tasks are the one registered exception: their deterministic diagnostic
    # table grows with the number of communities.  Treat only that table as
    # ranked/trimmable and retain every row referenced by the downstream
    # selection step.  Other operator arguments remain immutable.
    for trace_key in ("operator_trace", "derived_rows"):
        trace = context.get(trace_key)
        if not isinstance(trace, list):
            continue
        required_communities = _community_metric_requirements(trace)
        for index, step in enumerate(trace):
            if not isinstance(step, dict) or step.get("operator") != "community_aggregate":
                continue
            metrics = step.get("metrics")
            if not isinstance(metrics, list):
                continue
            required_indices = tuple(
                metric_index
                for metric_index, metric in enumerate(metrics)
                if isinstance(metric, dict)
                and metric.get("community") in required_communities
            )
            candidates.append(
                _TrimmableList(
                    label=f"{trace_key}[{index}].metrics",
                    path=(trace_key, index, "metrics"),
                    values=metrics,
                    required_indices=required_indices,
                )
            )
    return candidates


def _minimum_count(candidate: _TrimmableList) -> int:
    if candidate.label == "summaries" and candidate.values:
        return 1
    return 0


def truncate_context_to_token_budget(
    context: dict[str, Any],
    *,
    count_tokens: Callable[[str], int],
    token_budget: int,
) -> TokenBudgetResult:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    full = copy.deepcopy(context)
    full_tokens = count_tokens(canonical_context_json(full))
    # Preserve the exact legacy allocation whenever it was feasible.  Nested
    # operator compaction is a fallback only for contexts for which the legacy
    # top-level budgeter would have raised before sending a request.  This is
    # essential for resumable experiments: completed request identities remain
    # byte-identical.
    lists = _candidate_lists(full)
    original_counts = {candidate.label: len(candidate.values) for candidate in lists}
    if full_tokens <= token_budget:
        rendered = canonical_context_json(full)
        return TokenBudgetResult(
            context=full,
            full_tokens=full_tokens,
            budgeted_tokens=full_tokens,
            token_budget=token_budget,
            retained_records=original_counts,
            original_records=original_counts,
            context_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        )
    if not lists:
        lists = _candidate_lists(full, include_operator_metrics=True)
        original_counts = {
            candidate.label: len(candidate.values) for candidate in lists
        }
        if not lists:
            raise ValueError("Fixed context exceeds token budget and has no trimmable records")

    def candidate(
        fraction: float,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, set[int]]]:
        output = copy.deepcopy(full)
        retained = {}
        retained_indices: dict[str, set[int]] = {}
        for list_spec in lists:
            values = list_spec.values
            minimum = _minimum_count(list_spec)
            prefix_count = max(minimum, min(len(values), int(len(values) * fraction)))
            indices = set(range(prefix_count))
            indices.update(list_spec.required_indices)
            selected = [value for index, value in enumerate(values) if index in indices]
            _path_set(output, list_spec.path, selected)
            retained[list_spec.label] = len(selected)
            retained_indices[list_spec.label] = indices
        return output, retained, retained_indices

    empty, empty_counts, empty_indices = candidate(0.0)
    empty_tokens = count_tokens(canonical_context_json(empty))
    if empty_tokens > token_budget:
        expanded_lists = _candidate_lists(full, include_operator_metrics=True)
        if len(expanded_lists) == len(lists):
            raise ValueError(
                "Context metadata/operator trace exceeds token budget before retrievable records"
            )
        lists = expanded_lists
        original_counts = {
            candidate.label: len(candidate.values) for candidate in lists
        }
        empty, empty_counts, empty_indices = candidate(0.0)
        empty_tokens = count_tokens(canonical_context_json(empty))
        if empty_tokens > token_budget:
            raise ValueError(
                "Context metadata/operator trace exceeds token budget before retrievable records"
            )
    low, high = 0.0, 1.0
    best, best_counts, best_indices, best_tokens = (
        empty,
        empty_counts,
        empty_indices,
        empty_tokens,
    )
    for _ in range(24):
        middle = (low + high) / 2.0
        current, counts, indices = candidate(middle)
        tokens = count_tokens(canonical_context_json(current))
        if tokens <= token_budget:
            low = middle
            best, best_counts, best_indices, best_tokens = current, counts, indices, tokens
        else:
            high = middle

    # Use remaining slack deterministically while preserving each retriever's order.
    improved = True
    while improved:
        improved = False
        for list_spec in lists:
            key = list_spec.label
            values = list_spec.values
            if best_counts[key] >= len(values):
                continue
            retained_indices = set(best_indices[key])
            next_index = next(
                (index for index in range(len(values)) if index not in retained_indices),
                None,
            )
            if next_index is None:
                continue
            retained_indices.add(next_index)
            trial = copy.deepcopy(best)
            _path_set(
                trial,
                list_spec.path,
                [value for index, value in enumerate(values) if index in retained_indices],
            )
            tokens = count_tokens(canonical_context_json(trial))
            if tokens <= token_budget:
                best = trial
                best_counts[key] = len(retained_indices)
                best_indices[key] = retained_indices
                best_tokens = tokens
                improved = True
    rendered = canonical_context_json(best)
    return TokenBudgetResult(
        context=best,
        full_tokens=full_tokens,
        budgeted_tokens=best_tokens,
        token_budget=token_budget,
        retained_records=best_counts,
        original_records=original_counts,
        context_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
    )
