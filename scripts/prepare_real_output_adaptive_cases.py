from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from citeweave.formal_adaptive_review import FormalAdaptiveCase
from citeweave.judge_protocol import canonical_json

ROOT = Path(__file__).resolve().parents[1]
TASK_RISKS = {
    "network_size": ("low", 0.15),
    "highest_occurrence": ("low", 0.30),
    "strongest_connection": ("high", 0.70),
    "cluster_membership": ("medium", 0.60),
    "unanswerable_false_premise": ("medium", 0.55),
}
INTERPRETIVE_RISK_MARKERS = (
    "suggest",
    "indicat",
    "reflect",
    "because",
    "due to",
    "therefore",
    "consistent with",
    "may be",
    "likely",
    "underscor",
)


def _parse_content(value: str) -> str:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return value.strip()
    return canonical_json(parsed)


def _completed_candidates(path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "complete":
            completed[row["item_id"]] = _parse_content(
                str(row["response"]["choices"][0]["message"]["content"])
            )
    return completed


def _bounded_unique_scope(text: str, *, prefer_interpretive: bool = False) -> str:
    """Select one exact visible risk span that a person could locally inspect."""

    if not text.strip():
        raise ValueError("Cannot identify a risk scope in an empty candidate")
    if len(text) <= 500:
        return text
    fragments: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if 20 <= len(paragraph) <= 500 and text.count(paragraph) == 1:
            fragments.append(paragraph)
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            if 20 <= len(sentence) <= 500 and text.count(sentence) == 1:
                fragments.append(sentence)
    if fragments:
        if prefer_interpretive:
            fragments.sort(
                key=lambda value: (
                    sum(marker in value.casefold() for marker in INTERPRETIVE_RISK_MARKERS),
                    "[e" not in value.casefold(),
                    len(value),
                ),
                reverse=True,
            )
        return fragments[0]
    for start in range(0, len(text) - 19, 500):
        fragment = text[start : start + 500]
        if text.count(fragment) == 1:
            return fragment
    raise ValueError("Could not identify one unique <=500-character risk scope")


def _graph_context(workspace: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (
            workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    return {
        item["item_id"]: json.loads(
            (workspace / item["graph_path"]).read_text(encoding="utf-8")
        )
        for item in manifest["contexts"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("experiments/human_references.yml"),
    )
    parser.add_argument("--run-id", default="formal_v2_nonthinking_20260806")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/formal_adaptive_cases/real_outputs.jsonl"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite cases: {args.output}")

    registry = yaml.safe_load(args.references.read_text(encoding="utf-8"))
    references = [
        item
        for item in registry["references"]
        if item.get("role") in {"development", "locked"}
    ]
    cases: list[FormalAdaptiveCase] = []
    for reference in references:
        dataset = reference["id"]
        role = reference["role"]
        workspace = ROOT / "experiments" / "formal_workspaces" / dataset
        report_path = (
            ROOT
            / "experiments"
            / "formal_reports"
            / dataset
            / "citeweave_full"
            / "report.md"
        )
        evidence_path = workspace / "evidence" / "evidence_items.json"
        report_candidate = report_path.read_text(encoding="utf-8")
        report_risk_scope = _bounded_unique_scope(
            report_candidate,
            prefer_interpretive=True,
        )
        cases.append(
            FormalAdaptiveCase(
                sample_id=f"{dataset}:report:grounding",
                dataset_id=dataset,
                topic_role=role,
                artifact_type="report",
                canonical_evidence=json.loads(evidence_path.read_text(encoding="utf-8")),
                anonymous_candidate=report_candidate,
                stage="report_finalization",
                issue_signature="report_claim_grounding",
                risk_notice_message=(
                    "The system flagged the quoted report span for claim-grounding "
                    "review. Assess and, if necessary, edit only that span against "
                    "the visible evidence."
                ),
                risk_scope_text=report_risk_scope,
                severity="high",
                detector_score=0.65,
                auto_accept_context_ok=True,
            )
        )

        benchmark = json.loads(
            (
                workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"
            ).read_text(encoding="utf-8")
        )
        contexts = _graph_context(workspace)
        candidates = _completed_candidates(
            ROOT
            / "experiments"
            / "formal_runs"
            / dataset
            / args.run_id
            / "graph_rag"
            / "items.jsonl"
        )
        selected: dict[str, dict[str, Any]] = {}
        for item in benchmark:
            selected.setdefault(item["task_type"], item)
        missing = set(TASK_RISKS) - set(selected)
        if missing:
            raise ValueError(f"{dataset}: missing adaptive task types {sorted(missing)}")
        for task_type, (severity, detector_score) in TASK_RISKS.items():
            item = selected[task_type]
            item_id = item["item_id"]
            graph_candidate = candidates[item_id]
            cases.append(
                FormalAdaptiveCase(
                    sample_id=f"{item_id}:grounding",
                    dataset_id=dataset,
                    topic_role=role,
                    artifact_type="graph",
                    canonical_evidence={
                        "question": item["question"],
                        "answer_contract": item["answer_contract"],
                        "graph_evidence": contexts[item_id],
                    },
                    anonymous_candidate=graph_candidate,
                    stage=f"graph_explanation:{task_type}",
                    issue_signature=f"graph:{task_type}",
                    risk_notice_message=(
                        f"The system flagged the quoted {task_type} answer for "
                        "focused factual review. Assess and, if necessary, edit "
                        "only that span against the visible graph evidence."
                    ),
                    risk_scope_text=_bounded_unique_scope(graph_candidate),
                    severity=severity,
                    detector_score=detector_score,
                    auto_accept_context_ok=True,
                )
            )

    payload = "".join(
        canonical_json(case.model_dump(mode="json")) + "\n" for case in cases
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"{len(cases)} real-output cases -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
