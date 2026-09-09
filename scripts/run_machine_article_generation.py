from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from citeweave.article_generation import REQUIRED_SECTIONS
from citeweave.io import read_json, sha256_file, write_json

TOKEN_PATTERN = re.compile(r"\b(?:PH|REF)-[A-Za-z0-9_-]+\b")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _api_key(path: Path, env_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value.strip()
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, candidate = line.partition(":")
        if separator and key.strip().casefold() == "deepseek" and candidate.strip():
            return candidate.strip()
    raise RuntimeError(f"Missing {env_name}; no deepseek entry found in key file")


def _section_text(article: str, section: str) -> str:
    match = re.search(
        rf"^#+\s+{re.escape(section)}\s*$\n(.*?)(?=^#+\s+|\Z)",
        article,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _prose_paragraphs(section: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", section)
        if paragraph.strip()
        and not paragraph.lstrip().startswith(("#", "|", "```"))
    ]


def assess_generated_article(
    article: str,
    *,
    writer_input: dict[str, Any],
) -> dict[str, Any]:
    permitted_low, permitted_high = writer_input["writing_brief"]["permitted_range"]
    words = len(WORD_PATTERN.findall(article))
    missing_sections = [
        section for section in REQUIRED_SECTIONS if not _section_text(article, section)
    ]
    allowed_ph = {row["phenomenon_id"] for row in writer_input["graph_phenomena"]}
    phenomenon_by_type = {
        row["task_type"]: row["phenomenon_id"]
        for row in writer_input["graph_phenomena"]
    }
    required_types = {
        "multi_hop_connector",
        "bridge_counterfactual",
        "community_role_contrast",
        "hub_removal_resilience",
        "temporal_structural_shift",
    }
    if set(phenomenon_by_type) != required_types:
        raise ValueError("Article assessment requires all five registered phenomenon types")
    registered_synthesis_pairs = {
        "connectivity_redundancy": frozenset(
            {
                phenomenon_by_type["multi_hop_connector"],
                phenomenon_by_type["bridge_counterfactual"],
            }
        ),
        "centrality_resilience": frozenset(
            {
                phenomenon_by_type["community_role_contrast"],
                phenomenon_by_type["hub_removal_resilience"],
            }
        ),
        "temporal_topology": frozenset(
            {
                phenomenon_by_type["temporal_structural_shift"],
                phenomenon_by_type["community_role_contrast"],
            }
        ),
    }
    allowed_refs = {
        row["reference_id"] for row in writer_input["representative_sources"]
    }
    observed_tokens = set(TOKEN_PATTERN.findall(article))
    unexpected_tokens = sorted(observed_tokens - allowed_ph - allowed_refs)
    results_tokens = set(TOKEN_PATTERN.findall(_section_text(article, "Results")))
    discussion_tokens = set(TOKEN_PATTERN.findall(_section_text(article, "Discussion")))
    conclusion_tokens = set(TOKEN_PATTERN.findall(_section_text(article, "Conclusion")))
    phenomena_in_results = allowed_ph & results_tokens
    phenomena_in_discussion = allowed_ph & discussion_tokens
    phenomena_in_conclusion = allowed_ph & conclusion_tokens
    synthesis_hits: set[str] = set()
    synthesis_paragraphs: list[dict[str, Any]] = []
    for section_name in ("Results", "Discussion"):
        for paragraph_index, paragraph in enumerate(
            _prose_paragraphs(_section_text(article, section_name)), start=1
        ):
            paragraph_phenomena = allowed_ph & set(TOKEN_PATTERN.findall(paragraph))
            matched = sorted(
                synthesis_id
                for synthesis_id, pair in registered_synthesis_pairs.items()
                if pair.issubset(paragraph_phenomena)
            )
            if matched and 2 <= len(paragraph_phenomena) < len(allowed_ph):
                synthesis_hits.update(matched)
                synthesis_paragraphs.append(
                    {
                        "section": section_name,
                        "paragraph_index": paragraph_index,
                        "synthesis_ids": matched,
                        "phenomenon_ids": sorted(paragraph_phenomena),
                    }
                )
    gates = {
        "word_range": permitted_low <= words <= permitted_high,
        "required_sections": not missing_sections,
        "no_unregistered_evidence_tokens": not unexpected_tokens,
        "all_phenomena_used": allowed_ph.issubset(observed_tokens),
        "all_five_phenomena_in_results": allowed_ph.issubset(phenomena_in_results),
        "at_least_four_phenomena_in_discussion": len(phenomena_in_discussion) >= 4,
        "at_least_three_phenomena_in_conclusion": len(phenomena_in_conclusion) >= 3,
        "at_least_two_registered_cross_phenomenon_syntheses": len(synthesis_hits) >= 2,
        "cross_phenomenon_reasoning_spans_at_least_two_paragraphs": (
            len(synthesis_paragraphs) >= 2
        ),
        "no_rendered_image_markup": "![" not in article,
    }
    return {
        "schema_version": 1,
        "passed": all(gates.values()),
        "word_count": words,
        "permitted_word_range": [permitted_low, permitted_high],
        "missing_sections": missing_sections,
        "unexpected_evidence_tokens": unexpected_tokens,
        "phenomena_in_results": sorted(phenomena_in_results),
        "phenomena_in_discussion": sorted(phenomena_in_discussion),
        "phenomena_in_conclusion": sorted(phenomena_in_conclusion),
        "registered_synthesis_hits": sorted(synthesis_hits),
        "cross_phenomenon_paragraphs": synthesis_paragraphs,
        "quality_gates": gates,
    }


def _validate_plan(
    plan_path: Path,
    protocol_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    protocol_hash = sha256_file(protocol_path)
    plan_hash = sha256_file(plan_path)
    freeze = read_json(freeze_path)
    if freeze.get("sha256") != protocol_hash:
        raise RuntimeError("Machine article generation protocol differs from freeze")
    if freeze.get("plan_sha256") != plan_hash:
        raise RuntimeError("Machine article generation plan differs from freeze")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("frozen_plan", {}).get("sha256") != plan_hash:
        raise RuntimeError("Machine article protocol does not bind the selected plan")
    for label, artifact in (protocol.get("implementation") or {}).items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Machine article implementation mismatch: {label}")
    plan = read_json(plan_path)
    if plan.get("status") != "machine_article_generation_plan_frozen":
        raise RuntimeError("Machine article generation plan is not frozen")
    if len(plan.get("cells") or []) != 16:
        raise RuntimeError("Machine article generation plan must contain 16 cells")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path("apikey.md"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = _validate_plan(args.plan, args.protocol, args.protocol_freeze)
    selected = [
        cell
        for cell in plan["cells"]
        if (not args.dataset or cell["dataset_id"] in args.dataset)
        and (not args.condition or cell["condition"] in args.condition)
    ]
    if args.max_cells is not None:
        selected = selected[: args.max_cells]
    audit = {
        "schema_version": 1,
        "status": "ready_to_execute" if not args.execute else "executing",
        "plan_sha256": sha256_file(args.plan),
        "selected_cells": len(selected),
        "already_terminal": 0,
        "executed": 0,
        "transport_failures": 0,
        "generation_gate_failures": 0,
    }
    if not args.execute:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    key = _api_key(args.api_key_file, args.api_key_env)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=600, follow_redirects=True) as client:
        for cell in selected:
            output_dir = Path(cell["output_dir"])
            record_path = output_dir / "execution_record.json"
            if record_path.is_file():
                existing = read_json(record_path)
                if existing.get("request_sha256") != cell["request_sha256"]:
                    raise RuntimeError(f"Existing cell request mismatch: {cell['cell_id']}")
                if existing.get("response_received") is True:
                    audit["already_terminal"] += 1
                    continue
            request_path = Path(cell["request"])
            if sha256_file(request_path) != cell["request_sha256"]:
                raise RuntimeError(f"Request changed after plan freeze: {cell['cell_id']}")
            writer_input_path = Path(cell["writer_input"])
            if sha256_file(writer_input_path) != cell["writer_input_sha256"]:
                raise RuntimeError(f"Writer input changed after plan freeze: {cell['cell_id']}")
            request = read_json(request_path)
            started_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                )
                response.raise_for_status()
                raw = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                audit["transport_failures"] += 1
                write_json(
                    record_path,
                    {
                        "schema_version": 1,
                        "cell_id": cell["cell_id"],
                        "request_sha256": cell["request_sha256"],
                        "response_received": False,
                        "started_at": started_at,
                        "completed_at": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error_type": type(error).__name__,
                        "error": str(error).replace(key, "***")[:1000],
                    },
                )
                continue
            if key in json.dumps(raw, ensure_ascii=False):
                raise RuntimeError("Provider response unexpectedly contains the API key")
            raw_path = output_dir / "raw_response.json"
            write_json(raw_path, raw)
            article = str(raw["choices"][0]["message"]["content"]).strip() + "\n"
            article_path = output_dir / "draft.md"
            article_path.write_text(article, encoding="utf-8")
            assessment = assess_generated_article(
                article,
                writer_input=read_json(writer_input_path),
            )
            assessment_path = output_dir / "draft_assessment.json"
            write_json(assessment_path, assessment)
            terminal_status = (
                "draft_ready_for_human_review"
                if assessment["passed"]
                and cell["condition"] == "citeweave_graph_review"
                else "one_shot_candidate_complete"
                if assessment["passed"]
                else "generation_quality_gate_failed"
            )
            if not assessment["passed"]:
                audit["generation_gate_failures"] += 1
            write_json(
                record_path,
                {
                    "schema_version": 1,
                    "cell_id": cell["cell_id"],
                    "dataset_id": cell["dataset_id"],
                    "condition": cell["condition"],
                    "status": terminal_status,
                    "request_sha256": cell["request_sha256"],
                    "writer_input_sha256": cell["writer_input_sha256"],
                    "blueprint_sha256": cell["blueprint_sha256"],
                    "response_received": True,
                    "generation_requests": 1,
                    "rendered_figure_access": False,
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "raw_response_sha256": sha256_file(raw_path),
                    "draft_sha256": sha256_file(article_path),
                    "draft_assessment_sha256": sha256_file(assessment_path),
                    "model": raw.get("model", request["model"]),
                    "finish_reason": raw["choices"][0].get("finish_reason"),
                    "usage": raw.get("usage"),
                    "provider_response_id_sha256": hashlib.sha256(
                        str(raw.get("id", "")).encode()
                    ).hexdigest(),
                },
            )
            audit["executed"] += 1
    audit["status"] = (
        "execution_pass_complete"
        if audit["transport_failures"] == 0
        else "transport_failures_retry_same_requests"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
