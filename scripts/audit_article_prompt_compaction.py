from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from citeweave.article_compiler_v4 import (
    build_compiler_repair_request,
    build_compiler_section_request,
)
from citeweave.article_context_compaction import (
    audit_section_writer_view,
    section_writer_view,
)
from citeweave.io import read_json, sha256_file, write_json

CALL_SEQUENCE = (
    ("Methods", "base"),
    ("Results", "base"),
    ("Results", "repair"),
    ("Discussion", "base"),
    ("Discussion", "repair"),
    ("Limitations", "base"),
    ("Conclusion", "base"),
    ("Introduction", "base"),
    ("Abstract", "base"),
)


def _user_chars(request: dict[str, Any]) -> int:
    return len(str(request["messages"][1]["content"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.source_plan)
    records = []
    condition_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "original_prompt_chars": 0,
            "compact_prompt_chars": 0,
            "observed_original_prompt_tokens": 0,
        }
    )
    all_view_audits = []
    for article in plan["articles"]:
        writer_input = read_json(Path(article["writer_input"]))
        compiled: dict[str, str] = {}
        base_bodies: dict[str, str] = {}
        call_rows = []
        for index, (section, pass_type) in enumerate(CALL_SEQUENCE, start=1):
            call_dir = Path(article["output_dir"]) / "calls" / (
                f"{index:02d}_{section.casefold()}_{pass_type}"
            )
            original_request = read_json(call_dir / "request.json")
            execution = read_json(call_dir / "execution_record.json")
            body = (call_dir / "section.md").read_text(encoding="utf-8").strip()
            if pass_type == "base":
                compact_request = build_compiler_section_request(
                    writer_input,
                    condition=article["condition"],
                    section=section,
                    compiled_sections=compiled,
                )
                base_bodies[section] = body
            else:
                compact_request = build_compiler_repair_request(
                    writer_input,
                    condition=article["condition"],
                    section=section,
                    original_body=base_bodies[section],
                    compiled_sections=compiled,
                )
            if pass_type == "repair" or section not in {"Results", "Discussion"}:
                compiled[section] = body
            original_chars = _user_chars(original_request)
            compact_chars = _user_chars(compact_request)
            prompt_tokens = int(
                (execution.get("usage") or {}).get("prompt_tokens") or 0
            )
            call_rows.append(
                {
                    "section": section,
                    "pass_type": pass_type,
                    "original_prompt_chars": original_chars,
                    "compact_prompt_chars": compact_chars,
                    "char_reduction_fraction": 1 - compact_chars / original_chars,
                    "observed_original_prompt_tokens": prompt_tokens,
                }
            )
            totals = condition_totals[article["condition"]]
            totals["original_prompt_chars"] += original_chars
            totals["compact_prompt_chars"] += compact_chars
            totals["observed_original_prompt_tokens"] += prompt_tokens
            view = section_writer_view(writer_input, section=section)
            all_view_audits.append(audit_section_writer_view(writer_input, view))
        records.append(
            {
                "article_id": article["article_id"],
                "dataset_id": article["dataset_id"],
                "condition": article["condition"],
                "calls": call_rows,
            }
        )
    original_chars = sum(
        row["original_prompt_chars"] for row in condition_totals.values()
    )
    compact_chars = sum(
        row["compact_prompt_chars"] for row in condition_totals.values()
    )
    observed_tokens = sum(
        row["observed_original_prompt_tokens"] for row in condition_totals.values()
    )
    for totals in condition_totals.values():
        totals["char_reduction_fraction"] = 1 - (
            totals["compact_prompt_chars"] / totals["original_prompt_chars"]
        )
    report = {
        "schema_version": 1,
        "status": "article_prompt_compaction_offline_audited",
        "scientific_role": "cost_and_integrity_diagnostic_not_provider_result",
        "source_plan_sha256": sha256_file(args.source_plan),
        "articles": len(records),
        "calls": sum(len(row["calls"]) for row in records),
        "all_section_views_pass_integrity_audit": all(
            row["passed"] for row in all_view_audits
        ),
        "original_prompt_chars": original_chars,
        "compact_prompt_chars": compact_chars,
        "char_reduction_fraction": 1 - compact_chars / original_chars,
        "observed_original_prompt_tokens": observed_tokens,
        "linear_char_ratio_prompt_token_estimate": round(
            observed_tokens * compact_chars / original_chars
        ),
        "estimate_warning": (
            "The token estimate is a character-ratio diagnostic, not a provider tokenizer "
            "measurement and not a billing guarantee."
        ),
        "condition_totals": dict(condition_totals),
        "section_view_audits": all_view_audits,
        "records": records,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
