#!/usr/bin/env python3
"""Independent, conservative audit for the CiteWeave complex-graph experiment.

The experiment scorer evaluates structured JSON fields. This script adds checks that
are intentionally stricter and reports suspicious free-text relation claims for
manual review. It does not call an LLM and does not modify experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NODE_RE = re.compile(r"N\d+", re.IGNORECASE)
PAIR_RE = re.compile(
    r"(N\d+)\s*(?:-|–|—|→|↔|与|和|到|至)\s*(N\d+)", re.IGNORECASE
)
COMMUNITY_RE = re.compile(
    r"(N\d+)\s*[（(]\s*C?(\d+)\s*[）)]", re.IGNORECASE
)
NEGATION_WORDS = ("不存在", "无直连", "没有", "并非", "不是边", "不相连", "未连接", "无直接边")
CORRECTION_WORDS = ("修正", "更正", "错误", "有误", "应为", "重新")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left.upper(), right.upper())))


def parse_edge(value) -> tuple[str, str] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    left, right = str(value[0]).upper(), str(value[1]).upper()
    if not NODE_RE.fullmatch(left) or not NODE_RE.fullmatch(right):
        return None
    return edge_key(left, right)


def aliases(value) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        alias = str(item).upper()
        if NODE_RE.fullmatch(alias):
            output.append(alias)
    return output


def path_edges(path: list[str]) -> set[tuple[str, str]]:
    return {edge_key(left, right) for left, right in zip(path, path[1:])}


def structured_claims(parsed: dict, task_type: str, gold: dict) -> set[tuple[str, str]]:
    chosen = aliases(parsed.get("indirect_path" if task_type == "unsupported_edge" else "path"))
    claims = path_edges(chosen)
    for field in ("evidence_edges", "cross_community_edges"):
        for raw in parsed.get(field) or []:
            edge = parse_edge(raw)
            if edge:
                claims.add(edge)
    if task_type == "unsupported_edge" and parsed.get("direct_edge_supported") is True:
        queried = parse_edge(gold.get("queried_pair"))
        if queried:
            claims.add(queried)
    return claims


def text_relation_mentions(text: str) -> list[dict]:
    mentions = []
    for match in PAIR_RE.finditer(text or ""):
        start, end = match.span()
        before = (text or "")[max(0, start - 12) : start]
        after = (text or "")[end : min(len(text or ""), end + 14)]
        context = before + match.group(0) + after
        compact = re.sub(r"\s+", "", context)
        explicit_negation = any(word in compact for word in NEGATION_WORDS)
        explicit_negation = explicit_negation or bool(
            re.search(r"(?:^|[；，。])[^；，。]{0,8}无[^；，。]{0,8}N\d+", compact)
            or re.search(r"N\d+[^；，。]{0,8}无(?:直接)?边", compact)
        )
        explicit_negation = explicit_negation or "不成立" in compact or "断连" in compact
        removal_disconnect = (
            ("删除" in compact or "移除" in compact) and "断连" in compact
        )
        mentions.append(
            {
                "edge": edge_key(match.group(1), match.group(2)),
                "context": context,
                "negated": explicit_negation or removal_disconnect,
            }
        )
    return mentions


def audit_record(record: dict, sample: dict) -> list[dict]:
    findings = []
    parsed = record.get("parsed_answer") or {}
    task_type = sample["task_type"]
    graph_edges = {
        edge_key(edge["source"], edge["target"]) for edge in sample["graph"]["edges"]
    }
    communities = {
        node["alias"].upper(): str(node["community"]) for node in sample["graph"]["nodes"]
    }
    claims = structured_claims(parsed, task_type, sample["gold"])
    unsupported_structured = sorted(claims - graph_edges)
    stored_unsupported = int(record.get("score", {}).get("unsupported_claimed_edge_count") or 0)
    if len(unsupported_structured) != stored_unsupported:
        findings.append(
            {
                "severity": "error",
                "kind": "score_mismatch",
                "detail": f"stored unsupported={stored_unsupported}, recomputed={len(unsupported_structured)}",
            }
        )

    explanation = str(parsed.get("explanation") or "")
    for mention in text_relation_mentions(explanation):
        if mention["edge"] not in graph_edges and not mention["negated"]:
            duplicate = mention["edge"] in unsupported_structured
            findings.append(
                {
                    "severity": "review",
                    "kind": "unsupported_free_text_edge",
                    "detail": (
                        f"{mention['edge'][0]}-{mention['edge'][1]}; "
                        f"also_structured_claim={duplicate}; in: {mention['context']}"
                    ),
                }
            )

    for node, reported in COMMUNITY_RE.findall(explanation):
        node = node.upper()
        actual = communities.get(node)
        if actual is not None and actual != reported:
            findings.append(
                {
                    "severity": "error",
                    "kind": "wrong_community_in_text",
                    "detail": f"{node} reported C{reported}, actual C{actual}",
                }
            )

    correction_hits = [word for word in CORRECTION_WORDS if word in explanation]
    if correction_hits:
        findings.append(
            {
                "severity": "review",
                "kind": "self_correction_language",
                "detail": ", ".join(correction_hits),
            }
        )

    if task_type in {"path_trace", "cross_community_path", "unsupported_edge"}:
        field = "indirect_path" if task_type == "unsupported_edge" else "path"
        chosen_path = aliases(parsed.get(field))
        expected_path_edges = path_edges(chosen_path)
        reported_evidence = {
            edge for raw in parsed.get("evidence_edges") or [] if (edge := parse_edge(raw))
        }
        missing = sorted(expected_path_edges - reported_evidence)
        extra = sorted(reported_evidence - expected_path_edges)
        if missing or extra:
            findings.append(
                {
                    "severity": "error",
                    "kind": "path_evidence_mismatch",
                    "detail": f"missing={missing}; extra={extra}",
                }
            )

    if task_type == "bridge_node" and record.get("score", {}).get("task_correct") == 1:
        bridge = str(parsed.get("bridge_node") or "").upper()
        evidence = {
            edge for raw in parsed.get("evidence_edges") or [] if (edge := parse_edge(raw))
        }
        valid_incident = {edge for edge in evidence if edge in graph_edges and bridge in edge}
        if len(valid_incident) < 2:
            findings.append(
                {
                    "severity": "error",
                    "kind": "bridge_evidence_insufficient",
                    "detail": f"bridge={bridge}; valid incident evidence edges={sorted(valid_incident)}",
                }
            )

    if task_type == "cross_community_path":
        path = aliases(parsed.get("path"))
        expected = {
            edge_key(left, right)
            for left, right in zip(path, path[1:])
            if communities.get(left) != communities.get(right)
        }
        reported = {
            edge for raw in parsed.get("cross_community_edges") or [] if (edge := parse_edge(raw))
        }
        if expected != reported:
            findings.append(
                {
                    "severity": "info",
                    "kind": "cross_edge_field_inconsistent",
                    "detail": f"expected={sorted(expected)}; reported={sorted(reported)}",
                }
            )

    return findings


def fmt_rate(numerator: int, denominator: int) -> str:
    return "NA" if denominator == 0 else f"{numerator / denominator:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = {row["sample_id"]: row for row in load_jsonl(args.benchmark)}
    records = load_jsonl(args.records)
    findings_by_record: dict[tuple[str, str, int], list[dict]] = {}
    finding_counts = Counter()
    condition_totals = defaultdict(lambda: Counter(records=0, task_correct=0, strict_bridge=0))

    for record in records:
        key = (record["sample_id"], record["condition"], int(record.get("repeat", 1)))
        sample = samples.get(record["sample_id"])
        if sample is None:
            findings_by_record[key] = [
                {"severity": "error", "kind": "missing_sample", "detail": "sample not in benchmark"}
            ]
            continue
        findings = audit_record(record, sample)
        if findings:
            findings_by_record[key] = findings
        for finding in findings:
            finding_counts[finding["kind"]] += 1

        totals = condition_totals[record["condition"]]
        totals["records"] += 1
        totals["task_correct"] += int(record.get("score", {}).get("task_correct") or 0)
        strict_correct = int(record.get("score", {}).get("task_correct") or 0)
        if any(f["severity"] == "error" for f in findings):
            strict_correct = 0
        totals["strict_bridge"] += strict_correct

    lines = [
        "# Independent audit report",
        "",
        f"- Benchmark samples: {len(samples)}",
        f"- Result records: {len(records)}",
        f"- Records with one or more findings: {len(findings_by_record)}",
        "- `error` means a deterministic inconsistency; `review` is a conservative free-text flag and must be read manually.",
        "",
        "## Accuracy after additional deterministic checks",
        "",
        "| Condition | Original correct | Original accuracy | Conservative correct | Conservative accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in sorted(condition_totals):
        totals = condition_totals[condition]
        lines.append(
            f"| {condition} | {totals['task_correct']}/{totals['records']} | "
            f"{fmt_rate(totals['task_correct'], totals['records'])} | "
            f"{totals['strict_bridge']}/{totals['records']} | "
            f"{fmt_rate(totals['strict_bridge'], totals['records'])} |"
        )

    lines.extend(["", "## Finding counts", ""])
    if finding_counts:
        lines.extend(["| Kind | Count |", "|---|---:|"])
        for kind, count in sorted(finding_counts.items()):
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("No findings.")

    lines.extend(["", "## Record-level findings", ""])
    if not findings_by_record:
        lines.append("No record-level findings.")
    for key in sorted(findings_by_record):
        sample_id, condition, repeat = key
        lines.append(f"### {sample_id} / {condition} / repeat {repeat}")
        lines.append("")
        for finding in findings_by_record[key]:
            lines.append(
                f"- **{finding['severity']} / {finding['kind']}**: {finding['detail']}"
            )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Records={len(records)}; records_with_findings={len(findings_by_record)}")
    print("Finding counts:", dict(sorted(finding_counts.items())))


if __name__ == "__main__":
    main()
