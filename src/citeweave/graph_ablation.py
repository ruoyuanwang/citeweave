from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

import pandas as pd

from .analytics import AnalysisBundle
from .exceptions import ConfigurationError
from .graph_explanation import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    QwenVisionClient,
    _parse_json,
    alias_graph,
    displayed_graph,
    explain_network,
    score_response,
    verify_response,
)
from .io import atomic_write_bytes, read_json, sha256_file, write_json, write_jsonl
from .models import GraphExplanationPolicy
from .visualization import FigureArtifact

ABLATION_MODES = ("vlm", "flat_kg", "graph_rag")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return round(float(numerator) / float(denominator), 4) if denominator else None


def _mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return round(fmean(cleaned), 4) if cleaned else None


def _std(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return round(pstdev(cleaned), 4) if len(cleaned) > 1 else (0.0 if cleaned else None)


def _aggregate_mode(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    mode_records = [record for record in records if record["mode"] == mode]
    successful = [record for record in mode_records if record["status"] == "complete"]
    reported_claims = sum(int(record["metrics"]["reported_claims"]) for record in successful)
    verified_claims = sum(int(record["metrics"]["verified_claims"]) for record in successful)
    reported_edges = sum(int(record["metrics"]["reported_edges"]) for record in successful)
    unsupported_edges = sum(int(record["metrics"]["unsupported_edges"]) for record in successful)
    reported_paths = sum(
        int(record["metrics"]["reported_multi_hop_claims"]) for record in successful
    )
    valid_paths = sum(
        int(record["metrics"]["valid_multi_hop_claims"]) for record in successful
    )
    complex_values = [
        int(record["metrics"]["verified_complex_claims"]) for record in successful
    ]
    return {
        "mode": mode,
        "runs": len(mode_records),
        "successful_runs": len(successful),
        "failed_runs": len(mode_records) - len(successful),
        "run_completion_rate": _safe_ratio(len(successful), len(mode_records)),
        "reported_claims": reported_claims,
        "verified_claims": verified_claims,
        "claim_support_rate": _safe_ratio(verified_claims, reported_claims),
        "mean_claim_support_rate": _mean(
            [record["metrics"]["claim_support_rate"] for record in successful]
        ),
        "std_claim_support_rate": _std(
            [record["metrics"]["claim_support_rate"] for record in successful]
        ),
        "reported_edges": reported_edges,
        "unsupported_edges": unsupported_edges,
        "edge_hallucination_rate": _safe_ratio(unsupported_edges, reported_edges),
        "reported_multi_hop_claims": reported_paths,
        "valid_multi_hop_claims": valid_paths,
        "path_validity_rate": _safe_ratio(valid_paths, reported_paths),
        "mean_verified_complex_claims": _mean(complex_values),
        "std_verified_complex_claims": _std(complex_values),
        "mean_abstention_rate": _mean(
            [record["metrics"].get("abstention_rate") for record in successful]
        ),
        "mean_reported_slot_coverage": _mean(
            [record["metrics"].get("reported_slot_coverage") for record in successful]
        ),
        "mean_verified_slot_coverage": _mean(
            [record["metrics"].get("verified_slot_coverage") for record in successful]
        ),
        "mean_duration_seconds": _mean(
            [record.get("duration_seconds") for record in successful]
        ),
        "prompt_tokens": sum(
            int(record.get("usage", {}).get("prompt_tokens") or 0) for record in successful
        ),
        "completion_tokens": sum(
            int(record.get("usage", {}).get("completion_tokens") or 0)
            for record in successful
        ),
        "total_tokens": sum(
            int(record.get("usage", {}).get("total_tokens") or 0) for record in successful
        ),
    }


def _comparison(summary: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["mode"]: row for row in summary}
    baseline = lookup.get("vlm") or {}
    flat_kg = lookup.get("flat_kg") or {}
    graph_rag = lookup.get("graph_rag") or {}

    def delta(field: str, reference: dict[str, Any]) -> float | None:
        left, right = graph_rag.get(field), reference.get(field)
        if left is None or right is None:
            return None
        return round(float(left) - float(right), 4)

    flat_prompt_tokens = float(flat_kg.get("prompt_tokens") or 0)
    graph_prompt_tokens = float(graph_rag.get("prompt_tokens") or 0)

    return {
        "graph_rag_minus_vlm_claim_support_rate": delta("claim_support_rate", baseline),
        "graph_rag_minus_vlm_edge_hallucination_rate": delta(
            "edge_hallucination_rate", baseline
        ),
        "graph_rag_minus_vlm_path_validity_rate": delta("path_validity_rate", baseline),
        "graph_rag_minus_vlm_mean_verified_complex_claims": delta(
            "mean_verified_complex_claims", baseline
        ),
        "graph_rag_minus_vlm_verified_slot_coverage": delta(
            "mean_verified_slot_coverage", baseline
        ),
        "graph_rag_minus_flat_kg_claim_support_rate": delta(
            "claim_support_rate", flat_kg
        ),
        "graph_rag_minus_flat_kg_edge_hallucination_rate": delta(
            "edge_hallucination_rate", flat_kg
        ),
        "graph_rag_minus_flat_kg_path_validity_rate": delta(
            "path_validity_rate", flat_kg
        ),
        "graph_rag_minus_flat_kg_mean_verified_complex_claims": delta(
            "mean_verified_complex_claims", flat_kg
        ),
        "graph_rag_minus_flat_kg_verified_slot_coverage": delta(
            "mean_verified_slot_coverage", flat_kg
        ),
        "graph_rag_prompt_token_reduction_vs_flat_kg": (
            round((flat_prompt_tokens - graph_prompt_tokens) / flat_prompt_tokens, 4)
            if flat_prompt_tokens
            else None
        ),
        "interpretation": (
            "Descriptive pilot comparison only; repeated graph instances are required for "
            "inferential claims."
        ),
    }


def _format_rate(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _summary_markdown(
    summary: list[dict[str, Any]], comparison: dict[str, Any], manifest: dict[str, Any]
) -> str:
    lines = [
        "# CiteWeave 系统内图解释消融实验",
        "",
        (
            "三组执行同一个 CiteWeave 图解释节点，只改变模型可见的图证据。"
            "Direct VLM组同时获得仅用于统一输出标识的alias-label表。"
        ),
        "",
        (
            "| 方法 | 成功/总运行 | 声明支持率 | 边幻觉率 | 路径有效率 | "
            "有效槽位覆盖 | 每次有效复杂声明 | 弃答率 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        complex_value = row["mean_verified_complex_claims"]
        lines.append(
            "| {mode} | {successful_runs}/{runs} | {claim_rate} | {edge_rate} | "
            "{path_rate} | {slot_rate} | {complex_value} | {abstention_rate} |".format(
                **row,
                claim_rate=_format_rate(row["claim_support_rate"]),
                edge_rate=_format_rate(row["edge_hallucination_rate"]),
                path_rate=_format_rate(row["path_validity_rate"]),
                slot_rate=_format_rate(row["mean_verified_slot_coverage"]),
                complex_value="—" if complex_value is None else f"{complex_value:.2f}",
                abstention_rate=_format_rate(row["mean_abstention_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## 实验配置",
            "",
            f"- 模型：`{manifest['model']}`",
            f"- 温度：`{manifest['temperature']}`",
            f"- 网络：`{', '.join(manifest['networks'])}`",
            f"- 每组重复：`{manifest['repeats']}`",
            f"- 显示节点上限：`{manifest['max_nodes']}`",
            "",
            "## GraphRAG 相对 Direct VLM 的描述性差值",
            "",
            f"- 声明支持率：{comparison['graph_rag_minus_vlm_claim_support_rate']}",
            f"- 边幻觉率：{comparison['graph_rag_minus_vlm_edge_hallucination_rate']}",
            f"- 路径有效率：{comparison['graph_rag_minus_vlm_path_validity_rate']}",
            (
                "- 每次有效复杂声明："
                f"{comparison['graph_rag_minus_vlm_mean_verified_complex_claims']}"
            ),
            (
                "- 有效任务槽覆盖："
                f"{comparison['graph_rag_minus_vlm_verified_slot_coverage']}"
            ),
            "",
            "## GraphRAG 相对 Flat KG 的描述性差值",
            "",
            f"- 声明支持率：{comparison['graph_rag_minus_flat_kg_claim_support_rate']}",
            (
                "- 边幻觉率："
                f"{comparison['graph_rag_minus_flat_kg_edge_hallucination_rate']}"
            ),
            f"- 路径有效率：{comparison['graph_rag_minus_flat_kg_path_validity_rate']}",
            (
                "- 每次有效复杂声明："
                f"{comparison['graph_rag_minus_flat_kg_mean_verified_complex_claims']}"
            ),
            (
                "- 有效任务槽覆盖："
                f"{comparison['graph_rag_minus_flat_kg_verified_slot_coverage']}"
            ),
            (
                "- Prompt Token降幅："
                f"{comparison['graph_rag_prompt_token_reduction_vs_flat_kg']}"
            ),
            "",
            (
                "> 当前结果是初步描述性实验。单张图上的重复运行不能替代多个独立图实例，"
                "不能据此直接声称统计显著性。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _freeze_case(network: Any, figure: FigureArtifact, max_nodes: int) -> dict[str, Any]:
    graph, lookup = displayed_graph(network, max_nodes)
    _, alias_to_id, nodes, edges = alias_graph(graph, lookup)
    identity = {
        "network": network.name,
        "figure_sha256": sha256_file(figure.png),
        "max_nodes": max_nodes,
        "alias_to_id": alias_to_id,
        "nodes": nodes,
        "edges": edges,
    }
    # The absolute figure path is provenance, not identity.  Excluding it prevents a
    # copied project from being counted as a second independent graph instance.
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return {
        "case_id": hashlib.sha256(payload).hexdigest(),
        "figure": str(figure.png),
        **identity,
    }


def _balanced_execution_jobs(
    pairs: list[tuple[str, Any, FigureArtifact, str]], repeats: int
) -> list[tuple[str, str, Any, FigureArtifact, str, int]]:
    """Interleave methods with a deterministic Latin-square order.

    Each method occupies every order position once per three repeats.  The case hash
    rotates the starting method across independent graphs, reducing API time drift
    without sacrificing reproducibility.
    """

    jobs: list[tuple[str, str, Any, FigureArtifact, str, int]] = []
    for repeat in range(1, repeats + 1):
        for network_name, network, figure, case_id in pairs:
            offset = (int(case_id[:8], 16) + repeat - 1) % len(ABLATION_MODES)
            ordered_modes = ABLATION_MODES[offset:] + ABLATION_MODES[:offset]
            for position, mode in enumerate(ordered_modes, start=1):
                jobs.append((mode, network_name, network, figure, case_id, repeat))
    return jobs


def run_graph_ablation(
    analyses: AnalysisBundle,
    figures: list[FigureArtifact],
    policy: GraphExplanationPolicy,
    *,
    max_nodes: int,
    repeats: int,
    output: Path,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run three input ablations on the same in-system graph explanation task."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError(f"Ablation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figure_lookup = {figure.name: figure for figure in figures}
    pairs = []
    cases = []
    for network_name in policy.networks:
        network = analyses.networks.get(network_name)
        figure = figure_lookup.get(f"network_{network_name}")
        if network is None or figure is None:
            raise ConfigurationError(
                f"Missing saved network or figure for ablation: {network_name}"
            )
        case = _freeze_case(network, figure, max_nodes)
        pairs.append((network_name, network, figure, case["case_id"]))
        cases.append(case)

    active_policy = policy.model_copy(update={"mode": "graph_rag"})
    active_client = client or QwenVisionClient(active_policy)
    jobs = _balanced_execution_jobs(pairs, repeats)
    started_at = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "started_at": started_at,
        "modes": list(ABLATION_MODES),
        "repeats": repeats,
        "model": policy.model,
        "base_url": policy.base_url,
        "temperature": policy.temperature,
        "networks": list(policy.networks),
        "max_nodes": max_nodes,
        "max_paths": policy.max_paths,
        "max_hops": policy.max_hops,
        "execution_order_strategy": "deterministic_case_rotated_latin_square",
        "protocol_version": "2.1-copy-ready-path-focus-normalization",
        "request_parameters": {
            "temperature": policy.temperature,
            "stream": False,
            "enable_thinking": False,
            "seed": None,
            "max_tokens": None,
        },
        "execution_order": [
            {
                "position": position,
                "mode": mode,
                "network": network_name,
                "case_id": case_id,
                "repeat": repeat,
            }
            for position, (
                mode,
                network_name,
                _network,
                _figure,
                case_id,
                repeat,
            ) in enumerate(jobs, start=1)
        ],
        "audit_hashes": {
            "output_schema_sha256": hashlib.sha256(
                json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "graph_explanation_code_sha256": sha256_file(
                Path(__file__).with_name("graph_explanation.py")
            ),
            "graph_ablation_code_sha256": sha256_file(Path(__file__)),
        },
        "cases": [
            {
                "network": case["network"],
                "case_id": case["case_id"],
                "figure": case["figure"],
                "figure_sha256": case["figure_sha256"],
            }
            for case in cases
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    for case in cases:
        write_json(output / "cases" / f"{case['network']}.json", case)

    records: list[dict[str, Any]] = []
    for execution_position, (
        mode,
        network_name,
        network,
        figure,
        case_id,
        repeat,
    ) in enumerate(jobs, start=1):
        mode_policy = policy.model_copy(update={"mode": mode})
        run_dir = output / "runs" / mode / network_name
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / f"repeat-{repeat:02d}.json"
        started = perf_counter()
        try:
            result = explain_network(
                network,
                figure,
                mode_policy,
                max_nodes=max_nodes,
                client=active_client,
            )
            write_json(result_path, result)
            duration = round(perf_counter() - started, 4)
            prompt = str(result.get("prompt") or "")
            records.append(
                {
                    "mode": mode,
                    "network": network_name,
                    "case_id": case_id,
                    "repeat": repeat,
                    "execution_position": execution_position,
                    "status": result["status"],
                    "duration_seconds": duration,
                    "metrics": result.get("metrics") or {},
                    "verification": result.get("verification") or {},
                    "usage": result.get("usage") or {},
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "result_file": str(result_path.relative_to(output)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - persist partial runs for audit.
            error = {
                "mode": mode,
                "network": network_name,
                "case_id": case_id,
                "repeat": repeat,
                "execution_position": execution_position,
                "status": "error",
                "duration_seconds": round(perf_counter() - started, 4),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "result_file": str(result_path.relative_to(output)),
            }
            write_json(result_path, error)
            records.append(error)

    write_jsonl(output / "records.jsonl", records)
    summary = [_aggregate_mode(mode, records) for mode in ABLATION_MODES]
    comparison = _comparison(summary)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary.json", {"groups": summary, "comparison": comparison})
    manifest["finished_at"] = datetime.now(UTC)
    manifest["successful_runs"] = sum(record["status"] == "complete" for record in records)
    manifest["failed_runs"] = sum(record["status"] != "complete" for record in records)
    manifest["prompt_hashes"] = sorted(
        {
            record["prompt_sha256"]
            for record in records
            if record.get("prompt_sha256")
        }
    )
    write_json(output / "run_manifest.json", manifest)
    markdown = _summary_markdown(summary, comparison, manifest)
    atomic_write_bytes(output / "summary.md", markdown.encode("utf-8"))
    return {
        "output": str(output),
        "runs": len(records),
        "successful_runs": manifest["successful_runs"],
        "failed_runs": manifest["failed_runs"],
        "summary": str(output / "summary.csv"),
        "report": str(output / "summary.md"),
        "comparison": comparison,
    }


def rescore_graph_ablation(source: Path, output: Path | None = None) -> dict[str, Any]:
    """Reapply the current deterministic verifier without another model call."""
    source = source.resolve()
    manifest = read_json(source / "run_manifest.json")
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = source / f"rescored-{timestamp}"
    if output.exists() and any(output.iterdir()):
        raise ConfigurationError(f"Rescore output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        path.stem: read_json(path) for path in (source / "cases").glob("*.json")
    }
    if not cases:
        raise ConfigurationError(f"No frozen cases found in {source / 'cases'}")

    records: list[dict[str, Any]] = []
    for mode in manifest["modes"]:
        for network_name in manifest["networks"]:
            case = cases[network_name]
            source_dir = source / "runs" / mode / network_name
            target_dir = output / "runs" / mode / network_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_path in sorted(source_dir.glob("repeat-*.json")):
                target_path = target_dir / source_path.name
                original = read_json(source_path)
                if original.get("status") != "complete" or not original.get("raw_answer"):
                    error = {
                        "mode": mode,
                        "network": network_name,
                        "repeat": int(source_path.stem.removeprefix("repeat-")),
                        "status": "error",
                        "error_type": "MissingRawAnswer",
                        "error": "Source run was incomplete or has no raw answer.",
                        "case_id": case["case_id"],
                        "result_file": str(target_path.relative_to(output)),
                    }
                    write_json(target_path, error)
                    records.append(error)
                    continue
                parsed = _parse_json(original["raw_answer"])
                verified, rejected = verify_response(
                    parsed,
                    case["alias_to_id"],
                    case["nodes"],
                    case["edges"],
                )
                metrics = score_response(parsed, case["nodes"], case["edges"], verified)
                rescored = {
                    **original,
                    "verified_claims": verified,
                    "rejected_claims": rejected,
                    "verification": {
                        "reported_claims": metrics["reported_claims"],
                        "verified_claims": metrics["verified_claims"],
                        "rejected_claims": metrics["reported_claims"]
                        - metrics["verified_claims"],
                        "claim_support_rate": metrics["claim_support_rate"],
                    },
                    "metrics": metrics,
                    "rescored_at": datetime.now(UTC),
                    "original_verification": original.get("verification") or {},
                    "original_metrics": original.get("metrics") or {},
                }
                write_json(target_path, rescored)
                repeat = int(source_path.stem.removeprefix("repeat-"))
                records.append(
                    {
                        "mode": mode,
                        "network": network_name,
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "status": "complete",
                        "duration_seconds": original.get("duration_seconds"),
                        "metrics": metrics,
                        "verification": rescored["verification"],
                        "usage": original.get("usage") or {},
                        "result_file": str(target_path.relative_to(output)),
                    }
                )

    write_jsonl(output / "records.jsonl", records)
    summary = [_aggregate_mode(mode, records) for mode in ABLATION_MODES]
    comparison = _comparison(summary)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary.json", {"groups": summary, "comparison": comparison})
    rescore_manifest = {
        **manifest,
        "source_experiment": str(source),
        "rescored_at": datetime.now(UTC),
        "verifier_note": (
            "Raw model responses were not regenerated. The current deterministic verifier "
            "was applied to the frozen case."
        ),
        "known_prompt_issue": (
            "The source prompt allowed the model to emit hub_or_bridge; that unsupported "
            "combined type remains rejected in this rescore."
        ),
        "successful_runs": sum(record["status"] == "complete" for record in records),
        "failed_runs": sum(record["status"] != "complete" for record in records),
    }
    write_json(output / "run_manifest.json", rescore_manifest)
    for case in cases.values():
        write_json(output / "cases" / f"{case['network']}.json", case)
    markdown = _summary_markdown(summary, comparison, rescore_manifest)
    markdown += (
        "\n## 重评分说明\n\n"
        "本目录没有重新调用模型，而是对冻结的原始回答应用当前确定性核验器。"
        "源实验中三组均输出了不受支持的组合类型 `hub_or_bridge`；该槽位在重评分中仍被拒绝，"
        "没有进行事后类型推断。\n"
    )
    atomic_write_bytes(output / "summary.md", markdown.encode("utf-8"))
    return {
        "source": str(source),
        "output": str(output),
        "runs": len(records),
        "successful_runs": rescore_manifest["successful_runs"],
        "failed_runs": rescore_manifest["failed_runs"],
        "summary": str(output / "summary.csv"),
        "report": str(output / "summary.md"),
        "comparison": comparison,
    }
