from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets_openalex_title_abstract.yml"
DEFAULT_REPORT = ROOT / "experiments" / "final_report" / "end_to_end_report.md"
DEFAULT_MANIFEST = ROOT / "experiments" / "final_report" / "manifest.json"
DEFAULT_GRAPH_RUN_ID = "formal_v2_nonthinking_20260806"
FINAL_CHECK = "final_english_end_to_end_report"
REPORT_COMPARISONS = ("full_vs_oneshot", "full_vs_human", "oneshot_vs_human")
GRAPH_COMPARISONS = ("graph_vs_no", "graph_vs_flat", "graph_vs_figure")
STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".yaml", ".yml"}


class FinalReportError(RuntimeError):
    """Raised when the immutable final report cannot be generated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalReportError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalReportError(f"{path} must contain a JSON object")
    return value


def _read_registry(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FinalReportError(f"Cannot read registry {path}: {exc}") from exc
    datasets = value.get("datasets") if isinstance(value, dict) else None
    if not isinstance(datasets, list):
        raise FinalReportError("The frozen registry has no datasets list")
    ids = [str(item.get("id", "")) for item in datasets if isinstance(item, dict)]
    locked = [
        str(item.get("id"))
        for item in datasets
        if isinstance(item, dict) and item.get("role") == "locked"
    ]
    development = [
        str(item.get("id"))
        for item in datasets
        if isinstance(item, dict) and item.get("role") == "development"
    ]
    if (
        value.get("status") != "frozen"
        or len(ids) != 8
        or len(set(ids)) != 8
        or len(locked) != 6
        or len(development) != 2
    ):
        raise FinalReportError("Registry is not the frozen 8-topic (2+6) design")
    return datasets, locked, development


def _load_auditor() -> Any:
    path = ROOT / "scripts" / "audit_formal_experiment_completion.py"
    spec = importlib.util.spec_from_file_location("_final_report_completion_audit", path)
    if spec is None or spec.loader is None:
        raise FinalReportError(f"Cannot load completion auditor from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_completion_audit(
    root: Path,
    registry_path: Path,
    graph_run_id: str,
) -> dict[str, Any]:
    """Run the same read-only Inspector used by the standalone completion audit."""
    module = _load_auditor()
    return module.Inspector(root.resolve(), registry_path.resolve()).run(graph_run_id)


def _validate_gate(
    audit: dict[str, Any],
    *,
    report_path: Path,
    manifest_path: Path,
) -> str:
    checks = audit.get("checks")
    if not isinstance(checks, list) or not checks:
        raise FinalReportError("Completion audit returned no checks")
    final = [item for item in checks if item.get("item") == FINAL_CHECK]
    if len(final) != 1:
        raise FinalReportError("Completion audit must contain exactly one final-report check")
    unfinished = [
        str(item.get("item"))
        for item in checks
        if item.get("item") != FINAL_CHECK and item.get("status") != "complete"
    ]
    if unfinished:
        preview = ", ".join(unfinished[:8])
        suffix = "" if len(unfinished) <= 8 else f" (+{len(unfinished) - 8} more)"
        raise FinalReportError(
            f"Final report refused: non-final requirements are incomplete: {preview}{suffix}"
        )
    final_status = final[0].get("status")
    if final_status == "complete":
        if not report_path.is_file() or not manifest_path.is_file():
            raise FinalReportError("Audit says final report is complete but output files are absent")
        return "existing_complete"
    if final_status != "incomplete":
        raise FinalReportError(
            f"Final report refused: final-report audit status is {final_status!r}"
        )
    if report_path.exists() or manifest_path.exists():
        raise FinalReportError(
            "Final report refused: a partial or invalid prior output exists; "
            "the generator never overwrites it"
        )
    return "missing"


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FinalReportError(f"{label} is not finite")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalReportError(f"{label} is not an integer")
    return value


def _pct(value: Any) -> str:
    return f"{100 * _number(value, label='percentage'):.1f}%"


def _metric(result: dict[str, Any], name: str) -> str:
    value = result.get(name)
    if not isinstance(value, dict):
        raise FinalReportError(f"Missing metric {name}")
    estimate = _number(value.get("estimate"), label=f"{name}.estimate")
    interval = value.get("cluster_bootstrap_95_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise FinalReportError(f"Missing 95% CI for {name}")
    low = _number(interval[0], label=f"{name}.ci_low")
    high = _number(interval[1], label=f"{name}.ci_high")
    if name == "completeness":
        return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"
    return f"{100 * estimate:.1f}% [{100 * low:.1f}%, {100 * high:.1f}%]"


def _panel_index(statistics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    panels = statistics.get("panels")
    if not isinstance(panels, list):
        raise FinalReportError("formal_statistics.json has no panels list")
    indexed: dict[str, dict[str, Any]] = {}
    for panel in panels:
        if not isinstance(panel, dict) or not isinstance(panel.get("comparison"), str):
            raise FinalReportError("Malformed statistics panel")
        comparison = panel["comparison"]
        if comparison in indexed:
            raise FinalReportError(f"Duplicate statistics panel {comparison}")
        indexed[comparison] = panel
    expected = {*REPORT_COMPARISONS, *GRAPH_COMPARISONS}
    if set(indexed) != expected:
        raise FinalReportError(
            f"Statistics panels differ from frozen design: {sorted(set(indexed) ^ expected)}"
        )
    return indexed


def _panel_row(panel: dict[str, Any]) -> str:
    a = str(panel.get("condition_a"))
    b = str(panel.get("condition_b"))
    conditions = panel.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != {a, b}:
        raise FinalReportError(f"Malformed conditions for {panel.get('comparison')}")
    pairwise = panel.get("pairwise_for_condition_a")
    if not isinstance(pairwise, dict):
        raise FinalReportError(f"Missing pairwise results for {panel.get('comparison')}")
    wins = _integer(pairwise.get("wins"), label="wins")
    ties = _integer(pairwise.get("ties"), label="ties")
    losses = _integer(pairwise.get("losses"), label="losses")
    return (
        f"| {panel['comparison']} | {a} | {_metric(conditions[a], 'ucr')} | "
        f"{_metric(conditions[a], 'completeness')} | {b} | "
        f"{_metric(conditions[b], 'ucr')} | "
        f"{_metric(conditions[b], 'completeness')} | {wins}/{ties}/{losses} |"
    )


def _effect_sentence(panel: dict[str, Any], label: str) -> str:
    effect = panel.get("effects", {}).get("ucr_reduction")
    if not isinstance(effect, dict):
        raise FinalReportError(f"Missing UCR effect for {panel.get('comparison')}")
    estimate = _number(effect.get("estimate"), label="ucr_reduction")
    interval = effect.get("cluster_bootstrap_95_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise FinalReportError("Missing UCR reduction interval")
    low = _number(interval[0], label="ucr_reduction.ci_low")
    high = _number(interval[1], label="ucr_reduction.ci_high")
    direction = "lower" if estimate > 0 else ("higher" if estimate < 0 else "equal")
    crosses = "included zero" if low <= 0 <= high else "excluded zero"
    return (
        f"For {label}, condition A had {abs(estimate) * 100:.1f} percentage points "
        f"{direction} UCR than condition B; the topic-cluster 95% interval was "
        f"[{low * 100:.1f}, {high * 100:.1f}] points and {crosses}."
    )


def _collect_source_hashes(
    root: Path,
    audit: dict[str, Any],
    registry_path: Path,
    *,
    report_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    candidates = {registry_path.resolve()}
    for check in audit["checks"]:
        if check.get("item") == FINAL_CHECK or check.get("status") != "complete":
            continue
        for raw in check.get("evidence", []):
            path = (root / str(raw)).resolve()
            if path.suffix.lower() in STRUCTURED_SUFFIXES:
                candidates.add(path)
    required = {
        (root / "experiments" / "formal_statistics_manifest.json").resolve(),
        (root / "experiments" / "formal_statistics" / "formal_statistics.json").resolve(),
        (root / "experiments" / "formal_statistics" / "formal_metrics.csv").resolve(),
        (root / "experiments" / "formal_statistics" / "graph_holm.csv").resolve(),
        (root / "experiments" / "formal_statistics" / "formal_results.md").resolve(),
    }
    candidates.update(required)
    excluded = {report_path.resolve(), manifest_path.resolve()}
    output: dict[str, str] = {}
    root_resolved = root.resolve()
    for path in sorted(candidates):
        if path in excluded:
            continue
        try:
            relative = path.relative_to(root_resolved).as_posix()
        except ValueError as exc:
            raise FinalReportError(f"Audit evidence escapes experiment root: {path}") from exc
        if not path.is_file():
            raise FinalReportError(f"Authoritative source is missing: {relative}")
        output[relative] = _sha256(path)
    missing_required = [
        path.relative_to(root_resolved).as_posix()
        for path in required
        if path.relative_to(root_resolved).as_posix() not in output
    ]
    if missing_required:
        raise FinalReportError(f"Required statistics sources lack hashes: {missing_required}")
    return output


def _render_report(
    *,
    datasets: list[dict[str, Any]],
    locked: list[str],
    development: list[str],
    root: Path,
    statistics: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    panels = _panel_index(statistics)
    if statistics.get("topics") != locked or statistics.get("topic_clusters") != 6:
        raise FinalReportError("Formal statistics do not identify the exact locked topics")
    bootstrap = statistics.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise FinalReportError("Formal statistics omit bootstrap provenance")

    census_rows: list[str] = []
    total_received = 0
    total_unique = 0
    total_duplicates = 0
    total_processed = 0
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        audit_root = root / "experiments" / "formal_workspaces" / dataset_id / "audit"
        harvest = _read_json(audit_root / "harvest_manifest.json")
        processing = _read_json(audit_root / "processing_manifest.json")
        evidence = _read_json(audit_root / "evidence_preparation_manifest.json")
        received = _integer(harvest.get("received_records"), label=f"{dataset_id}.received")
        unique = _integer(harvest.get("unique_records"), label=f"{dataset_id}.unique")
        duplicates = _integer(
            harvest.get("duplicate_records"), label=f"{dataset_id}.duplicates"
        )
        processed = _integer(
            processing.get("records_processed"), label=f"{dataset_id}.processed"
        )
        evidence_items = _integer(
            evidence.get("evidence_items"), label=f"{dataset_id}.evidence_items"
        )
        graph = evidence.get("graph")
        if not isinstance(graph, dict):
            raise FinalReportError(f"{dataset_id} has no graph census")
        nodes = _integer(graph.get("nodes"), label=f"{dataset_id}.nodes")
        edges = _integer(graph.get("edges"), label=f"{dataset_id}.edges")
        total_received += received
        total_unique += unique
        total_duplicates += duplicates
        total_processed += processed
        census_rows.append(
            f"| {dataset_id} | {dataset['role']} | {dataset['year_from']}–"
            f"{dataset['year_to']} | {received:,} | {unique:,} | {duplicates:,} | "
            f"{processed:,} | {evidence_items:,} | {nodes:,}/{edges:,} |"
        )

    report_rows = "\n".join(_panel_row(panels[name]) for name in REPORT_COMPARISONS)
    graph_rows = "\n".join(_panel_row(panels[name]) for name in GRAPH_COMPARISONS)
    adaptive = statistics.get("adaptive")
    if not isinstance(adaptive, dict) or not isinstance(adaptive.get("conditions"), dict):
        raise FinalReportError("Formal statistics omit adaptive-review results")
    adaptive_rows: list[str] = []
    for name in ("baseline_original", "always_review", "static_review", "adaptive_review"):
        condition = adaptive["conditions"].get(name)
        if not isinstance(condition, dict):
            raise FinalReportError(f"Missing adaptive condition {name}")
        counts = condition.get("counts")
        metrics = condition.get("metrics")
        if not isinstance(counts, dict) or not isinstance(metrics, dict):
            raise FinalReportError(f"Malformed adaptive condition {name}")
        items = _integer(counts.get("items"), label=f"{name}.items")
        reviews = _integer(counts.get("review_requests"), label=f"{name}.reviews")
        passes = _integer(
            counts.get("final_quality_passed"), label=f"{name}.quality_passes"
        )
        rrr = metrics.get("rrr", {}).get("estimate")
        fqpr = metrics.get("fqpr", {}).get("estimate")
        unsafe = metrics.get("unsafe_auto_accept_rate", {}).get("estimate")
        unsafe_text = "not defined (no auto-accepts)" if unsafe is None else _pct(unsafe)
        adaptive_rows.append(
            f"| {name} | {items:,} | {reviews:,} | {passes:,} | "
            f"{_pct(rrr)} | {_pct(fqpr)} | {unsafe_text} |"
        )
    comparisons = adaptive.get("original_to_post_review")
    if not isinstance(comparisons, dict):
        raise FinalReportError("Missing original-to-post-review comparisons")
    adaptive_sentences = []
    for name in ("always_review", "static_review", "adaptive_review"):
        comparison = comparisons.get(name)
        if not isinstance(comparison, dict):
            raise FinalReportError(f"Missing error comparison for {name}")
        before = _number(
            comparison.get("baseline_original_quality_error_rate"),
            label=f"{name}.baseline_error",
        )
        after = _number(
            comparison.get("post_review_quality_error_rate"),
            label=f"{name}.post_error",
        )
        reduction = _number(
            comparison.get("absolute_quality_error_rate_reduction"),
            label=f"{name}.error_reduction",
        )
        adaptive_sentences.append(
            f"- **{name}:** quality-error rate changed from {_pct(before)} to "
            f"{_pct(after)}, an absolute reduction of {100 * reduction:.1f} "
            "percentage points."
        )

    always_rrr = _number(
        adaptive["conditions"]["always_review"]["metrics"]["rrr"]["estimate"],
        label="always_review.rrr",
    )
    adaptive_rrr = _number(
        adaptive["conditions"]["adaptive_review"]["metrics"]["rrr"]["estimate"],
        label="adaptive_review.rrr",
    )
    review_delta = always_rrr - adaptive_rrr
    review_statement = (
        f"Adaptive review requested {abs(review_delta) * 100:.1f} percentage points "
        f"{'fewer' if review_delta >= 0 else 'more'} reviews than always-review."
    )

    return f"""# End-to-End Evaluation of an Automated Bibliometric Analysis System

## Scope and protocol

This report closes the preregistered end-to-end experiment for an automated
bibliometric workflow. The system generated its own topic-aligned Boolean
queries, harvested OpenAlex title-and-abstract metadata through exhaustive cursor
pagination, transformed the records into canonical relational tables, produced
visualizations and graph representations, assembled bounded evidence bundles,
and generated English analytical reports. It then evaluated report generation,
graph-grounded explanation, and adaptive human-in-the-loop review. The experiment
used eight complete natural-year corpora: two development topics
({", ".join(development)}) and six locked evaluation topics
({", ".join(locked)}). Development results were used for calibration only; all
formal comparative statistics below use exactly the six locked topics.

The frozen search definition retained every unique OpenAlex work matching each
topic expression in title or abstract metadata during its specified year range.
No maximum-record cap was applied. Cursor exhaustion, raw-page hashes, staged
corpus hashes, processing reconciliation, figure checks, evidence hashes, report
call archives, graph-run coverage, blind-Judge resolution, and the statistical
manifest were independently checked before this report could be written. The
generator itself made no API or model calls.

## Data census and end-to-end artifacts

| Dataset | Role | Years | Received | Unique | Harvest duplicates | Processed | Evidence items | Graph nodes/edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(census_rows)}
| **Total** | — | — | **{total_received:,}** | **{total_unique:,}** | **{total_duplicates:,}** | **{total_processed:,}** | — | — |

For every dataset, the accepted artifact chain includes the full harvested
metadata, canonical bibliographic relations, deterministic figures, a
machine-readable evidence bundle, a graph-grounding benchmark, two English report
conditions, three text explanation conditions, and a visible-only Figure/VLM
condition. Received and unique counts are reported separately because records
can overlap across date slices or source identities; the manifests explicitly
reconcile received, duplicate, staged, and processed counts.

## Structured-one-shot benchmark

The feasible end-to-end baseline was not an impossible attempt to place hundreds
of megabytes of raw metadata into one prompt. Instead, both conditions received
the same frozen, already-computed structured evidence. `structured_one_shot`
generated the report with exactly one DeepSeek V4 Pro call, whereas
`citeweave_full` used the staged four-call report workflow. Independent,
condition-blind Judges scored supported and unsupported claims, completeness on a
1–5 scale, and pairwise preference; disagreements were resolved without modifying
the underlying reports.

| Comparison | Condition A | A UCR (95% CI) | A completeness (95% CI) | Condition B | B UCR (95% CI) | B completeness (95% CI) | A wins/ties/losses |
|---|---|---:|---:|---|---:|---:|---:|
{report_rows}

{_effect_sentence(panels["full_vs_oneshot"], "the full workflow versus the structured one-shot benchmark")}

## Comparison with published human bibliometric studies

The published articles are topic-aligned human reference outputs, not gold
annotations and not identical-corpus replications. The automated system generated
its own searches and harvested its own full-year metadata, so differences in
corpus size or individual findings are not treated as errors. The comparison asks
whether the evidential support, completeness, and overall usefulness of the
automated report approach the quality of a real bibliometric article on the same
subject. `full_vs_human` is the primary human-reference comparison;
`oneshot_vs_human` is supplementary and exposes how much of any apparent
human-level performance depends on the multi-stage workflow.

{_effect_sentence(panels["full_vs_human"], "CiteWeave Full versus the published human reference")}
{_effect_sentence(panels["oneshot_vs_human"], "the structured one-shot baseline versus the published human reference")}

## Graph grounding, flat evidence, no retrieval, and Figure/VLM

The graph experiment reused bibliometric network data in machine-readable form.
`graph_rag` retrieved graph facts and relations; `flat_structured` supplied
non-graph structured evidence; `no_rag` supplied no retrieval grounding; and the
cross-model `figure_vlm` condition interpreted only the rendered visualization.
This design separates whether an answer is visually plausible from whether its
claims are traceable to canonical graph evidence.

| Comparison | Condition A | A UCR (95% CI) | A completeness (95% CI) | Condition B | B UCR (95% CI) | B completeness (95% CI) | A wins/ties/losses |
|---|---|---:|---:|---|---:|---:|---:|
{graph_rows}

{_effect_sentence(panels["graph_vs_no"], "Graph RAG versus no retrieval")}
{_effect_sentence(panels["graph_vs_flat"], "Graph RAG versus flat structured retrieval")}
{_effect_sentence(panels["graph_vs_figure"], "Graph RAG versus Figure/VLM")}

The family-wise interpretation uses the recorded exact topic-level sign-flip tests
and Holm adjustment:

| Comparison | Raw p | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|---:|
{chr(10).join(f"| {row['comparison']} | {_number(row['raw_p_value'], label='raw_p'):.4f} | {_number(row['holm_adjusted_p_value'], label='holm_p'):.4f} | {str(bool(row['reject_at_0_05'])).lower()} |" for row in statistics["graph_primary_holm"])}

## Adaptive review and quality-error reduction

This is a constrained LLM-based Human Proxy experiment, not a real-user study.
Ordinary Judges only observed and scored outputs. Only the Human Proxy could act
after a visible risk notice. It received only the exact system-flagged excerpt
(at most 500 characters) rather than the complete artifact, plus the frozen
evidence exposed by the review card. Its action was limited to what a person
could do in that interface: accept, reject, or make one local edit of at most
500 characters within the flagged span. It could not search for new issues,
browse, call APIs, inspect hidden truth, alter queries or data, rerun analysis,
or rewrite an entire report.
The untouched `baseline_original` condition supplies the pre-review quality-error
rate needed to distinguish error correction from selective escalation.

| Condition | Items | Review requests | Final quality passes | Review-request rate | Final-quality pass rate | Unsafe auto-accept rate |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(adaptive_rows)}

{chr(10).join(adaptive_sentences)}

{review_statement} This result must be read jointly with final-quality pass rate
and unsafe auto-accept rate: reducing review requests is useful only when it does
not silently release low-quality outputs.

## Limitations and negative results

The experiment does not equate topic alignment with exact replication of a
published study, and published human reports are comparison targets rather than
claim-level ground truth. LLM-as-Judge outcomes can retain model-specific bias
despite independent blind judging and adjudication. The Human Proxy results
estimate behavior under a tightly constrained simulated reviewer and cannot
establish real analyst workload, trust, or usability. Figure/VLM is a cross-model
extension because the main DeepSeek report model was not used as a multimodal
model, so that comparison combines retrieval format and model differences.

Most importantly, the design does not presume that Graph RAG must dominate flat
structured retrieval. Direct factual questions can be fully answerable from a
flat evidence record, and any interval including zero is evidence against claiming
a reliable advantage for that comparison. Conversely, fluent no-retrieval or
visual answers can still contain unsupported explanations. The tabled outcomes
and Holm-adjusted tests, including ties, null intervals, or unfavorable effects,
are retained as measured rather than rewritten into a uniformly positive story.

## Statistical analysis and reproducibility

All UCR, completeness, and preference estimates use resolved Judge verdicts.
Intervals are 95% topic-cluster bootstrap intervals with
{_integer(bootstrap.get("samples"), label="bootstrap.samples"):,} resamples,
seed {_integer(bootstrap.get("seed"), label="bootstrap.seed")}, across six locked
topic clusters. Graph UCR comparisons use exact paired topic-level sign-flip tests
with Holm adjustment. The supplementary comparisons are identified in the
statistics rather than promoted after seeing their results.

This report is a deterministic rendering of accepted local artifacts. Its
provenance manifest records SHA-256 hashes for {len(source_hashes):,} structured
source files, including the frozen registry, eight-topic harvest and processing
manifests, report and graph artifacts, six-topic resolved comparisons, adaptive
baseline and post-review counts, and formal statistical outputs. Re-running the
generator on unchanged sources produces identical bytes; changed or incomplete
sources cause refusal rather than silent overwrite. Together with archived raw
page hashes, canonical relation hashes, prompt/call records, blind packet
exchanges, and the read-only completion audit, this supports exact artifact-level
traceability without claiming that stochastic model outputs can be regenerated
from external services indefinitely.
"""


def generate(
    *,
    root: Path,
    registry_path: Path,
    graph_run_id: str,
    report_path: Path,
    manifest_path: Path,
    audit: dict[str, Any] | None = None,
) -> str:
    root = root.resolve()
    registry_path = registry_path.resolve()
    report_path = report_path.resolve()
    manifest_path = manifest_path.resolve()
    audit_value = audit or run_completion_audit(root, registry_path, graph_run_id)
    gate = _validate_gate(
        audit_value,
        report_path=report_path,
        manifest_path=manifest_path,
    )
    datasets, locked, development = _read_registry(registry_path)
    statistics_path = root / "experiments" / "formal_statistics" / "formal_statistics.json"
    statistics = _read_json(statistics_path)
    source_hashes = _collect_source_hashes(
        root,
        audit_value,
        registry_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )
    report = _render_report(
        datasets=datasets,
        locked=locked,
        development=development,
        root=root,
        statistics=statistics,
        source_hashes=source_hashes,
    )
    report_bytes = report.encode("utf-8")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "language": "English",
        "generator": "scripts/generate_final_english_report.py",
        "generator_mode": "deterministic_local_no_api",
        "formal_graph_run_id": graph_run_id,
        "report_path": report_path.relative_to(root).as_posix(),
        "report_sha256": report_sha256,
        "dataset_ids": [str(item["id"]) for item in datasets],
        "locked_topic_ids": locked,
        "source_hashes": source_hashes,
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if gate == "existing_complete":
        if (
            report_path.read_bytes() != report_bytes
            or manifest_path.read_text(encoding="utf-8") != manifest_text
        ):
            raise FinalReportError(
                "Existing complete final report differs from deterministic rendering; "
                "refusing overwrite"
            )
        return "unchanged"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the deterministic English end-to-end report only after every "
            "other formal completion check passes."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--graph-run-id", default=DEFAULT_GRAPH_RUN_ID)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        outcome = generate(
            root=args.root,
            registry_path=args.registry,
            graph_run_id=args.graph_run_id,
            report_path=args.report,
            manifest_path=args.manifest,
        )
    except FinalReportError as exc:
        raise SystemExit(f"Final report generation refused: {exc}") from exc
    print(f"Final English end-to-end report: {outcome}")


if __name__ == "__main__":
    main()
