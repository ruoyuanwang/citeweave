from __future__ import annotations

# Malformed or unreadable user artifacts are audit findings, not auditor crashes.
# ruff: noqa: BLE001
import argparse
import gzip
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_datasets_openalex_title_abstract.yml"
TEXT_GRAPH_CONDITIONS = ("no_rag", "flat_structured", "graph_rag")
REPORT_CONDITIONS = {"structured_one_shot": 1, "citeweave_full": 4}
REPORT_COMPARISONS = ("full_vs_oneshot", "full_vs_human", "oneshot_vs_human")
GRAPH_COMPARISONS = ("graph_vs_no", "graph_vs_flat", "graph_vs_figure")
POST_REVIEW_CONDITIONS = ("always_review", "static_review", "adaptive_review")
BASELINE_ORIGINAL_CONDITION = "baseline_original"
ADAPTIVE_CONDITIONS = (BASELINE_ORIGINAL_CONDITION, *POST_REVIEW_CONDITIONS)
EXCLUDED_MARKERS = ("development", "rejected", "preflight", "pilot", "formal_v1")


@dataclass
class Check:
    item: str
    status: str
    evidence: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class Inspector:
    def __init__(self, root: Path, registry_path: Path) -> None:
        self.root = root.resolve()
        self.registry_path = registry_path.resolve()
        self.checks: list[Check] = []
        self.datasets: list[dict[str, Any]] = []
        self.locked: list[str] = []
        self.development: list[str] = []

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    def add(
        self,
        item: str,
        *,
        complete: bool,
        invalid: bool = False,
        evidence: Iterable[Path] = (),
        reasons: Iterable[str] = (),
        details: dict[str, Any] | None = None,
    ) -> Check:
        check = Check(
            item=item,
            status="invalid" if invalid else ("complete" if complete else "incomplete"),
            evidence=[self.rel(path) for path in evidence if path.exists()],
            reasons=list(reasons),
            details=details or {},
        )
        self.checks.append(check)
        return check

    @staticmethod
    def json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{path} is not a JSON object")
        return value

    @staticmethod
    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{number} is not a JSON object")
            rows.append(value)
        return rows

    def registry(self) -> None:
        reasons: list[str] = []
        invalid = False
        evidence: list[Path] = []
        if not self.registry_path.is_file():
            self.add("frozen_registry", complete=False, reasons=["registry is missing"])
            return
        evidence.append(self.registry_path)
        try:
            registry = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
            self.datasets = list(registry.get("datasets", []))
        except Exception as exc:
            self.add(
                "frozen_registry",
                complete=False,
                invalid=True,
                evidence=evidence,
                reasons=[f"cannot parse registry: {exc}"],
            )
            return
        ids = [str(row.get("id", "")) for row in self.datasets]
        roles = [row.get("role") for row in self.datasets]
        if registry.get("status") != "frozen":
            reasons.append("registry status is not frozen")
            invalid = True
        if len(ids) != 8 or len(set(ids)) != 8:
            reasons.append("registry must contain exactly eight unique topics")
            invalid = True
        if roles.count("development") != 2 or roles.count("locked") != 6:
            reasons.append("registry must distinguish exactly 2 development and 6 locked topics")
            invalid = True
        for row in self.datasets:
            if (
                row.get("source") != "openalex"
                or row.get("max_records") is not None
                or row.get("query_status") != "frozen"
                or row.get("search_scope") != "title_abstract"
            ):
                reasons.append(f"{row.get('id')}: not frozen uncapped title/abstract OpenAlex")
                invalid = True
        self.development = [row["id"] for row in self.datasets if row["role"] == "development"]
        self.locked = [row["id"] for row in self.datasets if row["role"] == "locked"]
        self.add(
            "frozen_registry",
            complete=not reasons,
            invalid=invalid,
            evidence=evidence,
            reasons=reasons,
            details={"dataset_ids": ids, "development": self.development, "locked": self.locked},
        )

    def harvest(self, row: dict[str, Any]) -> None:
        dataset = row["id"]
        workspace = self.root / "experiments" / "formal_workspaces" / dataset
        harvest = workspace / "audit" / "harvest_manifest.json"
        acquisition = workspace / "audit" / "acquisition_manifest.json"
        reasons: list[str] = []
        invalid = False
        evidence = [harvest, acquisition]
        if not harvest.is_file() or not acquisition.is_file():
            self.add(
                f"dataset:{dataset}:harvest",
                complete=False,
                evidence=evidence,
                reasons=["harvest/acquisition manifest is missing"],
            )
            return
        try:
            manifest = self.json(harvest)
            summary = self.json(acquisition)
            if manifest.get("status") != "complete" or not summary.get("complete"):
                reasons.append("harvest does not declare complete")
            if manifest.get("source") != "openalex" or summary.get("source") != "openalex":
                reasons.append("source is not OpenAlex")
                invalid = True
            if summary.get("truncated") is not False or summary.get("failed_pages") != 0:
                reasons.append("acquisition is truncated or has failed pages")
                invalid = True
            slices = manifest.get("slices")
            if not isinstance(slices, list) or not slices:
                reasons.append("no cursor slices")
                invalid = True
                slices = []
            raw_count = 0
            raw_hashes: list[str] = []
            page_count = 0
            for slice_ in slices:
                if (
                    slice_.get("status") != "complete"
                    or slice_.get("cursor_exhausted") is not True
                    or slice_.get("cursor") is not None
                ):
                    reasons.append(f"slice {slice_.get('slice_id')} did not exhaust its cursor")
                    invalid = True
                pages = slice_.get("pages") or []
                previous_out: Any = "*"
                for index, page in enumerate(pages):
                    page_count += 1
                    if page.get("page_index") != index + 1 or page.get("cursor_in") != previous_out:
                        reasons.append(f"slice {slice_.get('slice_id')} has a broken cursor chain")
                        invalid = True
                    previous_out = page.get("cursor_out")
                    raw_path = workspace / str(page.get("raw_path", ""))
                    evidence.append(raw_path)
                    if not raw_path.is_file():
                        reasons.append(f"raw page missing: {self.rel(raw_path)}")
                        invalid = True
                    else:
                        # Raw page hashes cover the canonical uncompressed API
                        # response bytes; gzip container metadata is not provenance.
                        with gzip.open(raw_path, "rb") as handle:
                            actual = hashlib.sha256(handle.read()).hexdigest()
                        if actual != page.get("raw_sha256"):
                            reasons.append(f"raw page hash mismatch: {self.rel(raw_path)}")
                            invalid = True
                        raw_hashes.append(actual)
                    raw_count += int(page.get("records", -1))
                if not pages or previous_out is not None:
                    reasons.append(f"slice {slice_.get('slice_id')} lacks terminal null cursor")
                    invalid = True
                if sum(int(page.get("records", 0)) for page in pages) != slice_.get(
                    "received_records"
                ):
                    reasons.append(f"slice {slice_.get('slice_id')} record total mismatch")
                    invalid = True
            received = manifest.get("received_records")
            unique = manifest.get("unique_records")
            duplicates = manifest.get("duplicate_records")
            if raw_count != received or unique + duplicates != received:
                reasons.append("raw/unique/duplicate counts do not reconcile")
                invalid = True
            if raw_hashes != summary.get("raw_sha256"):
                reasons.append("acquisition raw hash list differs from cursor manifest")
                invalid = True
            staged = workspace / str(manifest.get("staged_path", ""))
            evidence.append(staged)
            staged_lines = -1
            if not staged.is_file():
                reasons.append("staged corpus is missing")
                invalid = True
            else:
                if self.sha(staged) != manifest.get("staged_sha256"):
                    reasons.append("staged corpus hash mismatch")
                    invalid = True
                with gzip.open(staged, "rb") as handle:
                    staged_lines = sum(1 for line in handle if line.strip())
                if staged_lines != unique:
                    reasons.append("staged corpus line count differs from unique records")
                    invalid = True
            if (
                summary.get("received_records") != received
                or summary.get("unique_records") != unique
            ):
                reasons.append("acquisition summary counts differ from harvest manifest")
                invalid = True
            self.add(
                f"dataset:{dataset}:harvest",
                complete=not reasons,
                invalid=invalid,
                evidence=evidence,
                reasons=reasons,
                details={
                    "received_records": received,
                    "unique_records": unique,
                    "duplicate_records": duplicates,
                    "raw_pages": page_count,
                    "staged_records": staged_lines,
                    "cursor_exhausted": bool(slices)
                    and all(value.get("cursor_exhausted") is True for value in slices),
                },
            )
        except Exception as exc:
            self.add(
                f"dataset:{dataset}:harvest",
                complete=False,
                invalid=True,
                evidence=evidence,
                reasons=[f"harvest validation error: {exc}"],
            )

    def pipeline(self, row: dict[str, Any]) -> None:
        dataset = row["id"]
        workspace = self.root / "experiments" / "formal_workspaces" / dataset
        process = workspace / "audit" / "processing_manifest.json"
        quality = workspace / "quality" / "processing_report.json"
        visual = workspace / "audit" / "visualization_acceptance.json"
        evidence_manifest = workspace / "audit" / "evidence_preparation_manifest.json"
        evidence_items = workspace / "evidence" / "evidence_items.json"
        grounding = workspace / "evidence" / "graph_grounding_manifest.json"
        graph_manifest = workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
        benchmark = workspace / "evidence" / "formal_graph_experiment" / "benchmark.json"

        reasons: list[str] = []
        invalid = False
        files = [process, quality]
        if not all(path.is_file() for path in files):
            self.add(
                f"dataset:{dataset}:processing_acceptance",
                complete=False,
                evidence=files,
                reasons=["processing manifest/report is missing"],
            )
        else:
            try:
                manifest = self.json(process)
                report = self.json(quality)
                if manifest.get("status") != "complete" or report.get("passed") is not True:
                    reasons.append("processing is not accepted")
                harvest = self.json(workspace / "audit" / "harvest_manifest.json")
                if (
                    manifest.get("input_sha256") != harvest.get("staged_sha256")
                    or manifest.get("records_processed") != harvest.get("unique_records")
                    or report.get("input_records") != harvest.get("unique_records")
                ):
                    reasons.append("processing input counts do not match harvest")
                    invalid = True
                if report.get("canonical_records", 0) <= 0 or report.get("scope_filter", {}).get(
                    "input_records_accounted"
                ) != manifest.get("records_processed"):
                    reasons.append("processing quality counts do not reconcile")
                    invalid = True
                for name, expected in (
                    manifest.get("outputs", {}).get("canonical_sha256", {}).items()
                ):
                    path = workspace / "canonical" / f"{name}.parquet"
                    files.append(path)
                    if not path.is_file() or self.sha(path) != expected:
                        reasons.append(f"canonical hash mismatch: {name}")
                        invalid = True
                if any(value for value in report.get("foreign_key_orphans", {}).values()):
                    reasons.append("foreign-key orphan check failed")
                    invalid = True
                if report.get("scope_filter", {}).get("out_of_scope_in_canonical") != 0:
                    reasons.append("canonical corpus contains out-of-scope records")
                    invalid = True
            except Exception as exc:
                reasons.append(f"processing validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:processing_acceptance",
                complete=not reasons,
                invalid=invalid,
                evidence=files,
                reasons=reasons,
            )

        reasons = []
        invalid = False
        if not visual.is_file():
            self.add(
                f"dataset:{dataset}:visualization_acceptance",
                complete=False,
                reasons=["visualization acceptance is missing"],
            )
        else:
            try:
                value = self.json(visual)
                if value.get("passed") is not True:
                    reasons.append("visualization acceptance did not pass")
                if not value.get("checks") or not all(value["checks"].values()):
                    reasons.append("one or more visualization checks failed")
                    invalid = True
                for image in value.get("images", []):
                    if not all(
                        image.get(key) is True
                        for key in (
                            "png_exists",
                            "svg_exists",
                            "png_hash_matches",
                            "svg_hash_matches",
                            "minimum_dimensions",
                        )
                    ):
                        reasons.append(f"figure validation failed: {image.get('name')}")
                        invalid = True
            except Exception as exc:
                reasons.append(f"visualization validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:visualization_acceptance",
                complete=not reasons,
                invalid=invalid,
                evidence=[visual, workspace / "figures" / "figure_manifest.json"],
                reasons=reasons,
            )

        reasons = []
        invalid = False
        if not evidence_manifest.is_file() or not evidence_items.is_file():
            self.add(
                f"dataset:{dataset}:evidence_acceptance",
                complete=False,
                evidence=[evidence_manifest, evidence_items],
                reasons=["evidence bundle or acceptance manifest is missing"],
            )
        else:
            try:
                value = self.json(evidence_manifest)
                if value.get("passed") is not True:
                    reasons.append("evidence acceptance did not pass")
                if self.sha(evidence_items) != value.get("evidence_sha256"):
                    reasons.append("evidence bundle hash mismatch")
                    invalid = True
                if int(value.get("evidence_items", 0)) <= 0:
                    reasons.append("evidence bundle is empty")
                    invalid = True
            except Exception as exc:
                reasons.append(f"evidence validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:evidence_acceptance",
                complete=not reasons,
                invalid=invalid,
                evidence=[evidence_manifest, evidence_items],
                reasons=reasons,
            )

        reasons = []
        invalid = False
        graph_files = [grounding, graph_manifest, benchmark]
        if not all(path.is_file() for path in graph_files):
            self.add(
                f"dataset:{dataset}:graph_grounding_acceptance",
                complete=False,
                evidence=graph_files,
                reasons=["graph-grounding artifact is missing"],
            )
        else:
            try:
                high = self.json(grounding)
                value = self.json(graph_manifest)
                rows = json.loads(benchmark.read_text(encoding="utf-8"))
                if high.get("qa_items", 0) <= 0 or value.get("questions") != len(rows):
                    reasons.append("graph question counts do not reconcile")
                    invalid = True
                if self.sha(benchmark) != value.get("benchmark_sha256"):
                    reasons.append("graph benchmark hash mismatch")
                    invalid = True
                for context in value.get("contexts", []):
                    for prefix in ("flat", "graph"):
                        path = workspace / context[f"{prefix}_path"]
                        graph_files.append(path)
                        if not path.is_file() or self.sha(path) != context[f"{prefix}_sha256"]:
                            reasons.append(
                                f"context hash mismatch: {context.get('item_id')}:{prefix}"
                            )
                            invalid = True
            except Exception as exc:
                reasons.append(f"graph-grounding validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:graph_grounding_acceptance",
                complete=not reasons,
                invalid=invalid,
                evidence=graph_files,
                reasons=reasons,
            )

    def reports(self, row: dict[str, Any]) -> None:
        dataset = row["id"]
        root = self.root / "experiments" / "formal_reports" / dataset
        evidence_path = (
            self.root
            / "experiments"
            / "formal_workspaces"
            / dataset
            / "evidence"
            / "evidence_items.json"
        )
        evidence_hash = self.sha(evidence_path) if evidence_path.is_file() else None
        condition_hashes: set[str] = set()
        for condition, calls in REPORT_CONDITIONS.items():
            directory = root / condition
            completion = directory / "completion.json"
            report = directory / "report.md"
            run_manifest = directory / "run_manifest.json"
            reasons: list[str] = []
            invalid = False
            files = [completion, report, run_manifest]
            if not all(path.is_file() for path in files):
                self.add(
                    f"dataset:{dataset}:report:{condition}",
                    complete=False,
                    evidence=files,
                    reasons=["formal report condition is missing"],
                )
                continue
            try:
                value = self.json(completion)
                if value.get("status") != "complete":
                    reasons.append("report completion status is not complete")
                if value.get("dataset_id") != dataset or value.get("condition") != condition:
                    reasons.append("report identity mismatch")
                    invalid = True
                if (
                    value.get("model") != "deepseek-v4-pro"
                    or value.get("call_count") != calls
                    or value.get("report_language") != "English"
                ):
                    reasons.append("wrong model, call count, or report language")
                    invalid = True
                if self.sha(report) != value.get("report_sha256"):
                    reasons.append("report hash mismatch")
                    invalid = True
                if value.get("evidence_sha256") != evidence_hash:
                    reasons.append("report evidence hash differs from frozen bundle")
                    invalid = True
                condition_hashes.add(str(value.get("evidence_sha256")))
                call_files = sorted(directory.glob("calls/*/call.json"))
                files.extend(call_files)
                if len(call_files) != calls:
                    reasons.append("archived API call count differs from completion")
                    invalid = True
                text = report.read_text(encoding="utf-8")
                if len(text.split()) < 100:
                    reasons.append("English report is implausibly short")
                    invalid = True
                if len(re.findall(r"[\u4e00-\u9fff]", text)) > max(10, len(text) // 100):
                    reasons.append("report is not predominantly English")
                    invalid = True
            except Exception as exc:
                reasons.append(f"report validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:report:{condition}",
                complete=not reasons,
                invalid=invalid,
                evidence=files,
                reasons=reasons,
            )
        self.add(
            f"dataset:{dataset}:report_shared_evidence",
            complete=len(condition_hashes) == 1 and evidence_hash in condition_hashes,
            invalid=len(condition_hashes) > 1,
            evidence=[evidence_path],
            reasons=[]
            if len(condition_hashes) == 1 and evidence_hash in condition_hashes
            else ["two report conditions do not share the frozen evidence hash"],
        )

    def graph_runs(self, row: dict[str, Any], run_id: str) -> None:
        dataset = row["id"]
        root = self.root / "experiments" / "formal_runs" / dataset / run_id
        benchmark = (
            self.root
            / "experiments"
            / "formal_workspaces"
            / dataset
            / "evidence"
            / "formal_graph_experiment"
            / "benchmark.json"
        )
        benchmark_rows = (
            json.loads(benchmark.read_text(encoding="utf-8")) if benchmark.is_file() else []
        )
        expected_ids = {item["item_id"] for item in benchmark_rows}
        for condition in TEXT_GRAPH_CONDITIONS:
            directory = root / condition
            manifest = directory / "run_manifest.json"
            items = directory / "items.jsonl"
            score = directory / "score.json"
            reasons: list[str] = []
            invalid = False
            files = [manifest, items, score]
            if not manifest.is_file() or not items.is_file():
                self.add(
                    f"dataset:{dataset}:graph_text:{condition}",
                    complete=False,
                    evidence=files,
                    reasons=["formal v2 graph condition is missing"],
                )
                continue
            try:
                value = self.json(manifest)
                rows = self.jsonl(items)
                completed = [item for item in rows if item.get("status") == "complete"]
                ids = [item.get("item_id") for item in completed]
                if (
                    value.get("run_id") != run_id
                    or value.get("dataset_id") != dataset
                    or value.get("condition") != condition
                    or value.get("profile", {}).get("model") != "deepseek-v4-pro"
                ):
                    reasons.append("formal graph run identity/model mismatch")
                    invalid = True
                if value.get("benchmark_sha256") != self.sha(benchmark):
                    reasons.append("formal graph run benchmark hash mismatch")
                    invalid = True
                if set(ids) != expected_ids or len(ids) != len(expected_ids):
                    reasons.append("formal graph item coverage is incomplete or duplicated")
                    invalid = True
                if not score.is_file():
                    reasons.append("formal graph mechanical score is missing")
                else:
                    score_value = self.json(score)
                    score_metadata = score_value.get("formal_metadata", {})
                    score_ids = {
                        item.get("item_id") for item in score_value.get("rows", [])
                    }
                    if (
                        score_value.get("items") != len(expected_ids)
                        or score_value.get("predictions_received") != len(expected_ids)
                        or score_ids != expected_ids
                        or len(score_value.get("rows", [])) != len(expected_ids)
                        or score_metadata.get("dataset_id") != dataset
                        or score_metadata.get("condition") != condition
                        or score_metadata.get("run_id") != run_id
                        or score_metadata.get("run_manifest_sha256") != self.sha(manifest)
                        or score_metadata.get("checkpoint_sha256") != self.sha(items)
                    ):
                        reasons.append(
                            "formal graph mechanical score identity, coverage, or "
                            "source-hash binding differs"
                        )
                        invalid = True
            except Exception as exc:
                reasons.append(f"graph run validation error: {exc}")
                invalid = True
            self.add(
                f"dataset:{dataset}:graph_text:{condition}",
                complete=not reasons,
                invalid=invalid,
                evidence=files,
                reasons=reasons,
                details={"expected_items": len(expected_ids)},
            )

    def vision(self, row: dict[str, Any], run_id: str) -> None:
        dataset = row["id"]
        packet = self.root / "experiments" / "vision_packets" / f"{dataset}.json"
        output = self.root / "experiments" / "vision_outputs" / f"{dataset}.json"
        score_root = (
            self.root
            / "experiments"
            / "formal_runs"
            / dataset
            / run_id
            / "figure_vlm"
        )
        score_manifest = score_root / "run_manifest.json"
        score = score_root / "score.json"
        reasons: list[str] = []
        invalid = False
        if not packet.is_file() or not output.is_file():
            self.add(
                f"dataset:{dataset}:figure_vlm",
                complete=False,
                evidence=[packet, output, score_manifest, score],
                reasons=["Figure/VLM packet or visual-subagent output is missing"],
            )
            return
        try:
            packet_value = self.json(packet)
            output_value = self.json(output)
            packet_ids = [item.get("item_id") for item in packet_value.get("items", [])]
            output_ids = [item.get("item_id") for item in output_value.get("results", [])]
            if (
                packet_value.get("dataset_id") != dataset
                or output_value.get("dataset_id") != dataset
                or packet_value.get("visible_only") is not True
                or output_value.get("visible_only") is not True
            ):
                reasons.append("Figure/VLM identity or visible-only contract mismatch")
                invalid = True
            if output_value.get("packet_sha256") != packet_value.get("packet_sha256"):
                reasons.append("Figure/VLM packet hash linkage mismatch")
                invalid = True
            if set(packet_ids) != set(output_ids) or len(packet_ids) != len(output_ids):
                reasons.append("Figure/VLM item coverage is incomplete or duplicated")
                invalid = True
            workspace = self.root / "experiments" / "formal_workspaces" / dataset
            graph_manifest = self.json(
                workspace / "evidence" / "formal_graph_experiment" / "manifest.json"
            )
            if packet_value.get("benchmark_sha256") != graph_manifest.get("benchmark_sha256"):
                reasons.append("Figure/VLM benchmark hash mismatch")
                invalid = True
            for item in packet_value.get("items", []):
                figure = Path(str(item.get("figure_path", "")))
                if not figure.is_file():
                    reasons.append(f"Figure/VLM source image missing: {figure}")
                    invalid = True
            if not score_manifest.is_file() or not score.is_file():
                reasons.append("Figure/VLM mechanical score or score manifest is missing")
            else:
                score_manifest_value = self.json(score_manifest)
                score_value = self.json(score)
                score_metadata = score_value.get("formal_metadata", {})
                score_ids = [
                    item.get("item_id") for item in score_value.get("rows", [])
                ]
                expected_score_ids = [
                    item.get("item_id") for item in packet_value.get("items", [])
                ]
                if (
                    score_manifest_value.get("run_id") != run_id
                    or score_manifest_value.get("dataset_id") != dataset
                    or score_manifest_value.get("condition") != "figure_vlm"
                    or score_manifest_value.get("packet_sha256") != self.sha(packet)
                    or score_manifest_value.get("output_sha256") != self.sha(output)
                    or any(
                        score_metadata.get(field) != value
                        for field, value in score_manifest_value.items()
                    )
                    or score_metadata.get("primary_mechanical_metrics")
                    != [
                        "exact_answer_accuracy",
                        "structured_unsupported_answer_rate",
                    ]
                    or score_value.get("items") != len(expected_score_ids)
                    or score_value.get("predictions_received") != len(expected_score_ids)
                    or score_ids != expected_score_ids
                ):
                    reasons.append(
                        "Figure/VLM mechanical score identity, coverage, or "
                        "source-hash binding differs"
                    )
                    invalid = True
        except Exception as exc:
            reasons.append(f"Figure/VLM validation error: {exc}")
            invalid = True
        self.add(
            f"dataset:{dataset}:figure_vlm",
            complete=not reasons,
            invalid=invalid,
            evidence=[packet, output, score_manifest, score],
            reasons=reasons,
        )

    @staticmethod
    def _topic_from_resolved(row: dict[str, Any]) -> str:
        return str(row.get("sample_id", "")).split(":", 1)[0]

    def judging_family(self, family: str, comparisons: tuple[str, ...]) -> None:
        base = (
            self.root
            / "experiments"
            / ("formal_judging" if family == "report" else "formal_graph_judging")
        )
        for comparison in comparisons:
            directory = base / comparison
            reasons: list[str] = []
            invalid = False
            judge_a = directory / "judge_a.jsonl"
            judge_b = directory / "judge_b.jsonl"
            map_path = directory / "secret_blind_map.json"
            resolved_dirs = sorted(
                [
                    path
                    for path in directory.glob("resolved_v*")
                    if path.is_dir()
                    and not any(marker in path.as_posix().lower() for marker in EXCLUDED_MARKERS)
                ]
            )
            resolved_dir = resolved_dirs[-1] if resolved_dirs else directory / "resolved"
            resolved = resolved_dir / "resolved_judgments.jsonl"
            metrics = resolved_dir / "judge_metrics.json"
            adjudication_packets = resolved_dir / "adjudication_packets.jsonl"
            adjudications = resolved_dir / "adjudications.jsonl"
            files = [
                judge_a,
                judge_b,
                map_path,
                resolved,
                metrics,
                adjudication_packets,
                adjudications,
            ]
            required = [judge_a, judge_b, map_path, resolved, metrics]
            if not all(path.is_file() for path in required):
                self.add(
                    f"formal_{family}_judging:{comparison}",
                    complete=False,
                    evidence=files,
                    reasons=["dual-Judge resolved formal output is missing"],
                )
                continue
            try:
                rows_a = self.jsonl(judge_a)
                rows_b = self.jsonl(judge_b)
                rows = self.jsonl(resolved)
                ids_a = {item.get("packet_id") for item in rows_a}
                ids_b = {item.get("packet_id") for item in rows_b}
                ids_r = {item.get("packet_id") for item in rows}
                if not rows or ids_a != ids_b or ids_r != ids_a:
                    reasons.append("Judge A/B/resolved packet coverage differs")
                    invalid = True
                if {item.get("judge_id") for item in rows_a} != {"eval_a"}:
                    reasons.append("judge_a contains wrong judge identity")
                    invalid = True
                if {item.get("judge_id") for item in rows_b} != {"eval_b"}:
                    reasons.append("judge_b contains wrong judge identity")
                    invalid = True
                topics = {self._topic_from_resolved(item) for item in rows}
                if topics != set(self.locked):
                    reasons.append(
                        "formal resolved judgments must cover exactly the six locked topics"
                    )
                    invalid = True
                conflict_ids = {
                    item.get("packet_id")
                    for item in rows
                    if item.get("source") == "adjudicated" or bool(item.get("conflicts"))
                }
                if conflict_ids:
                    if not adjudications.is_file():
                        reasons.append("conflicts exist but adjudications are missing")
                        invalid = True
                    else:
                        adjudicated_ids = {
                            item.get("packet_id") for item in self.jsonl(adjudications)
                        }
                        if not conflict_ids <= adjudicated_ids:
                            reasons.append("not all conflicts have blind adjudications")
                            invalid = True
                if adjudication_packets.exists() != adjudications.exists():
                    reasons.append("adjudication packets/results are not paired")
                    invalid = True
            except Exception as exc:
                reasons.append(f"formal judging validation error: {exc}")
                invalid = True
            self.add(
                f"formal_{family}_judging:{comparison}",
                complete=not reasons,
                invalid=invalid,
                evidence=files,
                reasons=reasons,
                details={"locked_topics": self.locked},
            )

    def adaptive(self) -> None:
        root = self.root / "experiments" / "formal_adaptive_review"
        manifest_path = root / "manifest.json"
        reasons: list[str] = []
        invalid = False
        files = [manifest_path]
        if not manifest_path.is_file():
            self.add(
                "formal_adaptive_experiment",
                complete=False,
                reasons=["formal adaptive manifest is missing"],
            )
        else:
            try:
                manifest = self.json(manifest_path)
                topics = manifest.get("topic_sequence")
                expected_sequence = [*self.development, *self.locked]
                if topics != expected_sequence or manifest.get("formal_results_used") is not True:
                    reasons.append(
                        "adaptive run is not the frozen 2-development + 6-locked formal experiment"
                    )
                    invalid = True
                separation = manifest.get("judge_separation", {})
                if (
                    separation.get("feedback_updates_online_memory") is not True
                    or separation.get("evaluation_updates_online_memory") is not False
                ):
                    reasons.append("Human Proxy and evaluation memory are not separated")
                    invalid = True
                proxy = manifest.get("human_proxy_capability", {})
                if (
                    proxy.get("experiment_only") is not True
                    or proxy.get("active_only_after_visible_risk_notice") is not True
                    or proxy.get("visible_packet_only") is not True
                    or proxy.get("flagged_risk_only") is not True
                    or proxy.get("candidate_visibility") != "flagged_excerpt_only"
                    or proxy.get("edit_target_bound_to_flagged_text") is not True
                    or proxy.get("maximum_flagged_text_characters") != 500
                    or proxy.get("may_search_for_additional_issues") is not False
                    or proxy.get("may_use_model_knowledge_or_model_scale_synthesis")
                    is not False
                    or proxy.get("external_tools_or_retrieval") is not False
                    or proxy.get("full_artifact_rewrite") is not False
                    or proxy.get("maximum_edits_per_intervention") != 1
                    or proxy.get("maximum_edit_characters") != 500
                ):
                    reasons.append("Human Proxy capability boundary is not enforced")
                    invalid = True
                for condition in POST_REVIEW_CONDITIONS:
                    state_path = root / condition / "state.json"
                    files.append(state_path)
                    if not state_path.is_file():
                        reasons.append(f"{condition} state is missing")
                        continue
                    state = self.json(state_path)
                    if state.get("completed") is not True:
                        reasons.append(f"{condition} is incomplete")
                    records = state.get("records", [])
                    if {item.get("dataset_id") for item in records} != set(expected_sequence):
                        reasons.append(f"{condition} does not cover the frozen 2+6 topic sequence")
                        invalid = True
            except Exception as exc:
                reasons.append(f"adaptive validation error: {exc}")
                invalid = True
            self.add(
                "formal_adaptive_experiment",
                complete=not reasons,
                invalid=invalid,
                evidence=files,
                reasons=reasons,
            )

        baseline_root = self.root / "experiments" / "formal_adaptive_original_evaluation"
        baseline_manifest_path = baseline_root / "manifest.json"
        baseline_audit_path = baseline_root / "audits" / "semantic_audit.json"
        baseline_audit_archive = baseline_root / "audits" / "semantic_audit_archive"
        baseline_audit_archive_manifest = (
            baseline_audit_archive / "archive_manifest.json"
        )
        baseline_metrics_path = baseline_root / "metrics.json"
        baseline_result_manifest_path = baseline_root / "result_manifest.json"
        reasons = []
        invalid = False
        files = [
            baseline_manifest_path,
            baseline_audit_path,
            baseline_audit_archive_manifest,
            baseline_metrics_path,
            baseline_result_manifest_path,
        ]
        if not baseline_manifest_path.is_file():
            reasons.append("baseline-original evaluation manifest is missing")
        else:
            try:
                baseline_manifest = self.json(baseline_manifest_path)
                expected_sequence = [*self.development, *self.locked]
                if (
                    baseline_manifest.get("topic_sequence") != expected_sequence
                    or baseline_manifest.get("formal_results_used") is not True
                    or baseline_manifest.get("judge_may_modify_artifacts") is not False
                    or baseline_manifest.get("evaluation_updates_feedback_memory") is not False
                ):
                    reasons.append("baseline-original protocol or 2+6 split is invalid")
                    invalid = True
                items = baseline_manifest.get("items", [])
                if len({item.get("sample_id") for item in items}) != len(items):
                    reasons.append("baseline-original sample IDs are not unique")
                    invalid = True
                if not all(
                    path.is_file()
                    for path in (
                        baseline_audit_path,
                        baseline_metrics_path,
                        baseline_result_manifest_path,
                    )
                ):
                    reasons.append(
                        "baseline-original semantic audit or finalized metrics are missing"
                    )
                else:
                    semantic_audit = self.json(baseline_audit_path)
                    audit_items = semantic_audit.get("items", [])
                    audit_by_packet = {
                        str(item.get("packet_id")): item
                        for item in audit_items
                        if isinstance(item, dict)
                    }
                    expected_packet_ids = {
                        str(item.get("packet_id")) for item in items
                    }
                    if (
                        semantic_audit.get("protocol_version")
                        != "adaptive-original-semantic-audit-v1"
                        or semantic_audit.get("read_only") is not True
                        or semantic_audit.get("judge_modified_artifacts") is not False
                        or semantic_audit.get("all_consistent") is not True
                        or semantic_audit.get("packets") != len(items)
                        or set(audit_by_packet) != expected_packet_ids
                        or any(
                            item.get("semantic_consistent") is not True
                            or item.get("decision_defensible") is not True
                            for item in audit_items
                            if isinstance(item, dict)
                        )
                    ):
                        reasons.append(
                            "baseline-original results lack complete independent "
                            "semantic acceptance"
                        )
                        invalid = True
                    if not baseline_audit_archive_manifest.is_file():
                        reasons.append(
                            "baseline-original bound semantic-audit archive is missing"
                        )
                        invalid = True
                    else:
                        archive_manifest = self.json(
                            baseline_audit_archive_manifest
                        )
                        archive_artifacts = archive_manifest.get("artifacts", [])
                        archive_paths = {
                            str(row.get("path"))
                            for row in archive_artifacts
                            if isinstance(row, dict)
                        }
                        actual_archive_paths = {
                            path.relative_to(baseline_audit_archive).as_posix()
                            for path in baseline_audit_archive.rglob("*")
                            if path.is_file()
                            and path != baseline_audit_archive_manifest
                        }
                        if (
                            archive_manifest.get("protocol_version")
                            != "adaptive-original-semantic-audit-v1"
                            or archive_paths != actual_archive_paths
                            or len(archive_paths) != len(archive_artifacts)
                        ):
                            reasons.append(
                                "baseline-original semantic-audit archive coverage "
                                "is invalid"
                            )
                            invalid = True
                        for artifact in archive_artifacts:
                            if not isinstance(artifact, dict):
                                continue
                            archived_path = (
                                baseline_audit_archive / str(artifact.get("path", ""))
                            )
                            files.append(archived_path)
                            if (
                                not archived_path.is_file()
                                or artifact.get("sha256") != self.sha(archived_path)
                            ):
                                reasons.append(
                                    "baseline-original semantic-audit archive hash "
                                    f"differs: {artifact.get('path')}"
                                )
                                invalid = True
                    metrics = self.json(baseline_metrics_path)
                    if (
                        metrics.get("items") != len(items)
                        or metrics.get("evaluation_feedback_leakage") is not False
                        or metrics.get("judge_modified_artifacts") is not False
                        or metrics.get("result_manifest_sha256")
                        != self.sha(baseline_result_manifest_path)
                    ):
                        reasons.append("baseline-original finalized metrics are invalid")
                        invalid = True
                for item in items:
                    packet = baseline_root / str(item.get("packet_path", ""))
                    result = (
                        baseline_root / "results" / "evaluation" / f"{item.get('packet_id')}.json"
                    )
                    files.extend([packet, result])
                    if not packet.is_file() or not result.is_file():
                        reasons.append(
                            f"baseline-original packet/result missing: {item.get('sample_id')}"
                        )
                    else:
                        audit_item = audit_by_packet.get(str(item.get("packet_id")), {})
                        if (
                            audit_item.get("source_packet_sha256") != self.sha(packet)
                            or audit_item.get("source_result_sha256") != self.sha(result)
                        ):
                            reasons.append(
                                "baseline-original semantic audit hash binding differs: "
                                f"{item.get('sample_id')}"
                            )
                            invalid = True
            except Exception as exc:
                reasons.append(f"baseline-original validation error: {exc}")
                invalid = True
        self.add(
            "formal_adaptive_original_evaluation",
            complete=not reasons,
            invalid=invalid,
            evidence=files,
            reasons=reasons,
        )

        count_root = self.root / "experiments" / "formal_adaptive_topic_counts"
        reasons = []
        invalid = False
        files = []
        for topic in self.locked:
            path = count_root / f"{topic}.json"
            files.append(path)
            if not path.is_file():
                reasons.append(f"adaptive topic counts missing: {topic}")
                continue
            try:
                value = self.json(path)
                if value.get("topic_id") != topic or set(value.get("conditions", {})) != set(
                    ADAPTIVE_CONDITIONS
                ):
                    reasons.append(f"invalid adaptive topic counts: {topic}")
                    invalid = True
                for condition, counts in value.get("conditions", {}).items():
                    items = counts.get("items")
                    if (
                        not isinstance(items, int)
                        or items <= 0
                        or counts.get("review_requests", 0) > items
                        or counts.get("final_quality_passed", 0) > items
                        or counts.get("auto_accepts", 0) > items
                    ):
                        reasons.append(f"non-reconciling adaptive counts: {topic}:{condition}")
                        invalid = True
            except Exception as exc:
                reasons.append(f"adaptive count validation error {topic}: {exc}")
                invalid = True
        self.add(
            "formal_adaptive_topic_counts",
            complete=not reasons,
            invalid=invalid,
            evidence=files,
            reasons=reasons,
        )

    def statistics(self) -> None:
        manifest_path = self.root / "experiments" / "formal_statistics_manifest.json"
        output_root = self.root / "experiments" / "formal_statistics"
        statistics = output_root / "formal_statistics.json"
        files = [
            manifest_path,
            statistics,
            output_root / "formal_metrics.csv",
            output_root / "graph_holm.csv",
            output_root / "formal_results.md",
        ]
        reasons: list[str] = []
        invalid = False
        if not manifest_path.is_file() or not all(path.is_file() for path in files[1:]):
            self.add(
                "formal_statistics",
                complete=False,
                evidence=files,
                reasons=["formal statistics manifest or outputs are missing"],
            )
            return
        try:
            manifest = self.json(manifest_path)
            value = self.json(statistics)
            if manifest.get("topics") != self.locked or value.get("topics") != self.locked:
                reasons.append("statistics do not use exactly the six locked topics")
                invalid = True
            if value.get("topic_clusters") != 6:
                reasons.append("statistics topic cluster count is not six")
                invalid = True
            bootstrap = value.get("bootstrap", {})
            if (
                bootstrap.get("samples") != 10_000
                or bootstrap.get("method") != "topic-cluster bootstrap"
                or bootstrap.get("confidence_level") != 0.95
            ):
                reasons.append("statistics are not the frozen 10,000-sample cluster bootstrap")
                invalid = True
            holm = value.get("graph_primary_holm")
            if not isinstance(holm, list) or len(holm) < 2:
                reasons.append("Holm-adjusted graph comparisons are missing")
                invalid = True
            for row in holm or []:
                if "holm_adjusted_p_value" not in row:
                    reasons.append("graph comparison lacks Holm-adjusted p-value")
                    invalid = True
            referenced: list[Path] = []
            for panel in manifest.get("report_comparisons", []) + manifest.get(
                "graph_comparisons", []
            ):
                if set(panel.get("files", {})) != set(self.locked):
                    reasons.append(f"statistics panel lacks six topics: {panel.get('name')}")
                    invalid = True
                referenced.extend(
                    manifest_path.parent / str(path)
                    for path in panel.get("files", {}).values()
                )
            adaptive = manifest.get("adaptive_results", {})
            if set(adaptive) != set(self.locked):
                reasons.append("statistics manifest adaptive panel lacks six topics")
                invalid = True
            referenced.extend(
                manifest_path.parent / str(path) for path in adaptive.values()
            )
            if not all(path.is_file() for path in referenced):
                reasons.append("statistics manifest references missing input files")
                invalid = True
            if any(
                any(marker in self.rel(path).lower() for marker in EXCLUDED_MARKERS)
                for path in referenced
            ):
                reasons.append("statistics manifest references development/rejected/preflight data")
                invalid = True
            files.extend(referenced)
        except Exception as exc:
            reasons.append(f"formal statistics validation error: {exc}")
            invalid = True
        self.add(
            "formal_statistics",
            complete=not reasons,
            invalid=invalid,
            evidence=files,
            reasons=reasons,
        )

    def final_report(self) -> None:
        path = self.root / "experiments" / "final_report" / "end_to_end_report.md"
        manifest_path = self.root / "experiments" / "final_report" / "manifest.json"
        reasons: list[str] = []
        invalid = False
        if not path.is_file() or not manifest_path.is_file():
            self.add(
                "final_english_end_to_end_report",
                complete=False,
                evidence=[path, manifest_path],
                reasons=["final English report or its provenance manifest is missing"],
            )
            return
        try:
            text = path.read_text(encoding="utf-8")
            value = self.json(manifest_path)
            if value.get("status") != "complete" or value.get("language") != "English":
                reasons.append("final report manifest is not complete English output")
                invalid = True
            if self.sha(path) != value.get("report_sha256"):
                reasons.append("final report hash mismatch")
                invalid = True
            if value.get("dataset_ids") != [row["id"] for row in self.datasets]:
                reasons.append("final report provenance does not cover all eight datasets")
                invalid = True
            if value.get("locked_topic_ids") != self.locked:
                reasons.append("final report provenance does not identify six locked topics")
                invalid = True
            source_hashes = value.get("source_hashes")
            if not isinstance(source_hashes, dict) or not source_hashes:
                reasons.append("final report provenance has no source hashes")
                invalid = True
            else:
                report_files = {path.resolve(), manifest_path.resolve()}
                for raw, expected in source_hashes.items():
                    source = (self.root / str(raw)).resolve()
                    try:
                        source.relative_to(self.root)
                    except ValueError:
                        reasons.append(f"final report source escapes root: {raw}")
                        invalid = True
                        continue
                    if source in report_files:
                        reasons.append("final report provenance is circular")
                        invalid = True
                    elif not source.is_file():
                        reasons.append(f"final report source is missing: {raw}")
                        invalid = True
                    elif self.sha(source) != expected:
                        reasons.append(f"final report source hash mismatch: {raw}")
                        invalid = True
            if len(text.split()) < 500:
                reasons.append("final end-to-end report is implausibly short")
                invalid = True
            if len(re.findall(r"[\u4e00-\u9fff]", text)) > max(20, len(text) // 100):
                reasons.append("final report is not predominantly English")
                invalid = True
            required_terms = ("OpenAlex", "benchmark", "Graph", "adaptive", "human")
            missing = [term for term in required_terms if term.lower() not in text.lower()]
            if missing:
                reasons.append(f"final report omits required end-to-end sections: {missing}")
                invalid = True
        except Exception as exc:
            reasons.append(f"final report validation error: {exc}")
            invalid = True
        self.add(
            "final_english_end_to_end_report",
            complete=not reasons,
            invalid=invalid,
            evidence=[path, manifest_path],
            reasons=reasons,
        )

    def run(self, run_id: str) -> dict[str, Any]:
        self.registry()
        for row in self.datasets:
            self.harvest(row)
            self.pipeline(row)
            self.reports(row)
            self.graph_runs(row, run_id)
            self.vision(row, run_id)
        self.judging_family("report", REPORT_COMPARISONS)
        self.judging_family("graph", GRAPH_COMPARISONS)
        self.adaptive()
        self.statistics()
        self.final_report()
        counts = {
            status: sum(check.status == status for check in self.checks)
            for status in ("complete", "incomplete", "invalid")
        }
        all_complete = bool(self.checks) and counts["complete"] == len(self.checks)
        return {
            "schema_version": 1,
            "audit_type": "read_only_formal_experiment_completion",
            "generated_at": datetime.now(UTC).isoformat(),
            "root": str(self.root),
            "registry": self.rel(self.registry_path),
            "formal_graph_run_id": run_id,
            "development_topics": self.development,
            "locked_topics": self.locked,
            "excluded_artifact_markers": list(EXCLUDED_MARKERS),
            "all_complete": all_complete,
            "summary": {"checks": len(self.checks), **counts},
            "checks": [asdict(check) for check in self.checks],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict read-only audit of the complete eight-topic formal experiment."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run-id", default="formal_v2_nonthinking_20260806")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON audit report path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    inspector = Inspector(args.root, args.registry)
    report = inspector.run(args.run_id)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output.resolve())
    else:
        sys.stdout.write(serialized)
    raise SystemExit(0 if report["all_complete"] else 1)


if __name__ == "__main__":
    main()
