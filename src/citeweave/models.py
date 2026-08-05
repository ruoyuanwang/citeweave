from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class SourceName(str, Enum):
    openalex = "openalex"
    crossref = "crossref"
    europe_pmc = "europe_pmc"
    import_file = "import_file"


class SearchProtocol(BaseModel):
    """Versionable protocol describing exactly what the acquisition means."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3)
    keywords: list[str] = Field(min_length=1)
    year_from: int = Field(ge=1800, le=2200)
    year_to: int = Field(ge=1800, le=2200)
    source: SourceName = SourceName.crossref
    query_mode: Literal["all", "any", "phrase"] = "all"
    document_types: list[str] = Field(default_factory=list)
    language: str | None = None
    max_records: int | None = Field(default=None, ge=1)
    enrich_crossref: bool = False
    include_abstracts: bool = True
    include_references: bool = True
    notes: str = ""
    input_file: Path | None = None
    input_format: Literal["auto", "csv", "ris", "bibtex", "wos"] = "auto"

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty keyword is required")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def validate_years(self) -> SearchProtocol:
        if self.year_to < self.year_from:
            raise ValueError("year_to must be >= year_from")
        if self.source == SourceName.import_file and self.input_file is None:
            raise ValueError("input_file is required when source=import_file")
        return self

    @property
    def query_text(self) -> str:
        if self.query_mode == "phrase":
            return " ".join(self.keywords)
        joiner = " AND " if self.query_mode == "all" else " OR "
        return joiner.join(self.keywords)


class AcquisitionPolicy(BaseModel):
    """Operational policy for resumable, large-result metadata acquisition."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["standard", "bulk"] = "standard"
    partition_strategy: Literal["none", "year", "adaptive_date"] = "adaptive_date"
    target_slice_records: int = Field(default=25_000, ge=1_000, le=1_000_000)
    page_size: int | None = Field(default=None, ge=1, le=1_000)
    requests_per_second: float | None = Field(default=None, gt=0, le=100)
    max_retries: int = Field(default=8, ge=0, le=20)
    max_slice_restarts: int = Field(default=3, ge=0, le=10)
    compress_raw: bool = True


class ProcessingPolicy(BaseModel):
    """Bounded-memory policy for cleaning and structuring harvested metadata."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "in_memory", "disk"] = "auto"
    chunk_size: int = Field(default=1_000, ge=100, le=50_000)
    duckdb_memory_limit: str = "4GB"
    candidate_pool_size: int | None = Field(default=None, ge=50, le=10_000)
    edge_row_limit: int = Field(default=200_000, ge=1_000, le=5_000_000)
    keep_partitions: bool = True


class VisualizationPolicy(BaseModel):
    """Scalable and reproducible policy for bibliometric maps."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "in_memory", "disk"] = "auto"
    max_display_nodes: int = Field(default=72, ge=20, le=300)
    label_budget: int = Field(default=22, ge=5, le=80)
    candidate_multiplier: int = Field(default=6, ge=2, le=20)
    max_candidate_nodes: int = Field(default=640, ge=50, le=5_000)
    max_edges_per_node: int = Field(default=4, ge=1, le=12)
    max_display_clusters: int = Field(default=10, ge=2, le=30)
    layout_restarts: int = Field(default=4, ge=1, le=12)
    layout_iterations: int = Field(default=700, ge=100, le=5_000)
    layout_algorithm: Literal["vos", "forceatlas2"] = "vos"
    min_cluster_size: int = Field(default=3, ge=1, le=30)
    counting_method: Literal["full", "fractional"] = "full"
    dpi: int = Field(default=240, ge=120, le=600)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    protocol: SearchProtocol
    created_at: datetime = Field(default_factory=utc_now)
    crossref_mailto: str | None = None
    openalex_api_key_env: str = "OPENALEX_API_KEY"
    acquisition: AcquisitionPolicy = Field(default_factory=AcquisitionPolicy)
    processing: ProcessingPolicy = Field(default_factory=ProcessingPolicy)
    visualization: VisualizationPolicy = Field(default_factory=VisualizationPolicy)
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = "https://api.deepseek.com"
    min_completeness_ratio: float = Field(default=0.995, ge=0, le=1)
    visualization_max_nodes: int = Field(default=80, ge=10, le=500)
    visualization_label_budget: int = Field(default=24, ge=5, le=100)
    random_seed: int = 42


class AcquisitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceName
    query: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None = None
    expected_records: int | None = None
    received_records: int = 0
    unique_records: int = 0
    duplicate_records: int = 0
    pages: int = 0
    failed_pages: int = 0
    complete: bool = False
    truncated: bool = False
    drift: int | None = None
    raw_sha256: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HarvestPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_index: int
    cursor_in: str
    cursor_out: str | None
    records: int
    raw_path: str
    raw_sha256: str
    bytes_compressed: int


class HarvestSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_id: str
    date_from: str
    date_to: str
    expected_records: int
    status: Literal["pending", "running", "complete", "failed"] = "pending"
    cursor: str | None = "*"
    received_records: int = 0
    pages: list[HarvestPage] = Field(default_factory=list)
    restart_count: int = 0
    failure_count: int = 0
    last_error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class HarvestManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    source: SourceName
    query_fingerprint: str
    query: dict[str, Any]
    status: Literal["planned", "running", "partial", "complete", "failed"] = "planned"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    page_size: int
    partition_strategy: str
    target_slice_records: int
    root_expected_records: int
    planned_expected_records: int
    received_records: int = 0
    unique_records: int = 0
    duplicate_records: int = 0
    raw_bytes_compressed: int = 0
    staged_path: str | None = None
    staged_sha256: str | None = None
    slices: list[HarvestSlice] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    path: str
    sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    inputs: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    claim_type: str
    statement: str
    value: Any
    unit: str | None = None
    artifact_path: str
    selector: dict[str, Any] = Field(default_factory=dict)
    method: str
    caveat: str | None = None


class ProjectPaths:
    """Stable, content-oriented project layout."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.raw = self.root / "raw"
        self.staged = self.root / "staged"
        self.canonical = self.root / "canonical"
        self.quality = self.root / "quality"
        self.analyses = self.root / "analyses"
        self.figures = self.root / "figures"
        self.evidence = self.root / "evidence"
        self.report = self.root / "report"
        self.audit = self.root / "audit"

    def create(self) -> None:
        for path in (
            self.root,
            self.raw,
            self.staged,
            self.canonical,
            self.quality,
            self.analyses,
            self.figures,
            self.evidence,
            self.report,
            self.audit,
        ):
            path.mkdir(parents=True, exist_ok=True)
