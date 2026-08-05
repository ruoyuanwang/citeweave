from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .io import load_config, slugify
from .large_scale_visualization import render_large_project
from .models import ProjectConfig, SearchProtocol
from .processing_acceptance import verify_large_processing
from .visual_acceptance import verify_visualization
from .workflow import process_project, run_project

api = FastAPI(
    title="BibAgent API",
    version="0.1.0",
    description="Auditable acquisition-to-manuscript bibliometric workflow.",
)


class RunRequest(BaseModel):
    output_dir: str
    protocol: SearchProtocol
    use_llm: bool = False
    review_rounds: int = Field(default=1, ge=0, le=3)
    allow_truncated: bool = False


class ProcessRequest(BaseModel):
    project_dir: str
    resume: bool = True
    chunk_size: int | None = Field(default=None, ge=100, le=50_000)
    keep_partitions: bool | None = None
    refinalize: bool = False
    batch_budget: int | None = Field(default=None, ge=1)


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "bibagent"}


@api.post("/v1/projects/run")
def run(request: RunRequest) -> dict:
    try:
        config = ProjectConfig(
            project_id=slugify(request.protocol.title),
            protocol=request.protocol,
            crossref_mailto=os.getenv("CROSSREF_MAILTO"),
        )
        return run_project(
            Path(request.output_dir),
            config,
            use_llm=request.use_llm,
            review_rounds=request.review_rounds,
            allow_truncated=request.allow_truncated,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@api.post("/v1/projects/process")
def process(request: ProcessRequest) -> dict:
    """Materialize disk-backed canonical and visualization-ready metadata tables."""
    try:
        root = Path(request.project_dir)
        config = load_config(root / "project.yml")
        return process_project(
            root,
            config,
            resume=request.resume,
            chunk_size=request.chunk_size,
            keep_partitions=request.keep_partitions,
            refinalize=request.refinalize,
            batch_budget=request.batch_budget,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@api.get("/v1/projects/process/accept")
def process_accept(project_dir: str) -> dict:
    """Independently verify a completed large-scale processing run."""
    return verify_large_processing(Path(project_dir))


@api.post("/v1/projects/visualize")
def visualize(project_dir: str) -> dict:
    """Render bounded bibliometric maps from full-corpus aggregates."""
    return render_large_project(Path(project_dir))


@api.get("/v1/projects/visualize/accept")
def visualize_accept(project_dir: str) -> dict:
    """Verify scalable visualization output without invoking an LLM or VLM."""
    return verify_visualization(Path(project_dir))
