from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .acceptance import verify_project
from .harvest_acceptance import verify_bulk_harvest
from .io import load_config, slugify, write_json
from .large_scale_visualization import render_large_project
from .models import AcquisitionPolicy, ProjectConfig, SearchProtocol, SourceName
from .processing_acceptance import verify_large_processing
from .processing_benchmark import run_processing_benchmark
from .scale import run_duckdb_benchmark
from .visual_acceptance import verify_visualization
from .word_export import export_word_report
from .workflow import (
    create_project,
    harvest_project,
    process_project,
    promote_candidate,
    recompute_downstream,
    refine_staged_generation,
    resume_generation,
    run_project,
)

app = typer.Typer(
    name="bibagent",
    help="Evidence-first bibliometric research agent.",
    no_args_is_help=True,
)


@app.command()
def init(
    output: Annotated[Path, typer.Argument(help="Project directory")],
    title: Annotated[str, typer.Option("--title", "-t")],
    keyword: Annotated[list[str], typer.Option("--keyword", "-k")],
    year_from: Annotated[int, typer.Option("--from")],
    year_to: Annotated[int, typer.Option("--to")],
    source: Annotated[SourceName, typer.Option()] = SourceName.crossref,
    mode: Annotated[str, typer.Option()] = "all",
    input_file: Annotated[Path | None, typer.Option("--input-file")] = None,
    input_format: Annotated[str, typer.Option("--input-format")] = "auto",
    bulk: Annotated[bool, typer.Option("--bulk/--standard")] = False,
    target_slice_records: Annotated[
        int, typer.Option("--target-slice-records", min=1_000)
    ] = 25_000,
    include_references: Annotated[
        bool, typer.Option("--include-references/--no-references")
    ] = True,
) -> None:
    """Create a versioned project protocol without acquiring data."""
    config = ProjectConfig(
        project_id=slugify(title),
        protocol=SearchProtocol(
            title=title,
            keywords=keyword,
            year_from=year_from,
            year_to=year_to,
            source=source,
            query_mode=mode,
            input_file=input_file,
            input_format=input_format,
            include_references=include_references,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk" if bulk else "standard",
            target_slice_records=target_slice_records,
        ),
        crossref_mailto=os.getenv("CROSSREF_MAILTO"),
    )
    paths = create_project(output, config)
    typer.echo(f"Created {paths.root / 'project.yml'}")


@app.command("run")
def run_command(
    project: Annotated[Path, typer.Argument(help="Project directory containing project.yml")],
    llm: Annotated[bool, typer.Option("--llm/--no-llm")] = False,
    review_rounds: Annotated[int, typer.Option(min=0, max=3)] = 1,
    allow_truncated: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run acquisition, normalization, analysis, figures, evidence, and manuscript."""
    config = load_config(project / "project.yml")
    result = run_project(
        project,
        config,
        use_llm=llm,
        review_rounds=review_rounds,
        allow_truncated=allow_truncated,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def harvest(
    project: Annotated[Path, typer.Argument(help="Project directory containing project.yml")],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    page_budget: Annotated[
        int | None,
        typer.Option(
            "--page-budget",
            min=1,
            help="Optional page limit for a controlled partial run; rerun to resume.",
        ),
    ] = None,
) -> None:
    """Run resumable, date-sharded bulk metadata acquisition only."""
    config = load_config(project / "project.yml")
    if config.acquisition.mode != "bulk":
        config = config.model_copy(
            update={"acquisition": config.acquisition.model_copy(update={"mode": "bulk"})}
        )
    result = harvest_project(
        project,
        config,
        resume=resume,
        page_budget=page_budget,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("harvest-accept")
def harvest_accept(
    project: Annotated[Path, typer.Argument(help="Bulk-harvest project directory")],
) -> None:
    """Verify slice coverage, page hashes and staged-corpus completeness."""
    result = verify_bulk_harvest(project)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command()
def process(
    project: Annotated[Path, typer.Argument(help="Project with staged source metadata")],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    chunk_size: Annotated[
        int | None,
        typer.Option(
            "--chunk-size",
            min=100,
            max=50_000,
            help="Maximum source records normalized in one Python batch.",
        ),
    ] = None,
    keep_partitions: Annotated[
        bool | None,
        typer.Option("--keep-parts/--remove-parts"),
    ] = None,
    refinalize: Annotated[
        bool,
        typer.Option(
            "--refinalize",
            help="Rebuild canonical/visualization outputs from completed partitions.",
        ),
    ] = False,
    batch_budget: Annotated[
        int | None,
        typer.Option(
            "--batch-budget",
            min=1,
            help="Controlled partial run for checkpoint/resume testing.",
        ),
    ] = None,
) -> None:
    """Clean, deduplicate and materialize visualization-ready Parquet tables."""
    config = load_config(project / "project.yml")
    result = process_project(
        project,
        config,
        resume=resume,
        chunk_size=chunk_size,
        keep_partitions=keep_partitions,
        refinalize=refinalize,
        batch_budget=batch_budget,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("process-accept")
def process_accept(
    project: Annotated[Path, typer.Argument(help="Processed project directory")],
) -> None:
    """Verify canonical hashes, keys, relationships, counts and visualization tables."""
    result = verify_large_processing(project)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command()
def visualize(
    project: Annotated[Path, typer.Argument(help="Processed project directory")],
) -> None:
    """Render scalable publication charts and sparse bibliometric maps."""
    result = render_large_project(project)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))


@app.command("visualize-accept")
def visualize_accept(
    project: Annotated[Path, typer.Argument(help="Visualized project directory")],
) -> None:
    """Verify figure files, sparse-map disclosures, caps and deterministic layout QA."""
    result = verify_visualization(project)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command()
def quickstart(
    output: Annotated[Path, typer.Argument(help="New project directory")],
    title: Annotated[str, typer.Option("--title", "-t")],
    keyword: Annotated[list[str], typer.Option("--keyword", "-k")],
    year_from: Annotated[int, typer.Option("--from")],
    year_to: Annotated[int, typer.Option("--to")],
    source: Annotated[SourceName, typer.Option()] = SourceName.crossref,
    llm: Annotated[bool, typer.Option("--llm/--no-llm")] = False,
    review_rounds: Annotated[int, typer.Option(min=0, max=3)] = 1,
    input_file: Annotated[Path | None, typer.Option("--input-file")] = None,
    input_format: Annotated[str, typer.Option("--input-format")] = "auto",
    bulk: Annotated[bool, typer.Option("--bulk/--standard")] = False,
    target_slice_records: Annotated[
        int, typer.Option("--target-slice-records", min=1_000)
    ] = 25_000,
    include_references: Annotated[
        bool, typer.Option("--include-references/--no-references")
    ] = True,
) -> None:
    """Create and execute a complete project in one command."""
    config = ProjectConfig(
        project_id=slugify(title),
        protocol=SearchProtocol(
            title=title,
            keywords=keyword,
            year_from=year_from,
            year_to=year_to,
            source=source,
            input_file=input_file,
            input_format=input_format,
            include_references=include_references,
        ),
        acquisition=AcquisitionPolicy(
            mode="bulk" if bulk else "standard",
            target_slice_records=target_slice_records,
        ),
        crossref_mailto=os.getenv("CROSSREF_MAILTO"),
    )
    result = run_project(
        output,
        config,
        use_llm=llm,
        review_rounds=review_rounds,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
) -> None:
    """Start the local HTTP API."""
    import uvicorn

    uvicorn.run("bibagent.api:api", host=host, port=port, reload=False)


@app.command()
def resume(
    project: Annotated[Path, typer.Argument(help="Project directory with saved artifacts")],
    llm: Annotated[bool, typer.Option("--llm/--no-llm")] = True,
    review_rounds: Annotated[int, typer.Option(min=0, max=3)] = 1,
) -> None:
    """Resume manuscript generation without reacquiring or recomputing data."""
    result = resume_generation(
        project,
        use_llm=llm,
        review_rounds=review_rounds,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def refine(
    project: Annotated[Path, typer.Argument(help="Project with reviewed generation stages")],
) -> None:
    """Repair reviewed sections against evidence and publish the final manuscript."""
    result = refine_staged_generation(project)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def promote(
    project: Annotated[Path, typer.Argument(help="Project directory")],
    candidate: Annotated[Path, typer.Argument(help="Candidate Markdown file")],
    model: Annotated[str, typer.Option()] = "deepseek-v4-flash",
) -> None:
    """Promote an already generated candidate only after full validation."""
    result = promote_candidate(project, candidate, model=model)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def recompute(
    project: Annotated[Path, typer.Argument(help="Project directory")],
) -> None:
    """Rebuild all deterministic downstream artifacts without network acquisition."""
    typer.echo(json.dumps(recompute_downstream(project), ensure_ascii=False, indent=2))


@app.command()
def benchmark(
    documents: Annotated[int, typer.Option(min=1)] = 100_000,
    terms_per_document: Annotated[int, typer.Option(min=1, max=100)] = 5,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Benchmark full-count aggregation and bounded network candidate selection."""
    result = run_duckdb_benchmark(documents=documents, terms_per_document=terms_per_document)
    if output is not None:
        write_json(output, result)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("benchmark-processing")
def benchmark_processing(
    output: Annotated[Path, typer.Argument(help="New benchmark project directory")],
    documents: Annotated[int, typer.Option(min=1)] = 100_000,
    references_per_document: Annotated[
        int, typer.Option("--references-per-document", min=0, max=100)
    ] = 10,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=100, max=50_000)] = 2_000,
) -> None:
    """Run synthetic records through full normalization, dedup and materialization."""
    result = run_processing_benchmark(
        output,
        documents=documents,
        references_per_document=references_per_document,
        chunk_size=chunk_size,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["acceptance_passed"]:
        raise typer.Exit(code=1)


@app.command()
def accept(
    project: Annotated[Path, typer.Argument(help="Completed project directory")],
    require_model: Annotated[str | None, typer.Option("--require-model")] = None,
    minimum_figures: Annotated[int, typer.Option(min=1)] = 12,
) -> None:
    """Verify completeness, schemas, visuals, evidence paths and manuscript claims."""
    result = verify_project(
        project,
        require_llm_model=require_model,
        minimum_figures=minimum_figures,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command()
def word(
    project: Annotated[Path, typer.Argument(help="Completed project directory")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    native_word: Annotated[
        bool,
        typer.Option(
            "--native-word/--no-native-word",
            help="On Windows, round-trip the DOCX through Microsoft Word before delivery.",
        ),
    ] = False,
) -> None:
    """Export the manuscript as an academic Word document with inline figures."""
    typer.echo(str(export_word_report(project, output, native_word=native_word)))


if __name__ == "__main__":
    app()
