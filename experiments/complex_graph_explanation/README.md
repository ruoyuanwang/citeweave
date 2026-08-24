# Complex graph explanation experiment

This directory contains a preliminary CiteWeave experiment for graph-grounded
explanation. It compares the same multimodal model under three evidence settings:

1. `vlm`: PNG network visualization only.
2. `vlm_flat_kg`: PNG plus the complete structured node/edge table.
3. `vlm_graph_rag`: PNG plus anchor-aware multi-hop path candidates or
   node-removal probes retrieved from the KG.

The included benchmark has 24 automatically generated samples (six per task):
multi-hop shortest path, cross-community path, bridge node, and unsupported
direct edge. Each sample contains eight nodes and is derived from one CiteWeave
keyword-co-occurrence graph.

## Preliminary result

Model: `qwen3-vl-plus-2025-12-19`, temperature 0, one run per sample.

| Condition | Strict accuracy | Unsupported structured edges |
|---|---:|---:|
| VLM | 10/24 (41.7%) | 7/57 (12.3%) |
| VLM + full KG | 19/24 (79.2%) | 0/66 (0.0%) |
| VLM + multi-hop GraphRAG | 20/24 (83.3%) | 2/64 (3.1%) |

These results support the value of structured graph evidence relative to an
image-only VLM. They do **not** yet establish that the current GraphRAG method is
better than providing the full KG: the difference is only one correct answer in
24 paired samples (exploratory exact McNemar p = 1.0).

The current multi-hop retriever enumerates paths up to four edges and guarantees
that a BFS shortest path is present among at most 12 unranked candidates. It does
not label that candidate as correct, but this is a strong graph-algorithm-assisted
setting. Claims should therefore use “anchor-aware graph retrieval” or
“graph-algorithm-assisted VLM,” not pure semantic GraphRAG performance.

## Reproduce the included benchmark

From this directory, using Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
python validate_benchmark.py
python run_complex_experiment.py --dry-run
$env:DASHSCOPE_API_KEY="your-key"
python run_complex_experiment.py --conditions vlm vlm_flat_kg vlm_graph_rag
```

To rebuild the benchmark from a CiteWeave run:

```powershell
python build_complex_benchmark.py --run-dir "../../runs/pilot-llm-bibliometrics"
```

The expected run directory must contain the keyword co-occurrence node and edge
tables produced by CiteWeave.

## Files

- `build_complex_benchmark.py`: constructs subgraphs, images, questions, and gold answers.
- `run_complex_experiment.py`: builds prompts, calls the compatible Qwen API, and scores outputs.
- `validate_benchmark.py`: validates graph, answer, retrieval, and scorer consistency.
- `audit_results.py`: performs stricter post-hoc checks, including free-text relation review.
- `generated_benchmark/`: the 24-sample benchmark and PNG images.
- `results/qwen3-vl-plus-2025-12-19_seed42/`: the reported run and audit report.

No API key is stored in this directory.

## Limitations

- The 24 samples come from one bibliometric topic graph, not 24 independent corpora.
- Results use one model, one seed, one repeat, and temperature 0.
- Questions and gold answers are generated from graph structure and need human
  review before use as a publication benchmark.
- Structured hallucination scoring does not automatically judge every semantic
  statement in free text; `audit_results.py` produces conservative manual-review flags.
- The next experiment should compare ranked retrieval, retrieval without forced
  shortest-path inclusion, and deterministic relation verification on multiple
  CiteWeave topic graphs.
