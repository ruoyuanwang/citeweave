# CiteWeave Paper Roadmap

## Recommended paper claim

**CiteWeave is a provenance-first bibliometric automation framework that
combines deterministic pipeline validation, graph-grounded interpretation, and
adaptive selective review.**

The paper should not be framed as merely a software description. Its central
research contribution is a reliability architecture and benchmark that
separates factual support, structured-output correctness, evidence validity,
coverage, and review effort.

## Defensible contributions

1. A reproducible end-to-end bibliometric workflow with machine-verifiable
   acquisition, canonicalization, network, figure, evidence, and package gates.
2. A unified bibliometric property graph in which analytical facts are
   first-class provenance-bearing nodes.
3. A graph QA benchmark with answerable and false-premise questions, explicit
   evidence nodes and edges, and non-vacuous evidence-path scoring.
4. A matched Graph RAG ablation showing higher structured accuracy and lower
   unsupported atomic-claim rate than question-only generation on three
   untouched topics.
5. A guarded feedback-memory policy that reduces simulated review intervention
   while maintaining a fixed audited precision criterion.
6. A measurement contribution: factual hallucination, answer-schema error,
   format failure, evidence validity, and claim coverage are reported
   separately.

## Claims that require additional evidence

- **Reduced human labor:** repeat the adaptive study with independent human
  reviewers and measure actual time.
- **Graph superiority over visual grounding:** run a matched figure/VLM arm with
  comparable information and model capability.
- **Open-ended interpretation reliability:** add narrative tasks and blinded
  atomic-claim annotation.
- **Broad generalization:** add more locked topics and at least one larger and
  one stronger model.

## Minimum study before submission

1. Recruit at least two independent reviewers for a stratified sample of
   escalated and auto-accepted items.
2. Report agreement before adjudication, reviewer time, override rate, and a
   risk-coverage curve.
3. Repeat the text ablation with one stronger open model or commercial model.
4. Add local-neighborhood, bridge, temporal, and multi-hop graph questions.
5. Deposit code, frozen benchmark artifacts, prompts, model hashes, and a data
   availability statement in a repository with a persistent identifier.

The figure/VLM experiment is strongly recommended for a broad AI or information
processing venue, but it is not mandatory for a focused scientometrics methods
paper if the visual comparison is removed from the confirmatory claims.

## Suggested manuscript structure

1. Introduction: reliability gap in automated bibliometric interpretation.
2. Related work: bibliometric automation, provenance, Graph RAG, selective
   prediction, and human-in-the-loop learning.
3. CiteWeave architecture and threat model.
4. Benchmark and registered metrics.
5. Data and experimental design.
6. Pipeline validity results.
7. Graph-grounding ablation.
8. Adaptive-review pilot and human replication.
9. Error analysis and metric decomposition.
10. Limitations, reproducibility, and conclusion.

## Journal strategy

### Best topical fit

- **Journal of Informetrics** — strongest fit when the contribution is framed
  as a rigorous new informetric method and reliability benchmark.
- **Quantitative Science Studies** — strong fit for an openly reproducible
  methodological study about science and scholarly communication; the journal
  explicitly considers novel methodological approaches and requires open
  sharing of essential reproducibility data.
- **Scientometrics** — pragmatic fit for a quantitative science-of-science
  methods paper with a substantial empirical evaluation.

### Broader, more demanding alternatives

- **Journal of the Association for Information Science and Technology** —
  appropriate after strengthening the information-behavior, human-review, or
  socio-technical contribution.
- **Information Processing & Management** — appropriate after adding stronger
  models, a matched VLM arm, broader information-retrieval baselines, and a more
  general computing contribution.

### Resource-oriented alternative

- **Data Intelligence** — suitable if the paper is reorganized around the open
  benchmark, knowledge graph, workflow, and reproducibility resource rather
  than the scientometric substantive contribution.

## Recommended submission order

For the current project, the recommended route is:

1. strengthen human validation and model replication;
2. submit to **Journal of Informetrics** or **Quantitative Science Studies**;
3. use **Scientometrics** as the lower-risk topical alternative.

If the VLM comparison and broader retrieval experiments are completed
convincingly, **Information Processing & Management** becomes a plausible
high-ambition target.
