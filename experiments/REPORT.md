# CiteWeave Experimental Report

## Executive summary

This report evaluates CiteWeave, an English-language bibliometric automation
system, at module, end-to-end, adaptive-review, and graph-grounded generation
levels. Four accepted cross-domain snapshots contain 1,072 works, 39,130
knowledge-graph nodes, 214,758 edges, 119 structured graph questions, and four
independently validated research packages. All 40 mandatory pipeline checks
passed.

A matched local-model experiment compared graph-grounded and question-only
generation on 89 paired items from three topics untouched during prompt
development. Graph RAG raised structured exact accuracy from 0.202 to 0.730
(difference 0.528; topic-stratified bootstrap 95% CI 0.427-0.629). The atomic
unsupported-claim rate was 0.000 for Graph RAG and 1.000 for no graph among
scorable statements. This result is bounded to short, template-constrained
graph claims; it is not evidence of zero hallucination in open-ended prose.

The adaptive-review pilot reduced intervention from 1.000 in two cold-start
rounds to 0.091 in a corrective round and 0.000 in the corrected and locked
rounds. All 92 retrospectively audited auto-accepts were correct, with a
one-sided 95% exact lower confidence bound of 0.968. Because the assistant
served as the reviewer proxy, this validates the mechanism but does not yet
establish reduced labor for real human reviewers.

## 1. Data and pipeline validity

| Dataset | Role | Works | Relevance | Abstracts | KG nodes | KG edges | QA |
|---|---|---:|---:|---:|---:|---:|---:|
| Graph RAG 2022-2025 | Prompt development | 172 | 0.983 | 0.610 | 5,661 | 20,371 | 30 |
| CRISPR editing 2018-2020 | Evaluation | 300 | 0.930 | 0.313 | 9,532 | 75,699 | 30 |
| Quantum ML 2019-2021 | Evaluation | 300 | 1.000 | 0.443 | 10,947 | 70,385 | 30 |
| Climate adaptation 2018-2020 | Locked evaluation | 300 | 0.973 | 0.373 | 12,990 | 48,303 | 29 |

Relevance is the proportion whose normalized title and abstract contain all
registered core terms. Raw responses, curated JSONL files, SHA-256 hashes,
acquisition manifests, and rejected candidates are retained. The datasets are
relevance-ranked Crossref samples subject to registered caps, not complete
censuses of their research areas.

Each accepted dataset passed ten independent checks: acquisition disclosure,
canonical tables, registered year scope, foreign-key integrity, network
invariants, figure existence and hashes, English report structure, raw-to-
evidence traceability, KG/QA construction, and package-file hashes. The
experimental audit found and corrected twelve implementation or benchmark
defects, each covered by a regression test or reproducible rerun.

## 2. Adaptive review

Eligible units combine dataset-quality alerts and atomic graph claims. The
policy always escalates high or critical issues. Low and medium issues may be
auto-accepted only after consistent accepts from at least two distinct prior
datasets, a posterior accept fraction of at least 0.95, and satisfaction of
the current context guard.

| Round | Items | Interventions | HIR | Auto coverage |
|---|---:|---:|---:|---:|
| Graph RAG curated | 31 | 31 | 1.000 | 0.000 |
| CRISPR | 32 | 32 | 1.000 | 0.000 |
| Quantum candidate | 33 | 3 | 0.091 | 0.909 |
| Quantum corrected | 31 | 0 | 0.000 | 1.000 |
| Climate locked | 31 | 0 | 0.000 | 1.000 |

The failed quantum candidate is informative: the policy accepted its registered
truncation but escalated low topic relevance, after which deterministic
curation raised relevance to 1.000. A context guard also prevented low abstract
coverage from being auto-accepted before correction. The 92 auto-accepted
outcomes were evaluated only after the online sequence and were not added to
memory.

The pilot meets its registered system-level criterion, but publication claims
should be limited to an adaptive review simulation until independent reviewers
repeat the study. A human study should use at least two blinded domain-capable
reviewers per sampled claim, adjudicate disagreements, report Krippendorff's
alpha or Cohen's kappa, and compare actual review minutes as well as HIR.

## 3. Graph-grounding benchmark

KG schema v2 represents works, authors, institutions, sources, keywords,
references, six bibliometric networks, clusters, graph facts, and evidence
relations. Answerable QA items cite a fact node plus its network and supporting
entities or edges. False-premise items use an explicit node-absence operation.

### 3.1 Deterministic metric checks

Across all 119 items:

| Condition | Exact accuracy | UCR | Evidence-path validity | Evidence recall |
|---|---:|---:|---:|---:|
| Structured graph oracle | 1.000 | 0.000 | 1.000 | 1.000 |
| No-context abstain stress test | 0.193 | 0.000 | 0.000 | 0.000 |
| No-context forced-answer stress test | 0.000 | 1.000 | 0.000 | 0.000 |

These rows validate the scorer and illustrate the coverage-safety trade-off.
They are not model treatment effects.

### 3.2 Matched local-model experiment

The experiment used Qwen2.5-1.5B-Instruct GGUF Q4_K_M through llama.cpp b9632
on CPU. Both conditions used prompt version `graph-qa-v3`, temperature 0, seed
42, a 4,096-token context, a 384-token completion cap, and the same JSON schema.
The graph condition received retrieved structured facts; the no-graph condition
received only the question. Raw requests and responses are retained.

The Graph RAG topic was used for prompt development and excluded from the
confirmatory pool. Results on the three untouched topics were:

| Metric | Graph RAG | No graph |
|---|---:|---:|
| Structured exact accuracy | 0.730 | 0.202 |
| Atomic unsupported-claim rate | 0.000 | 1.000 |
| Statement-claim coverage | 0.986 | 0.577 |
| Format-failure rate | 0.270 | 0.798 |
| Structured unsupported-answer rate | 0.338 | 1.000 |
| Evidence-path validity | 0.845 | 0.000 |

Graph RAG produced the exact structured answer on 47 pairs where no graph did
not; the reverse occurred on zero pairs. The exact pooled McNemar p-value was
1.42109e-14. Holm-adjusted topic-level p-values were below 0.001 for all three
topics.

The distinction between factual and structural failure is substantive. Most
Graph RAG errors contained the correct copied natural-language graph fact but
returned `answer: null`, malformed JSON, or a truncated evidence array. The old
scorer incorrectly counted these format failures as hallucinations. Benchmark
version 0.3 now reports:

- atomic UCR over non-abstained, scorable empirical statements;
- structured unsupported-answer rate over attempted structured answers;
- format-failure rate;
- statement-claim coverage.

Atomic support uses exact normalized matching because the prompt instructs the
model to copy a supplied fact. This deterministic metric is conservative and
reproducible for the present task, but open-ended interpretations require
independent semantic annotation.

## 4. What the evidence supports

The current evidence supports three bounded claims:

1. CiteWeave's deterministic pipeline can produce internally consistent,
   traceable bibliometric packages on four cross-domain samples.
2. A guarded feedback-memory policy can reduce simulated review interventions
   without lowering audited item precision in this sequential pilot.
3. On short graph-fact questions with a small frozen local model, retrieved
   graph context substantially improves structured correctness and prevents
   unsupported scorable statements relative to question-only generation.

It does not yet support claims that CiteWeave reduces real human labor, that
Graph RAG eliminates hallucination in unrestricted scholarly interpretation,
or that graph retrieval is superior to a matched visual-language baseline.

## 5. Limitations and next publication steps

- The local model is small and CPU-quantized; repeat with at least one stronger
  open model and one commercial model.
- The planned figure/VLM condition was not executed because the frozen model is
  text-only. A fair VLM comparison requires the same information budget,
  matched model family where possible, and blinded atomic-claim annotation.
- Abstract coverage is 0.313-0.610, so abstract-dependent semantic synthesis is
  outside the accepted scope.
- The QA benchmark emphasizes global network facts and false-premise
  abstention. Add local path, bridge, temporal trend, and multi-hop questions.
- The adaptive pilot used an assistant reviewer proxy. Add independent humans,
  inter-rater agreement, a stratified audit of auto-accepts, and measured time.
- Four topics are insufficient for stable mixed-effects inference; add more
  locked domains before claiming broad generalization.
- Because QA facts and the KG derive from the same frozen network tables, the
  graph oracle is a pipeline consistency ceiling, not an external gold standard.

## 6. Reproducibility

The authoritative protocol is `PLAN.md`, metrics are defined in
`BENCHMARK.md`, dataset registrations are in `datasets.yml`, local model hashes
are in `local_model_manifest.json`, and machine-readable outputs are under
`runs/` and `reviews/results/`. Frozen project packages are under
`experiments/workspaces/` and are intentionally excluded from Git because of
size.
