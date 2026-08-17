# Experiment Status and Decision Log

**Last updated:** 2026-08-06  
**Protocol:** 0.3  
**Overall status:** Core text-only experimental program complete

## Current state

| Work package | State | Evidence |
|---|---|---|
| Baseline audit | Complete | Initial baseline, defect log, regression tests |
| Plan and benchmark | Complete v0.3 | `PLAN.md`, `BENCHMARK.md`, `datasets.yml` |
| Dataset acquisition | Complete | Four accepted snapshots; 1,072 works |
| Pipeline validity | Complete | Four independent 10/10 validity reports |
| Adaptive review | Complete pilot | Five sequential rounds, 158 eligible items |
| Graph/KG infrastructure | Complete | KG schema v2 and 119 QA items |
| Deterministic stress baselines | Complete | Oracle, abstain, forced-answer conditions |
| Matched text-only LLM ablation | Complete | 89 untouched-topic paired items |
| Figure/VLM ablation | Not executed | Text-only local model; retained as future study |
| Final English report | Complete | `REPORT.md` |

## Accepted datasets

| Dataset | Role | Works | Relevance | Abstracts | KG nodes | KG edges | QA |
|---|---|---:|---:|---:|---:|---:|---:|
| Graph RAG 2022-2025 | Prompt development | 172 | 0.983 | 0.610 | 5,661 | 20,371 | 30 |
| CRISPR editing 2018-2020 | Evaluation | 300 | 0.930 | 0.313 | 9,532 | 75,699 | 30 |
| Quantum ML 2019-2021 | Evaluation | 300 | 1.000 | 0.443 | 10,947 | 70,385 | 30 |
| Climate adaptation 2018-2020 | Locked evaluation | 300 | 0.973 | 0.373 | 12,990 | 48,303 | 29 |

All accepted runs passed acquisition integrity, registered year scope,
canonical schema and foreign-key integrity, network endpoint/weight invariants,
figure hashes, English report structure, evidence traceability, KG/QA
construction, and package-hash integrity.

## Matched text-only LLM ablation

The frozen prompt (`graph-qa-v3`) was developed only on the Graph RAG topic.
The confirmatory pool contains the other three topics. Both conditions used the
same local Qwen2.5-1.5B-Instruct Q4_K_M model, temperature 0, seed 42, and JSON
schema. The only intended treatment difference was retrieved graph context.

| Metric | Graph RAG | No graph | Difference/reduction |
|---|---:|---:|---:|
| Structured exact accuracy | 0.730 | 0.202 | +0.528 |
| Atomic unsupported-claim rate | 0.000 | 1.000 | 1.000 reduction |
| Statement-claim coverage | 0.986 | 0.577 | +0.409 |
| Format-failure rate | 0.270 | 0.798 | -0.528 |
| Structured unsupported-answer rate | 0.338 | 1.000 | -0.662 |

The topic-stratified bootstrap 95% CI for the accuracy difference was 0.427 to
0.629. The exact pooled McNemar test had 47 graph-only correct pairs, zero
no-graph-only correct pairs, and p=1.42109e-14. All three topic-level McNemar
tests remained significant after Holm correction.

The 0.000 graph UCR has a deliberately narrow interpretation. It means every
scorable atomic statement copied from retrieved graph facts matched its gold
statement after Unicode, case, whitespace, and terminal-punctuation
normalization. It does not establish zero hallucination for unrestricted
narrative generation. One graph answer emitted no scorable statement, giving
0.986 rather than 1.000 statement coverage.

## Adaptive-review pilot

| Round | Eligible items | Review interventions | HIR |
|---|---:|---:|---:|
| R1 Graph RAG curated | 31 | 31 | 1.000 |
| R2 CRISPR | 32 | 32 | 1.000 |
| R3 Quantum candidate | 33 | 3 | 0.091 |
| R4 Quantum corrected | 31 | 0 | 0.000 |
| R5 Climate locked | 31 | 0 | 0.000 |

Measured online review time was 55.2 seconds. Retrospective annotation found
92/92 auto-accepted items correct; the one-sided 95% Clopper-Pearson lower bound
was 0.968. Outcome labels were not inserted into feedback memory.

This is a system-development pilot in which the assistant supplied the
reviewer decisions. It demonstrates policy mechanics and leakage controls, not
human-subject validity. A paper must add independent human reviewers, blinded
labels, agreement, and reviewer-time measurements before describing the result
as reduced human labor.

## Decisions and deviations

1. Atomic claims are the primary factuality unit.
2. Deterministic validity is separated from generative quality.
3. Capped API results are benchmark samples, never complete source censuses.
4. Core-term relevance below 0.90 triggers correction or rejection.
5. Feedback confirmations are counted by distinct prior datasets.
6. Auto-accept transfer requires the current context to satisfy prior guards.
7. Evidence-path validity is non-vacuous: an answered claim needs cited evidence.
8. The climate topic remained locked during prompt development.
9. The Graph RAG prompt-development topic was excluded from confirmatory pooling.
10. Version 0.3 corrected a scoring implementation error: JSON/answer-schema
    failure had been counted as factual hallucination. The registered benchmark
    already defined UCR over atomic empirical statements, so the correction
    restores protocol compliance. Structured answer error and format failure
    are now reported separately.
11. The planned figure/VLM condition was not run because the frozen local model
    is text-only. It must not be represented as completed evidence.

## Retained failed and excluded runs

- Crossref Graph RAG candidate: rejected for low core-term relevance.
- OpenAlex Graph RAG attempt: rejected after anonymous quota exhaustion.
- Over-strict Graph RAG import: rejected after exact-phrase filtering left three works.
- Initial quantum candidate: rejected at 0.823 relevance and corrected with a
  deterministic `quantum AND learning` filter.
- Prompt pilots v0-v3 on the Graph RAG topic: retained but excluded from pooling.
- Earlier ablation summaries using structured error as UCR: superseded by
  benchmark version 0.3 and retained in Git history rather than reported.

## System defects found and fixed

- unsupported Crossref cursor parameter;
- Crossref online-first/print-year precedence mismatch;
- missing English figure directives in deterministic reports;
- English terminal punctuation falsely marked as truncation;
- English headings falsely reported missing;
- empty evidence sets counted as valid paths;
- aggregate graph facts lacked explicit fact nodes;
- double-prefixed KG entity identifiers prevented entity unification;
- same-title works were ambiguous in strongest-edge statements;
- adaptive confirmations counted records rather than distinct datasets;
- adaptive transfer lacked a context guard;
- structured output failure was conflated with unsupported narrative claims.
