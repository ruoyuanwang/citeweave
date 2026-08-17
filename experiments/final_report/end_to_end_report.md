# End-to-End Evaluation of an Automated Bibliometric Analysis System

## Scope and protocol

This report closes the preregistered end-to-end experiment for an automated
bibliometric workflow. The system generated its own topic-aligned Boolean
queries, harvested OpenAlex title-and-abstract metadata through exhaustive cursor
pagination, transformed the records into canonical relational tables, produced
visualizations and graph representations, assembled bounded evidence bundles,
and generated English analytical reports. It then evaluated report generation,
graph-grounded explanation, and adaptive human-in-the-loop review. The experiment
used eight complete natural-year corpora: two development topics
(gnn_drug_discovery_2017_2023, crispr_extracellular_vesicles_2015_2022) and six locked evaluation topics
(machine_learning_climate_change_2008_2022, climate_change_risks_1990_2021, digital_twins_healthcare_2012_2024, plant_heat_drought_2008_2021, gene_editing_als_2004_2024, global_microplastics_2004_2019). Development results were used for calibration only; all
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
| gnn_drug_discovery_2017_2023 | development | 2017–2023 | 456 | 456 | 0 | 456 | 72 | 416/1,654 |
| crispr_extracellular_vesicles_2015_2022 | development | 2015–2022 | 259 | 259 | 0 | 259 | 70 | 412/1,669 |
| machine_learning_climate_change_2008_2022 | locked | 2008–2022 | 7,720 | 7,686 | 34 | 7,686 | 72 | 477/1,875 |
| climate_change_risks_1990_2021 | locked | 1990–2021 | 152,865 | 152,030 | 835 | 152,030 | 72 | 473/1,886 |
| digital_twins_healthcare_2012_2024 | locked | 2012–2023 | 796 | 796 | 0 | 796 | 72 | 422/1,696 |
| plant_heat_drought_2008_2021 | locked | 2008–2021 | 1,604 | 1,599 | 5 | 1,599 | 72 | 444/1,762 |
| gene_editing_als_2004_2024 | locked | 2004–2023 | 211 | 211 | 0 | 211 | 70 | 409/1,679 |
| global_microplastics_2004_2019 | locked | 2004–2019 | 4,011 | 3,995 | 16 | 3,995 | 72 | 460/1,908 |
| **Total** | — | — | **167,922** | **167,032** | **890** | **167,032** | — | — |

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
| full_vs_oneshot | citeweave_full | 1.7% [0.0%, 5.4%] | 5.000 [5.000, 5.000] | structured_one_shot | 8.3% [4.9%, 10.3%] | 4.000 [4.000, 4.000] | 6/0/0 |
| full_vs_human | citeweave_full | 4.5% [0.0%, 6.4%] | 5.000 [5.000, 5.000] | published_human_reference | 6.6% [0.0%, 8.7%] | 3.083 [3.000, 3.250] | 6/0/0 |
| oneshot_vs_human | structured_one_shot | 10.9% [5.6%, 15.8%] | 4.000 [4.000, 4.000] | published_human_reference | 0.0% [0.0%, 0.0%] | 3.000 [3.000, 3.000] | 6/0/0 |

For the full workflow versus the structured one-shot benchmark, condition A had 6.6 percentage points lower UCR than condition B; the topic-cluster 95% interval was [2.9, 10.2] points and excluded zero.

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

For CiteWeave Full versus the published human reference, condition A had 2.1 percentage points lower UCR than condition B; the topic-cluster 95% interval was [-5.3, 3.6] points and included zero.
For the structured one-shot baseline versus the published human reference, condition A had 10.9 percentage points higher UCR than condition B; the topic-cluster 95% interval was [-15.8, -5.6] points and excluded zero.

## Graph grounding, flat evidence, no retrieval, and Figure/VLM

The graph experiment reused bibliometric network data in machine-readable form.
`graph_rag` retrieved graph facts and relations; `flat_structured` supplied
non-graph structured evidence; `no_rag` supplied no retrieval grounding; and the
cross-model `figure_vlm` condition interpreted only the rendered visualization.
This design separates whether an answer is visually plausible from whether its
claims are traceable to canonical graph evidence.

| Comparison | Condition A | A UCR (95% CI) | A completeness (95% CI) | Condition B | B UCR (95% CI) | B completeness (95% CI) | A wins/ties/losses |
|---|---|---:|---:|---|---:|---:|---:|
| graph_vs_no | graph_rag | 1.5% [0.5%, 2.6%] | 4.839 [4.726, 4.952] | no_rag | 89.6% [88.2%, 91.1%] | 1.851 [1.775, 1.927] | 143/27/4 |
| graph_vs_flat | graph_rag | 1.4% [0.4%, 2.5%] | 4.839 [4.726, 4.952] | flat_structured | 0.8% [0.2%, 1.6%] | 4.908 [4.821, 4.978] | 2/167/5 |
| graph_vs_figure | graph_rag | 0.0% [0.0%, 0.0%] | 5.000 [5.000, 5.000] | figure_vlm | 0.0% [0.0%, 0.0%] | 5.000 [5.000, 5.000] | 0/35/0 |

For Graph RAG versus no retrieval, condition A had 88.1 percentage points lower UCR than condition B; the topic-cluster 95% interval was [86.4, 89.7] points and excluded zero.
For Graph RAG versus flat structured retrieval, condition A had 0.6 percentage points higher UCR than condition B; the topic-cluster 95% interval was [-1.6, 0.2] points and included zero.
For Graph RAG versus Figure/VLM, condition A had 0.0 percentage points equal UCR than condition B; the topic-cluster 95% interval was [0.0, 0.0] points and included zero.

The family-wise interpretation uses the recorded exact topic-level sign-flip tests
and Holm adjustment:

| Comparison | Raw p | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|---:|
| graph_vs_no | 0.0312 | 0.0938 | false |
| graph_vs_flat | 0.5000 | 1.0000 | false |
| graph_vs_figure | 1.0000 | 1.0000 | false |

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
| baseline_original | 36 | 0 | 27 | 0.0% | 75.0% | 25.0% |
| always_review | 36 | 36 | 28 | 100.0% | 77.8% | not defined (no auto-accepts) |
| static_review | 36 | 24 | 28 | 66.7% | 77.8% | 8.3% |
| adaptive_review | 36 | 20 | 28 | 55.6% | 77.8% | 0.0% |

- **always_review:** quality-error rate changed from 25.0% to 22.2%, an absolute reduction of 2.8 percentage points.
- **static_review:** quality-error rate changed from 25.0% to 22.2%, an absolute reduction of 2.8 percentage points.
- **adaptive_review:** quality-error rate changed from 25.0% to 22.2%, an absolute reduction of 2.8 percentage points.

Adaptive review requested 44.4 percentage points fewer reviews than always-review. This result must be read jointly with final-quality pass rate
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
10,000 resamples,
seed 20260806, across six locked
topic clusters. Graph UCR comparisons use exact paired topic-level sign-flip tests
with Holm adjustment. The supplementary comparisons are identified in the
statistics rather than promoted after seeing their results.

This report is a deterministic rendering of accepted local artifacts. Its
provenance manifest records SHA-256 hashes for 1,029 structured
source files, including the frozen registry, eight-topic harvest and processing
manifests, report and graph artifacts, six-topic resolved comparisons, adaptive
baseline and post-review counts, and formal statistical outputs. Re-running the
generator on unchanged sources produces identical bytes; changed or incomplete
sources cause refusal rather than silent overwrite. Together with archived raw
page hashes, canonical relation hashes, prompt/call records, blind packet
exchanges, and the read-only completion audit, this supports exact artifact-level
traceability without claiming that stochastic model outputs can be regenerated
from external services indefinitely.
