# Formal Experiment Results

Results cover 6 topic clusters. All 95% confidence intervals use 10,000 topic-cluster bootstrap replicates.

## Report and graph comparisons

| Panel | Target vs comparator | Target UCR | Target completeness | Win / tie / loss | UCR reduction (95% CI) |
|---|---|---:|---:|---:|---:|
| full_vs_oneshot | citeweave_full vs structured_one_shot | 0.017 | 5.000 | 6 / 0 / 0 | 0.066 (0.029 to 0.102) |
| full_vs_human | citeweave_full vs published_human_reference | 0.045 | 5.000 | 6 / 0 / 0 | 0.021 (-0.053 to 0.036) |
| oneshot_vs_human | structured_one_shot vs published_human_reference | 0.109 | 4.000 | 6 / 0 / 0 | -0.109 (-0.158 to -0.056) |
| graph_vs_no | graph_rag vs no_rag | 0.015 | 4.839 | 143 / 27 / 4 | 0.881 (0.864 to 0.897) |
| graph_vs_flat | graph_rag vs flat_structured | 0.014 | 4.839 | 2 / 167 / 5 | -0.006 (-0.016 to 0.002) |
| graph_vs_figure | graph_rag vs figure_vlm | 0.000 | 5.000 | 0 / 35 / 0 | 0.000 (0.000 to 0.000) |

## Human-reference quality gap

The gap is defined as CiteWeave Full minus the published human reference for completeness; positive values favor CiteWeave.

| Metric | Estimate | Topic-cluster bootstrap 95% CI |
|---|---:|---:|
| Completeness Difference | 1.917 | 1.750 to 2.000 |
| Ucr Reduction | 0.021 | -0.053 to 0.036 |
| Pairwise Preference Score | 1.000 | 1.000 to 1.000 |

## Adaptive review

| Policy | Review request rate | Final quality pass rate | Unsafe auto-accept rate |
|---|---:|---:|---:|
| baseline_original | 0.000 | 0.750 | 0.250 |
| always_review | 1.000 | 0.778 | NA |
| static_review | 0.667 | 0.778 | 0.083 |
| adaptive_review | 0.556 | 0.778 | 0.000 |

### Original-to-post-review quality-error reduction

The untouched original candidate was independently evaluated once per case before any Human Proxy intervention.

| Policy | Original error rate | Final error rate | Absolute reduction | Relative reduction |
|---|---:|---:|---:|---:|
| always_review | 0.250 | 0.222 | 0.028 | 0.111 |
| static_review | 0.250 | 0.222 | 0.028 | 0.111 |
| adaptive_review | 0.250 | 0.222 | 0.028 | 0.111 |

## Holm-adjusted preregistered graph comparisons

| Comparison | Raw p | Holm-adjusted p | Reject at 0.05 |
|---|---:|---:|:---:|
| graph_vs_no | 0.0312 | 0.0938 | No |
| graph_vs_flat | 0.5000 | 1.0000 | No |
| graph_vs_figure | 1.0000 | 1.0000 | No |

UCR, completeness, and pairwise results were computed from resolved LLM-as-Judge records. No report-text exact-match proxy was used.
