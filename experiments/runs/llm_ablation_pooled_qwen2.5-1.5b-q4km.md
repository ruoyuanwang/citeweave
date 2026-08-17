# Pooled LLM Graph-Grounding Ablation

- Model: `qwen2.5-1.5b-q4km`
- Topics: 3
- Paired items: 89
- Prompt-development topic excluded from pooling: `rag_graph_2022_2025`
- Accuracy: graph 0.730; no graph 0.202
- Accuracy difference: 0.528 (topic-stratified bootstrap 95% CI 0.427 to 0.629)
- Unsupported-claim rate: graph 0.000; no graph 1.000
- Statement-claim coverage: graph 0.986; no graph 0.577
- Format-failure rate: graph 0.270; no graph 0.798
- Structured unsupported-answer rate: graph 0.338; no graph 1.000
- UCR reduction: 1.000 (topic-stratified bootstrap 95% CI 1.000 to 1.000)
- Exact pooled McNemar test: graph-only correct 47, no-graph-only correct 0, p=1.42109e-14

## Holm-adjusted topic tests

| Dataset | Adjusted p |
|---|---:|
| quantum_machine_learning_2019_2021 | 4.57764e-05 |
| crispr_editing_2018_2020 | 0.00012207 |
| climate_adaptation_2018_2020 | 0.00012207 |
