# LLM Graph-Grounding Ablation: crispr_editing_2018_2020

- Model: `qwen2.5-1.5b-q4km`
- Paired items: 30
- Accuracy: graph 0.700; no graph 0.200
- Accuracy difference: 0.500 (paired bootstrap 95% CI 0.300 to 0.667)
- Unsupported-claim rate: graph 0.000; no graph 1.000
- Statement-claim coverage: graph 1.000; no graph 0.708
- Format-failure rate: graph 0.300; no graph 0.800
- Structured unsupported-answer rate: graph 0.375; no graph 1.000
- UCR reduction: 1.000 (paired bootstrap 95% CI 1.000 to 1.000)
- Exact McNemar test: graph-only correct 15, no-graph-only correct 0, p=6.10352e-05
