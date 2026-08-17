# LLM Graph-Grounding Ablation: quantum_machine_learning_2019_2021

- Model: `qwen2.5-1.5b-q4km`
- Paired items: 30
- Accuracy: graph 0.767; no graph 0.200
- Accuracy difference: 0.567 (paired bootstrap 95% CI 0.400 to 0.733)
- Unsupported-claim rate: graph 0.000; no graph 1.000
- Statement-claim coverage: graph 0.958; no graph 0.458
- Format-failure rate: graph 0.233; no graph 0.800
- Structured unsupported-answer rate: graph 0.292; no graph 1.000
- UCR reduction: 1.000 (paired bootstrap 95% CI 1.000 to 1.000)
- Exact McNemar test: graph-only correct 17, no-graph-only correct 0, p=1.52588e-05
