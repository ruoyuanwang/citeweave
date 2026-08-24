# CiteWeave 复杂图解释实验（初步结果）

## 实验目标

在同一批由 CiteWeave 关键词共现图派生的复杂子图上比较三种证据条件：

- `VLM`：只读取 PNG 网络图；
- `VLM + Flat KG`：读取 PNG 和完整节点边表，作为结构化信息基线；
- `VLM + Multi-hop GraphRAG`：读取 PNG 和按问题锚点检索的多跳候选链或节点删除探针。

Benchmark 共 24 题，每类 6 题，包括多跳最短路径、跨社区路径、桥接节点和不存在直接边。每题使用一张 8 节点子图，标准答案由图结构自动计算。

## 初步结果

模型为 `qwen3-vl-plus-2025-12-19`，temperature=0，每题运行一次。

| 条件 | 严格准确率 | 结构化关系幻觉率 |
|---|---:|---:|
| VLM | 10/24（41.7%） | 7/57（12.3%） |
| VLM + 完整 KG | 19/24（79.2%） | 0/66（0.0%） |
| VLM + Multi-hop GraphRAG | 20/24（83.3%） | 2/64（3.1%） |

这个结果说明：结构化图证据相对只看图片的 VLM 有明显帮助。但目前不能声称 GraphRAG 明显优于完整 KG，因为两者只差 1 题，配对精确 McNemar 检验 p=1.0。

还需特别说明：当前多跳检索器枚举不超过 4 跳的路径，并保证 BFS 最短路径出现在最多 12 条未排序候选链中。模型不知道哪条是正确答案，但检索阶段提供了很强的图算法辅助。因此现阶段更准确的叫法是“锚点感知的图检索”或“图算法增强 VLM”，而不是纯语义 GraphRAG。

## 文件说明

- `build_complex_benchmark.py`：从 CiteWeave 节点表和边表生成子图、PNG、题目与标准答案；
- `run_complex_experiment.py`：构造三组输入、调用 Qwen API 并自动评分；
- `validate_benchmark.py`：检查图、答案、检索输入和评分器的一致性；
- `audit_results.py`：复算结构化评分，并对自由文本关系和桥接证据做保守审计；
- `generated_benchmark/`：24 题及配套 PNG；
- `results/qwen3-vl-plus-2025-12-19_seed42/`：本次结果和审计报告。

## PowerShell 复现

在本目录执行：

```powershell
python -m pip install -r requirements.txt
python .\validate_benchmark.py
python .\run_complex_experiment.py --dry-run
$env:DASHSCOPE_API_KEY="替换成你自己的Key"
python .\run_complex_experiment.py --conditions vlm vlm_flat_kg vlm_graph_rag
```

如果要从 CiteWeave 已有运行重新生成题目：

```powershell
python .\build_complex_benchmark.py --run-dir "..\..\runs\pilot-llm-bibliometrics"
```

## 结论边界

- 24 题来自一个文献计量主题图，不是 24 个独立语料库；
- 目前只有一个模型、一个 seed、一次重复；
- 自由文本中的所有语义陈述无法完全依靠规则评分，审计脚本输出的可疑项仍需人工复核；
- 下一步应在多个 CiteWeave 主题图上比较排序检索、取消强制包含最短路径、关系验证和纠错机制。

本目录不保存 API Key。
