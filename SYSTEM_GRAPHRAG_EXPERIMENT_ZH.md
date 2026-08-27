# CiteWeave 系统内 GraphRAG 图解释实验

## 这次改的不是外置问答任务

原系统链路是：文献数据 → 计量分析 → 网络图 → EvidenceBundle → 论文正文。

新增链路是：文献数据 → 计量分析 → 网络图 → **图解释节点** → 关系核验 → EvidenceBundle → 论文 3.4 节。

图解释节点读取 CiteWeave 自己生成的关键词共现网络和 PNG，产出可用于论文写作的多节点关系声明。每条声明必须带节点、社区、路径和证据边；不存在的边、错误社区和无效路径会被确定性核验器拒绝，不能进入正文证据包。

## 三组消融

三组都执行同一个 CiteWeave 图解释任务，只改变节点能看到的证据：

1. `vlm`：给网络 PNG 和仅用于统一输出标识的 `alias-label` 表，不提供社区、边或中心性。
2. `flat_kg`：给 PNG 和完整显示子图的节点、边表。
3. `graph_rag`：给 PNG 和按显著节点、社区锚点、跨社区边、多跳路径检索的结构化证据。

消融运行器会冻结同一张图、显示子图和按标签字典序生成的别名映射，并保存 `case_id` 与哈希。三组使用同一模型、同一温度、同一任务槽和同一输出结构；实验文件写入独立目录，不覆盖系统正文与 EvidenceBundle。

## 初步指标

- 声明支持率：通过结构核验的声明数 / 模型报告的声明数。
- 边幻觉率：不存在或格式错误的证据边 / 模型报告的证据边。
- 路径有效率：真实、无循环且证据边完全匹配的多跳路径 / 模型报告的多跳路径。
- 有效任务槽覆盖：通过核验的固定任务槽 / 5 个任务槽。
- 复杂关系产出量：每次运行通过核验且不重复的跨社区、多跳和桥接声明数量。
- 弃答率、API成功率、输入输出Token与运行时间。
- 系统落地结果：通过核验的声明是否进入 `EvidenceBundle`，并被论文 3.4 节引用。

单张关键词图足以证明新增节点已经接入系统并得到初步案例结果，但不能作为论文的最终统计结论。正式实验应扩展到多个查询主题或多个网络实例，并重复运行。

## 运行

### 系统节点与正文案例

在 PowerShell 中执行：

```powershell
cd "<citeweave-repo>"
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
.\.venv\Scripts\citeweave.exe explain-graphs ".\runs\pilot-system-graphrag"
```

节点结果写入：

```text
runs/pilot-system-graphrag/evidence/graph_explanations.json
```

确认结果后，再让 CiteWeave 使用刷新后的证据包生成正文：

```powershell
.\.venv\Scripts\citeweave.exe resume ".\runs\pilot-system-graphrag" --llm --review-rounds 0
```

`resume` 会复用已经生成的图解释结果，不会再次调用 Qwen 图解释接口。

### 三组消融实验

```powershell
cd "<citeweave-repo>"
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
.\.venv\Scripts\citeweave.exe graph-ablation ".\runs\pilot-system-graphrag" --repeats 5
```

运行器自动执行：

```text
Direct VLM × 5
Flat KG × 5
GraphRAG × 5
```

默认输出到项目的 `experiments/system-graph-ablation-时间戳/`，其中包括：

- `cases/keyword_cooccurrence.json`：冻结的图、别名与真值；
- `runs/<mode>/.../repeat-XX.json`：每次完整模型输出与核验结果；
- `records.jsonl`：逐次指标；
- `summary.csv`、`summary.json`、`summary.md`：三组汇总与描述性差值；
- `run_manifest.json`：模型、温度、图哈希、重复次数和成功/失败调用数。

三组消融只调用 Qwen，不调用 DeepSeek，也不会重新生成论文正文。
