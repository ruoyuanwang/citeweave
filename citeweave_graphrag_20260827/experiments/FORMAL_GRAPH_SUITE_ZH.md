# CiteWeave 多主题 GraphRAG 正式实验方案

## 研究问题

本实验评估的不是“大模型智力是否提升”，而是 CiteWeave 的图解释节点在复杂关系描述中，
能否通过结构化图检索降低幻觉并提高可核验解释的产出。

预先固定两个主要假设：

1. GraphRAG 相比 Direct VLM，具有更高的有效任务槽覆盖率和更低的边幻觉率。
2. GraphRAG 相比 Flat KG，在可靠性相当或更好的前提下，使用更少的 Prompt Token。

路径有效率、声明支持率、有效复杂声明数量和弃答率作为次要指标。

## 三组输入

- `vlm`：PNG 和统一输出标识所需的 alias-label 表，不提供边、社区或中心性。
- `flat_kg`：同一 PNG 和完整显示子图的节点、边表。
- `graph_rag`：同一 PNG 和检索出的社区边、跨社区边、多跳路径、hub 与割点证据。

三组使用相同模型快照、温度、任务槽、JSON Schema、图片和确定性核验器。调用顺序按
case 哈希执行轮换 Latin square，避免始终先跑某一种方法造成时段偏差。

## 测试集与准入规则

- 数据源：Europe PMC。
- 固定时间：2020—2025年，避开尚未结束的2026年。
- 网络类型：只使用关键词共现图。
- 显示规模：50个节点、最多40个标签。
- 12个候选主题均未用于 Prompt 或核验器调试；原 LLM＋bibliometrics 图只作开发集。
- 准入规则在查看正式模型输出前固定：至少40节点、40边、2社区、1条跨社区边、
  2条2—4跳路径。
- 任意两主题语料 work_id 的 Jaccard 重叠率不得超过0.20。

当前准备结果：12/12主题通过，12个 case_id 均不相同；每图50节点、150—152条边、
4—9个社区，语料重叠最大值为0.149712。

## 样本量和统计

正式规模为：

```text
12张独立主题图 × 3种方法 × 3次重复 = 108次模型调用
```

统计独立样本量是12张主题图，不是108次调用。程序先在每个“主题×方法”内部平均3次，
再计算 GraphRAG 相对两种基线的图内配对差值，并报告图级 bootstrap 95%置信区间、
Wilcoxon配对检验和Holm校正结果。

## 正式运行前的开发集 Smoke

先在已经用于调试的旧图上确认真实API接受新的 `focus_node` 和 `hub/bridge` 协议。
该结果不进入正式统计：

```powershell
cd "<citeweave-repo>"
$env:DASHSCOPE_API_KEY="你的百炼API Key"
.\.venv\Scripts\citeweave.exe graph-ablation `
  ".\runs\pilot-system-graphrag" `
  --repeats 1 `
  --output ".\runs\pilot-system-graphrag\experiments\hub-bridge-smoke-20260827"
```

## 正式运行

12个项目已经准备完毕，不需要再次抓取：

```powershell
cd "<citeweave-repo>"
$env:DASHSCOPE_API_KEY="你的百炼API Key"
.\.venv\Scripts\citeweave.exe graph-suite-run `
  ".\experiments\formal_graph_suite.yml" `
  --repeats 3
```

本实验只调用 Qwen 图解释接口，不调用 DeepSeek。API Key 只通过当前 PowerShell 环境变量
读取，不写入代码、配置或结果文件。

结果目录包括：

- `records.jsonl`：所有逐次调用与指标；
- `topic_mode_summary.csv`：先按主题聚合的三组结果；
- `paired_differences.csv`：每张图的配对差值；
- `aggregate_summary.csv`：跨图均值和置信区间；
- `statistics.json`：Wilcoxon与Holm校正；
- `summary.md`：可直接阅读的总表；
- `run_manifest.json`：模型、调用顺序、case、Prompt、Schema和源码哈希。

正式输出应表述为“图证据编排提高复杂图解释的可核验性”，不能表述为“模型自主推理能力
被训练或增强”。
