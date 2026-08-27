# CiteWeave 系统内 GraphRAG 图解释实验（2026-08-27）

## 实验定位

本实验把 GraphRAG 加入 CiteWeave 原有的“网络图生成 → 图解释 → 证据包 → 正文生成”链路，
不是仓库之外的独立问答任务。三组执行相同的复杂图解释任务，只改变模型可见的图证据：

- `vlm`：网络 PNG 与节点别名表；
- `flat_kg`：PNG 与完整显示子图的节点、边表；
- `graph_rag`：PNG 与检索出的社区锚点、跨社区边、多跳路径、hub/bridge 证据。

模型输出均经过同一套确定性结构核验器；不存在的边、错误社区和无效路径不能进入
`EvidenceBundle`。

## 已完成的 smoke 结果

开发图上各方法运行 1 次，模型为 `qwen3-vl-plus-2025-12-19`，温度为 0：

| 方法 | 声明支持率 | 边幻觉率 | 路径有效率 | 有效槽位覆盖 | 有效复杂声明 |
|---|---:|---:|---:|---:|---:|
| Direct VLM | 0% | 64.7% | 0% | 0% | 0 |
| Flat KG | 40% | 50.0% | 50% | 40% | 2 |
| GraphRAG | 100% | 0% | 100% | 100% | 3 |

GraphRAG 本次使用 11,051 个 Prompt Token，Flat KG 使用 24,231 个，降幅为 54.39%。
这些数值用于证明系统接入、提示协议和核验链路可以工作；因为只有一张开发图、每组一次调用，
不能据此声称统计显著性或普遍优越性。

`smoke/` 保存冻结图案例、PNG、逐次模型原始输出、核验结果、汇总指标和运行清单，可用于审计。

## 正式实验准备状态

`formal_preparation/` 记录了正式多主题数据的准备结果：12/12 个主题满足准入条件，
12 个 `case_id` 均不同，任意两主题语料的最大 Jaccard 重叠率为 0.149712（阈值 0.20）。

正式实验设计为：

```text
12 张独立主题图 × 3 种方法 × 3 次重复 = 108 次模型调用
```

这里提交的是准备清单，不包含约 188 MB 的完整主题项目，也不代表 108 次正式调用已经执行。

## 复现命令

在仓库根目录设置 API Key（不要写入源码或提交记录）：

```powershell
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
```

单项目三组消融：

```powershell
.\.venv\Scripts\citeweave.exe graph-ablation `
  ".\runs\pilot-system-graphrag" `
  --repeats 1
```

已有 12 个正式项目时运行正式套件：

```powershell
.\.venv\Scripts\citeweave.exe graph-suite-run `
  ".\experiments\formal_graph_suite.yml" `
  --repeats 3
```

正式项目目录体积较大，需从本地实验存档恢复到配置指定位置，或重新执行数据准备步骤。
