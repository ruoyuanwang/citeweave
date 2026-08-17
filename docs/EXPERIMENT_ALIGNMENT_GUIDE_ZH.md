# CiteWeave 正式实验对齐说明

这份说明用于让协作者在不改变原实验定义的前提下复核现有结果、补跑新实验或产出可合并的数据。开始前先阅读 `docs/EXPERIMENT_RESULTS_SUMMARY_ZH.md`。

## 1. 对齐目标

“对齐”分为两层：

1. **产物级复核**：使用现有冻结语料、盲评结果和统计清单，重新生成统计与最终报告。这个层级应当字节级或数值级一致，不需要访问外部 API。
2. **外部服务重跑**：重新采集 OpenAlex 或重新调用模型。它只能复现协议，不能保证返回相同语料或相同模型文本，因为数据库和在线模型会变化。

默认先完成第 1 层。除非明确分配了完整重跑任务，不要覆盖当前 `experiments/formal_*` 目录。

## 2. 基准快照

| 项目 | 固定值 |
|---|---|
| 仓库提交 | `5f8c16c0ee5cb5574112b7910e07d1d7f6a8dc74` |
| Python | 3.12.7（项目最低要求 3.11） |
| CiteWeave | 0.1.0 |
| DuckDB | 1.5.5 |
| pandas | 3.0.5 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| scikit-learn | 1.9.0 |
| PyArrow | 25.0.0 |
| Pydantic | 2.13.4 |
| 冻结 registry SHA-256 | `7c0ea6f10726e986793780ba5469879255f034eeefdb1c506af54cde9e07d3cd` |
| 正式统计 manifest SHA-256 | `9be40cb77b6eae010c35160c768b47e39b803ab5b31ed46d461e6b10a2c54829` |
| 正式统计 JSON SHA-256 | `e7257d202b19a86054af6f8301d87c1d4b8179e006acb3025a9e9cd506f4c880` |
| 最终英文报告 SHA-256 | `d0b903034320a21a21cfd3b9ce07952c14f47c1ad2ace95e29d55de5cde2d82c` |
| 图实验 run ID | `formal_v2_nonthinking_20260806` |

注意：当前工作树包含大量未提交的实验代码和产物，所以只有 Git commit 不足以定义实验快照。交接时必须连同冻结 registry、统计 manifest、协议文件及所需 `experiments/` 产物一起传递，并用 SHA-256 核对。

## 3. 环境安装

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

若要最大限度贴近原运行环境，应安装上表版本；只执行产物级复核时，优先使用项目随附的原 `.venv` 或由负责人导出的完整 `pip freeze`。当前仓库没有 lockfile，仅执行 `pip install -e ".[dev]"` 可能安装更新版本，不能视为严格环境复现。

需要外部服务时，通过环境变量传密钥，绝不能写入源码、registry、日志或结果包：

```powershell
$env:DEEPSEEK_API_KEY="<your-key>"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
$env:OPENALEX_API_KEY="<optional-openalex-key>"
```

## 4. 数据与检索配置

唯一正式 registry 是：

```text
experiments/formal_datasets_openalex_title_abstract.yml
```

不要使用同目录中的 round2–round6、默认 fulltext、failed、rejected 或旧 `formal_datasets.yml` 作为正式结果输入。

冻结规则：

- 数据源：OpenAlex；
- 检索范围：title + abstract；
- 文献类型：article；
- 语言：不限制；
- 时间：每个主题的完整自然年区间；
- `max_records: null`，必须游标遍历到耗尽，不能抽样或截断；
- 保留 raw responses、检索时间和 SHA-256；
- 保留参考文献关系；
- 先按 OpenAlex 身份去重，再在 canonical 阶段按规范化 DOI 或“规范化标题（至少 20 字符）+年份”精确去重；
- 正式统计只使用 6 个 `role: locked` 主题；两个 `role: development` 主题只能用于校准。

处理配置：

| 配置 | 值 |
|---|---:|
| acquisition mode | bulk |
| partition strategy | adaptive_date |
| target slice records | 25,000 |
| page size | 1,000 |
| max retries | 8 |
| processing mode | disk |
| chunk size | 1,000 |
| DuckDB memory limit | 2GB |
| candidate pool size | 640 |
| edge row limit | 200,000 |
| keep partitions | true |

计数时必须同时报告：received records、source-unique/staged records、canonical works、duplicates removed。下游统计的样本量采用 canonical works，不能把 staged records 当作最终分析文献数。

## 5. 报告条件配置

正式报告主模型均为 `deepseek-v4-pro`，API base 为 `https://api.deepseek.com`，报告语言为英文：

| 配置 | 值 |
|---|---:|
| temperature | 0.1 |
| requested seed | 42 |
| max tokens/call | 8,000 |
| Evidence Bundle | 两条件必须 SHA-256 完全相同 |

条件定义：

- `structured_one_shot`：同一冻结 Evidence Bundle，只调用一次；
- `citeweave_full`：固定 4 次调用，依次为 `editorial_plan`、`complete_draft`、`internal_review`、`revised_report`；
- 内部审阅只用于生成修订稿，不能复用为外部评价；
- 每条经验性声明应引用附近的 `[E###]`；只能使用冻结证据，不能引入模型记忆中的事实。

报告运行示例：

```powershell
.\.venv\Scripts\python.exe scripts\run_report_conditions.py `
  --dataset-id machine_learning_climate_change_2008_2022 `
  --evidence experiments\formal_workspaces\machine_learning_climate_change_2008_2022\evidence\evidence_items.json `
  --output-root tmp\aligned_formal_reports `
  --model deepseek-v4-pro `
  --temperature 0.1 `
  --seed 42 `
  --max-tokens 8000
```

在线模型可能不支持或不严格执行 seed，因此外部服务重跑只要求配置和完整 trace 对齐，不要求文本哈希一致。

## 6. 图问答条件配置

严格同模型文本面板：

| 配置 | 值 |
|---|---|
| provider profile | `experiments/provider_profiles/deepseek_v4_pro.json` |
| model | `deepseek-v4-pro` |
| response format | JSON object |
| temperature | 0 |
| max tokens | 512 |
| thinking | disabled |
| seed | provider 不支持，不发送 |
| prompt version | `formal-graph-qa-v2-nonthinking` |

三种文本条件：

- `no_rag`：不提供检索证据；
- `flat_structured`：提供扁平结构化事实；
- `graph_rag`：提供等义的图实体、关系或路径；
- 三者必须使用同模型、同问题、同答案 contract，只有上下文表示不同。

任务族包括 network size、highest occurrence、strongest connection、cluster membership 和 false-premise abstention；共有 bibliographic coupling、citation、coauthorship、cocitation、institution collaboration、keyword co-occurrence 六类网络。不是每个小语料都具备全部有效任务，所以每主题可能是 24–30 项；正式 6 个锁定主题合计 174 项。

Figure/VLM 只看渲染图片，由独立视觉代理完成，属于 cross-model extension；锁定主题共 35 个 network-size 项。不要把它和文本面板写成严格的模型内比较，也不要将 network-size 结果外推到其他任务。

文本图实验示例：

```powershell
.\.venv\Scripts\python.exe scripts\run_formal_graph_experiment.py `
  --dataset machine_learning_climate_change_2008_2022 `
  --condition graph_rag `
  --text-profile experiments\provider_profiles\deepseek_v4_pro.json `
  --run-id <new-run-id> `
  --execute
```

新实验必须使用新 run ID 和新输出目录，禁止写入 `formal_v2_nonthinking_20260806`。

## 7. Judge 与盲评配置

冻结文件为 `experiments/judge_calibration_freeze.json`。核心规则：

- 2 个独立 Judge；
- 条件名盲化，A/B 映射保密；
- 分歧触发独立 blind adjudication；
- Judge 只能观察和评分，不能修改 pipeline 或 source artifacts；
- claim unit 是独立的事实、数值、关系、方法或解释性声明；
- `supported` 必须给出合法 evidence ID；`contradicted` 与 `not_in_evidence` 都计入 unsupported；
- 系统证据使用 `E` 前缀，人类参考证据使用 `H` 前缀；
- 人类参考与系统输出分别相对各自证据评分，人类论文中的数值不是系统语料的 gold value；
- 报告 rubric：`judge-formal-v1-addressable-evidence`；
- 图 rubric：`graph-judge-formal-v1`。

开发校准只使用两个 development topics。`rejected_judge_calibration_*` 中的旧协议存在证据不可寻址或条件标签泄漏问题，绝不能纳入正式结果。

## 8. 自适应审核配置

协议文件：`experiments/human_proxy_protocol.yml`。正式配置：

| 配置 | 值 |
|---|---:|
| protocol | `formal-adaptive-v3-scoped-human-proxy` |
| seed | 42 |
| minimum confirmations | 2 |
| auto-accept threshold | 0.95 |
| audit rate | 0.10 |
| static detector threshold | 0.50 |
| static review severities | high, critical |

条件包括 untouched `baseline_original`、`always_review`、`static_review`、`adaptive_review`。Human Proxy 只能处理系统已经标出的风险，最多查看和编辑 500 字符、每次最多一次局部编辑；不能搜索其他问题、查看隐藏真值、调用外部工具、使用完整产物或重跑分析。反馈 Judge 可更新在线记忆，评价 Judge 不得更新记忆，防止 evaluation-feedback leakage。

原始 `formal_adaptive_review/metrics.json` 含 8 个主题共 48 项；论文正式统计必须通过 `formal_adaptive_topic_counts/*.json` 只汇总 6 个锁定主题共 36 项。

## 9. 指标定义与报告要求

### 9.1 核心指标

- `UCR = unsupported claims / all empirical claims`。越低越好；`Claim Precision = 1 - UCR`。
- Completeness：Judge 的 1–5 分完整度均值。
- Pairwise preference：A 胜、平、负，同时报告 preference score，不能替代 UCR。
- Review Request Rate：请求审核数 / 全部可审核项。
- Final Quality Pass Rate：最终通过项 / 全部项。
- Unsafe Auto-accept Rate：不安全自动放行数 / 自动放行数；Always-review 没有自动放行，分母为 0，必须记为 NA，不能记为 0。
- 图机械指标还应报告 exact answer accuracy、structured unsupported-answer rate、abstention F1、schema-valid rate、证据有效性与 claim coverage。

### 9.2 统计方法

- 统计单元：6 个锁定 topic cluster；
- 置信区间：10,000 次 topic-cluster bootstrap；
- 随机种子：`20260806`；
- UCR reduction 定义：`UCR(comparator) - UCR(target)`，正值有利于 target；
- Completeness difference 定义：`Completeness(target) - Completeness(comparator)`；
- 图的 3 个预注册主比较：基于 topic-level UCR reduction 的精确双侧符号翻转检验，再做 Holm 校正；
- 同时报告估计值、95% CI、原始 p、校正 p 和胜/平/负；不能只报告显著结果。

### 9.3 解释边界

- 置信区间跨 0 时，不得写成确定优势；
- `graph_vs_no` 虽效果量很大，但 Holm-adjusted p=0.0938，不得写成“经多重比较校正后显著”；
- `graph_vs_flat` 当前是近乎全平，不得声称 Graph RAG 优于等信息量扁平证据；
- Human reference 是 topic-aligned comparator，不是同语料 gold truth；
- Adaptive 的 0 observed unsafe auto-accept 不等于安全率已经被证明为 100%；
- 开发结果、failed/rejected/invalidated 结果、`formal_v1` 和锁定正式结果必须分开存放、分开汇报。

## 10. 产物级复核命令

以下命令全部写入 `tmp/`，不会覆盖冻结正式产物：

```powershell
# 1. 完成性审计
.\.venv\Scripts\python.exe scripts\audit_formal_experiment_completion.py `
  --registry experiments\formal_datasets_openalex_title_abstract.yml `
  --run-id formal_v2_nonthinking_20260806 `
  --output tmp\formal_completion_audit_alignment.json

# 2. 重新计算正式统计
.\.venv\Scripts\python.exe scripts\analyze_formal_experiment.py `
  --manifest experiments\formal_statistics_manifest.json `
  --output-dir tmp\formal_statistics_alignment `
  --bootstrap-samples 10000 `
  --seed 20260806

# 3. 确定性重建最终英文报告
.\.venv\Scripts\python.exe scripts\generate_final_english_report.py `
  --registry experiments\formal_datasets_openalex_title_abstract.yml `
  --graph-run-id formal_v2_nonthinking_20260806 `
  --report tmp\end_to_end_report_alignment.md `
  --manifest tmp\end_to_end_report_alignment.manifest.json

# 4. 哈希核对
Get-FileHash tmp\formal_statistics_alignment\formal_statistics.json -Algorithm SHA256
Get-FileHash tmp\end_to_end_report_alignment.md -Algorithm SHA256
```

预期：在 resolved 镜像目录可读时，审计为 108/108 完整；直接使用冻结 manifest 时统计 JSON 哈希为 `e7257d...c880`；英文报告哈希为 `d0b903...2d82c`。如果统计 JSON 因路径、JSON 属性顺序或非语义 metadata 导致哈希不同，至少应逐项比对 `formal_metrics.csv`、`graph_holm.csv` 和各 section 的数值完全一致。

如果 Windows 对 `experiments/formal_judging/report_resolved` 或 `graph_resolved` 返回 `Permission denied`，先检查 ACL 或重新传递这些普通目录。也可以从可读的总 resolved 文件在 `tmp/` 重建只读镜像，再生成临时统计 manifest：

```powershell
$check = "tmp\alignment_fallback"

foreach ($c in @("full_vs_oneshot", "full_vs_human", "oneshot_vs_human")) {
  .\.venv\Scripts\python.exe scripts\split_resolved_judgments_by_topic.py `
    --input "experiments\formal_judging\$c\resolved_v1\resolved_judgments.jsonl" `
    --references experiments\human_references.yml `
    --output-root "$check\report_resolved\$c"
}

foreach ($c in @("graph_vs_no", "graph_vs_flat", "graph_vs_figure")) {
  .\.venv\Scripts\python.exe scripts\split_resolved_judgments_by_topic.py `
    --input "experiments\formal_graph_judging\$c\resolved_v1\resolved_judgments.jsonl" `
    --references experiments\human_references.yml `
    --output-root "$check\graph_resolved\$c"
}

.\.venv\Scripts\python.exe scripts\prepare_formal_statistics_manifest.py `
  --references experiments\human_references.yml `
  --report-root "$check\report_resolved" `
  --graph-root "$check\graph_resolved" `
  --adaptive-root experiments\formal_adaptive_topic_counts `
  --output "$check\formal_statistics_manifest.json"

.\.venv\Scripts\python.exe scripts\analyze_formal_experiment.py `
  --manifest "$check\formal_statistics_manifest.json" `
  --output-dir "$check\formal_statistics" `
  --bootstrap-samples 10000 `
  --seed 20260806
```

此 fallback 只重建索引镜像，不改变判决数据。由于临时 manifest 的路径和 JSON 属性顺序可能不同，整体 JSON 哈希不一定相同；核心指标、置信区间、`formal_metrics.csv` 和 `graph_holm.csv` 必须完全一致。最终英文报告生成器是 fail-closed 的，若正式 resolved 镜像不可读会拒绝生成；应先修复读取权限，不能绕过完整性检查。

## 11. 完整流水线与资源提醒

从冻结 registry 重新采集和处理全部主题会下载约 16.8 万条源记录，其中最大主题超过 15 万条，耗时、磁盘和 API 配额开销都较大。先 dry-run：

```powershell
.\.venv\Scripts\python.exe scripts\run_remaining_formal_experiments.py `
  --registry experiments\formal_datasets_openalex_title_abstract.yml `
  --dry-run
```

只有获得明确任务后才执行完整采集：

```powershell
.\.venv\Scripts\python.exe scripts\run_formal_pipeline.py `
  --registry experiments\formal_datasets_openalex_title_abstract.yml `
  --all-datasets `
  --stage all
```

外部服务重跑时必须保存 raw request/response、模型标识、temperature、thinking、token 上限、用量、延迟、检索时间、cursor exhaustion、所有输入输出 SHA-256 和错误重试记录。数据库或模型快照改变时，使用新的 run ID，并把结果标为 protocol replication，不得与当前冻结 formal run 直接拼接。

## 12. 提交给负责人的最小交付包

每次补实验至少提交：

1. 一页实验说明：假设、数据角色、条件、唯一变化项；
2. 运行命令和环境版本；
3. 输入 registry/manifest 及 SHA-256；
4. 每条件的 run manifest、raw trace、completion/checkpoint；
5. 机械评分和盲评原始结果；
6. resolved judgments 与 adjudication 记录；
7. 汇总 CSV/JSON，含分子、分母、主题级结果、CI 和校正 p；
8. 完成性审计；
9. 异常、失败、排除项和理由；
10. 明确声明是否使用了 development 数据，以及是否发生任何配置偏离。

合入前检查：

- [ ] 没有覆盖原 formal 目录；
- [ ] 没有把开发主题放入正式统计；
- [ ] 两个报告条件的 evidence hash 相同；
- [ ] 三个文本图条件只有 evidence representation 不同；
- [ ] Judge 条件盲化且两名 Judge 独立；
- [ ] 分歧已经 adjudicate；
- [ ] UCR 的分子分母可追溯；
- [ ] Always-review 的 unsafe auto-accept 为 NA；
- [ ] 同时报效果量、CI、原始 p 和 Holm-adjusted p；
- [ ] 所有 raw/processed/statistical artifacts 有 SHA-256；
- [ ] 失败和负结果没有被删除或改写。
