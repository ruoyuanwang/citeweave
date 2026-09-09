# CiteCalibrator 前置诊断实验与 Benchmark 交接文档

> 版本：v1.0
> 日期：2026-09-08
> 状态：待执行的前置研究方案
> 当前决策：**暂不训练模型，先证明问题存在且值得用模型解决**

## 1. 交接目标

本工作的目标不是立即微调一个 8B 模型，而是先回答一个更基础的问题：

> CiteWeave 当前在没有专用小模型的情况下，是否会稳定产生“事实表面正确、但不受证据支持或解释越界”的研究论断；这些问题是否无法被现有规则或通用 API 以足够低的代价解决？

只有答案为“是”，才进入 CiteCalibrator 的数据构造与训练阶段。

本交接文档说明：

1. 原计划训练什么模型；
2. 为什么目前还不能断言该模型有必要；
3. 前置诊断实验应如何开展；
4. Discovery Bench 应如何构造和标注；
5. 用哪些指标判断问题是否存在；
6. 达到什么程度才允许启动模型训练；
7. 现有项目中哪些代码和实验资产可以复用。

完整训练配置另见 `docs/CITECALIBRATOR_TRAINING_PLAN_ZH.md`。本文是训练前的决策门禁，优先级高于训练计划。

## 2. 背景

### 2.1 当前系统的工作方式

CiteWeave 目前通过 Agent 完成文献检索、图构建、图程序执行、来源组织和文章生成。简化流程为：

```text
用户问题
  ↓
文献检索与图构建
  ↓
路径、社区、中心性、删点、时间变化等图程序
  ↓
结构化现象与来源证据
  ↓
通用 API 生成研究文章
```

系统已经能够保存较丰富的计算结果和 provenance。潜在薄弱点位于最后一步：通用模型需要把结构化图结果解释成自然语言研究结论。

### 2.2 假设中的问题

假设程序得到：

> 在当前语料构建、边权阈值为 3 的共现图中，A 位于 B 与 C 之间的一条最短路径上；阈值改为 5 时该路径消失。

文章可能写成：

> A 是驱动 B 与 C 两个研究方向融合的核心机制。

后一句可能包含：

- 从特定语料外推到整个领域；
- 从特定参数下的结果外推到稳定结论；
- 从图结构关系升级为因果关系；
- 把“路径上的节点”实体化为“领域机制”。

这类问题不能简单归为数字错误。它更接近“证据—论断边界失配”。

### 2.3 目前为什么不能直接开始训练

上述问题在逻辑上可能存在，但项目当前还没有足够真人数据证明：

- 它在真实系统文章中出现得足够频繁；
- 它在多个主题上重复出现，而非个别案例；
- 现有规则无法解决；
- 通用 API 自检无法以合理成本解决；
- 人类根据现有证据包可以稳定判断；
- 错误类型足够集中，适合训练一个专用模型。

如果没有这些证据，直接训练模型容易形成循环论证：先定义一种错误，再合成大量这种错误，最后证明模型能识别自己定义的错误。这不能证明真实系统需要该模型。

## 3. 原计划训练的模型

### 3.1 模型定位

模型暂定名为 **CiteCalibrator-8B**。它不是文章生成模型，也不是通用事实核查模型，而是一个“图程序与来源证据驱动的原子论断校准器”。

模型处于文章初稿之后：

```text
现有 Agent 与图程序
  ↓
通用 API 生成初稿
  ↓
抽取高风险原子论断
  ↓
CiteCalibrator-8B
  ├─ supported：保留
  ├─ qualify：补充限制或降低表述强度
  ├─ reject：删除或重写
  └─ abstain：交给人工或更强 API
  ↓
只修改受影响的句子或段落
```

### 3.2 模型输入

每次只校验一个原子论断。输入包括：

- 用户问题；
- 待校验论断及局部上下文；
- 图中节点、边和指标的语义；
- 图程序、参数和执行结果；
- 对应来源与 evidence ID；
- 阈值、时间窗、随机种子等扰动结果；
- 当前算子允许和禁止的解释方式。

### 3.3 模型输出

模型输出受约束 JSON：

```json
{
  "decision": "supported | qualify | reject | abstain",
  "risk_types": ["causal_overclaim"],
  "unsupported_spans": ["驱动", "核心机制"],
  "evidence_ids": ["E1", "E2"],
  "revised_claim": "保留正确内容的最小修订",
  "required_limitation": "必须说明的语料、参数或解释边界",
  "confidence": 0.86
}
```

### 3.4 为什么可能值得训练

2–4 张 RTX 4090 无法训练一个在通用写作上超过前沿 API 的模型，但足以使用 QLoRA 微调 8B 模型。该模型的潜在优势不是参数量，而是能够使用本项目独有的监督：

- 可执行的图程序结果；
- 可回放的数值和路径；
- 来源链；
- 参数扰动；
- 明确的解释契约。

如果前置实验成立，专用模型可能比通用 API 自检更稳定、更便宜，并且更擅长发现范围、因果和参数稳定性问题。

## 4. 当前已有证据与缺口

### 4.1 已有信号

现有开发实验已经观察到一些值得继续调查的信号：

- graph 文章相对 flat 文章出现了更多解释风险标记；
- 机器文章的校准性措辞多于人类参考，但数值事实密度更低；
- 现有系统能够产出带 PH/REF 的可审核 claim；
- matched compiler 开发稿中已有 4 篇达到 claim-review readiness，每篇包含 23–38 条候选 claim。

这些结果说明“解释冲动与证据边界”值得评测，但不能证明存在稳定、严重的真实错误。

### 4.2 关键缺口

当前正式文章审核协议计划 8 个主题、每主题 20 条 claim、共 160 条 claim 和 320 次首审，但现状仍是：

- 正式有效机器稿不足；
- 真实审核者为 0；
- 真人审核结果为 0；
- 当前不能作任何发生率或有效性结论。

此外，项目中的平衡 claim 面板是为了比较实验条件而构造的固定 case 谱，不代表自然部署类别率。人为平衡的 `qualify/reject` 数据不能用于回答“当前系统实际有多少问题”。

### 4.3 可以复用的资产

- `src/citeweave/article_claim_review.py`：claim 抽取和审核包；
- `src/citeweave/review_ui.py`：盲化审核界面；
- `src/citeweave/article_review_revision.py`：依赖限定的局部修订；
- `src/citeweave/operator_verification.py`：程序结果校验；
- `src/citeweave/graph_robustness.py`：参数扰动；
- `experiments/article_quality_v2/article_claim_review_protocol.yml`：双审和第三人裁决协议；
- `scripts/prepare_article_claim_review.py`：审核包准备入口；
- `scripts/finalize_article_claim_review.py`：审核结果合并入口。

复用上述实现时，不得静默修改 frozen v2/v3 工件、哈希或原有结论。新的诊断实验应使用独立目录、独立协议 ID 和独立 manifest。

## 5. 前置研究的核心问题

前置实验依次回答四个问题。

### Q1：问题是否真实存在

在当前准备部署的系统输出中，有多少原子论断存在：

- 事实或数值错误；
- 引用不支持；
- 范围外推；
- 因果夸大；
- 参数敏感性遗漏；
- 图概念实体化；
- 证据不足。

### Q2：现有方案能否解决

对相同 claim 和证据包，比较：

- 确定性规则；
- 生成文章的同一 API 自检；
- 独立前沿 API 校验；
- 规则 + API。

### Q3：该问题是否可由当前输入解决

给人类审核者与未来模型完全相同的证据包。如果合格审核者仍无法一致判断，则应先补充 evidence packet，而不是训练模型。

### Q4：剩余问题是否值得专门训练

需要同时考虑：

- 错误频率和严重程度；
- 现有基线的剩余错误；
- 错误是否跨主题重复；
- 可归纳的数据规模；
- API 自检成本和延迟；
- 小模型部署的潜在节省。

## 6. 两阶段 Benchmark 设计

### 6.1 Discovery Bench

用途：

- 估计真实错误率；
- 发现主要错误类型；
- 测试标注规范和 evidence packet；
- 比较非训练基线；
- 作出 Go / Extend / No-Go 决策。

Discovery Bench 可以被研究者查看和分析，因此以后只能作为开发材料、训练候选或补充报告，不能作为最终无偏测试集。

### 6.2 Frozen Confirmation Bench

只有 Discovery 阶段通过训练门禁后才构造。它使用全新的主题、问题、文章和人工标签，用于：

- 独立复核问题确实存在；
- 评测训练模型；
- 与规则、API 自检进行最终对比；
- 支撑论文结论。

Frozen Confirmation Bench 在模型结构、标签体系、主要指标和阈值确定后再冻结。测试标签不得用于改提示词、选检查点或调整路由阈值。

## 7. Discovery Bench 的数据来源

### 7.1 目标总体

Bench 要代表的是“当前最佳、准备部署的 CiteWeave 流程自然生成的文章”，而不是人为制造的错误句子。

主发生率分析只纳入：

- 正常执行的真实研究问题；
- 通过机器结构、长度、证据 ID 和完成度门禁的文章；
- 未经人工修改的原始草稿；
- Results 和 Discussion 中真实出现的 evidence-bearing claims。

以下内容不进入主发生率分母：

- 为凑类别比例而合成的错误；
- 已知失败且被截断的文章；
- 研究者挑选的典型坏案例；
- 经人工修订后的文章；
- 已用于确定标签规则的示例。

失败文章可以单独报告工程失败率，但不能与论断语义错误混为一个指标。如果当前系统无法稳定生成满足机器门禁的文章，应先解决生成问题；CiteCalibrator 不能修复截断、章节缺失或非法证据 ID。

### 7.2 MVP 规模

建议第一轮：

| 项目 | 最低规模 |
|---|---:|
| 主题 | 6–8 个 |
| 每主题真实研究问题 | 5 个 |
| 合格文章 | 30–40 篇 |
| 每篇 evidence-bearing claims | 15–20 条 |
| 主要审核 claims | 450–800 条 |
| 每条首审 | 2 人 |
| 分歧裁决 | 1 名独立第三人 |

主题应覆盖至少四个学科大类。主题是主要独立复制单位，不得把数百条同主题 claim 当作数百个独立实验样本。

### 7.3 文章与 claim 抽样

为了估计自然发生率，不能只抽“看起来危险”的句子。建议：

1. 对 Results 和 Discussion 中所有带 PH/REF 或其他证据标识的句子建立完整清单；
2. 将复合句拆成原子论断，但保留原句和段落 ID；
3. 每篇不超过上限时全量审核；
4. 超过上限时按预先固定的随机种子分层抽样；
5. 保存抽样概率，在总体发生率估计中使用权重；
6. 每篇随机抽取 5 条未被“高风险筛选器”选中的 evidence-bearing claims，估计筛选器漏检。

如果只审核高风险句子，可以估计“高风险池中的错误率”，但不能声称它是整篇文章的错误率。两种分母必须分开报告。

### 7.4 不要使用平衡采样估计发生率

为了训练分类器，可以将 supported、qualify、reject、abstain 调成接近平衡；为了估计当前系统自然错误率，不可以这样做。

Discovery Bench 应保留自然类别分布。若为了分析罕见错误另做过采样，必须：

- 单独标记为 challenge subset；
- 保存原始抽样概率；
- 不把 challenge subset 的类别比例报告为部署发生率。

## 8. Benchmark 样本结构

评测单位是一个原子 ClaimCase：

```json
{
  "case_id": "discovery_case_0001",
  "topic_id": "topic_x",
  "task_id": "task_x_03",
  "article_id": "article_x_03",
  "article_condition": "hidden_from_reviewer",
  "section": "Discussion",
  "paragraph_id": "p12",
  "original_sentence": "原始系统句子",
  "atomic_claim": "拆分后的单一论断",
  "claim_extraction": {
    "method": "rule_or_model_version",
    "source_span": [120, 156],
    "sampling_probability": 1.0
  },
  "question": "原始研究问题",
  "local_context": "前后必要上下文",
  "graph_scope": {
    "node_semantics": "节点含义",
    "edge_semantics": "边含义",
    "corpus_scope": "语料范围",
    "time_range": [2015, 2025]
  },
  "operator_trace": {
    "program": [],
    "parameters": {},
    "execution_result": {}
  },
  "evidence": [],
  "perturbation_profile": {},
  "review": {
    "factual_supported": null,
    "interpretation_calibrated": null,
    "alternative_adequate": null,
    "evidence_sufficient": null,
    "action": null,
    "risk_types": [],
    "severity": null,
    "decisive_evidence_ids": [],
    "invalid_dependency_ids": [],
    "replacement": null
  },
  "lineage": {
    "generator_commit": "git sha",
    "article_sha256": "sha256",
    "packet_sha256": "sha256"
  }
}
```

审核者界面不得显示文章条件、自动分数、预期标签或其他审核者答案。

## 9. 人工标注规范

### 9.1 一级判断

优先复用现有 article claim review 协议中的四项判断：

- `factual_supported`：事实和数值是否得到支持；
- `interpretation_calibrated`：解释强度是否与证据匹配；
- `alternative_adequate`：是否充分说明合理替代解释；
- `evidence_sufficient`：当前证据是否足以作出判断。

### 9.2 最终动作

- `accept`：原论断可原样保留；
- `rewrite`：核心内容可保留，但必须最小修订；
- `reject_claim`：核心论断不能保留；
- `abstain`：证据不足或冲突，需要额外信息。

映射到未来模型标签：

| 人工动作 | 模型标签 |
|---|---|
| accept | supported |
| rewrite | qualify |
| reject_claim | reject |
| abstain | abstain |

### 9.3 风险类型

建议记录：

```text
numeric_error
entity_error
path_error
provenance_error
missing_evidence
scope_overclaim
causal_overclaim
semantic_reification
parameter_sensitivity_omitted
temporal_instability_omitted
community_instability_omitted
graph_construction_omitted
corpus_scope_omitted
unsupported_domain_mechanism
conflicting_evidence
unanswerable
```

### 9.4 严重度

- `minor`：不影响主要结论的局部不精确；
- `major`：会改变读者对结果范围、可靠性或机制的理解；
- `critical`：核心事实、来源或主结论错误。

Go / No-Go 判断应优先使用 major + critical，而不是把标点或措辞偏好计入问题率。

### 9.5 标注流程

1. 用 50–100 条开发样本培训审核者；
2. 冻结 rubric 和正反例；
3. 两人独立首审；
4. 任一关键字段不一致时交给第三人盲裁；
5. 提交后不可修改原始判断；
6. 保存审核时间、分歧率和证据访问记录；
7. 以 topic 为聚类单位进行统计。

目标一致性：主要动作的 Cohen's kappa 或 Krippendorff's alpha ≥ 0.70。若低于 0.60，必须暂停正式标注，修改证据包或 rubric。

## 10. 前置对比实验

对每个 adjudicated claim，用相同证据包评测以下方案：

| 编号 | 条件 | 目的 |
|---|---|---|
| B0 | 不校验 | 估计当前原始系统问题率 |
| B1 | 确定性规则 | 判断简单规则可以解决多少 |
| B2 | 生成文章的同一 API 自检 | 判断生成者自检能力与锚定效应 |
| B3 | 独立前沿 API 校验 | 判断强通用模型的现实上限 |
| B4 | 规则 + 独立 API | 形成训练前最强基线 |
| H | 合格人类 + 相同证据包 | 判断当前输入下任务是否可解 |

所有基线必须：

- 使用相同 claim 和相同 evidence packet；
- 使用固定提示词和输出 Schema；
- 在看正式标签前冻结配置；
- 记录 token、延迟、费用和解析失败；
- 不允许为单个测试主题调整提示词；
- API 自检不得看到人工标签或未来模型输出。

B0 的错误率来自人工对原文的裁决，不是让 B0 自己输出判断。

## 11. 主要指标

### 11.1 问题发生率

- actionable defect rate：`rewrite + reject_claim` 占全部自然抽样 claim 的比例；
- major/critical defect rate；
- 每 1,000 字 major/critical 缺陷数；
- 至少有一个 major/critical 缺陷的文章比例；
- 各风险类型发生率；
- 各主题发生率及主题间差异；
- 高风险筛选器召回率。

发生率同时报告点估计和按主题 cluster bootstrap 的 95% 置信区间。

### 11.2 非训练基线效果

- 四分类 macro-F1；
- major/critical 缺陷召回率；
- accept claim 的误拒率；
- 修订后仍有缺陷的比例；
- 修订引入新错误的比例；
- abstain 率；
- 每纠正一个 major/critical 缺陷的成本；
- 每篇文章延迟和 API token。

### 11.3 可解性与数据质量

- 人工首审一致性；
- 第三人裁决比例；
- `evidence_sufficient=true` 比例；
- evidence ID 可解析率；
- 程序回放成功率；
- 需要额外领域知识才能判断的比例；
- 能归入预定义风险类型的缺陷比例。

## 12. Go / Extend / No-Go 门禁

所有阈值应在查看 Discovery 正式标签前冻结。下列数值是建议初稿，可以在 pilot 后、正式 Discovery 标注前调整一次并记录理由。

### 12.1 Gate A：问题达到实质规模

满足以下任一主条件，并且问题出现在至少 4 个主题中：

- 自然抽样 claim 中 major/critical 的 `rewrite + reject` 比例 ≥ 8%；
- 至少 30% 的合格文章包含一个或以上 major/critical 缺陷；
- 平均每篇文章 ≥ 1 个 major/critical 缺陷。

如果比例为 3%–8%，进入 Extend：扩大主题和文章数后重新估计。低于 3%，且受影响文章少于 20%，原则上 No-Go。

### 12.2 Gate B：现有方案仍有明显缺口

最强非训练基线 B4 至少满足一个问题条件：

- 对 major/critical 缺陷的召回率 < 0.80；
- 修订后仍残留 ≥ 3% 的 major/critical claim；
- accept claim 误拒率 > 0.10；
- 虽然质量足够，但校验 API token、延迟或费用不满足部署要求。

如果 B4 已达到高召回、低误拒和可接受成本，不应仅为了增加“模型创新点”训练 CiteCalibrator。

### 12.3 Gate C：任务可以从现有证据中学习

必须同时满足：

- evidence packet 可解析率 ≥ 99%；
- 程序回放成功率 ≥ 99%；
- `evidence_sufficient=true` ≥ 70%；
- 主要动作标注一致性 ≥ 0.70；
- 至少 60% 的 major/critical 缺陷属于计划中的风险类型；
- 缺陷不能主要来自截断、检索完全失败或实体身份错误等上游故障。

若 Gate C 不通过，应先修改系统输入、图构建或审核协议，不能直接用更多合成数据掩盖问题。

### 12.4 Gate D：训练具有实际价值

至少满足一项：

- 现有规则和 API 在某些重复风险类型上有稳定性能缺口；
- 用本地 8B 替代大部分 API 校验预计可减少 ≥ 50% 的校验 API token；
- API 输出方差或 JSON 失败影响自动化；
- API 自检存在明显的同源锚定错误，而独立判断可以纠正；
- 系统需要离线、隐私或固定版本推理。

### 12.5 总决策

| 结果 | 动作 |
|---|---|
| A、B、C、D 全部通过 | Go：开始训练数据构造和 8B MVP |
| A 接近阈值，其他通过 | Extend：扩大 Discovery Bench，不训练 |
| A 不通过 | No-Go：问题规模不足，不训练 |
| B 不通过 | No-Go：规则/API 已足够，不训练 |
| C 不通过 | Repair：先修 evidence packet 或上游系统 |
| D 不通过 | No-Go：模型没有实际部署价值 |

禁止通过改变分母、只报告 challenge subset 或事后降低阈值，把 No-Go 改写成 Go。

## 13. 如果通过门禁，后续训练数据怎么来

通过 Go 门禁后，Discovery Bench 的作用是确定真实错误谱和标签规范，而不是直接充当最终测试集。

训练数据按以下方式扩展：

1. 从可回放图程序生成 canonical supported claims；
2. 对阈值、时间窗、删边和社区随机性执行扰动；
3. 根据真实 Bench 中高频错误类型生成单点受控变异；
4. API 只负责语言改写，不负责决定金标签；
5. 人工重点审核 `qualify/reject` 边界、复杂语义和模型不确定样本；
6. 按主题划分 Train/Dev/Test；
7. 另建从未参与设计的 Frozen Confirmation Bench。

训练规模和 QLoRA 配置见 `docs/CITECALIBRATOR_TRAINING_PLAN_ZH.md`。

## 14. 建议的目录与工件

不要写入现有 frozen formal 目录。建议新增：

```text
experiments/citecalibrator_need_assessment_v1/
  protocol.yml
  topic_manifest.json
  task_manifest.json
  article_manifest.json
  claim_sampling_manifest.json
  evidence_packets/
  reviewer_roster.json
  assignments/
  adjudication/
  baselines/
  analysis/
  freeze_manifest.json
```

最低交付物：

- 结果前冻结的协议；
- 主题、问题、文章和 claim 抽样 manifest；
- 每个工件的 SHA256；
- 审核者资格和冲突声明；
- 双审与裁决结果；
- B1–B4 的逐 claim 原始输出；
- 发生率、基线能力和成本分析；
- 明确的 Go / Extend / No-Go 决策报告；
- 未通过门禁时的停止记录。

## 15. 推荐执行顺序

### 阶段 0：协议和可行性 pilot

1. 从现有 claim-ready 开发稿中选 50–100 条，仅用于修订 rubric 和界面；
2. 检查 evidence packet 是否足以支持真人判断；
3. 检查原子论断拆分质量；
4. 培训审核者并计算初始一致性；
5. 冻结正式 Discovery 协议和门槛。

现有 4 篇 matched compiler 开发稿已经被研究者查看，适合 pilot，不适合无偏发生率确认。

### 阶段 1：生成自然分布文章

1. 确定当前最佳可部署生成配置；
2. 选择 6–8 个主题、每主题 5 个真实问题；
3. 保存全部成功和失败记录；
4. 只有通过机器资格门的文章进入语义 claim 主分析；
5. 冻结文章和 claim sampling manifest。

### 阶段 2：真人双审和裁决

1. 生成匿名 evidence packet；
2. 每条 claim 分配两名独立审核者；
3. 冲突进入预注册第三人裁决；
4. 合并最终金标；
5. 在揭盲前完成完整性审计。

### 阶段 3：非训练基线

1. 冻结 Rules、Self-check API、Independent API、Rules + API；
2. 对相同 claim packet 执行；
3. 记录准确率、误拒、成本和延迟；
4. 不在正式结果上迭代提示词。

### 阶段 4：作出决策

按 Gate A–D 逐项填写证据，不使用“整体看起来不错”代替门禁。如果决定 Go，再申请云端 GPU 和正式训练数据预算。

## 16. 交接验收清单

执行者开始前应确认：

- [ ] 理解当前任务是“验证是否需要模型”，不是“证明模型一定有用”；
- [ ] 没有使用平衡面板估计自然错误率；
- [ ] pilot 数据与正式 Discovery 数据隔离；
- [ ] 发生率分母和抽样概率已冻结；
- [ ] 文章必须来自当前真实系统自然输出；
- [ ] 机器失败和语义错误分开统计；
- [ ] 审核者看得到完整证据，但看不到实验条件；
- [ ] 每条正式 claim 双审，分歧独立裁决；
- [ ] 规则和 API 基线在正式标签前冻结；
- [ ] topic 而不是 claim 被视为主要独立复制单位；
- [ ] Go / Extend / No-Go 阈值在查看结果前冻结；
- [ ] No-Go 是允许且有价值的研究结果。

## 17. 最终结论

CiteCalibrator 当前只是一个待验证的模型假设。最先需要完成的研究贡献不是训练模型，而是建立一个能够回答“问题是否真实存在”的自然分布 Discovery Bench。

只有同时证明以下四点，训练才合理：

1. 当前系统在多个主题上产生了非偶发、实质性的论断支持或解释校准问题；
2. 规则和通用 API 自检仍有明显质量、成本或稳定性缺口；
3. 人类能够根据系统已有证据包稳定判断这些问题；
4. 问题类型足够集中，可以被程序化构造为高质量训练监督。

如果任一关键门禁不成立，应停止 CiteCalibrator 训练，把资源用于更真实的上游瓶颈。这样可以避免为了增加一个模型创新点而制造不必要的模型。
