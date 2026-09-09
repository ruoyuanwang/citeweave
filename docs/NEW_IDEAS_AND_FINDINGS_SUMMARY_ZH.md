# CiteWeave v2：新思路与阶段性发现总结

**更新时间：** 2026-08-20

**文档用途：** 组会讨论、实验规划与论文主线梳理
**当前状态：** 已完成机制设计和真实探索性实验，尚未完成正式的人类实验与全部锁定主题评估

---

## 1. 核心结论

本轮工作的最大变化，不是简单增强现有 Graph RAG，而是重新定义了论文的技术主线：

> **将“图作为额外上下文”升级为“可执行的 Graph Program RAG”，将“保存人类反馈”升级为“可迁移、可验证的人类审核编译器”。**

旧实验已经证明 grounding 有效，但没有证明图结构本身有效。新的实验进一步说明：真正产生优势的是**全图计算、查询规划、反事实分析和可验证的派生事实**，而不是把相同信息序列化成图 JSON。

在人类审核方面，旧系统只有审核记录，没有可靠的反馈学习证据。新的设计要求每次审核都生成可复用规则、依赖传播结果和回归测试，从而能够测量反馈是否真正迁移、是否减少人工成本，以及是否引入新的安全风险。

建议论文主线为：

> **Human-Guided Graph Program RAG for Auditable Scientific Landscape Synthesis**

---

## 2. 上一轮实验暴露出的核心问题

### 2.1 图的作用过小

上一轮 Graph RAG 实验主要包含以下任务：

- 节点数查询；
- 最大节点查询；
- 最强边查询；
- 聚类归属查询；
- 错误前提识别。

这些任务大多属于单跳查表问题。只要扁平证据中包含相同事实，模型就不需要真正利用图结构。

正式实验结果也验证了这一点：

| 比较 | 主要结果 | 解释 |
|---|---:|---|
| Graph RAG vs. 无参考 | Graph RAG 明显更好 | 证明外部 grounding 有价值 |
| Graph RAG vs. 扁平结构化证据 | 没有优势 | 未证明图结构本身有贡献 |
| Graph RAG vs. Figure/VLM | 35 项全部打平 | 简单视觉查表无法体现图推理能力 |

因此，旧实验最多支持“有证据比无证据好”，不能支持“Graph RAG 比普通 RAG 更好”。

### 2.2 人类审核机制过于简单

旧审核机制主要是：

1. 保存审核记录；
2. 根据历史记录调整后续审核请求；
3. 将历史经验作为上下文再次使用。

对现有审核记忆的审计发现：

- 共 32 条记录，覆盖 8 个主题；
- 只有一个审核者身份：`CODEX-HUMAN-PROXY`；
- 所有 `review_seconds` 都为 0；
- 可形成 4 条候选规则；
- 满足独立验证条件的可启用规则为 0。

这意味着旧实验不能证明：

- 系统学习了真实人类偏好；
- 反馈能够跨主题迁移；
- 审核减少了真实人力；
- 自动放行策略在新数据上仍然安全。

### 2.3 文章偏向“结构完整”，缺少“科学发现”

旧机器文章的优势主要是：

- 结构规范；
- 方法透明；
- 证据密度高；
- provenance 清晰；
- 易于审计。

与优秀人类文章相比，主要差距是：

- 图表分析停留在描述层；
- 缺少复杂现象与结构机制；
- 缺少多张图之间的综合；
- 缺少反事实和替代解释；
- 与具体代表文献、领域机制和现实问题连接不足；
- 容易从描述性模式跳到“成熟”“驱动”“热点”等阶段性或因果判断。

---

## 3. 新思路一：Scale-aware Graph Program RAG

### 3.1 从“检索图数据”转向“执行图程序”

新的处理流程是：

```text
问题分解
  → 全图算子执行
  → 预算内连通证据检索
  → 反事实与稳健性检查
  → 现象卡生成
  → 跨章节科学论证
```

系统不再要求 LLM 在上下文中手算大规模图，而是执行注册过的图算子，例如：

- `shortest_path`
- `k_hop_expand`
- `community_aggregate`
- `boundary_edge_filter`
- `edge_betweenness`
- `weighted_degree_argmax`
- `delete_edge_counterfactual`
- `delete_node`
- `threshold_sensitivity`
- `community_stability`
- `cross_layer_join`
- `temporal_shift`

每个算子输出带 provenance 的派生事实。LLM 负责选择、综合、解释和校准，而不是直接在大量原始节点和边上进行不可靠计算。

### 3.2 同时扩大图规模和任务复杂度

Graph RAG 的优势需要在图无法被完整扁平化、问题必须依赖关系计算时才能显现。

#### 图规模设计

| 层级 | 目标规模 | 实验作用 |
|---|---:|---|
| Small | 约 100 节点 | 与旧任务衔接，作为机制负对照 |
| Medium | 约 300–1,000 节点 | 图开始难以完整放入上下文 |
| Large | 完整 canonical graph | 测试真实全局结构和规模鲁棒性 |

当前真实关键词网络已达到：

| 层级 | 节点数 | 边数 |
|---|---:|---:|
| Small | 100 | 4,235 |
| Medium | 300 | 22,427 |
| Large | 640 个非孤立节点 | 50,118 |

#### 任务复杂度设计

| 等级 | 任务类型 | 主要能力 | 预期图优势 |
|---|---|---|---|
| C1 | 单节点、单边查询 | 事实定位 | 不应明显领先，作为负对照 |
| C2 | 2–4 跳路径、共同邻居 | 局部拓扑推理 | 中等 |
| C3 | 社区角色、跨社区桥 | 全局聚合 | 强 |
| C4 | 删边、删点、阈值扰动 | 反事实与稳健性 | 强 |
| C5 | 跨网络、跨时间综合 | 多图程序与科学叙事 | 最强 |

每道题需要同时保存：

- 结构化答案；
- 全部合法答案集合；
- 证据路径；
- 图算子执行轨迹；
- 替代解释；
- 禁止推断；
- 必要局限。

---

## 4. Graph Program 的阶段性实验发现

> 本节结果属于冻结 prompt 下的跨主题探索性实验。主题和样本数量仍不足以支撑最终论文结论。

### 4.1 复杂任务能够显著拉开 Graph Program 与普通检索的差距

在 5 个主题、24 个复杂现象问题上：

| 条件 | 准确率 |
|---|---:|
| `graph_program` | 100.0% |
| `flat_tfidf` | 12.5% |
| `graph_retrieval` | 37.5% |
| `flat_program` | 100.0% |

主要比较：

- Graph Program 相对 TF-IDF：准确率差 **+87.5 个百分点**；
- Graph Program 相对普通图检索：准确率差 **+62.5 个百分点**；
- Graph Program 相对同信息 Flat Program：准确率差 **0**。

BM25 强文本基线实验得到相似结果：

| 条件 | 准确率 |
|---|---:|
| `graph_program` | 100.0% |
| `flat_bm25` | 12.5% |

当前结果说明：复杂图任务确实可以体现图计算相对词法 RAG 的优势。

### 4.2 图格式仍然不是优势来源

`graph_program` 与 `flat_program` 获得完全相同的准确率，因为两者包含相同的：

- 节点与边；
- 图算子结果；
- 派生事实；
- 解释约束。

二者只在组织形式上不同。

因此目前最稳妥的结论是：

> **优势来自图程序和结构计算，不来自图 JSON 或连通子图的序列化形式。**

论文不应声称“图表示天然优于扁平表示”，而应强调：

- 查询规划；
- 全图算子；
- 反事实计算；
- provenance 回链；
- 面向科学现象的结构化综合。

### 4.3 算子输出贡献了大部分能力，原始图可能主要用于验证

机制消融结果：

| 条件 | 准确率 |
|---|---:|
| Graph Program | 100.0% |
| Operator Only | 79.2% |
| Oracle Graph Retrieval | 33.3% |
| Query Graph Retrieval | 25.0% |

这表明：

1. 正确的图算子结果承担了大部分答案能力；
2. 仅检索相关子图不足以稳定完成全局和反事实任务；
3. 原始图与证据路径可能主要贡献验证、纠错和 provenance，而不只是答案信息；
4. 后续需要通过故障注入实验检验模型能否利用原始图识别错误算子结果。

### 4.4 图优势可能随规模增加，但当前证据不足

配对规模 pilot 中：

- Small：Graph Program 与强基线打平；
- Medium：Graph Program 与强基线打平；
- Large：Graph Program 开始出现优势。

但当前每档只有极少样本，尚不能证明稳定的规模交互。

正式实验需要检验：

```text
outcome ~ condition × graph_scale × complexity
        + context_tokens
        + (1 | topic)
        + (1 | item)
```

最重要的研究假设应当是：

> **Graph Program 的优势随图规模和任务复杂度共同增加，而在 C1 简单任务上不应明显领先。**

---

## 5. 新思路二：从图注升级为 PhenomenonCard

最终文章生成单位不再是“一段图解释”，而是复杂现象卡：

```text
PhenomenonCard
├── claim
├── supporting_graph_program
├── evidence_paths
├── counterfactual_result
├── alternative_explanation
├── scope_and_limitation
└── article_slots
```

每篇文章至少使用 4 张现象卡，并满足：

- 至少一张跨图卡；
- 至少一张时间变化卡；
- 至少一张反事实或稳健性卡；
- 图派生内容分布于 Results、Discussion 和 Conclusion；
- 每个复杂解释都能回溯到具体图算子和证据路径。

### 5.1 第一篇真实多现象文章

已在 `digital_twins_healthcare_2012_2024` 上完成第一篇 v2 文章。

文章包含 5 张现象卡：

1. 多跳连接；
2. 桥边删除反事实；
3. 社区角色对比；
4. 枢纽删除韧性；
5. 时间—结构变化。

确定性审计结果：

| 指标 | 结果 |
|---|---:|
| 文章长度 | 3,576 词 |
| 现象卡使用 | 5/5 |
| 正文代表文献 | 4 篇 |
| CARD 引用 | 59 次 |
| WORK 引用 | 17 次 |
| 可审计论证段落 | 26 段 |
| 合法 provenance 覆盖 | 24/26，92.3% |
| 结构与 provenance gate | 全部通过 |

单主题盲化 LLM 预审中，新稿相对旧机器稿的变化为：

| 维度 | 旧稿 | 新稿 |
|---|---:|---:|
| Phenomenon complexity | 3/5 | 5/5 |
| Cross-evidence synthesis | 3/5 | 5/5 |
| Graph-derived insight | 3/5 | 5/5 |
| Research value | 3/5 | 5/5 |
| Domain specificity | 3/5 | 4/5 |

### 5.2 尚未解决的文章质量问题

新稿仍有两个重要缺口：

1. `maturing field` 等阶段性判断仍可能超过结构证据；
2. 代表文献覆盖不足，尚未深入连接具体临床应用、领域机制和争议。

因此，目前只能认为 PhenomenonCard 显著改善了机器文章的结构性分析，不能据此声称超过人类专家文章。

---

## 6. 新思路三：Human Review Compiler

### 6.1 审核不再只是保存历史记录

新的审核机制要求每次人类操作至少产生一种可复用资产：

- 当前 claim 或 paragraph 的局部 patch；
- evidence 撤销、替换或强度调整；
- operator 参数或查询计划修订；
- 分析计划修订；
- 带上下文条件的 guard rule；
- 语义邻域回归用例。

处理流程变成：

```text
细粒度审核
  → 守卫规则编译
  → 证据依赖传播
  → 语义邻域验证
  → 无关任务回归测试
  → 主动审核调度
```

### 6.2 建立可传播的审核依赖图

系统维护以下依赖关系：

```text
source → evidence → operator → claim → paragraph
```

如果审核者否定上游证据或算子：

- 相关 claim 自动失效；
- 依赖该 claim 的段落进入重新审核；
- 系统记录影响范围；
- 修订前后版本可以回滚和比较。

这使人类反馈产生可测量的系统效应，而不是只影响当前文本或下一个 prompt。

### 6.3 规则启用必须经过多重验证

候选规则只有同时通过以下门槛才能进入自动策略：

#### 支持门

- 至少两个独立主题；
- 至少两个独立审核者；
- 相同 issue type 和 guard；
- 无冲突或冲突率低于预注册阈值。

#### 迁移门

- 在未用于规则形成的留出主题上验证；
- 语义邻域迁移精度不低于预注册门槛。

#### 安全门

- 无关样本的 regression rate 不高于预注册门槛；
- critical claim 无论是否命中规则都必须人审。

每次自动操作还必须保存：

- rule ID；
- 支持样本；
- 独立验证指标；
- 自动修改前后版本；
- 可撤销记录。

### 6.4 正式人类审核实验条件

建议比较四个条件：

| 条件 | 作用 |
|---|---|
| `always_review` | 全量审核质量与成本上限 |
| `static_risk` | 固定规则审核基线 |
| `raw_memory_prompt` | 旧式“历史记录作为上下文”机制 |
| `review_compiler_active` | 新规则编译、传播和主动选择机制 |

核心成功标准是：

> 在最终质量不低于 `always_review` 的前提下，显著减少审核时间，同时将 unsafe auto-accept 控制在预注册阈值以内。

### 6.5 需要记录的新指标

- Correction Lag；
- Post-feedback Accuracy；
- Rule Transfer Precision；
- Unrelated Regression；
- Unsafe Auto-accept Rate；
- risk–coverage / AURC；
- 审核秒数；
- 每分钟纠正错误数；
- reviewer override rate；
- Cohen's kappa 或 Krippendorff's alpha；
- adjudication rate。

---

## 7. 论文可以主张什么，暂时不能主张什么

### 7.1 当前可以支持的阶段性判断

1. 简单图解释任务不足以检验 Graph RAG 的独立价值；
2. 全图算子和反事实任务能够明显拉开 Graph Program 与普通词法检索的差距；
3. 图程序优于普通检索，但图序列化本身没有显示独立优势；
4. PhenomenonCard 能把图证据从孤立图注扩展到跨章节科学论证；
5. 旧审核记忆不满足独立审核者、时间测量和规则迁移验证要求；
6. 审核规则编译和依赖传播提供了比 raw memory 更强、可测量的研究机制。

### 7.2 当前不能支持的结论

1. 不能声称所有任务上 Graph RAG 都优于扁平 RAG；
2. 不能声称图 JSON 或连通图格式天然优于信息等价的扁平表示；
3. 不能根据当前小样本证明稳定的规模效应；
4. 不能声称系统已经从真实人类反馈中有效学习；
5. 不能声称已经减少真实人类审核劳动；
6. 不能根据单主题 LLM 预审声称文章质量超过人类专家。

---

## 8. 建议的论文创新点

### 创新点一：Human-Guided Graph Program RAG

在真实大规模文献计量图上执行：

- 多跳检索；
- 全局社区分析；
- 桥接结构识别；
- 节点与边反事实；
- 时间变化分析；
- 跨网络三角验证。

系统输出的不只是答案，而是带算子轨迹、证据路径、替代解释和局限的可审计科学现象。

### 创新点二：Human Review Compiler

将细粒度人类反馈编译成：

- 有适用范围的可执行规则；
- 可传播的证据依赖更新；
- 跨主题迁移验证；
- 无关任务回归测试；
- 成本敏感的主动审核策略。

论文需要证明它优于简单的 `raw_memory_prompt`，而不只是展示一个审核界面。

### 创新点三：从图问答到可审计长文综合

将验证过的图现象组织成完整文章，使图派生声明进入：

- Results；
- Discussion；
- Limitations；
- Conclusion。

这是本项目区别于一般 KGQA、GraphRAG 问答和单图解释工作的系统级价值。

---

## 9. 下一阶段最重要的实验

建议按以下顺序推进：

1. **扩充复杂任务族**：加入跨网络三角验证、阈值敏感性和 topology perturbation；
2. **完成规模交互实验**：在 Small、Medium、Large 上使用同构问题，检验 `condition × scale × complexity`；
3. **加强文本 RAG 基线**：加入严格 token budget、dense embedding、hybrid retrieval 和 RAPTOR 类方法；
4. **补强算子验证**：为派生事实生成独立 provenance ID，并进行 corrupted-operator 故障注入；
5. **增加锁定主题**：至少新增 4 个未用于开发的正式测试主题；
6. **开展真实双人审核**：至少两名独立审核者完成盲审，再对分歧进行裁决；
7. **比较四种审核策略**：Always、Static、Raw Memory、Review Compiler；
8. **开展领域专家文章评审**：覆盖多个主题，不再只依赖单主题 LLM Judge；
9. **统一记录成本**：token、延迟、API 成本、审核时间和每分钟纠错数；
10. **预注册统计方案**：明确主比较、效应量、混合效应模型、多重比较校正和失败处理规则。

---

## 10. 最终定位

本轮工作的真正成果，不是已经证明“Graph RAG 全面优越”，而是完成了三个关键转变：

1. 找到了旧实验无法体现图优势的根本原因；
2. 将研究对象从图格式升级为可执行、可验证的图程序；
3. 将人类审核从简单记忆升级为可编译、可迁移、可回归验证的反馈机制。

如果后续正式实验能够证明：

- Graph Program 的优势随规模和复杂度稳定增加；
- Review Compiler 在保持质量的同时显著降低真实审核成本；
- 多现象文章在多个主题上获得领域专家偏好；

那么这套工作就能够形成一篇结构完整、机制清楚、负结果诚实、实验链条完善的论文。

---

## 11. 相关本地产物

- 总体实验方案：`docs/PAPER_EXPERIMENT_V2.md`
- 旧实验总结：`docs/EXPERIMENT_RESULTS_SUMMARY_ZH.md`
- Complex Phenomena Panel：`experiments/graph_discovery_v2/complex_phenomena_panel_20260820/analysis.json`
- Mechanism Panel：`experiments/graph_discovery_v2/mechanism_panel_20260820/analysis.json`
- BM25 Panel：`experiments/graph_discovery_v2/bm25_panel_20260820/analysis.json`
- Scale Pilot：`experiments/graph_discovery_v2/paired_scale_panel_20260820/analysis.json`
- 旧审核记忆审计：`experiments/review_learning_v2/existing_memory_audit.json`
- 多现象文章：`experiments/phenomenon_article_v2/digital_twins_healthcare_2012_2024/article.md`
- 文章 provenance 审计：`experiments/phenomenon_article_v2/digital_twins_healthcare_2012_2024/verification.json`
