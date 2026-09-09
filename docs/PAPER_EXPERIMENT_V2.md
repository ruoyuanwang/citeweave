# CiteWeave v2：面向论文的图程序 RAG 与人类审核学习实验方案

**状态（2026-09-08更新）：** 正式GraphRAG主面板、复制、复杂度扩展和同计算量机制补充均已完成；复杂任务支持注册图计算和原始provenance价值，但规模放大为0，图形序列化本身无独立增益。新增40题连续难度稳健性综合基准已通过冻结与精确token审计，尚未运行。另有8个从未用于开发的确认主题完成58,358篇论文采集并全部通过自动数据门槛；结果前冻结的100条/主题盲法相关性双审与独立第三裁决链已可执行，正在等待真实领域审核者。文章与真人效应仍未形成可投稿结论。实时事实以`docs/CURRENT_FORMAL_EXPERIMENT_STATUS_ZH.md`为准，最新文献定位和可主张边界见`docs/PAPER_POSITIONING_GRAPH_RAG_AND_HUMAN_OVERSIGHT_20260907_ZH.md`。
**建议论文主线：** *Human-Guided Graph Program RAG for Auditable Scientific Landscape Synthesis*

## 1. 结论先行

上一轮实验不能支持“图结构本身有效”或“系统从人类审核中学习”这两个论文主张。

1. 图实验的题目主要是节点数、最大节点、最强边、所属簇和错误前提等单跳查表题。`flat_structured` 与 `graph_rag` 还被构造成完全相同的语义原子，因此图结构没有发挥空间。正式结果中，Graph RAG 相对无参考更好，但相对扁平结构的 UCR 差为 -0.006（方向不利于图，置信区间跨 0）。后续可重复范围审计进一步确认：旧 Figure/VLM 的46/46题都只是读取图副标题中的节点数和边数；同期Graph条件227题中181题没有视觉对应项，正式盲评只比较35个简单重叠题并全部打平。因此旧视觉结果是任务天花板，不是视觉与图结构能力等价的证据。
2. 旧审核记忆有 32 条记录和 8 个主题，但只有一个 `CODEX-HUMAN-PROXY`，所有 `review_seconds` 都为 0。组件级 credit audit 显示 32/32 信号全部落在 generator（14 个 pairwise rewrite、18 个 pointwise），retriever/planner/risk-router 均为 0；按跨数据集、跨审核者、无冲突守卫编译后，可启用规则仍为 0。旧结果只能证明流程可运行，不能证明减少真实人力或学习了系统级偏好。
3. 机器文章的优势是结构、可审计性、方法透明和证据密度；主要差距是领域具体性、图上复杂现象、与具体文献/机制的连接，以及从描述性指标到科学论证的深度。

v2 不再把“图 JSON”作为创新点。核心处理单元改为一个可执行图程序：

`问题分解 → 全图算子执行 → 预算内连通证据检索 → 反事实/稳健性检查 → 现象卡 → 文章论证`

人类审核也不再是把原始意见追加到 prompt，而是：

`细粒度审核 → 守卫规则编译 → 依赖图传播 → 语义邻域回归测试 → 主动审核调度`

## 2. 论文可以成立的两个创新点

### 2.1 Scale-aware Graph Program RAG

Graph RAG 的价值来自无法在固定上下文内扁平化的全局结构，而不是三元组的排版方式。系统需要同时处理：

- 大规模图：约 10²、10³、10⁴–10⁵ 条关系的规模层级；
- 多跳问题：2–4 跳证据链，并接受多个等价最短路径；
- 全局问题：社区角色、桥接结构、跨社区覆盖和结构集中度；
- 反事实问题：删边、删点、阈值扰动和社区参数扰动后的结论稳定性；
- 跨图问题：生产力、合作、关键词、引用和时间层之间的三角验证；
- 文章问题：把多个现象卡组合成有替代解释、有局限的段落，而非单独解释一张图。

这里的“program”不是让 LLM 猜一个图算法，而是由系统生成并执行注册算子，例如：

- `shortest_path`、`k_hop_expand`；
- `community_aggregate`、`boundary_edge_filter`；
- `edge_betweenness`、`weighted_degree_argmax`；
- `delete_edge_counterfactual`、`delete_node`；
- `threshold_sensitivity`、`community_stability`；
- `cross_layer_join`、`temporal_shift`。

每个算子输出带 provenance 的派生事实；LLM 的工作是选择、综合、解释和校准，而不是在上下文中手算 50,000 条边。

### 2.2 Human Review Compiler

每一次人审都必须产生至少一种可复用资产：

- 局部修订：当前 claim/paragraph 的 patch；
- 证据修订：撤销错误 evidence 或改变证据强度；
- 算子修订：参数、查询计划或检索范围的 correction；
- 计划修订：要求补做反事实、补充替代解释或降级结论；
- 守卫规则：仅在满足上下文特征时自动应用；
- 回归用例：相似问题必须改正，无关问题不得退化。

审核依赖图使用 `source → evidence → operator → claim → paragraph`。上游证据被否定后，下游声明和段落自动进入重新审核；这比“下次把历史记录放进上下文”有可测量的系统效应。

规则启用现在采用两道门，必须同时满足：

- **支持门：** 至少两个独立主题、两个独立审核者，相同 issue type/guard，无相反 action；
- **迁移门：** 至少两个留出主题的语义邻域样本，迁移精度不低于 0.8；
- **安全门：** 无关留出样本的回归率不高于 0.05；
- critical 声明无论是否命中规则都必须人审；
- 每次自动接受或自动修订必须保存 rule ID、支持样本、验证指标和可撤销版本。

这套门控已经成为可执行代码，而不是 prompt 约定。没有留出验证数据时，跨主题/跨审核者共识也只能生成 candidate rule，不能进入自动策略。

人机互补的正式统计也以主题而不是审核条目为重复单位。结果前修正案002规定至少8个独立主题：先分别计算各主题中 `active policy - raw memory` 的质量效应及每题节省时间，再在主题间等权；质量非劣（绝对正确率界值 -0.02）和劳动优势（每题节省时间大于0）分别做单侧精确 sign-flip，只有两者同时 `p<0.05` 才通过联合门。大量同主题条目只能提高主题内估计精度，不能伪装成更多独立重复；题级 McNemar 和汇总总秒数仅保留为描述性结果。

互补审核的2×2 factorial同样以8个主题为主推断单位，而不是把96个case或重复审核当成96个主题。每个主题至少12例且四个arm齐全；O1为two-sided packet的准确率主效应，O2为能力路由相对随机合格审核者的“正确结局/审核分钟”效应，O3为准确率交互。三者分别从主题内arm汇总得到，再做等主题权重的精确sign-flip与Holm校正。case/reviewer mixed model仍可报告，但只作为诊断性敏感度分析。

## 3. 新基准：同时扩大图规模和任务复杂度

### 3.1 三个规模层级

主实验每个主题选关键词、机构、作者三个网络，复制实验为关键词图。层级不是复制图片节点，而是从 canonical 语料构建的候选截断图；它不等于未截断的全语料实体图。

| 层级 | 目标规模 | 用途 |
|---|---:|---|
| Small | 约 100 节点 | 与旧可视化题衔接，作为负对照 |
| Medium | 正式冻结为 300 节点 | 图已不能可靠地全部放入 prompt |
| Large | 最多 2,000 候选节点的注册图 | 注册范围内全局检索，常见 10⁴–10⁵ 条边 |

开发集曾只有 640 个非孤立节点。v3 在任何正式模型调用前把候选池扩到 2,000，并在四个全新主题上得到：

| 网络/层级 | 节点范围 | 边范围 |
|---|---:|---:|
| Keyword Small | 100 | 4,519–4,910 |
| Keyword Medium | 300 | 23,437–33,730 |
| Keyword Large | 2,000 | 109,682–186,956 |
| Institution Large | 1,974–1,999 | 18,288–101,077 |
| Coauthor Large | 1,760–1,997 | 5,710–38,635 |

四个主题共含 79,968 篇规范论文。Large是对全量 occurrence 统计做确定性的2,000候选节点展开，准确名称为candidate-capped corpus graph。节点上限相对开发集扩大3.125倍，边规模最高18.7万；这些规模不能支持“遍历了全部原始实体”的表述，稀疏存储也不要求构造稠密邻接矩阵。

2026-08-27独立范围核算覆盖8主题、32个注册图。关键词Large保留原始实体的25.2%–47.5%，但保留89.3%–98.0%的共现pair-incidence mass；机构Large保留42.9%–75.2%的mass，作者Large仅3.3%–11.5%。分母由canonical的每篇唯一实体数计算`sum choose(k_work,2)`，分子为注册图边权总和；这不是未加权unique-edge recall，更不是答案准确率。原作者层随后查出placeholder身份污染，因此作者覆盖值仅描述修复前工件，不可当作有效科学图的覆盖结论。见`formal_v3_registered_graph_scope_audit.json`。

后续选择数据集时不能只看文献量，还要预注册全图节点/边规模和连通性，确保 scale manipulation 真正成立。

### 3.2 五级任务复杂度

| 等级 | 任务 | 主要能力 | 是否应体现图优势 |
|---|---|---|---|
| C1 | 单节点/单边查询 | 事实定位 | 否，作为负对照 |
| C2 | 2–4 跳路径、共同邻居 | 局部拓扑 | 是 |
| C3 | 社区角色、跨社区桥 | 全局聚合 | 强 |
| C4 | 删边/删点/阈值扰动 | 反事实和稳健性 | 强 |
| C5 | 跨网络、跨时间现象综合 | 多图程序与科学叙事 | 最强 |

每道题必须包含：结构化答案、全部合法答案集合、证据路径、算子轨迹、替代解释、禁止推断和必要局限。多个同长路径必须全部计为正确，不能使用单一任意 gold。

### 3.3 现象卡而非图注

最终生成单元为 `PhenomenonCard`：

```text
claim
supporting_graph_program
evidence_paths
counterfactual_result
alternative_explanation
scope_and_limitation
article_slots
```

每篇文章至少使用 4 张卡，并满足：

- 至少一张跨图卡；
- 至少一张时间变化卡；
- 至少一张反事实/稳健性卡；
- 图派生声明分布在 Results、Discussion 和 Conclusion，而不是只放在一个网络解释段。

### 3.4 连续难度稳健性综合开发集

既有复杂度扩展证明简单题不需要图程序、复杂题需要注册计算，但出现flat条件全错与Graph Program全对，无法继续区分更强方法。新的开发集把每个主题扩为1个基线图与17个阈值、删边、社区参数和时间窗扰动图，要求模型跨扰动聚合比例、处理不可比案例、分解枢纽删除前后的两类损失，并在五种非同质图算子之间识别冲突而不是写一段平均化图解。

当前8主题共40题、5条件、200格；32个rate gold中19个为内部取值，共13个不同值。`graph_program`与`flat_program`使用完全相同的20条以内原始证据和相同算子轨迹，`graph_hierarchical_retrieval_v2`与`flat_bm25`不见轨迹，`operator_only`不见原始证据。主指标为5–7个答案字段的字段准确率，并保留exact、evidence F1与必要局限。全部上下文通过8,192-token硬门，Graph Program最大7,586 tokens。由于任务设计使用了已经查看过扰动结局的8个主题，该面板只能校准难度与机制；确认性版本必须在至少8个新主题的扰动和gold生成前冻结。

## 4. Graph RAG 正式对照条件

在相同基础模型、温度、prompt、数据快照和输出 schema 下比较：

1. `no_reference`：无外部证据；
2. `flat_random`：哈希随机抽取的预算控制，只用于检验上下文存在性，不能作为主 baseline；
3. `flat_tfidf` / `flat_bm25`：带端点标签的强词法 top-k RAG；
4. `flat_lsa` / `flat_hybrid`：classic latent-dense 诊断与 BM25+TF-IDF+LSA RRF；不得称为神经 dense；
5. `flat_neural_dense`：外部预训练 embedding 的 query-independent index；它是正式执行前的硬性必备基线，当前正在按冻结的 BGE-M3 revision 物化，未全量完成和资格晋升前不得启动正式面板；
6. `graph_query_retrieval`：只使用问题中的显式实体锚点；实体对问题执行路径遍历，全局问题退化为显著节点检索；
7. `graph_oracle_support`（旧代码名 `graph_retrieval`）：保证包含 gold support 的连通子图，只作为证据充分时的上界，不得称为端到端检索器；
8. `operator_only`：只给算子轨迹，不给原始节点边，用于检验 provenance 回链价值；
9. `flat_program`：与 graph program 完全相同的节点、边、派生事实和解释约束，但序列化为无序表；
10. `graph_program`：连通子图 + 注册算子轨迹 + 派生事实；
11. `figure_vlm` / `hybrid_graph_figure`：只作为独立模态扩展；若未来执行，必须覆盖与结构条件相同的复杂任务并单独报告跨模型混杂，不进入当前同模型 GraphRAG 主检验。

不能再以随机 flat 作为主要证据。`graph_program vs flat_tfidf/BM25` 回答相对稀疏文本 RAG 的端到端效果，`graph_program vs graph_query_retrieval` 回答图计算的增量，`graph_program vs graph_oracle_support` 是证据充分上界，`graph_program vs operator_only` 回答原始图/provenance 回链价值，`graph_program vs flat_program` 才回答表示组织的独立贡献。旧开发结果中最后一个效应接近0；正式结果尚未产生，因此论文现在既不能声称“图形 JSON 本身优于扁平序列”，也不能声称已经证明等价。创新假设应落在查询规划、图算子执行、provenance 和审核依赖上。

结果前答案暴露审计显示，120个复杂题中，`operator_trace`和`operator_only`已有96题的全部答案字段可确定性派生，平均字段覆盖为0.96；完整`flat_program`与`graph_program`上下文均为120/120可派生。这是Graph Program把确定性计算前置的直接结果，不是LLM自行遍历大图。为把“计算内容”“序列化形式”和“原始图证据”拆开，另行冻结8主题×21题×3条件的504-cell同计算量机制补充：

- M1：`graph_program - flat_program`答案准确率做±0.05等价检验，回答图形序列化是否有独立价值；
- M2：`graph_program - operator_only`必须同时满足evidence F1优越与答案准确率非劣，回答原始图/provenance是否有独立价值；
- M3：桥边反事实这类轨迹不完整题，相比四类轨迹完整任务，是否对原始图表现出更大的答案增益。

补充共复用288格、新增216调用，所有检验以8个科学主题为独立重复并采用精确sign-flip。它是机制识别，不改变原主实验、复制、复杂度扩展的注册结论，也不能在C1/C2失败时充当补救分析。

必须记录并控制：

- 实际输入 token，而不只记录条数；
- 检索到的 gold evidence recall；
- 连通率和路径完整率；
- 派生算子是否泄漏最终自然语言答案；
- 模型调用 token、延迟和成本；
- 图规模和任务复杂度。

## 5. Graph RAG 指标和统计模型

### 5.1 指标

检索层：

- Evidence Recall@Budget；
- Path Completeness；
- Connected Evidence Ratio；
- Distractor Rate；
- Retrieval latency/cost。

推理层：

- structured answer accuracy（含等价答案）；
- operator execution accuracy；
- evidence-path validity；
- counterfactual consistency；
- abstention F1；
- scale robustness。

文章层：

- atomic claim precision/recall；
- graph-derived claim recall；
- cross-evidence synthesis；
- epistemic calibration；
- domain specificity；
- phenomenon-card utilization；
- expert pairwise preference。

### 5.2 主统计模型

主要模型使用混合效应 logistic/ordinal regression：

```text
outcome ~ condition * graph_scale * complexity
        + context_tokens + (1 | topic) + (1 | item)
```

预注册主检验：

- `graph_program > flat_retrieval` 在 C3–C5 上；
- `graph_program > graph_retrieval` 在 C4–C5 上；
- `graph_retrieval > flat_retrieval` 的效应随规模增大；
- C1 上 graph 不应有显著优势，作为机制负对照。

主题聚类 bootstrap 报告置信区间；多个主比较用 Holm 校正；同时报告绝对风险差、odds ratio 和成本差。不能只报告 p 值。

## 6. 人类审核实验

### 6.1 四个条件

1. `always_review`：全量审核上限；
2. `static_risk`：固定规则审核；
3. `raw_memory_prompt`：旧式历史记录上下文；
4. `review_compiler_active`：规则编译 + 依赖传播 + 主动选择。

这四组可以直接证明新机制是否超越“简单维护记录”。

### 6.2 审核单位和界面

审核者不直接面对整篇文章，而是看到：

- 一个原子 claim；
- 它依赖的 evidence path 和 graph operator；
- 原图的局部视图；
- 系统置信度与替代解释；
- accept / rewrite / reject evidence / reject operator / abstain；
- 可选 guard 和严重程度。

最后再做文章级连贯性审核。这样既能定位错误来源，也能测量局部反馈如何传播到全文。

当前已经实现可直接运行的本地双盲审核界面，而不是用研究者代填 JSON：

- reviewer 只持有一次性明文 token；服务端只保存 SHA-256 摘要；
- packet 接口不会暴露 condition、gold answer、自动评分或另一位审核者结果；
- factual 与 semantic 表单分离，分别记录证据充分性/答案正确性和解释、过度推断、修改动作；
- 审核时间由服务端根据可见页面 heartbeat 累计，不能由客户端自行填报；
- 每项提交原子写入并冻结，重复提交返回冲突，避免结果后改；
- 两名审核者的返回先独立校验，再计算一致性并生成裁决清单；只要存在分歧，反馈规则编译就保持锁定。

启动一次审核轮次：

```powershell
python scripts/prepare_review_access.py `
  --manifest experiments/human_review_v2/mechanism_panel_20260820/internal_manifest.json `
  --output experiments/human_review_v2/mechanism_panel_20260820/access_hashes.json
python scripts/serve_human_review.py `
  --packet-root experiments/human_review_v2/mechanism_panel_20260820 `
  --access experiments/human_review_v2/mechanism_panel_20260820/access_hashes.json
```

第一条命令只在终端显示一次明文 token，应分别线下交给两位审核者，不写入实验目录。审核结束后冻结统计：

```powershell
python scripts/finalize_human_review.py `
  --packet-root experiments/human_review_v2/mechanism_panel_20260820 `
  --output experiments/human_review_v2/mechanism_panel_20260820/finalized
```

`--allow-incomplete` 只供中期一致性诊断；正式统计必须使用完整返回。裁决不能用多数票静默覆盖，需由预注册的第三位裁决者生成独立 adjudicated artifact，之后才允许进入规则校准。

### 6.3 核心指标

- Correction Lag：一次反馈后多少个相似案例才稳定改正；
- Post-feedback Accuracy：语义邻域上的反馈后准确率；
- Unrelated Regression：无关任务退化率；
- Rule Transfer Precision；
- Unsafe Auto-accept Rate；
- risk–coverage/AURC；
- 审核秒数、每分钟纠正错误数；
- reviewer override rate；
- 两位审核者的 Krippendorff’s alpha 或 Cohen’s kappa；
- adjudication rate。

主要成功门槛：active compiler 相对 raw memory 的最终质量以2个百分点为非劣界且节省审核时间；同时报告相对 `always_review` 的质量—劳动差，并要求 unsafe auto-accept 的双侧95% Wilson 区间上端低于预注册阈值。

### 6.4 分层抽样和审核覆盖

全量双审所有 packet 会把实验变成人力堆叠，也无法体现主动审核的价值。新增的抽样协议按隐藏的 `condition × task_type × complexity × deterministic outcome` 平衡 factual 样本，semantic 样本不使用 outcome；deterministic outcome 只允许用于离线评估样本分层，严禁作为在线策略特征或向审核者显示。每个 stratum 记录总体数、唯一抽样数和 inverse-probability weight，允许恢复总体错误率而不是把平衡样本误当自然患病率。

两位审核者采用 partial-overlap：共同样本用于 Cohen’s kappa 和分歧裁决，其余样本互斥以提高覆盖面。机制面板的可执行轮次为每人 60 个 factual、40 个 semantic；factual 共同双审 30 个、唯一覆盖 90/96，semantic 因总体只有 57 个而自动把共同数提高到 23，唯一覆盖 57/57。自动提高 overlap 是由 `2n-N` 可行性约束决定，不是看结果后调整。审核包位于 `experiments/human_review_v2/mechanism_stratified_round_20260820`。

### 6.5 审核不是 memory，而是多粒度策略学习

借鉴 listwide feedback、document/list/response 多粒度对齐和 corrective retrieval evaluator，正式系统把一次审核拆成三个可归因的学习信号：

- evidence/document level：证据是否相关、充分、冲突，训练或校准检索/重排评分；
- plan/list level：路径、社区摘要和算子序列是否组成了正确分析计划，学习 listwise 排序与查询规划；
- claim/response level：最终声明是否正确、过度推断或需要降级，训练 risk/accept/review policy。

每个反馈先进入离线 replay：在语义邻域、无关邻域和关键声明集合上重放，测 transfer precision、unrelated regression 和 unsafe auto-accept。只有通过支持门、迁移门和安全门的版本才能 promotion；旧版本及完整决策轨迹保留，可一键回滚。在线主动选择只使用提交前可见的风险、分歧预测、结构新颖度和预估审核成本；隐藏 gold/outcome 只能用于离线分层评估，禁止策略泄漏。

因此论文的审核创新不应写成“把人类意见放进 RAG”，而应写成 `typed feedback → dependency-aware credit assignment → held-out replay → guarded policy promotion`，并用四条件消融直接验证每一层的增量。

### 6.6 四策略配对 replay 与 promotion gate

现已在真实人审开始前冻结 `review_policy_protocol.yml`（SHA-256 `6c613d4422cf81af7b78cca425ce3131f8208d8ec21f7728ee83c4c0e4c92ed5`）。至少180个 case 只由人类盲审一次；两位审核者 partial overlap，分歧由第三人裁决。之后同一份 adjudicated label 只在某策略已请求审核时才可见，按完全相同的 case 顺序 replay `always_review`、`static_risk`、`raw_memory_prompt` 和 `review_compiler_active`，从而把“人类标签质量”和“策略何时调用人类”分开。

`review_policy_experiment.py` 对每个 case 强制四条件完整配对，并验证 dataset、顺序、severity、原始正确性和 issue type 不变；route/risk/policy version 必须在揭示裁决标签前冻结。Critical case 永不自动化，自动 route 不能申报人工时间。分析同时输出数据集簇 bootstrap、McNemar、AURC、错误纠正/引入、post-feedback accuracy，以及“连续3个语义邻居正确”定义下的 correction lag 和右删失数。

主比较是 active compiler 对 raw memory：最终准确率以2个百分点为非劣界，同时必须节省服务端计时的人审秒数。Promotion 还需通过独立数据集、全部 auto-accept 的 Wilson 上界、critical 零自动化和 unrelated regression 门。若零 unsafe error，要让双侧95% Wilson 区间上端低于5%，仍至少需要73个 auto-accepted audit case；没有足够安全暴露时必须保持 `hold`，不能因“没有自动接受高风险样本”而宣称安全。

### 6.7 能力互补路由与对抗式双面审核

原协议仍把合格审核者近似视为可互换资源，也没有正面检验 claim-first 呈现造成的锚定。为此在任何真实审核结果产生前冻结补充协议 `review_policy_amendment_001_complementary_oversight.yml`（SHA-256 `7fb864caf7b813f9a890365ebabd08234b9d7b51103cd9ca9401e528fb9b4219`）。审核者能力不是自报标签，而是在排除出正式分析的热身题上，以 `reviewer × domain × issue type` 的层级 Beta 后验估计；证据不足时依次回退到领域和全局层。路由器联合选择 `auto_accept`、单审、双审加独立裁决或硬阻断，目标为最小化严重度加权的预期错误与服务端计时成本。Critical case 至少两名独立首审者，分歧必须由不同的第三人裁决；合格人数不足时禁止静默退化为自动接受。

审核材料把支持与反证随机映射为不知道角色的 evidence set A/B，并同时展示图算子轨迹、替代解释和禁止推断。审核者必须提交结论、决定性 evidence IDs、无效算子步骤、失败模式、最小可辩护改写和理由；服务端会拒绝不存在的证据 ID、重复或越界的算子步骤，并冻结带可见性 heartbeat 的服务器计时提交。这使反馈可以分别归因到 evidence、operator 和 claim，而不是只留下“赞成/反对”记录。

补充实验为 `packet（standard/adversarial） × assignment（qualified-random/capability-router）` 的 2×2 blocked crossover。至少96个 held-out 真实候选输出、6名审核者，每人12个已裁决热身题且覆盖至少2个领域；至少50%的正式 case 双审，同一审核者不得重复看到同一 case。O1检验双面材料是否提高裁决后正确率，O2检验能力路由是否提高每审核分钟正确率，O3检验二者交互。最终团队相对case内平均单次首审的跨主题增益另作C2检验；它只回答聚合与盲裁价值。广义“人机互补”还必须有同包、同输出schema的AI-only自审对照，不能把生成器没有输出的四分类决策机械补成弱基线。

正式96个任务及真实候选输出均已物化，但首次质量审计发现原面板金标为90个`qualify`、6个`reject`，其中4个主题没有类内变化；永远输出`qualify`即可达到93.75%。该面板保留为自然输出分布诊断，不承担干预效果检验。修正案008在0名真人和0条人类结果时保留相同96个任务，从真实、已完成且哈希绑定的候选输出中确定性选择每主题6个`qualify`与6个`reject`；生成条件和金标对审核者隐藏。由此得到48/48的固定平衡benchmark。由于候选条件选择读取了机器结果，随机化O1/O2/O3只识别这一固定case谱上的干预效应，不能估计自然部署类别率。当前真正阻塞项已收缩为6名真人、264条校准判断、能力冻结和四臂分配；程序和论文仍不得把结构就绪冒充人体实验结果。

### 6.8 来源相关性校准不是人工记忆

文章证据包的开发审计表明，仅靠泛化图节点或自动主题词无法证明一篇文献真正支持某个图现象。为此冻结独立的 source-relevance warmup 协议（SHA-256 `d34d70c7ba4934274955c716ddc8cf1d25ef7639b1c0e561f4b9dd1fb1fe45de`），把v4中40个现象×3篇来源=120个关系全部做成条件盲化的人审任务。审核者必须分别判断主题相关性与证据角色，复制精确决定性摘要片段，标注 generic-anchor/query-too-broad/off-topic 等失败类型，并建议检索词；来源的原选择依据、BM25 rank、命中关键词和其他审核者判断全部隐藏。

这些标签不会作为“审核历史”拼回提示词。结果前冻结的 source-retriever promotion gate 要求留一数据集 replay：direct/contextual 分级计入 NDCG@3，challenges 作为必须保留的反证而不是负样本，irrelevant 用于 off-topic 控制，cannot-assess 排除。候选校准器只有同时满足跨域 NDCG@3 至少提升0.02、正例召回不降、challenge recall 下降不超过0.02、off-topic 不升、无关查询回归率不超过5%，且不存在训练折泄漏时才能上线；缺反证或无关对照直接 `hold`。因此人类反馈改变的是带版本、可撤销的检索策略，而不是不可审计的上下文记忆。

这些标签只监督 evidence retriever/reranker：直接或背景相关且有精确片段的关系作为正例，无关或generic-anchor作为hard negative，`challenges` 单独保留为反证，绝不错误转换成负样本；建议词只能进入held-out replay，不能直接拼进后续prompt。120个pack的哈希、盲化和每主题15题审计全部通过。分配器要求6名真实审核者各30题、至少跨2领域、共180个判断，其中60/120题独立双审；每领域必须至少3名无冲突合格者以支持第三人裁决。当前roster为0人，readiness按预期blocked，所有warmup结果永久排除出正式2×2互补实验。

真人标签出现前的实现复核又冻结了promotion完整性修正案002。除原有留出泄漏和query内重复rank检查外，系统现在拒绝重复`query-reference`行、零/负/非整数rank，以及同一query混合dataset、held-out fold或training set；无效rank不参与指标算术且独立触发hold。该修正不改变gain、NDCG@3增益、正例/反证召回、off-topic或无关查询回归阈值，只防止格式或身份错误产生虚假的上线结论。

## 7. 样本量与数据划分

推荐正式设计：

- 2 个 prompt/算子开发主题；
- 2 个审核规则校准主题；
- 8 个完全锁定测试主题；
- 共 12 个主题，来自至少 4 个领域；
- 每个锁定主题 3 个网络 × 3 个规模 × 5 类复杂任务 × 2 个实例，约 90 项；
- 主测试约 720 项 × 6 条件；
- 人类双盲审核至少覆盖 180 个现象卡/claim，另加分层抽取的 auto-accept 审计。

当前 8 个主题可作为 v2 pilot。为了避免沿用旧 prompt 和选题偏差，正式 test 最好新增 4 个主题，并在任何新提示词开发前冻结。

## 8. 已完成的真实 pilot

### 8.1 多跳任务，规模三档

模型：`deepseek-v4-pro`，温度 0，thinking disabled。每个文本条件 160 条记录，flat 与 graph 的字符差约 4%。

| 条件 | Small | Medium | Large | 总准确率 |
|---|---:|---:|---:|---:|
| no_reference | 0/1 | 0/1 | 0/1 | 0/3 |
| flat_retrieval | 0/1 | 0/1 | 0/1 | 0/3 |
| graph_retrieval | 1/1 | 1/1 | 1/1 | 3/3 |
| graph_program | 1/1 | 1/1 | 1/1 | 3/3 |

早期评分只接受一条最短路径，曾把 Large 图中另一条合法 2-hop 路径误判为错；v2 scorer 已枚举等价答案。这一修复必须保留为 benchmark construction 的回归测试。

### 8.2 全局与反事实任务，Large 图

- Bridge deletion：graph retrieval 与 graph program 均正确；
- Community role contrast：只有 graph program 正确；
- Hub removal resilience：定义澄清为“最大连通分量占原始节点数”，只有 graph program 正确；
- Temporal–structural shift（时间聚合 + 拓扑）：只有 graph program 正确；
- Productivity–centrality rank divergence（生产力层 + 合作层）：只有 graph program 正确；
- 无参考与扁平检索在这些 pilot 项上均未给出 exact answer。

样本极小，不能报告显著性或泛化结论。其价值是验证任务能区分局部连通检索和全局图程序，并发现了预算装配、等价答案、指标定义三个会污染正式实验的缺陷。

### 8.3 第一轮冻结 prompt 的跨主题面板（弱 flat，仅作开发证据）

在一个 v2 开发主题完成实现修正后，冻结 `graph-discovery-v2-pilot3-20260820`，在 5 个未用于 v2 调参的主题上运行 Large 图面板。面板包含 14 道跨关键词/机构/作者网络的多跳题和 5 道时间—结构联合题，共 19 项、76 次调用，无 parse failure。该轮 flat 是哈希随机预算控制，不是强检索器，因而不再用于主论文效果量。

| 条件 | Exact accuracy | Evidence F1 | 平均上下文字符 |
|---|---:|---:|---:|
| no_reference | 0/19 | 0.000 | 0 |
| flat_retrieval | 0/19 | 0.008 | 23,294 |
| graph_retrieval | 9/19 | 0.476 | 23,300 |
| graph_program | 19/19 | 0.546 | 24,047 |

分层结果：

- C2 multi-hop：graph retrieval 9/14，graph program 14/14；
- C5 temporal–structural：graph retrieval 0/5，graph program 5/5；
- graph program 在关键词、机构合作、作者合作三个网络上均为全对；
- graph program 相对 graph retrieval 的配对差为 +0.526，主题聚类 bootstrap 95% CI 为 0.383–0.683，exact McNemar p=0.00195。

该结果仍是探索性证据：只有 5 个主题，而且 graph program 的算子轨迹包含确定性派生事实。因此它支持“系统级图分析工具有效”，不能被表述成“LLM 自己学会了图推理”。正式实验必须加入 topology perturbation、operator-only/oracle 上界和非图分析工具对照，区分检索、计算与语言综合三个贡献来源。

### 8.4 强词法与同信息表示对照

重新构建 benchmark 后，edge row 显式包含两个端点标签；`flat_tfidf` 在完整节点/边表上做 unigram+bigram TF-IDF top-160，`flat_program` 与 `graph_program` 使用完全相同的证据记录和算子输出，只改变序列化组织。冻结提示版本为 `graph-discovery-v2-strong-controls-20260820`。5 个主题、19 项、四条件共 76 个有效响应；原始调用有 1/76 次因输出达到长度上限而 parse failure，随后按预定断点续跑补齐。

| 条件 | Exact accuracy | Evidence F1 | 平均上下文字符 |
|---|---:|---:|---:|
| `flat_tfidf` | 4/19 = 21.1% | 0.081 | 33,686 |
| `graph_retrieval` | 12/19 = 63.2% | 0.499 | 29,545 |
| `flat_program` | 19/19 = 100% | 0.569 | 33,157 |
| `graph_program` | 19/19 = 100% | 0.569 | 30,292 |

检索审计显示，TF-IDF 平均 gold evidence recall 为 0.377，旧 `graph_retrieval` 为 1.0。后者后来被确认是 oracle-support 装配，因此本轮相对它的结果只能解释为“给足支持子图仍不足以完成全局计算”，不能解释为真实检索优势；非 oracle 结果见 8.7。

- graph program 相对 TF-IDF：+0.789，主题 bootstrap 95% CI 0.700–0.900，exact McNemar p=6.10e-5；
- graph program 相对原始图检索：+0.368，95% CI 0.267–0.467，p=0.0156；
- graph program 相对同信息 flat program：0.000，p=1.0。

最后一个结果是核心消融：效果来自图检索/计算管线，不来自 JSON 的图形组织。

### 8.5 复杂现象面板

为避免只测“解释简单图”，又在 5 个未调参主题的 Large 关键词图上构建 24 项复杂现象任务：多跳连接、最高介数跨社区桥删边反事实、社区内/跨社区角色对比、枢纽删除后的韧性重算、时间—结构漂移。图规模为 461–640 节点、7,091–116,671 条边。四条件获得 96 个有效响应，另有 1 次原始 parse failure 后续跑补齐。

| 条件 | 总准确率 | Bridge C4 | Community C3 | Hub removal C4 | Temporal C5 |
|---|---:|---:|---:|---:|---:|
| `flat_tfidf` | 3/24 = 12.5% | 0/5 | 0/5 | 0/5 | 0/5 |
| `graph_retrieval` | 9/24 = 37.5% | 5/5 | 0/5 | 0/5 | 0/5 |
| `flat_program` | 24/24 = 100% | 5/5 | 5/5 | 5/5 | 5/5 |
| `graph_program` | 24/24 = 100% | 5/5 | 5/5 | 5/5 | 5/5 |

采用预注册 `1e-4` 数值容差接受四位小数等价舍入；例如 0.997831 与 0.9978 视为一致，但 0.98 仍为错误。graph program 相对 TF-IDF 的准确率差为 +0.875（主题 bootstrap 95% CI 0.800–0.960，McNemar p=9.54e-7），相对原始图检索为 +0.625（95% CI 0.600–0.690，p=6.10e-5），相对同信息 flat program 仍为 0。

这组实验把任务从“读出图中一个值”提升为“对全图执行注册算子、重算反事实、比较全局结构并输出有边界的现象”。不过确定性 operator trace 接近可执行 oracle，正式论文仍需加入算子扰动、错误算子检测和自然语言综合的独立人工评分，避免把复制派生事实误称为 LLM 推理。

### 8.6 同问题配对的规模 pilot

规模效应不能用不同端点、不同难度的 Small/Medium/Large 题直接比较。本轮只保留 4 个在三档嵌套关键词图中端点和 2-hop 难度完全相同的主题，共 12 个配对 item、36 个调用。图规模从 100 节点/2,247–3,566 边，增加到 461–640 节点/7,091–28,853 边。

| Scale | TF-IDF gold recall（检索审计） | `flat_tfidf` accuracy | `graph_retrieval` | `graph_program` |
|---|---:|---:|---:|---:|
| Small | 0.750 | 4/4 | 4/4 | 4/4 |
| Medium | 0.550 | 4/4 | 4/4 | 4/4 |
| Large | 0.500 | 3/4 | 4/4 | 4/4 |

这只显示了一个方向正确但很弱的 ceiling-effect pilot：词法检索的路径证据召回随规模下降，答案错误只在 Large 出现 1 次；Large 上准确率差 +0.25 的 bootstrap CI 为 0–0.75，McNemar p=1.0。它不构成显著的 scale interaction 证据。正式规模实验需要更多锁定主题、3–4 hop 查询、多个等价路径和 C3–C5 配对任务，避免 2-hop 题过易。

### 8.7 非 oracle 检索与 operator-only 机制消融

复查代码后发现，早期 `graph_retrieval` 会先放入任务的 gold evidence，再做一跳扩展，因此它是 oracle-support 上界，不能作为真实查询检索。新实现增加 `graph_query_retrieval`：只从问题文本匹配显式实体；多跳问题执行端点间路径遍历，全局问题没有实体时只使用高显著性种子。又加入 `operator_only`，检验派生结果离开原始图后能否独立工作。

在同一 5 主题、24 项复杂现象任务上获得：

| 条件 | Exact accuracy | Evidence F1 | 平均上下文字符 |
|---|---:|---:|---:|
| `graph_query_retrieval` | 6/24 = 25.0% | 0.272 | 27,947 |
| `graph_oracle_support` | 8/24 = 33.3% | 0.330 | 27,758 |
| `operator_only` | 19/24 = 79.2% | 0.095 | 744 |
| `graph_program` | 24/24 = 100% | 0.403 | 28,430 |

分任务机制清晰：query graph 在 4 个多跳题上全对，但无法从原始子图手算社区聚合、枢纽删除和时间聚合；operator-only 在这些派生数值题上全对，却在 5 个桥接题上全错，因为轨迹保留内部 node ID，缺少原始图中的 ID→label 与 evidence 映射。完整 graph program 同时完成计算结果和 provenance 回链。相对 query retrieval 的差为 +0.750（主题 bootstrap 95% CI 0.640–0.880，McNemar p=7.63e-6）；相对 operator-only 为 +0.208（95% CI 0.200–0.230，p=0.0625）。

此外实现 canonical-graph deterministic replay：重新执行注册算子并核对问题、答案、等价答案、evidence IDs、operator trace 和解释约束。24 个干净任务零误拒；对每个任务注入一个轨迹故障后，24/24 全部检出。该验证器用于在 LLM 和人类审核之前阻断损坏的图程序输出，但由于重放器与生成器目前共享实现，正式论文还应增加独立实现或属性型不变量测试。

### 8.8 BM25 强稀疏检索基线

为避免主要结论依赖 TF-IDF，又实现标准 BM25（`k1=1.2, b=0.75`），对同一完整节点/边行表做 top-160 检索。BM25 的 gold evidence recall 为 0.125，高于同批 TF-IDF 的 0.083；但 24 项复杂面板上的答案准确率仍为 3/24（12.5%），仅在显式实体多跳题上达到 3/4，在 Bridge、Community、Hub-removal、Temporal 四类全局/反事实任务上均为 0/5。graph program 为 24/24，配对差 +0.875，主题 bootstrap 95% CI 0.800–0.960，McNemar p=9.54e-7。

TF-IDF 与 BM25 得到相同的总体准确率并不证明所有文本 RAG 都弱。正式实验仍需加入 dense embedding、hybrid retrieval 和预生成层级/社区摘要基线；否则结论只能写成“图程序优于行级稀疏检索”，不能泛化为“优于所有 RAG”。

### 8.9 非 oracle 层级 GraphRAG 基线

进一步实现 query-independent 的层级图索引：离线为完整图生成全局摘要、社区规模/重要性/内外连边摘要、社区边界及代表边；在线阶段只根据问题文本检索摘要，并与 query-anchored 局部子图共享 160-record budget。索引构建函数不接收 benchmark answer、gold evidence 或 operator trace，因此它是端到端非 oracle GraphRAG，而不是把答案换名塞入上下文。

在同一 5 主题、24 项复杂面板上，v1 层级检索得到 10/24（41.7%），高于 query-subgraph 的 6/24 和 TF-IDF/BM25 的 3/24。分任务为：multi-hop 4/4、bridge 2/5、community 0/5、hub-removal 4/5、temporal 0/5。它说明层级摘要确实比局部图和行级稀疏检索更适合全局问题，但仍无法稳定执行全局选择、精确反事实和时间聚合。

v1 错误审计发现摘要字段 `external_edge_share` 被实现为边数占比，而注册 community 算子使用边权占比。保留 v1 artifact 后，开发版 v2 显式区分 count-share 与 weight-share，并加入按 node importance 选择的社区代表。v2 得到 12/24（50.0%）：community 从 0/5 提升到 2/5，hub-removal 从 4/5 到 5/5，但 bridge 仍为 2/5、temporal 仍为 0/5，multi-hop 由 4/4 波动到 3/4。v2 相对 v1 的配对差仅 +0.083，McNemar p=0.625；相对 query-subgraph 为 +0.250，p=0.0703。

graph program 对 v2 层级检索仍为 24/24 对 12/24，配对差 +0.500，主题 bootstrap 95% CI 0.320–0.660，McNemar p=0.000488。错误逐项检查还发现：部分 community 题的正确摘要已经被召回，模型仍把 `member_count` 输出成成员列表或给 community ID 添加字符串前缀；temporal 题则缺少时间分层索引。这支持把贡献拆成三层：层级 GraphRAG 提高全局证据可见性，query planner 选择所需分析，注册图算子负责精确计算与 provenance 回链。

由于 v2 是查看 v1 错误后的开发迭代，这一提升只能作为机制诊断，不能作为锁定测试主结果。正式实验必须在全新主题上冻结 v2；并补充 temporal community summaries、dense/hybrid summary retrieval，以及与标准社区报告式 GraphRAG 的同预算对照。

### 8.10 Latent-dense 与 hybrid 文本 RAG

当前本地环境没有 sentence-transformers、Torch、ONNX 或 DeepSeek embeddings 接口，因此不能把任何现有条件写成“神经 dense”。本轮增加的是命名严格的 classic latent-dense baseline：在全部 node/edge rows 上离线拟合 TF-IDF + 48维 randomized LSA（词表上限4096、seed=42），测试问题不参与索引训练；hybrid 使用 BM25、TF-IDF、LSA 各自 top-640 候选的 reciprocal-rank fusion（`k=60`），最终仍截断到160条记录。

24项复杂面板的 gold evidence recall 为：TF-IDF 0.083、BM25 0.125、LSA 0.067、hybrid 0.092。真实模型结果为 LSA 2/24（8.3%）、hybrid 4/24（16.7%）；hybrid 的4个正确项全部是显式实体 multi-hop，Bridge、Community、Hub-removal、Temporal 均为0。hybrid 相对 BM25 只多1项，McNemar p=1.0；因此不能声称 latent dense 显著改善。

层级 GraphRAG v2 为12/24，对 hybrid 的配对差 +0.333，主题 bootstrap 95% CI 0.210–0.480，McNemar p=0.0215；graph program 为24/24，对 hybrid 的差为 +0.833，95% CI 0.800–0.920，p=1.91e-6。这构成一个更强的机制链：行级相似度混合适合实体路径，层级图摘要改善全局可见性，注册算子才稳定完成全局选择、反事实和时间聚合。正式论文仍必须补一个真正的神经 dense/hybrid baseline。

### 8.11 v3 真实数据与 100 题冻结基准

采集前冻结协议 SHA-256 为 `2e59e2029df034470b93a10cb86548fda1322ee3a322d4200c6ffeb692761b8f`。四个新领域的 OpenAlex exhaustive cursor harvest 共得到 80,008 条去重源记录，规范化后 79,968 篇论文；四套 harvest/process acceptance 全部通过，年份边界完整、外键孤儿为 0。

采集后、任何正式模型调用前发现两项构造问题：默认 640 候选池没有实现规模扩展；高密度关键词图的无权最短路会退化为一步直连。修正案 `formal_v3_amendment_001` 将候选池提高到 2,000、边上限提高到 1,000,000，并把多跳定义为 `distance = 1 / edge_weight` 的加权最短路；修正案 SHA-256 为 `4ba476b5cf7506761f116a903552150a22bdd0e6aa4a61702510163f24ca1a9b`，明确记录 `model_outcomes_inspected: false`。

每个数据集严格生成 25 题：关键词网络 3 个尺度 × 5 类任务 = 15，机构与合著网络各 1 个 Large × 5 类任务 = 10，总计 100/100。题型包含加权跨社区多跳、桥删边反事实、社区角色对比、枢纽删点韧性、时序—拓扑联合变化和生产力—中心性分歧。所有 benchmark 均有独立 SHA-256，构造状态为 `constructed_not_executed`。

`formal_v3_readiness.json` 已机器验证协议、修正案、benchmark hash 和题型分布，`benchmark_structure_valid=true`、`no_confirmatory_results_present=true`。官方 DeepSeek V4-Pro tokenizer 已在固定 revision 上下载，并用6项真实API完整消息探针逐项精确复现 prompt token；当前状态仍为 `blocked`，剩余原因是真正神经 dense index/contexts 尚未全部物化。正式 runner 已加入硬门禁，缺 `ready` 审计时在读取 API key 前退出。

为避免拿到 tokenizer 后只验证裸字符串、却漏掉 DeepSeek V4 专用消息边界编码，已用当前 `deepseek-v4-pro` API 采集并冻结6个不含任何benchmark内容的message usage探针（artifact SHA-256 `4e97d33c6fe2afcf17cce3abfc86c9aa86860dc524dbcc925294432ae89cb01b`）。empty user、英文、中文、graph JSON、system/user边界和multi-turn边界的真实 `prompt_tokens` 分别为4、10、11、21、17、20。主实验、复制实验和复杂度扩展readiness现在都要求官方message encoder逐项精确复现这6个计数，同时还要通过独立raw-text探针；正式runner会再次执行两层验证。该探针是tokenizer校准，不包含任务答案，也不构成confirmatory outcome。

神经 dense 的模型选择也已在结果前冻结为 `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`。第二修正案 SHA-256 为 `0c9e3c47a93f5e9c56cc0b2426f920f94229e61c27f1a20da8fcd3c843a5a39a`；它固定原始注册问题 query、L2 normalization/cosine、top-160、evidence-ID hash tie-break 和无微调 query-independent index。选择依据是其多语、dense retrieval 和最长8192-token输入能力；该选择发生在任何 neural retrieval output 之前。最初登记的 full-row canonical JSON 后经前结果表示审计发现会把无意义的序号 ID 和规模依赖统计量送入语义编码器，已由下述前瞻性修订取代。

正式执行使用该 revision 的官方 FP32 ONNX 图（SHA-256 `f8425123…435b`）与外置权重（SHA-256 `1eebfb28…16b4`）。DirectML/AMD 780M 与 CPU-FP32 的实现等价性探针不比较条件、不读取 gold：160条真实序列化图记录与25个注册问题上，文档向量最小 cosine 为0.99999988，最大绝对差为2.39e-6，重复运行最大差为0，25/25个查询的Top-10集合完全一致；吞吐由4.88提升到17.85条/秒。该实现选择、provider、batch、fallback和探针披露冻结在 `formal_v3_neural_runtime_implementation.yml`，发生在任何 confirmatory sidecar 或生成结果之前。

为使 query-independent index 不只是名义设定，sidecar builder 现在按 dataset/network/scale 把全量 float32 embedding、row-ID hash、模型与运行时身份、内容 SHA-256 写成可恢复缓存。主实验与复杂度扩展遇到同一张图时复用这个文档索引，只重算注册 query 和 top-k 排序。INT8 虽有更高吞吐，但独立精度探针的最小向量 cosine 仅0.889、Top-10 overlap最低0.1，已在任何正式输出前明确否决。

进一步的跨规模审计发现，若直接编码完整证据行，Small/Medium 共191,656条可行性记录仅4条能与Large精确复用（0.0021%）：中心性、社区、度和边权会随规模变化，序号型边 `evidence_id` 甚至会在不同规模指向不同端点。结果前冻结的 `semantic_entity_relation_json_v2` 因此只把可读实体标签或关系两端标签送入 BGE-M3；节点身份、证据ID和全部结构统计仍原样保留在返回证据中，供图检索、重排、算子和生成使用。这样既防止 `flat_neural_dense` 偷带图结构，又让图条件的结构增益可解释。跨全部主实验网络的 broad feasibility 审计复用了191,514/191,656条（99.926%）；按真正注册执行范围重审后，8主题×keyword×Small/Medium 的16个单元共247,995条全部从Large复用，新增编码为0。运行预算只报告这个 execution-scope 结果；同名不同身份只共享语义向量，不合并图节点。该表示修订 SHA-256 为 `dca3905cfe1402a1333279dace2c5c2d0d361c2bdb495fd8edf161ace8f09367`，范围澄清不改变任务、向量或检索结果，旧缓存隔离保留。

规模交互不能把不同语义锚点的 Small/Large 题硬配对。结构审计后只保留三档中实体锚点完全一致的 14 组、42 个 item，并冻结为 `formal_v3_scale_pairs.json`（SHA-256 `061e85391fa56ecf61fe24714a8d1df4d4f643a2dbbe29bf2c30408ed22d3339`）。第三修正案进一步锁定 H1/H2 的精确 McNemar、数据集簇 bootstrap、H3 的单侧数据集簇精确符号翻转和 H1–H3 Holm 校正；修正案 SHA-256 为 `879fce248bc6ecd5906ffff7a40acbcab5c6ff19083649627174c40ac4b8e598`。四个数据集簇使 H3 可达到的最小单侧 p 值只有 1/16=0.0625，因此本轮能估计 effect/CI，却不能靠 item 伪重复制造显著性；若要检验 H3，需增加独立预冻结主题作为 replication。

结果前实现复核进一步发现，旧cluster bootstrap在抽中主题后拼接其题目行，因此点估计仍按题数加权；而两阶段每主题冻结配对数并不完全相同。这与exact sign-flip的等主题权重不一致。正式provider结果仍为0时冻结修正案007（SHA-256 `4ad307c691a4d768379b1d3ec7b20a5295607f73da24ff79d2fda242952bc097`）：所有相关区间先计算主题内效应，再对主题等权平均和重采样。假设、配对、方向、精确检验、Holm、seed、调用和任务均不变；匹配题只增加主题内精度，不能成为独立推断重复。

同时，H1/H2的题级exact McNemar保留为原注册配对结果，但不再被允许单独支撑跨主题显著性，因为同一主题内任务可能相关。结果前修正案008（SHA-256 `955ee81ec7baefa1035cf9cd0a24ef26ae26776181eb01fbd607caa748863bfb`）在不增加调用的前提下，利用完整8主题扩展面板注册E1（复杂度3–5的`graph_program-flat_hybrid`）和E2（四类全局/反事实任务的`hierarchical-flat_hybrid`）两个主题级单侧exact sign-flip，并对二者做Holm校正。论文级跨主题优越性主张必须通过相应E1/E2；C1/C2仍分别回答“是否复杂度选择性”和“是否随规模放大”，不能互相替代。

正式运行面板已由 `run_formal_v3_panel.py` 固化为 6 个主条件、`flat_program` 诊断和 `flat_neural_dense` 外部基线，共 100 题 × 8 条件 = 800 次调用。`formal_v3_panel_plan.json` 逐数据集绑定 benchmark hash；`analyze_formal_v3_results.py` 要求 800 个 item-condition 唯一且完整，parse failure 保留并计错，缺任一配对即停止，再输出 accuracy/evidence-F1/cost、cluster CI、matched odds ratio 和冻结的三项主检验。当前实际执行演练按预期在读取 `apikey.md` 前被 readiness 拦截，未产生正式结果。

结果前另冻结统一执行监督规则：每个 JSON parse failure 最多使用完全相同请求重试3次，第3次仍失败则保留并计错；传输或进程中断最多恢复10轮，已完成单元按身份跳过，禁止基于答案正确性重试。监督器先验证800/120/312三套调用计划，再按主实验、复制、增量扩展顺序运行，最后进行672格严格合并和冻结分析。终端审计拒绝重复、意外条件、benchmark hash变化和未注册状态；每个结果 manifest 还直接记录 semantic-v2 表示修订与执行范围修订 SHA，而非只依赖间接路径。

### 8.12 独立规模交互 replication

四数据集主实验使 H3 的最小精确单侧 p 值受限于1/16。为避免用 item 伪重复追显著性，在查看任何主实验或 replication outcome 前另行冻结四个新领域：钠离子电池正极、类器官疾病建模、海水淡化膜污染、可解释AI医学影像。预采集 query judge 首轮否决了连续短语 `membrane desalination` 的系统性漏召回；保留原否决 artifact 后，版本化修正为 `desalination AND "membrane fouling"`，第二轮 judge 四项全部批准。协议 SHA-256 为 `872c28cd84e56cc98d6522ed36db01ee24026f58c639695b127d6cdebc999fec`，query修正案 SHA-256 为 `0e051fc11e86895250256f5b2d47a603f771680d17e4904f5cc525c4258969c5`。

真实 OpenAlex exhaustive cursor harvest 新增50,732条、510页、零重复；规范化后50,716篇论文，四套采集与处理门禁均为6/6，年份完整且外键孤儿为0。四个关键词Large图均为2,000节点，边数85,238–127,827；每个主题严格生成3尺度×5复杂任务=15题，总计60/60，状态均为 `constructed_not_executed`。

按三尺度实体锚点完全相同的规则，replication 再冻结14组、42个item（SHA-256 `e74af08327bffa391027767c76b9bada28fa524d983f42919ee8b0ed6b190e6a`）。主实验与复制实验合并后为8个独立主题簇、28组、84个尺度配对item，精确符号翻转的最小单侧 p 值降至1/256≈0.0039。复制实验只执行 `flat_hybrid` 与 `graph_program`，共60题×2=120次调用；主结果、复制结果和合并结果必须分别报告，不能只展示合并显著性。

replication readiness 已通过协议、query judge、采集处理、benchmark hash、scale-pair hash和零结果污染检查，只剩DeepSeek V4 exact tokenizer/token cap。加上主实验，当前真实数据总量为130,740条源记录、130,684篇规范论文，冻结复杂任务160题，计划模型调用920次。

### 8.13 锚点匹配的规模 × 复杂度扩展

仅在复杂题上观察到图优势，仍不能证明优势来自任务复杂度，因为缺少同一实体上的简单负对照。为此在任何模型 outcome 前冻结独立扩展协议（SHA-256 `32cd49e3335b74cba48524e4c7d769e4f0b7e41d9a2720d7451542a27e386988`），覆盖上述8个独立主题、100/300/2,000三档关键词图和四个条件：`flat_hybrid`、`flat_neural_dense`、`graph_hierarchical_retrieval_v2`、`graph_program`。

简单题不是另选容易实体，而是严格共享复杂题锚点：同一条桥边形成“直接边权查找 ↔ 删边反事实”，同一个加权度枢纽形成“节点属性查找 ↔ 删点韧性”。每主题每尺度2个简单题和5个复杂题，共21题；真实物化结果为8主题×21=168题，其中48个简单负对照、120个已有复杂题，全部绑定 `matched_complex_item_id` 并通过原始 evidence anchor 校验。C1检验 `(graph_program-flat_neural_dense)_complex - (graph_program-flat_neural_dense)_simple`；C2再检验该复杂度选择性是否从Small到Large增强。没有C1就不能把图条件主效应称为复杂推理优势；有C1无C2也不能声称规模放大。

扩展共有672个逻辑单元格，但结果前执行修正案（SHA-256 `9eaded9d966becdc02fe49b23b1262757c9a911e409c3848d3a6eca6ce750c4f`）禁止重跑与主/复制实验字节等价的单元格。逐项核验 task、context、prompt、model、tokenizer 和 token cap 后可复用360格；真正新增调用为312次：48个简单题×4条件=192，以及复制复杂题补 neural/hierarchical 60×2=120。任何等价字段不一致都保留为缺失，不允许看过 outcome 后静默重跑。分析脚本要求完整672格、按数据集簇 bootstrap/符号翻转，并在混合 logistic、C1/C2 Holm 和简单题 TOST 完成前只输出 `descriptive_ready_mixed_model_pending`。

该复用方案现已实现为逐单元格可验证执行链，而不只是预算说明。正式 runner 为每条结果写入去除contexts后的task payload hash、exact-token裁剪后的context hash、完整messages hash与tokenizer manifest hash；extension readiness 在任何API调用前预物化672个期望identity。执行计划固定为12个job和312次调用；merger分别读取主实验、复制实验、简单题新结果与复制补充结果，逐格核验identity和源文件SHA后才生成8×84=672格面板，并为每格记录是否复用、来源路径、来源SHA和等价性审计。当前真实dry-run中协议、修正案、8个benchmark、题型分布、exact tokenizer和零结果污染均通过，仍只被8套BGE-M3 sidecar及其672格身份物化阻断；带`--execute --api-key-file apikey.md`的演练已确认在读取或调用API前退出。

为避免“复杂题”只是文字更长，另做了不读取任何模型答案的图必要性证书审计。48/48个简单负对照在任务定义上都是单记录属性查找；120/120个复杂题都含至少两个注册算子阶段，且没有一题能从 `flat_hybrid` 的任一单行直接取得全部决定性答案。复杂题覆盖加权多跳、全局边介数选择+删边反事实、社区检测+全局角色聚合、枢纽删除+连通性/替代枢纽重算、时间窗口聚合+拓扑跨层连接。后续实现完整性复核发现原验证器只检查轨迹长度，没有验证具体算子能否覆盖预注册能力类别；修正后的独立v2审计逐项映射和检查`path/community_constraint/global_selection/intervention/community_detection/global_aggregation/recompute/temporal_aggregation/cross_layer_join`，120/120复杂题仍覆盖全部必需类别，0项缺失，反例测试会拒绝由两个无关算子伪装的复杂轨迹。原冻结审计未覆盖或改写，正式题目、条件和预算也未变化。冻结上下文的描述性诊断显示，复杂题 gold-evidence recall 为 `flat_hybrid=0.388`、`graph_hierarchical_retrieval_v2=0.990`、`graph_program=1.000`；该诊断在 neural sidecar 和生成结果出现前冻结，禁止据此删题或调参。因而正式机制链可区分“图提高证据可见性”和“可执行算子完成非局部计算”，不再把一段图解释误称为图推理。

该差异随规模单调放大：在每档40个复杂题上，`flat_hybrid` 的 gold-evidence recall 从 Small 0.811 降至 Medium 0.281、Large 0.071，完整覆盖题数从22降至6再降至0；层级 GraphRAG 则为0.984、0.990、0.997，完整覆盖35、37、39题，Graph Program 三档均为1.000。这个 pre-generation retrieval curve 是固定160-record budget下的机制性次要终点，不等于最终答案质量，也不能替代 neural-dense 对照；查看后禁止改题、改检索器或改预算。

## 9. 现有文章质量审计

8 组机器稿与已发表人类稿进行了盲化 LLM 预审。注意：两者语料库不完全相同，此实验只比较写作和科学推理，不比较数字正确性，也不能替代专家评价。

| 维度（1–5） | 机器 | 人类 |
|---|---:|---:|
| 方法透明 | 5.00 | 1.88 |
| 认识论校准 | 4.38 | 2.38 |
| 论证连贯 | 4.25 | 2.50 |
| 跨证据综合 | 3.88 | 2.50 |
| 图派生洞见 | 3.25 | 2.38 |
| 研究价值 | 3.50 | 2.88 |
| 学术文风 | 4.38 | 2.50 |
| **领域具体性** | **2.63** | **3.75** |

诊断性文本指标：

- 平均篇幅：机器约 1,535 词，人类约 4,639 词；
- 数字密度：机器每千词 83.8，人类 50.9；
- 解释性风险词：机器每千词 5.78，人类 2.94；
- 机器稿有大量 evidence IDs，人类稿更多连接具体文献。

因此改进目标不是单纯扩写。应该：

1. 用现象卡组织 Results–Discussion，而不是按表格逐项汇报；
2. 为每个图现象检索具体代表文献/摘要，解释可能机制；
3. 强制提供替代解释和反事实结果；
4. 把“dominant / hot / mature / driven by”等词交给 review compiler；
5. 增加领域专家评价，避免 LLM 因 evidence ID 和规范结构产生偏好偏差。

### 9.1 多现象文章真实生成

已在 `digital_twins_healthcare_2012_2024` 上完成第一篇 v2 多现象文章。5 张 `PhenomenonCard` 分别覆盖多跳连接、桥删除反事实、社区角色、枢纽删除韧性和时间—结构漂移；每张卡都通过 canonical graph 重放，并从 `canonical/keywords.parquet + works.parquet` 连接最多 3 篇代表文献。

生成稿为 3,576 词，图派生内容分布在 Results、Discussion、Limitations 和 Conclusion，而不是一个孤立图注段。确定性文章审计结果：

- 5/5 card 全部使用，无未知 CARD/WORK ID；
- 4 篇代表文献进入正文；
- CARD 引用 59 次、WORK 引用 17 次；
- 26 个含数值、具体实体图指标或解释风险词的论证段落中，24 个具有合法 provenance，覆盖率 92.3%；
- Results 至少使用 4 张卡，Discussion 和 Conclusion 各至少使用 3 张卡；
- 通过全部结构与 provenance gate。

单主题盲化 LLM 预审中，新稿相对旧机器稿在 phenomenon complexity、cross-evidence synthesis、graph insight 和 research value 上从 3/5 提升到 5/5，domain specificity 从 3/5 提升到 4/5。预审同时指出两个剩余问题：`maturing field` 仍是超过结构证据的阶段性解释；领域连接只有 4 篇文献，尚不足以覆盖具体临床应用与争议。该结果是一次 LLM 诊断，不是专家证据；当前人类参考稿过短，也不能据此声称超过人类文章。

### 9.2 同证据领域专家盲评

已在任何 confirmatory 专家评分和同证据人类稿产生前冻结 `expert_evaluation_protocol.yml`（SHA-256 `7e8267bd91b69ac12b563aa1e5af47921cd67d521a4f17a28bb652fb1fd875a5`）。正式比较不再把不同语料的已发表文章当作等价金标准，而是在至少8个主题上让 CiteWeave、一次性LLM和独立领域研究者使用完全相同的文献、结构化图分析、研究问题、字数和截止时间写作；渲染图按结果前修正案统一在草稿冻结后附入盲审包，不作为任何作者的写作输入。

同证据输入已具体化为8个通过门禁的writer pack，而不是只在协议中写“相同证据”。每个主题固定5类Large图复杂现象（多跳连接、桥删除反事实、社区角色对照、枢纽删除韧性、时间—结构漂移）、15条互异且摘要非空的代表来源和1张经可视检查的图谱概览，共40个注册现象、120条来源摘要、8张图，计划形成8主题×3条件=24篇约3000词文章。所有条件必须把五种现象分布到Results与Discussion，不能只用一段文字解释图。来源检索经三次结果前版本化修正：v1暴露泛化答案关键词污染，v2暴露单一主题词过宽，v3暴露不同work ID的重复题名；候选正式v4按主题词覆盖数再按任务BM25排序、排除引用量排序并跨现象规范化题名去重。v4结构审计为8/8主题、40/40现象、120/120非空摘要、120个独立规范化题名和8/8图像hash全部通过；旧版本保留为开发工件，不进入正式写作。

同证据协议随后发现一个结果前的模态漏洞：生产记录原本绑定writer pack，却没有证明机器请求真正携带渲染图。现有 `deepseek-v4-pro` profile 为text-only，既有 `deepseek-v4-flash` image-input probe也以HTTP 400拒绝。修正案001先因此fail closed；在0篇正式文章、0个专家结局时又冻结修正案002（SHA-256 `eb144b0d6f5cac41f5ee009d5e8af6a87474f97ff03836b5aadb6483a1e1212e`），将焦点明确为图结构推理而非视觉识图。8份正式作者输入仍包含相同的5类现象、完整算子轨迹、来源摘要和非视觉元数据，但递归审计确认0份含PNG路径、URL或图像字节；三类作者均不得看渲染图。草稿冻结后，packet builder把同一figure SHA原样复制给三条件的盲审包，生产记录绑定writer-input hash、无图访问声明、draft hash、请求/交付回执和posthoc insertion receipt。`figure_access_readiness.json` 现为ready；VLM纯视觉比较保持独立，不能混入文章质量主对比，也不能据此否定GraphRAG。

每篇文章至少由3名专家盲评，且每个主题至少2名领域专家；格式、引用和provenance token做条件无关映射。整体层评估事实准确、证据可追溯、现象深度、替代解释、认识论校准、领域具体性、论证连贯和研究效用；claim层每篇分层抽20个Results/Discussion/图派生/数值/因果风险声明，至少双审并裁决分歧。

盲评材料生成也已实现为硬门禁。每篇稿件必须同时提交字节绑定的production record和由不知道条件的claim abstractor制作的候选清单；系统验证人类作者未看机器稿、one-shot只有一次请求且无图算子、CiteWeave稿确实使用图程序和validated review日志。随后去除作者/模型/条件元数据，把相同PH/REF证据映射成主题内一致的opaque ID，并从五个stratum各选4条互异claim。分配采用可复现的混合不完全区组：每主题1人共评三条件，另3人分别共评一个条件对，因此每篇恰有3名整体评分者、每个条件对恰有2名共评者。`expert_packet_intake/readiness.json` 当前如实标记为blocked，因为真实24篇文章、领域专家和claim abstraction尚未产生；程序不会用伪造文章或模拟专家填补这一缺口。

结果前执行审计进一步发现，opaque ID若没有配套证据查看器，专家只能看到`EV-*`标签而无法核验其对应的图计算或来源，claim支持性评分不成立。修正案007现要求每主题三条件共用一个哈希绑定的匿名证据查看器：`EV-*`可展开到注册图问题、确定性答案、算子轨迹、解释边界，或来源题名、年份、DOI与冻结摘要；原始PH/REF、条件、自动分数和检索排名保持隐藏。专用服务器采集八维整体评分、claim判断及决定性`EV-*`、不可判断和强制成对偏好，并使用页面可见heartbeat计时和不可重复提交。每项任务同时进入结果前注册表；汇总器只接受全部任务恰好一次、服务器身份/hash/计时一致的返回，之后才恢复私有条件，并把任一claim字段分歧物化为排除两名首审者的盲裁worklist。确定性负载均衡路由器从该主题其余合格专家中选择第三人，裁决者看不到首审身份/答案；每项争议恰有一次有效盲裁后才输出分析输入。缺少可解析证据的claim不能进入collection。这修复的是测量有效性与执行闭环，不改变A1/A2/A3、样本量、非劣界值或分析方法。

三个主检验分别是：CiteWeave相对one-shot的“正确且有支持”claim率；相对同证据人类稿研究效用的0.35分非劣；相对人类稿证据可追溯性的优势。即使可追溯性更高，若研究效用未达非劣或领域具体性明显下降，也禁止声称“超过人类文章”。不同语料的旧人类稿只保留为ecological descriptive benchmark。

该协议现已接入可执行分析门禁。`article_expert_evaluation.py` 会拒绝主题数不足、条件缺失、每篇少于3评、每主题少于2名领域专家、20个claim未覆盖五个分层、claim未双审、分歧未由恰好一名独立第三人裁决或强制两两偏好缺失的数据。由于独立主题只有8个，任何文章与专家结果产生前已冻结分析修正案003（SHA-256 `3a8e707ae9f45705cbe3cfb08287ad41d31c8804bdd2f5efd76fdc1d7ff75100`）：claim先按一致意见或裁决形成每claim一个结局，整体评分先在同一专家内做条件配对，再汇总为8个主题效应；A1/A2/A3完全枚举2^8种sign-flip并做Holm校正。A1同时报告cannot-assess对Graph最不利/最有利极端情形；A2按-0.35分边界检验研究效用非劣，A3检验可追溯性优势，领域具体性另设-0.35非退化门禁。CLMM/混合logistic保留为诊断敏感性，不能把claim数或评分者数冒充独立样本，也不能推翻主题级精确检验。

### 9.2 文章生成与claim级人审闭环

正式机器写作已冻结为16个单次响应cell：8个主题分别生成one-shot和CiteWeave pre-review draft。两条件接收相同的text-only writer input；CiteWeave唯一增加的是由同一证据确定性构造的5张现象卡、三类跨现象综合和`REF → PH → synthesis`依赖计划。任一provider response使cell终止，质量门禁失败也保留而不重采样，因此结果不能通过best-of-N获得。

在0个机器响应时冻结的v2协议进一步封闭了形式化漏洞：原门禁只数PH是否在章节出现，五个标识堆进一段也可能通过。v2不改变16个请求或任何输入，只把两条件共同的终止门禁加强为五类现象全部进入Results、至少四类进入Discussion、三类进入Conclusion；至少两组注册现象配对必须分布在两个不同的Results/Discussion正文段落。含全部五个PH的单一段落不计为聚焦综合。该检查只能拒绝明显的“引用堆叠”，不能证明段落语义正确，后续真人claim审核与专家盲评仍是质量结论的必要条件。

`article_claim_review_protocol.yml`（SHA-256 `c3d74f5a6d1d8de130813c780cf16aafd4f1acedb52aca10e0d4992509beb8f2`）在0篇机器稿、0名真人审核者、0个审核结局时冻结。每主题从Results/Discussion选择20条带PH证据的claim，按因果词、数字、多现象综合和Discussion解释提高风险优先级，同时保证5个现象各至少2条；8主题共160个claim和320次独立首审。每主题至少3名无冲突领域合格者，两名首审者必须对四个二值判断、action、决定证据、失效依赖、replacement和guard完全一致，否则只把该claim交给预注册第三人独立裁决。

裁决后的反馈不会追加进prompt。系统建立`source/graph evidence → claim → paragraph`依赖DAG；上游证据否定会自动传播到全部下游claim和paragraph。修订器只接受受影响段落的原始hash绑定replacement，其他正文按字节锁定，并拒绝任何未在该段落审核包注册的PH/REF。过程指标包括分歧/裁决率、每claim和每修订段落的服务端有效审核时间、每个上游否定影响的段落数与正文精确保留比例；效果指标由独立专家盲评pre/post claim的严格支持与校准变化、修改精度、未修改claim伤害率，以及最终三条件文章盲评共同给出。没有真人结果时只能声称该闭环可执行，不能声称人审有效。

完整性修正案001在所有文章和真人结局仍为0时冻结。代码审计发现初版把依赖传播事件算出来，却只把直接缺陷段落写入修订清单；修复后以全文显式引用构图，未抽样但共享失效证据的段落也进入worklist。保证范围限于显式PH/REF和现象—来源/图证据依赖，不包括自动推断所有无引用的语义依赖。改写仅产生候选稿，不跳过独立专家检验。

新增的独立修订评估将修改前后的段落分别盲化，并随机排列到独立领域专家的同一审核界面；每版本双审、分歧单独裁决，原写作和审核参与者不能评自己的主题。主结局是四项严格充分性全通过的变化，弃权记为未成功；先汇总到主题，再对8主题枚举256个sign-flip，并把5000次主题bootstrap区间作为小簇数描述性不确定性。无改动主题贡献0效应；相似措辞可能暴露配对，不能声称完美盲化或随机化因果效应。

审核选择消融在packet产生后、任何真人标签返回前冻结：25%/50%/75%的每主题题数预算，100次随机排名、固定风险分与风险×共享依赖影响三策略，另有全审上限。标签收集一次后仅做离线回放，报告可纠正缺陷捕获量、直接涉及的修订段落数、两位首审者与裁决者的有效秒数。相同题数不保证相同人力时间；这组结果不能代替未实际生成的反事实文章质量或真实劳动节省。

## 10. 与相关工作的关系

- Edge et al. 的 GraphRAG 使用社区摘要回答全局 corpus 问题，说明图的优势应在全局 sensemaking 而非单条事实上体现：<https://arxiv.org/abs/2404.16130>
- HippoRAG 用知识图与 Personalized PageRank 处理 multi-hop QA，说明图检索器应与强 dense/iterative retrieval 比较，而不是只对比随机或稀疏行检索：<https://arxiv.org/abs/2405.14831>
- BGE-M3 同时支持多语、dense/sparse/multi-vector 与长文本检索，为冻结的 neural dense 外部基线提供公开可复现模型：<https://arxiv.org/abs/2402.03216>
- CRAG 用 retrieval evaluator 触发 corrective action，支持把人审结果编译成可校准的“接受/改检索/升级人审”路由，而不是被动 memory：<https://arxiv.org/abs/2401.15884>
- Pistis-RAG 将整列输入上的人类偏好建模为 listwide learning-to-rank，支持 plan/list 级审核信号：<https://arxiv.org/abs/2407.00072>
- TRACE 构造知识支撑的多跳 reasoning chains，支持把 evidence path 作为一等评估对象：<https://aclanthology.org/2024.findings-emnlp.496/>
- ReasonGraphQA 明确评估多链、多跳图检索和证据解释：<https://aclanthology.org/2024.lrec-main.1437/>
- DIVKNOWQA 使用结构化与非结构化知识的两跳多源问题，说明跨源任务比简单 KGQA 更能测到结构贡献：<https://aclanthology.org/2024.findings-naacl.5/>
- HADAS 用错误多样性主动选择人类标注，为主动审核提供直接方法依据：<https://aclanthology.org/2024.naacl-long.479/>
- Ewe 用在线 fact-checking feedback 更新显式工作记忆并改善长文事实性：<https://aclanthology.org/2025.acl-long.548/>
- Feedback Adaptation for RAG 提出 correction lag 和 post-feedback performance，正好可替代“记忆条数”这种弱指标：<https://arxiv.org/abs/2604.06647>
- DMA 把 document/list/response 多粒度反馈用于在线 RAG 对齐，提示我们分别处理证据、排序和最终声明：<https://arxiv.org/abs/2511.04880>

本项目可区分于这些工作的点是：把可执行图分析、科学现象发现、证据依赖传播和人类审核规则编译放入同一个可审计长文生成系统，并在真实 bibliometric 全图上做规模–复杂度交互实验。

## 11. 实施状态与下一步

已实现：

- `src/citeweave/graph_discovery.py`：规模化全图加载、六类复杂任务、TF-IDF/BM25/query-graph/层级 GraphRAG/oracle/program 条件、固定记录预算和等价答案评分；
- `src/citeweave/review_learning.py`：审核规则编译、依赖传播、成本敏感主动审核；
- `src/citeweave/operator_verification.py`：canonical graph 算子重放和故障注入审计；
- `src/citeweave/phenomenon_cards.py`：经验证的复杂现象卡和代表文献依赖；
- `src/citeweave/article_verification.py`：完整文章的 CARD/WORK provenance 与跨章节覆盖审计；
- `scripts/run_graph_discovery_experiment.py`：真实模型条件实验、断点续跑和 parse-failure 保留；
- `scripts/run_formal_v3_panel.py`：按冻结 benchmark hash 编排 4 主题 × 25 题 × 8 条件的 800 次正式调用；
- `scripts/analyze_formal_v3_results.py`：完整性硬门禁、配对效应、数据集簇 bootstrap、精确检验和 Holm 校正；
- `scripts/build_formal_v3_replication_benchmarks.py`、`run_formal_v3_replication_panel.py`：4个独立复制主题、60题与120次双条件调用；
- `scripts/analyze_formal_v3_scale_replication.py`：分别报告主实验、复制实验与8数据集簇合并的规模交互；
- `scripts/build_formal_v3_complexity_extension.py` 与 `analyze_formal_v3_complexity_extension.py`：168题锚点匹配的简单/复杂×三尺度扩展及数据集簇分析；
- `scripts/audit_formal_v3_complexity_extension_readiness.py`、`run_formal_v3_complexity_extension_panel.py` 与 `merge_formal_v3_complexity_extension_results.py`：672格identity预物化、312次增量执行计划和360格严格等价复用；
- `scripts/rescore_graph_discovery_results.py`：冻结 raw response 后重评分；
- `scripts/audit_review_learning.py`：旧审核记忆可发表性审计；
- `scripts/audit_article_quality_v2.py`：文章盲化 rubric 预审；
- `src/citeweave/article_evidence_pack.py` 与 `scripts/build_article_same_evidence_packs.py`：8主题、40个复杂现象、120条来源摘要和8张图的同证据源包；
- `src/citeweave/article_writer_inputs.py` 与 `scripts/build_article_text_writer_inputs.py`：移除渲染图定位符、保留结构化图证据并绑定草稿后统一附图的作者输入；
- `src/citeweave/article_generation.py` 与 `scripts/run_machine_article_generation.py`：16-cell同证据单次生成、确定性图蓝图和不可重采样质量门禁；
- `src/citeweave/article_claim_review.py`：160个高风险/现象覆盖claim包、真实审核者资格与双审分配；
- `src/citeweave/article_review_revision.py`：分歧裁决、evidence–claim–paragraph传播和hash限定段落修订；
- `src/citeweave/review_packets_v2.py`：事实层/语义层双层盲审 packet；
- `src/citeweave/human_review_v2.py`：审核返回校验、独立一致性和裁决 worklist；
- `src/citeweave/review_ui.py`：条件盲化、哈希令牌、服务端计时和冻结提交的本地审核界面；
- `src/citeweave/complementary_oversight.py`：审核者×领域×错误类型能力后验、严重度—成本联合路由、critical双审加独立裁决与盲化双面材料；
- `src/citeweave/oversight_readiness.py` 与 `scripts/audit_complementary_oversight_readiness.py`：96案例、6审核者、热身排除、能力冻结、四格crossover和50%双审覆盖的失败关闭门禁；
- `src/citeweave/source_relevance_packets.py` 与 `source_review_assignment.py`：120个现象—来源校准包、选择元数据盲化、6×30平衡分配、60题双审和证据检索组件监督；
- `src/citeweave/review_sampling.py`：隐藏分层、partial-overlap、逆概率权重和审核覆盖审计；
- `scripts/finalize_human_review.py`：双审结果校验、一致性统计、裁决清单和规则编译锁；
- `src/citeweave/review_policy_experiment.py` 与 `scripts/analyze_review_policy_experiment.py`：四策略完整配对 replay、质量—劳动联合主检验、correction lag 和 promotion audit；
- `experiments/article_quality_v2/expert_evaluation_protocol.yml`：至少8主题、三类同证据文章、整体评分与claim级双审的专家盲评设计；
- `src/citeweave/article_expert_packets.py`、`prepare_article_expert_evaluation.py` 与 `build_article_expert_packets.py`：生产条件审计、文章盲化、20-claim分层匹配和专家平衡配对分配；
- `src/citeweave/article_expert_evaluation.py` 与 `scripts/analyze_article_expert_evaluation.py`：专家面板完整性门禁、主题簇效应、一致性、不可判断率、计时和强制偏好分析；
- `src/citeweave/article_expert_collection.py`、`src/citeweave/article_expert_review_ui.py`、`scripts/prepare_article_expert_collection.py`、`scripts/serve_article_expert_review.py`与`scripts/validate_article_expert_primary_returns.py`：匿名证据查看器、整体/claim/成对偏好采集、服务端计时、不可变提交、任务完整性验证与真实分歧盲裁worklist；
- 相应单元测试和真实 pilot artifacts。

新增强控制与安全机制：

- `flat_tfidf` 强词法基线和 `flat_program` 同信息表示消融；
- 5 主题 19 项强对照面板与 24 项复杂现象面板；
- rule support gate、独立 holdout transfer gate、unrelated regression gate；
- `GuardedReviewPolicy` 的可审计自动接受/修订/人审路由，critical 永不自动化；
- 五组新面板共 352 个 factual packet、233 个 semantic packet，均已为两位独立审核者生成双盲任务包。
- 双盲 packet 已能通过本地审核服务分发；任何双审分歧都会阻断规则编译，真实审核尚未执行，当前不能报告人工一致性或节省时间。
- 审核服务已支持对抗式A/B证据、决定性证据ID、无效图算子步骤和最小改写的强校验；40个原型包审计通过，但正式2×2互补实验readiness仍因无真实候选输出和审核者而硬阻断。
- 120个v4来源相关性warmup包已通过审计并接入审核服务；真实6人roster尚不存在，因此不会把自动主题覆盖当作专家相关性标签。
- LSA 与 BM25+TF-IDF+LSA hybrid 已完成48次新增真实调用、零 parse failure；它们是 classic latent baseline，不是神经 dense。
- v3 confirmatory protocol 已在采集前冻结4个全新领域和3个主假设；真实数据采集、规范化、图扩容和100/100复杂题构造已完成，正式模型结果仍为零。
- `scripts/audit_formal_v3_readiness.py` 对协议、修正案、benchmark、结果污染、tokenizer 和 neural dense 做硬门禁；正式 runner 无 ready artifact 时拒绝执行。
- `src/citeweave/token_budget.py` 通过冻结的本地 tokenizer command 执行 probe 校验和真实 token-cap 裁剪；operator trace 不允许被静默截断。
- `scripts/probe_deepseek_v4_tokenizer_usage.py` 与冻结的六项API usage probes：验证DeepSeek V4完整消息编码边界，防止raw-text token计数假通过；
- `scripts/build_neural_dense_sidecars.py` 用固定预训练 embedding/revision 流式构建 top-k sidecar，以可恢复、带SHA的 query-independent FP32 index 跨面板复用文档编码，并以 benchmark SHA 和 item 全覆盖绑定，不修改冻结题目文件。
- `formal_v3_scale_pairs.json` 只保留三档图中语义锚点一致的14组任务；统计修正案明确披露四簇 H3 的最小精确 p 值为0.0625。
- `formal_v3_panel_plan.json` 已验证正式面板为100题、8条件、800次调用；exact tokenizer 门禁已通过，首个正式 BGE-M3 index 正在可恢复构建，尚未读取 API key、未产生 confirmatory outcomes。
- 独立 replication 已新增50,716篇规范论文、60个复杂任务与14组尺度配对；主+复制共8簇，使H3最小精确 p 值由0.0625降至0.0039。
- 复杂度扩展已在8主题上生成48个锚点匹配简单负对照，并复用120个冻结复杂题；672个逻辑单元格中仅312次为新增API调用。

下一阶段应按以下顺序推进：

1. 完成8个主题可复用 BGE-M3 FP32 index 与 `flat_neural_dense` sidecar 的物化和 SHA 审计；
2. 重新运行主实验与复杂度扩展 readiness，冻结全部 item-condition request identity；
3. readiness 转为 ready 后，运行主实验800次与replication 120次调用，并分别完成 H1–H3、R1和8簇合并统计；再按执行修正案补312次复杂度扩展新调用，合并672逻辑格检验C1–C4；
4. 增加 cross-network triangulation、阈值敏感性与 topology perturbation 作为独立扩展实验，不回填已冻结100题；
5. 为 operator outputs 生成独立 provenance IDs，并用独立实现或属性型不变量补强 corrupted-operator replay；
6. 招募至少6名真实审核者，完成每人12题且跨2领域的排除式热身校准，并在任何held-out标签揭示前冻结能力后验；
7. 从真实模型输出构建至少96个正式case，冻结standard/adversarial × random/router四格分配，完成至少50%双审和第三人裁决；
8. 比较 always-review、static-risk、raw-memory 和 validated-review-policy 四种审核策略，并检验互补团队是否超过AI-only与最佳单审核者；
9. 用至少 4 张现象卡重新生成完整文章并进行领域专家盲评。

正式论文在人审与策略实验完成前不能声称“人机互补”或“减少人类劳动”，在同证据领域专家实验完成前不能声称“文章质量超过人类”。

## 12. 序贯依赖价值信息审核（2026-09-08新增）

现有`dependency_active`只做结果前静态排序，仍不能体现审核如何在线改变系统状态。新增策略把每个文章的`PH/REF/graph-evidence/synthesis → claim → paragraph`表示为独立routing scope。一次真人双审/裁决若报告可见上游依赖无效，系统不会把这条记录追加到prompt，而是执行三步：更新依赖无效性Beta后验；把同一依赖的全部未审下游claim标记为传播修订目标；按剩余预测人工秒预算重算下一条claim的严重度加权边际覆盖价值。

实验用四个策略共享相同的依赖传播和真人标签，只改变选择机制：uniform hash、static risk、static dependency fanout和sequential dependency VOI。预算为各主题结果前预测always-review秒数的25%/50%/75%，实际服务器秒数只能事后报告，不能泄漏给当前选择。主要比较为50%预算下VOI相对最佳固定策略的主题内actionable capture差，以及每实际真人分钟捕获量差；8主题等权精确sign-flip并对两个主终点做Holm校正。未解决critical、过传播nonactionable和弃权是安全结局。

完整标签只用于隐藏回放评价；策略在第t步只能读取前t−1步自己选择并揭示的裁决。因此该实验能检验序贯策略价值，但仍不是live随机部署的因果劳动节省。真实8主题claim、预冻结耗时估计和双审/裁决返回缺失时必须fail closed。当前80条开发特征没有真人结局，只证明依赖schema与传播执行可运行。

## 13. Development-only实时VOI审核服务（2026-09-08新增）

离线回放之外，系统新增独立服务入口`serve_live_voi_review.py`，用于证明人类发现能够真正改变后续交互状态，而不只是作为下一次prompt上下文。article packet、每条claim的结果前耗时估计、逐审核者秒预算必须先生成`live_voi_config.json`并相邻SHA冻结；存在任何真人返回后禁止补建或覆盖配置。正式双审/裁决仍使用原静态服务，避免自适应呈现影响独立金标。

实时服务每次只使用同一审核者此前亲自提交的结果，构建`routing_scope × dependency`后验。选择事件记录揭示前VOI分数、预估秒数和前一事件SHA。若当前审核发现PH、REF或graph-evidence无效，未审下游claim从直接审核队列转入修订集合；已经审核的下游claim也被重新打开，避免较晚的上游证据发现无法撤销早期accept。随后剩余预算重新分配给边际覆盖价值最高的claim。任何返回顺序、配置hash、不可见依赖ID或事件链不一致都会fail closed。

当前合成端到端测试验证了“先接受、后发现共享依赖无效”的非平凡路径：1条未审claim被传播，1条已审claim被重新打开，所有受影响claim进入revision target，且只计入两次直接审核的预测预算。协议SHA256为`75a33b435782357b7e678429f9afcd7e1bc43d136fb09a25d06226e91a8233e0`，冻结时真实live VOI结果为0。因此论文当前可写“实现了可重放的在线依赖审核机制”，不能写“实时路由已降低真人成本”。
