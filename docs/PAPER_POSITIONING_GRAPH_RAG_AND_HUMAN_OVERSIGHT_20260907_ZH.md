# CiteWeave论文定位：复杂GraphRAG与能力感知人审

更新日期：2026-09-07。本文用于相关工作定位和论文叙事，不改变已经冻结的确认性实验、任务、指标或统计方案。

## 1. 文献给出的直接结论

### GraphRAG

1. **“有图”本身不是贡献。** 2025年的GraphRAG-Bench系统分析明确指出，GraphRAG在不少真实任务上会输给vanilla RAG；需要按事实检索、复杂推理、上下文总结等难度分层，才能回答“什么时候图有用”。这与本项目早期简单图解释、flat/VLM几乎打平的观察一致，而不是反常结果。
   来源：[When to use Graphs in RAG](https://arxiv.org/abs/2506.05690)

2. **必须排除答案可由单个三元组直接取回。** BRINK指出，许多KG-RAG benchmark的问题可以由现有三元组直接回答，因而无法区分推理与检索；知识不完整还会暴露模型依赖参数记忆的问题。
   来源：[What Breaks Knowledge Graph based RAG?](https://aclanthology.org/2026.eacl-long.114/)

3. **结构优势通常出现在多跳、多实体与路径连接。** GRAG联合文本视图和图视图；GNN-RAG用候选实体间最短路径提供深层图上下文，并在多跳/多实体问题上报告明显优势。这意味着仅把邻接表或图截图塞进prompt，不足以形成有竞争力的方法贡献。
   来源：[GRAG](https://aclanthology.org/2025.findings-naacl.232/)、[GNN-RAG](https://aclanthology.org/2025.findings-acl.856/)

4. **图计算和图形序列化必须拆开。** 现有工作常同时改变检索子图与prompt表示，难以判断提升来自结构计算还是额外文本。CiteWeave的`graph_program / flat_program / operator_only`同计算量机制补充正好针对这个混淆：M1检验图形序列化，M2检验原始图证据相对答案承载trace的增量价值，M3检验trace不完整时的选择性收益。

### 人机互补

1. **能力路由不是空白领域。** Learning-to-Defer已有多专家准确率—成本优化、校准和一致性理论；因此论文不能把“按历史正确率分配给专家”单独声称为新算法。
   来源：[Mastering Multiple-Expert Routing](https://proceedings.mlr.press/v267/mao25c.html)、[Learning to Defer to Multiple Experts](https://proceedings.mlr.press/v206/verma23a.html)

2. **少量校准样本刻画新专家已有先例。** Population L2D用小context set适应未见专家。CiteWeave需要强调的不是“我们也有warmup”，而是图证据审核具有问题类型、领域、证据依赖和严重度四个结构维度，并对后验下界、冲突、工作量、双审和裁决做联合约束。
   来源：[Learning to Defer to a Population](https://proceedings.mlr.press/v238/tailor24a.html)

3. **简单反馈记忆也已有先例。** ReportGPT用可验证DSL和用户选择形成few-shot反馈。因此“保存人类意见供下次prompt使用”不能成为主要创新点。CiteWeave应将创新放在可执行图证据合约、错误依赖传播、局部修订、能力感知分派和确认性劳动—质量评估的闭环。
   来源：[ReportGPT](https://aclanthology.org/2024.emnlp-industry.39/)

## 2. CiteWeave可成立的论文主张

下面三层必须同时成立，任何单层都不够：

| 层 | 可检验主张 | 当前证据设计 | 失败时必须如何降级表述 |
|---|---|---|---|
| 图任务 | 图优势随结构复杂度增加，并可能随图规模放大 | 8主题、small/medium/large、simple/complex锚定配对，C1/C2主题级检验 | 若C1失败，只能报告结构输入未转化为复杂任务收益 |
| 图机制 | 收益来自确定性图计算和可追溯原始证据，而非JSON形状或泄露答案的trace | `graph_program-flat_program-operator_only` M1/M2/M3 | 若只有graph/flat等价，承认表示不是创新；若M2失败，承认原始图证据无增量 |
| 人机互补 | 问题类型×领域能力路由在相同合格team空间内提高正确结果/人审分钟，并与双面证据呈现产生可检验交互 | 96 held-out、2×2、全局MILP、6人跨四臂、≥50%双审、盲裁、O1/O2/O3 | 若O2失败，能力模型仅作审计基础设施，不宣称效率优势 |

建议论文的一句话定位：

> CiteWeave不是把图作为另一段上下文，也不是把人类反馈作为另一条聊天记忆；它把文献图转成可执行、可反事实检验、可追溯的证据程序，并把人类审核转成经校准、受约束、可裁决、可传播修复的决策系统。

## 3. 与现有工作的真正差异

### 差异A：bibliometric graph phenomenon discovery，而非通用KGQA

CiteWeave的复杂任务不是从图中取一个实体答案，而是生成并核验：多跳连接体、桥接边反事实、社区角色对比、枢纽移除韧性、时间结构转移。这些输出连接到文献综述的“现象—证据—claim—段落”，比普通KGQA更接近科学综合，但也要求更严格地限制因果语言。

### 差异B：evidence program，而非graph-shaped prompt

系统保留原始节点/边、确定性算子、operator trace、解释合约和禁止推断。文章中的图结论必须能回放到程序和证据，而不是由LLM自由解释可视化。机制补充专门检验程序、表示和原始出处各自贡献。

### 差异C：审核结果进入依赖图，而非追加到memory

审核对象是typed claim。错误会沿`evidence → claim → paragraph`传播，只允许修改受影响且hash绑定的段落；无关段落要求字节不变。反馈只有在跨主题、跨审核者和held-out安全门禁通过后，才能升级为可复用规则。

### 差异D：multi-expert routing作为实验处理，而非默认正确

能力后验来自排除在确认性结果之外的120条来源观测和144条多维观测。正式2×2把能力路由与同一合格team空间中的冻结随机路由直接比较，并将所有首审和真实裁决时间计入成本。这样可以检验路由是否真的创造互补，而不是只展示一个看似聪明的调度器。

## 4. 审稿风险与现有防线

| 高风险质疑 | 当前防线 | 尚需真实数据证明 |
|---|---|---|
| 图只是把答案写进trace | answer-exposure audit + operator-only条件 + M2/M3 | 原始图证据F1增量与不完整trace选择性 |
| flat baseline太弱 | BM25 flat、BGE-M3 dense、flat-program同计算量三层基线 | 各复杂度/规模上的主题级配对效果 |
| VLM对比不公平 | 旧VLM只覆盖读取副标题的`network_size`，降为范围审计 | 不再把旧VLM平局解释为结构等价 |
| 96题被当作96个独立样本 | 8个dataset为推断单位，exact sign-flip，题级模型仅诊断 | 至少8主题方向的一致性 |
| 人审只是记录反馈 | 多维私有金标校准、能力后验下界、MILP、双审盲裁、依赖传播修订 | O1/O2/O3与真实劳动时间 |
| 人类结果由研究者或LLM代写 | roster、角色隔离、服务器计时、完整返回哈希、零模拟替代 | 真实6人及后续作者/专家池 |
| 文章看似流畅但信息密度低 | 同证据人类作者、长度门禁、每千词正确支持claim、overclaim/cannot-assess | 24篇文章及独立专家评分 |

## 5. 文章质量：预期的人类差距与评估重点

现阶段正式16篇机器稿尚未生成，不能直接给CiteWeave打质量分；但最新对照研究已经清楚说明应重点检查什么：

1. OpenScholar的专家研究表明，检索增强系统可能在覆盖与信息深度上优于人类答案，但代表性、时效性和citation quality仍可能落后；整体偏好也可能被流畅度掩盖，因此citation precision/recall必须与主观有用性分开评估。
   来源：[Synthesizing scientific literature with retrieval-augmented language models](https://www.nature.com/articles/s41586-025-10072-4)
2. SurveyGen在4200多篇人类综述上发现，全自动综述的主要弱点仍是citation quality和critical analysis；半自动流程可以部分追平，但不能据此假定全自动稿达到人类水平。
   来源：[SurveyGen](https://aclanthology.org/2025.emnlp-main.136/)
3. 人类与AI临床综述的直接比较报告了AI在参考文献真实性/全面性/准确性、评价深度、逻辑、创新性和总体质量上的不足。领域虽不同，但它给出了需要防范的典型失效模式。
   来源：[Cross sectional pilot study on clinical review generation](https://www.nature.com/articles/s41746-025-01535-z)
4. 长文本不能只把原子事实正确率相加；多个真实事实可能因实体混淆组合成错误段落。因而CiteWeave的专家评估必须保留段落级关系一致性与跨claim矛盾检查。
   来源：[Merging Facts, Crafting Fallacies](https://aclanthology.org/2024.findings-acl.160/)

基于这些证据，CiteWeave与人类文章的差距应拆成四类，而不是一个“总体质量”分数：

| 维度 | 机器稿可能占优 | 人类稿可能占优 | 本项目对应测量 |
|---|---|---|---|
| 覆盖 | 更快枚举更多主题与关系 | 更会选择真正重要、经典和代表性文献 | phenomenon coverage、代表性来源、遗漏关键claim |
| 证据 | 可强制每条claim绑定证据ID | 更能判断来源质量、语境和领域争议 | citation precision/recall/F1、cannot-assess、source support |
| 推理 | 可稳定执行预定义图算子 | 更擅长解释机制、权衡冲突、识别不可形式化限制 | graph consistency、causal overreach、counterevidence、逻辑评分 |
| 写作 | 结构整齐、术语一致、生成速度快 | 更有取舍、批判性、叙事弧线和原创综合 | organization、critical synthesis、innovation、每千词正确claim |

因此，最终文章结论必须同时报告：盲化整体评分、claim级裁决、每千词正确且受支持claim、overclaim、cannot-assess、引用质量、长度/密度公平性和真人写作时间。流畅度高但证据错误，或引用准确但没有批判性综合，都不能称为达到人类文章质量。

## 6. 结果出来后的论文决策树

1. C1和C2通过、M2通过：可以主张复杂度选择性、规模放大和原始图证据价值。
2. C1通过但C2失败：主张复杂任务价值，不主张随规模放大。
3. C1失败但某些复杂任务稳定为正：降级为任务类型异质性探索，不做总体GraphRAG优势主张。
4. M1等价且M2通过：最理想的机制解释是“价值来自图计算与出处，不来自图形化序列”。
5. M2失败：必须承认当前operator trace已经承载大部分有效信息，图谱在生成阶段的边际作用不足。
6. O2通过：支持能力感知审核提高正确结果/人审分钟。
7. O1通过而O2失败：双面证据界面有效，但个性化路由没有证明增益。
8. 文章A1/A2失败：不能以QA正确率代替文章质量，只能将文章模块报告为尚未达到人类写作水平。

## 7. 当前结论边界

截至本文更新，机器正式面板仍在执行，正式文章和真人结果尚不存在。因此本文只证明研究问题与实验设计在最新文献中的位置，不证明CiteWeave已经优于flat RAG、VLM或人类作者。所有效果结论必须等待冻结分析。
