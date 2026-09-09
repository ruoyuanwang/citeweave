# GraphRAG 与人类审核：论文定位和可证伪实验

更新时间：2026-09-08

## 2026-09-08稿件级补充核验

本轮把文献定位从内部设计备忘录同步进中英文稿，并新增文章质量与人机协作的直接证据。SurveyGen以4200余篇人类综述和242143条参考文献为资源，指出全自动综述仍受引用质量和批判性分析限制；因此本文将文章评价拆为声明支持、领域解释、批判性综合和整体写作，并把非同证据人类文章限定为描述性诊断。Bayesian Online Learning for Consensus Prediction说明基于后验不确定性决定是否继续询问人类也已有先例；CiteWeave不能把“顺序VOI”单独作为创新，而必须检验它与claim–evidence依赖传播、重开旧结论和局部修订约束的组合效果。Vaccaro等对106项人机实验的元分析表明，人机组合平均不优于两类单独基线中的较优者；因此正式设计必须同时保留human-only、AI-only和human–AI，而不能只比较协作系统与较弱基线。

本轮引用均重新核对官方会议、PMLR或期刊页面。这里的检索用于收紧主张，不构成“没有其他相似方法”的系统综述证明。

## 2026-08-31新增检索后的定位修正

近两年文献进一步抬高了两条主线的门槛。以下结论来自论文和会议正式页面，不把“尚未见到完全相同组合”当作新颖性证明；最终仍需系统性检索和审稿人判断。

| 已有工作已经覆盖 | 不能再作为独立创新 | CiteWeave必须额外证明 |
|---|---|---|
| [GraphRAG-Bench](https://arxiv.org/abs/2506.02404) 已覆盖领域复杂问题、数学/编程式推理以及图构造—检索—生成全链评价 | “复杂题”“不只看最终答案”“全流程评价” | 固定科学图上的注册算子、反事实、规模×复杂度选择性以及同计算量机制识别 |
| [GRAG](https://aclanthology.org/2025.findings-naacl.232/) 已联合文本视图和图视图，并在多跳图推理上比较RAG基线 | “图和文本双视图”“多跳子图检索” | `graph_program`与`flat_program`共享完全相同计算内容；`operator_only`进一步隔离答案轨迹和原始provenance |
| [HyPER](https://aclanthology.org/2025.acl-long.355/) 已学习把反馈实例路由给人或模型，并报告质量—成本混合收益 | “按成本选择人/AI反馈”“混合比单方更好” | 路由到具体审核者/审核深度，审核动作作用于claim–evidence–operator依赖图，并通过真实主题外迁移和组件promotion门 |
| [Learning to Defer to a Population](https://proceedings.mlr.press/v238/tailor24a.html) 可用少量context适配未见过的专家；[Exploiting Human-AI Dependence](https://proceedings.mlr.press/v235/wei24a.html) 已显式建模人和AI错误依赖 | “估计审核者能力”“利用人机互补错误模式” | 证明能力路由相对随机合格审核者提高正确结局/分钟，并对主题聚类做前瞻随机实验 |
| [Bayesian Online Learning for Consensus Prediction](https://proceedings.mlr.press/v238/showalter24a.html) 已按人类查询成本和后验不确定性顺序选择反馈 | “动态选择下一条人审”“后验更新” | 与依赖传播、旧结论重开和局部修订边界形成机制消融，并用真实服务器计时验证节省劳动 |
| [Just Put a Human in the Loop?](https://aclanthology.org/2025.findings-acl.1323/) 的350名标注者、7,000次标注实验显示，展示LLM建议未让人更快，却改变标签分布并可能抬高被评模型表现 | “人看过并批准，所以是独立人类金标准” | 先独立判断、隐藏条件、两侧证据随机化、双审/裁决以及机器稿与人稿的独立盲评 |
| [SurveyGen](https://aclanthology.org/2025.emnlp-main.136/) 已系统比较自动与人类指导的综述生成；[Vaccaro et al.](https://doi.org/10.1038/s41562-024-02024-1) 显示人机组合并不天然优于最好单方 | “有人指导就接近人类”“human-in-the-loop天然协同” | 同证据作者基线、独立盲评，并同时报告human-only、AI-only与human–AI质量—时间—成本 |

因此更稳健的候选贡献不是“GraphRAG + HITL”，而是：

1. **Graph computation identification：**在同模型、同token预算和同派生计算内容下，把检索、确定性图计算、序列化形式和原始provenance拆成可证伪效应。
2. **Dependency-aware oversight intervention：**人类不是给整段文本打分，而是修改source/evidence/operator/claim节点；系统传播失效范围，并把反馈编译为带留出门禁的组件变更。
3. **Cluster-valid prospective evaluation：**图效果和人审效果都以8个科学主题为独立重复，不能把同主题问题、多个审核者或重复评判伪装成独立样本。

这三个贡献必须由正式结果共同支撑。若M1/M2/M3或O1/O2/O3失败，论文应把对应部分报告为边界发现，而不是继续用“系统复杂”替代效果证据。

## 1. 文献对照后的结论

论文不能把“使用图”“多跳检索”“按风险决定是否找人”本身写成创新。这些方向已有明确先例。可成立的主线应收缩为两个可证伪命题：

1. **Graph Program 的增益具有复杂度与规模选择性。**在共享实体锚点的简单任务上不应明显优于强 neural RAG；在必须执行全图聚合、反事实和跨层连接的任务上，增益应随复杂度和图规模上升。
2. **审核编译器实现的是互补监督，而不是反馈记忆。**系统联合决定是否审核、由谁审核、需要几人，并把审核结果编译到检索器、规划器、生成器或风险路由器；只有在真实 held-out 人类实验中同时超过 AI-only 和最佳单人审核基线，才称为互补。

## 2. GraphRAG 相关工作的边界

### Microsoft GraphRAG

[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) 使用实体图、社区摘要和 map-reduce 式汇总回答全局 sensemaking 问题。它证明社区层级对全局问题有价值，但核心任务仍是 query-focused summarization。

我们的区别不能写成“能回答全局问题”，而应是：

- 题目有可执行结构答案与合法答案集合；
- 使用注册图算子而不是仅汇总社区文本；
- 检验删边、删点、阈值变化和跨网络连接；
- 每个派生事实有 operator trace 和 provenance；
- 用共享锚点简单题识别图优势是否真的来自复杂计算。

### HippoRAG 与 LightRAG

[HippoRAG](https://arxiv.org/abs/2405.14831) 用知识图谱和 Personalized PageRank 完成多跳知识整合，并报告相对迭代检索的效率优势。[LightRAG](https://arxiv.org/abs/2410.05779) 结合低层与高层检索，并支持增量更新。

因此，“图支持多跳”“图与向量混合”“图可增量更新”均不能单独作为创新。我们的有效区别是把图计算输出变成可验证的科学现象对象，并对其必要性做同预算、同锚点、同模型消融。

### 新的 GraphRAG 基准

[When to use Graphs in RAG](https://arxiv.org/abs/2506.05690) 明确指出 GraphRAG 在许多真实任务上可能弱于 vanilla RAG，并按事实检索、复杂推理、上下文总结和生成任务研究“何时图有用”。[GraphRAG-Bench](https://arxiv.org/abs/2506.02404) 进一步强调领域多跳问题、图构造、检索和推理过程的整体评估。[RAG vs. GraphRAG](https://arxiv.org/abs/2502.11371) 也显示两类方法的优势依任务而变。

这与我们的负结果一致，也提高了论文门槛：

- 必须保留简单任务上的零优势或负优势，不能只选复杂题；
- 主检验必须是 `condition × complexity × graph_scale`，而不是 GraphRAG 主效应；
- 必须区分图构造、检索、算子执行和生成四段误差；
- 必须报告强 neural dense 和 flat hybrid，而不能只和无参考/BM25 比。

2026-08-31的结果前答案暴露审计又增加一层约束：120个复杂题中，`operator_trace`对96题已经包含可确定性派生的全部答案字段，平均字段覆盖0.96。这说明`graph_program`对纯检索的优势即使成立，也主要识别“外部确定性图计算 + grounded synthesis”，不能直接称为LLM图推理。新增504-cell机制补充分别注册：M1 `graph_program`与同计算内容`flat_program`的±0.05等价检验；M2 `graph_program`相对`operator_only`的evidence F1优越与答案非劣联合门；M3轨迹不完整桥边反事实相对四类轨迹完整任务的选择性。

## 3. 人类审核相关工作的边界

### Learning to defer 与多审核者

[Learning to Defer with Limited Expert Predictions](https://arxiv.org/abs/2304.07306) 已研究在专家标签有限时估计专家能力并学习何时交给专家。[Learning to Complement and to Defer to Multiple Users](https://arxiv.org/abs/2407.07003) 更直接地联合研究 AI、自主补充、人类接管以及应邀请多少用户。[Complementarity in Human-AI Collaboration](https://arxiv.org/abs/2404.00029) 将信息不对称和能力不对称列为互补的两个主要来源。

因此，仅做风险阈值、选择审核者或双审不是足够的新意。CiteWeave 的差异必须依靠组合机制和真实结果：

- 能力按 `reviewer × domain × issue_type` 建模，而不是单一平均准确率；
- 路由对象不是分类样本，而是 claim/evidence/operator/plan 依赖图中的节点；
- 决策同时选择自动放行、单审、独立双审和第三人裁决；
- 人类动作传播到下游依赖，并被编译为组件特定训练/规则信号；
- 规则跨审核者、跨数据集、held-out transfer 和 unrelated regression 全部通过后才能启用；
- critical 永不自动化，资格不足时硬阻断。

[Hybrid Preferences](https://aclanthology.org/2025.acl-long.355/)进一步表明，“在人类反馈和AI反馈之间学习路由，并优化质量—成本混合”本身已有强先例。[Learning to Defer to a Population](https://proceedings.mlr.press/v238/tailor24a.html)也已处理少量context下对未见专家的快速适配；[Exploiting Human-AI Dependence for Learning to Defer](https://proceedings.mlr.press/v235/wei24a.html)说明人和AI错误依赖是已有理论对象。因此我们的能力后验与路由只能算必要组件，不能单独列为算法创新。真正需要实验识别的是：依赖图上的审核动作是否产生可验证的下游修复，以及组件级promotion是否在留出主题上优于raw-memory prompt而不伤害安全。

### 丰富语言反馈与可扩展监督

[Training Language Models with Language Feedback at Scale](https://arxiv.org/abs/2303.16755) 表明自然语言反馈比单纯偏好比较包含更多信息。[On scalable oversight with weak LLMs judging strong LLMs](https://arxiv.org/abs/2407.04622) 比较 debate、consultancy 和直接回答，发现 debate 的收益依任务而异，并非稳定成立。

这支持两个设计，但不预先保证结果：

- 审核返回必须包含 decisive evidence、失败类型、最小改写和理由，而不是一个 accept/reject；
- 对抗审核包把支持与反证随机放在 A/B 两侧，隐藏其角色，要求审核者先独立检查两侧再裁决。

对抗呈现是否减少 anchoring 只能通过随机实验验证，不能从 scalable-oversight 文献直接外推。

[Just Put a Human in the Loop?](https://aclanthology.org/2025.findings-acl.1323/)提供了直接的有效性警告：在其预注册实验中，看到LLM建议没有提高标注速度，却显著改变标注分布，并可能让后续模型评价看起来更好。因此CiteWeave不能把“审核者看到机器结论后点击同意”当作独立金标准。正式协议必须保留条件隐藏、独立首判、支持/反证A/B随机化、双审与第三人裁决；文章级专家评价也必须与claim修订人员分离。

2026-09-01进一步冻结了“同反馈修复机制试验”，专门回答用户提出的“高级人审是否只是把记录塞回上下文”。真实双审/裁决产生同一份反馈后，每个段落案例生成两个独立版本：`dependency_compiled`把反馈编译成证据—claim—段落依赖、允许证据集合和不可越界guard；`raw_memory_prompt`只把完全相同的反馈内容按时间顺序附入prompt。8主题×12个互不组合的段落案例×2条件共192格，同模型、温度、token预算且每格只接受一次provider响应。独立双审/裁决检验严格修复成功与旁损伤，R1/R2以8主题等权精确sign-flip并做Holm校正。这样，创新主张不再依赖“系统结构看起来复杂”，而是依赖依赖编译相对raw memory的可证伪因果对照。当前真实反馈worklist不存在，readiness正确保持blocked，不能用模拟反馈冒充结果。

2026-09-08又补上了静态主动审核的缺口。旧`dependency_active`只在审核开始前计算一次`risk × shared-evidence fanout`，随后不会因新发现的无效证据而改变队列。新增序贯依赖VOI将审核过程视为在claim–evidence图上的部分可观测决策：每次只揭示当前所选claim的真人裁决，更新依赖无效性的Beta后验；被确认无效的PH/REF/graph-evidence会把全部下游claim传播到修订集合，剩余人工预算再按严重度、缺陷先验、预期新增覆盖和结果前耗时估计重新分配。对照策略共享同一依赖传播，只消融后验更新和边际价值选择，从而不会把“有传播”和“会主动选择”混成一个效果。

该机制仍需真实数据证明。当前只有4篇开发稿的80条无标签特征，用于验证132个文章隔离依赖和最大fanout 20的schema；正式8主题、真人双审/裁决和耗时估计均未产生。离线完整标签回放只能回答“在这批已裁决claim上策略本会优先发现多少问题”，不能直接声称live部署节省劳动。

为避免论文只停留在“假设系统会在线调整”，又增加了与正式金标采集严格隔离的development-only实时审核服务。它按审核者维护私有后验，不能读取其他审核者的结论；每次提交后重算下一条claim，并用SHA链绑定结果揭示前分数。无效上游依赖不仅传播到未审claim，也会重新打开此前已审核的同依赖claim，解决晚发现证据缺陷无法追溯早期accept的问题。配置必须在任何真人返回前绑定packet hash、耗时估计和预算，实际服务时间不得用于选择。该服务目前只有合成集成测试，不加入正式人审效果结果，也不改变离线四策略的前瞻分析。

### 风险保证

[Conformal Tail Risk Control for Large Language Model Alignment](https://arxiv.org/abs/2502.20285) 研究人类评分与机器评分失配下的尾部风险控制。我们的 Wilson 上界和 critical hard rule 比完整 conformal guarantee 弱，因此论文应称为“经验安全门禁”，不能称为分布无关保证。后续若样本量足够，可增加按严重度分层的 conformal risk control 作为扩展。

## 4. 冻结实验矩阵

### GraphRAG

- 8 个独立主题；
- 100/300/2,000 节点三尺度；
- 48 个共享锚点简单负对照 + 120 个复杂任务；
- `flat_hybrid`、`flat_neural_dense`、`graph_hierarchical_retrieval_v2`、`graph_program`；
- 主检验为 Graph Program 相对 neural dense 的复杂度选择性及其尺度放大；
- 额外报告 operator-only、flat-program 和 corrupted-operator 机制消融。

### 人类审核

原冻结 replay：

- always-review；
- static-risk；
- raw-memory-prompt；
- review-compiler-active。

新增 prospective 2×2：

- standard packet × qualified random reviewer；
- adversarial packet × qualified random reviewer；
- standard packet × capability-cost router；
- adversarial packet × capability-cost router。

新增同反馈修复机制试验：

- dependency-compiled repair；
- raw-memory-prompt repair；
- 每主题12个paragraph-distinct案例，共96个配对案例、192个生成cell；
- 联合门禁为严格修复成功率优势和旁损伤率优势，成本只作次级描述。

新增序贯审核策略消融：

- uniform-hash + 相同依赖传播；
- static-risk + 相同依赖传播；
- static-dependency + 相同依赖传播；
- sequential dependency VOI：逐步更新依赖后验和剩余队列；
- 25%/50%/75%结果前预测秒预算，50%为主分析；
- 正式推断以8主题等权，而不是把160条claim当160个独立样本。

新增实时机制验证（开发性，与正式金标隔离）：

- 每名审核者只用自己的既往揭示结果更新依赖后验；
- 上游无效会传播未审claim，并重新打开早先已审claim；
- 每次选择、传播和重新打开事件形成可重放SHA链；
- 只能支持系统可执行性，必须另做随机live实验才能支持真实节省劳动。

暖启动案例只估计审核者能力，不进入正式结果。held-out 阶段按数据集、领域、错误类型和严重度分层；同一审核者不重复看到同一案例。真实 reviewer return、服务器计时和第三人裁决缺一不可。

## 5. 可发表与不可发表的表述

在正式结果前可以表述：

- 提出可执行、可审计的 bibliometric Graph Program RAG；
- 提出共享锚点的规模 × 复杂度评价设计；
- 提出 dependency-aware feedback compiler 与 capability-aware team oversight；
- 系统和协议已在结果前冻结。

只有相应结果通过后才能表述：

- Graph Program 在复杂大图上优于 neural RAG；
- 图优势具有复杂度选择性；
- 审核编译器减少人力且质量非劣；
- 审核者能力路由实现人机互补；
- 对抗证据呈现降低确认偏差；
- 依赖编译修复相对raw-memory prompt提高严格修复成功并减少旁损伤；
- 机器文章达到或超过人类专家质量。

无论结果如何都不应表述：

- 图格式天然优于扁平格式；
- 多跳任务天然证明科学发现；
- 模拟审核等价于真实专家；
- 单次 LLM judge 等价于文章质量金标准。
