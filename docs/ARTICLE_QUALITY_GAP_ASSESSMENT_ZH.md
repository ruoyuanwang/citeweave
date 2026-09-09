# CiteWeave 文章质量差距评估

更新时间：2026-09-08

## 正式文章实验的功效边界

在注册的机器文章结果产生前、同证据人类文章和专家评分均为0时，已完成不读取任何结果的前瞻功效审计。此后16/16个冻结机器请求已获得单次响应，但16/16均未通过原生成质量门禁；原结果不作事后重分类或重采样。A1每篇抽20个claim：若one-shot严格正确且受支持率为0.60，图审核带来20个百分点绝对提升时，8主题exact sign-flip在三重检验保守阈值下的估计功效为0.819；15个百分点时只有0.521。因此A1足以检测较大的claim质量提升，但对较小提升不敏感。

A2/A3的人类比较更弱：8主题下，标准化topic效应为1.0时估计功效仅0.565，达到1.5时才约0.902；增加到12个独立主题后，标准化效应1.0的功效约0.830。多名专家和每篇20个claim只能降低主题内测量误差，不能把8个主题变成更多独立复制。修正案006因此冻结表述门禁：A2失败只能说“未建立非劣”，不能说机器与人类等价，也不能仅凭未显著就断言人类更好；任何广泛的“达到或超过人类科学写作”主张必须新增至少4个独立主题并单独预注册复制。

## 最新外部证据对实验门槛的影响

[OpenScholar的Nature论文](https://www.nature.com/articles/s41586-025-10072-4)是当前最接近的强比较：108个由专家撰写的文献综述问答，另由16名博士级领域专家对机器与人类答案做coverage、relevance、organization、usefulness和pairwise preference评价，总计400余次细粒度评估。其检索增强系统在超过一半案例中优于人类答案，主要优势来自覆盖广度；但机器答案也比人类长2.0–2.4倍，作者明确把长度列为潜在混杂。这意味着“RAG文献综合超过人类”并非不可实现，也不再是新颖结论；CiteWeave必须证明图程序带来的机制性增益，并控制同证据、篇幅和作者访问，而不能只报告更长、更全。

[真实复杂医疗查询的人机报告比较](https://www.nature.com/articles/s41598-025-21689-w)则给出相反边界：20篇人类报告与56篇LLM报告中，人类报告更常满足医生预期、被认为更可靠和更专业；人类引用更多且相关性更高，研究还发现LLM存在幻觉或不忠实引用。该研究同时发现主观满意度与客观质量没有有意义的相关。因此CiteWeave不能只用总体偏好或“看起来有用”作为成功标准，必须同时核查claim支持、引用忠实、来源相关性和领域效用。

[OpenScholar](https://www.nature.com/articles/s41586-025-10072-4)与[FrontierScience Bench](https://aclanthology.org/2025.realm-1.31/)共同说明“文献综合回答”和“生成可信研究方法”是不同能力；后者在88篇遮蔽方法章节的论文上仍发现强模型难以恢复方法细节。CiteWeave当前只生成基于冻结证据的综述/科学景观文章，不应包装成从零完成原创研究或方法设计。

[科学摘要盲评研究](https://www.nature.com/articles/s41746-023-00819-6)发现，人工评审认为被识别出的机器摘要往往表浅、模糊，且模型可生成看似可信但虚构的数据。我们的冻结结构答案和证据ID可以减少凭空造数，却不能自动解决空泛、公式化或领域解释不足；这些维度仍需真人逐段评价。

[Mind the Blind Spots](https://aclanthology.org/2025.emnlp-main.1805/)发现现成LLM审稿器相对人类专家明显忽视novelty维度。因此旧单一LLM评分只能作诊断，不能承担“文章达到人类论文质量”或“创新性更强”的主张。

## 可防守的结论

当前生成文章已经明显超出“把图解释成一段话”的版本：单篇开发样例为3,576词，五类图现象全部进入 Results，Discussion 进行了跨现象综合，并带算子级 provenance、替代解释和限制。新完成的8主题同证据机器实验进一步显示，层级解析纠偏后两条件16/16稿都在Results和Discussion中使用全部五类现象；CiteWeave 8/8稿命中全部三类预注册跨现象综合，而one-shot只有4/8达到至少两类。这是结构覆盖诊断，不是论证正确性或文章优越性的真人证据。

正式生成同时暴露了严重失败：两条件16/16稿都超过2700–3300词范围；CiteWeave 8/8达到8000 completion-token上限，6/8缺完整Conclusion、2/8同时缺Limitations，另有1稿出现截断的未注册PH标识。原门禁还存在一个已冻结保留的解析缺陷：它在`###`子标题处终止`## Results/Discussion`，导致把真实存在的内容误报为缺失。后生成修正案只做层级边界纠偏诊断，不覆盖原assessment、不改变confirmatory失败、不释放真人claim包。综合结论是：图蓝图显著加强了跨现象组织，但当前prompt/输出预算使文章过长且更易截断。

结果后的V3开发实验进一步确认，继续微调一个全局字数prompt不能解决问题：第二轮单次检查点已让4/4稿的章节、五类现象、三类综合和证据ID全部通过，但仍有2/4稿低于2700词。图依赖文章编译器按章节DAG分节生成；配合一次Results/Discussion完整性修复和确定性句子预算器后，早期两主题开发稿达到3240/3217词并通过全部机器门禁。预算器只删除无PH/REF的完整句子并记录删除哈希。

随后完成的matched compiler开发v1把人工审核接口直接加入生成契约，并设置同调用量flat控制。四篇稿全部达到23–38条独立可审核claim，4/4通过readiness；两个主题中graph相对flat的候选claim分别多15和3条。但36次分节调用有9次触及token上限，严格整体验收0/4，两篇稿出现半截证据标识；因此它只证明“可审核对象密度”问题已在开发样本中解决，尚未证明graph条件的文章质量优势。只提高token上限的下一轮在0个有效响应时因HTTP 402停止。

但目前不能得出“文章优于人类”的结论。8主题旧审计使用的是单一 LLM 评分者，机器稿与已发表人类稿使用不同语料、不同篇幅和不同任务；开发样例与人类参考的单主题比较也仍是一次 LLM 预审。目前只有16篇机器单次响应，且全部未通过冻结门禁；8篇同证据人类文章、可进入盲评的24篇合格稿和领域专家评分仍不存在，readiness 正确地保持 `blocked`。

## 非LLM的文章表面诊断

为避免继续依赖LLM judge，先对matched compiler v1的4篇开发稿和8篇已缓存人类参考论文做了纯确定性v1诊断，指标实现、输入路径与SHA均写入`experiments/article_quality_v3/deterministic_machine_human_quality_diagnostic_v1.json`。复核代码后发现，v1虽声明比较Abstract、Results、Discussion与Conclusion，实际上只对机器稿调用了统一章节抽取，对人类缓存使用了整个`reference_report.md`。因此下述v1字数、段落CV和数值密度只能描述各自缓存文本，不能视为严格统一章节比较。

最明显的新发现不是机器文章“更会写”，而是结构过度规则：机器段落长度变异系数均值只有0.15，人类参考为0.71；平均段落长度却相近（136.58 vs 141.07词）。这说明差异来自段落节奏和展开层次异常均匀，而不是简单的段落偏短。机器稿平均2048.75词，人类参考平均4800词；两者平均句长较接近（27.13 vs 25.26词），词汇多样性MSTTR-100也接近（0.74 vs 0.71）。因此“机器只会短句、词汇贫乏”不是当前数据支持的主要差距，篇幅、段落层级和论证展开不足更关键。

证据表达也出现双向偏差。机器稿的校准标记为6.52/千词，高于人类参考的2.15/千词，说明系统确实更频繁显式写出may/could/cannot/alternative/limitation；但数值密度只有18.22/千词，低于人类的48.38/千词。开发graph稿相对flat稿更长（2131.5 vs 1966词），解释风险标记更高（3.46 vs 1.53/千词），校准标记也更高（8.02 vs 5.03/千词）。这提示显式依赖图同时放大了“解释冲动”和“自我限定”，并未自动增加领域事实密度；下一轮应优化每个解释claim的来源锚定与反证质量，而不能只增加谨慎措辞。

精确重复句率整体很低，近重复段落对在该阈值下为0，故“逐字复读”不是主要问题；更可能的问题是五类现象以不同措辞反复覆盖、段落模板过度一致。后续真人量表应单独评价semantic redundancy和argument progression，不能用字面去重率代替。

在读取新结果前另行冻结v2论证结构诊断协议，固定12个输入及实现SHA，并真正对两组文本都只抽取Abstract、Results、Discussion和Conclusion。结果揭示缓存人类参考本身不适合作为matched基线：8篇中4篇在这一视图里只有Abstract，另外多篇缺Results或Conclusion。因而章节熵（机器0.794、人类0.286）和最大章节占比（0.482 vs 0.879）主要反映缓存章节缺失，而非写作质量，不能用于优劣判断。

仍可用于生成器调试的信号是模板化表面特征。机器稿六内容词段首重复率为0.117，人类缓存为0.023；最大段落词频余弦相似度为0.780 vs 0.593；通用脚手架词为6.07 vs 2.34/千词。两篇graph稿的段首重复率均值0.200，高于flat的0.033，说明依赖图编译器更容易用相同PH起句和固定论证槽位。机器稿对照词、替代解释词和失败条件词更多，但机制词只有1.64/千词，人类缓存为2.77/千词；这进一步支持“批判性措辞充分、领域机制仍浅”的诊断。由于主题、来源、引用语法和章节可得性均不匹配，这些数值不作显著性检验，也不进入论文的人机质量主结论。

## 已观测的优势

旧8主题盲化 LLM 诊断中，机器稿在方法透明（5.00 vs 1.88）、认识论校准（4.38 vs 2.38）、论证连贯（4.25 vs 2.50）和图派生洞见（3.25 vs 2.38）上更高。该结果主要反映系统化结构：每个数字和图结论有显式依赖，反事实与限制不是作者临时补写。

开发样例进一步把图能力从中心性/频次描述扩展为五种可执行问题：跨社区多跳、桥边删除反事实、社区角色对照、枢纽删除韧性和时间—结构联合。这些结果被组织成“冗余连接而非单点脆弱”的跨现象论证，而不是五个互不相关的答案。

## 与成熟人类论文的主要差距

1. **领域解释仍浅。** 旧8主题诊断的领域具体性为2.63，低于人类的3.75。开发样例只实际使用4篇领域文献，主要用综述摘要为图结果配背景，尚未深入连接具体应用、争议、机制和反例。
2. **结构证据被语言放大。** “maturing field”“field in transition”“dominant core”等表述超过了单一关键词共现图可直接支持的范围。即便后文加入限定，这类框架词仍会引导读者作阶段性或实质性解释。
3. **主稿仍缺少稳健性论证，但已有真实补充数据。** 原文章只有局限性声明。2026-08-27已完成8主题×18设置的独立分析：72组拓扑扰动中路径距离69组不变，但社区seed变动下“对外联系角色”所选社区的成员Jaccard均值只有0.337，时间窗口变化使5/16次的新兴关键词改变。这说明数字稳定不等于角色/趋势解释稳定。新增结果未注入已冻结的同证据主稿，也不能替代真人盲评。详见`docs/TEMPORAL_PARITY_AND_GRAPH_ROBUSTNESS_ZH.md`。
4. **图像与图结构证据需要分离评价。** 开发文章的核心证据来自结构化现象卡，而不是渲染 Figure。正式文章实验现已前瞻性规定三类作者都不看 PNG，只接收相同的结构化图记录；冻结图在草稿完成后原样附入盲审包供专家复核。这样不会把 VLM 看图能力误当作 GraphRAG 能力，但视觉异常定位仍需在独立模态实验中验证。
5. **重复多于扩展。** 相同的五个数字在 Abstract、Results、Discussion、Conclusion 和 Provenance Map 多次出现。篇幅达到论文长度，但新增领域解释和反证的速度低于文字增长速度。
6. **文献覆盖存在选择偏差。** 初始代表来源按答案关键词和高被引度抽取，实际把 `Chemistry`、`Biology`、`Computer science` 等泛节点当作相关性证据。该问题已通过三次结果前版本化修正，v4改为主题词覆盖优先、任务BM25次序、引用量不参与排序且规范化题名跨现象去重；结构审计已通过。但这仍是检索启发式，不等于领域专家确认每篇来源真正支持相应图解释。
7. **没有真实同行评议行为。** 当前没有领域专家的 claim 级 supported/correct/overclaim 判断，也没有作者如何响应审稿意见、修改是否引入新错误的数据。
8. **覆盖广度与篇幅混杂已设门槛，但尚无正式文章数据。** OpenScholar的人机比较显示，机器答案更长可影响偏好。2026-09-01冻结的修正004要求每个主题最长/最短稿不超过1.10倍，三条件跨主题平均字数最长/最短不超过1.05倍；盲评前记录证据、候选claim和重复claim密度，盲评后报告单位千词的已裁决正确且受支持claim、overclaim和cannot-assess。该控制在任何正式文章和专家结果出现前冻结；当前24篇文章仍缺失，因此只能证明设计已补齐，不能证明长度混杂在真实数据中已经消失。
9. **研究效用与写作流畅度仍混在一起。** 组织性、语言流畅和引用格式容易被机器门禁优化；真正难的是来源是否支持具体解释、是否遗漏关键争议、结论是否帮助研究者改变判断。专家量表必须把这两层分开。
10. **不是原创研究能力。** 当前文章根据已完成的图计算与文献包做综合，没有独立提出并验证新实验方法。即使专家偏好超过人类同证据稿，也只能支持“可审计文献综合”，不能支持“自动科学发现”或“完整论文作者替代”。

## 下一版文章必须满足的生成约束

- Results 至少包含四类复杂图操作，并把每个结论绑定到算子轨迹和原始文献证据；禁止只解释渲染图。
- 至少形成两个跨现象命题，每个命题必须同时给支持证据、反证或替代解释、以及在何种图构造下会失效。
- 机器门禁必须拒绝“只把多个PH标识堆进一段”的伪综合：五类现象全部进入Results，至少四类进入Discussion、三类进入Conclusion；至少两组预注册现象配对分布在两个不同的Results/Discussion正文段落。该规则只做结构筛查，不能替代专家对论证有效性的判断。
- 至少运行一组图构造敏感性和一组时间/社区划分敏感性；不能用一段 limitations 代替稳健性实验。
- 正式文章比较中，作者不得接触渲染 Figure；正文中的图结论必须来自结构化算子记录。冻结 Figure 在草稿后统一附入三种条件的盲审包。视觉—结构不一致案例另设 VLM 模态实验，不能替代图算子计算，也不进入文章质量主对比。
- 每个主要领域解释至少连接两篇内容直接相关的来源，其中至少一篇不是高被引综述；fallback 来源必须单独标记并允许审核者拒绝。
- 120个现象—来源关系先完成人类双层判断（topic relevance与evidence role）；无关来源训练reranker hard negative，反证来源必须作为counterevidence保留，不能被清洗掉。
- 删除无法由结构证据直接支持的阶段性词汇，或把它们改写成可证伪的候选解释。
- 同证据专家实验必须以研究效用非劣和可追溯性优势共同成立为成功条件；领域具体性若明显下降，即使 provenance 更强也不能称为优于人类。
- 作者与评价者必须分离，同一专家不得评价自己撰写的human-same-evidence稿；机器稿、人类稿和one-shot稿隐藏来源并随机顺序。
- 同时报告篇幅、引用数、有效且受支持的原子claim/千词、重复claim率；盲评前必须通过主题内1.10与条件均值1.05的长度比硬门槛。失败稿在原条件下修改并重新冻结，禁止看到评分后截断正文。
- 总体偏好之外，预注册至少四个独立门：claim支持/引用忠实、领域具体性、跨证据综合、研究效用；organization和fluency只能作为次级写作指标。
- LLM judge只用于发现格式错误和准备人工包；novelty、领域效用、科学过度推断与最终人机比较必须由冲突筛查后的真人专家判断。

## 证据位置

- 8主题旧诊断：`experiments/article_quality_v2/blind_audit_20260820/summary.json`
- 非LLM文章表面诊断：`experiments/article_quality_v3/deterministic_machine_human_quality_diagnostic_v1.json`
- 复杂现象开发文章：`experiments/phenomenon_article_v2/digital_twins_healthcare_2012_2024/article.md`
- 文章机器门禁：`experiments/phenomenon_article_v2/digital_twins_healthcare_2012_2024/verification.json`
- 16篇冻结机器响应：`experiments/article_quality_v2/machine_generation_v1/`
- 层级解析纠偏诊断：`experiments/article_quality_v2/machine_generation_v1/secondary_hierarchical_parser_diagnostic.json`
- 后生成诊断修正案：`experiments/article_quality_v2/article_generation_diagnostic_amendment_001_hierarchical_section_parser.yml`
- V3文章编译开发结果：`docs/ARTICLE_GENERATION_V3_DEVELOPMENT_RESULTS_ZH.md`
- 同证据候选正式v4包：`experiments/article_quality_v2/same_evidence_writer_packs_v4/manifest.json`
- 无图像泄漏的正式作者输入：`experiments/article_quality_v2/same_evidence_text_writer_inputs_v1/manifest.json`
- 作者访问修正与冻结：`experiments/article_quality_v2/expert_evaluation_amendment_002_text_only_writer_access.yml`
- 同证据正式协议：`experiments/article_quality_v2/expert_evaluation_protocol.yml`
- 8主题精确配对主分析修正：`experiments/article_quality_v2/expert_evaluation_analysis_amendment_003.yml`
- 长度公平性与信息密度修正：`experiments/article_quality_v2/expert_evaluation_amendment_004_length_parity_and_density.yml`
- 前瞻功效与结论范围修正：`experiments/article_quality_v2/expert_evaluation_amendment_006_design_sensitivity_and_claim_scope.yml`
- 前瞻功效审计：`experiments/article_quality_v2/article_design_sensitivity_20260908.json`
- 正式文章/专家当前缺口：`experiments/article_quality_v2/expert_packet_intake/readiness.json`
- 来源相关性人审当前缺口：`experiments/human_review_v2/source_relevance_warmup_v4/readiness.json`
