# CiteWeave 面向 IP&M 的论文结构设计

版本：2026-08-27，工作稿 v0.1。先固定论证结构，再撰写英文前置章节。本文档不改变任何已冻结实验。

## 1. 期刊定位与核验边界

**直接回答“是否按照期刊要求”：目前不是逐条对照现行作者指南完成的合规稿，而是参考期刊定位和已发表文章设计的研究论文结构。** 2026-08-27 再次访问官方作者指南仍返回 403，未取得新的指南正文。

| 判断层次 | 已有依据 | 本稿如何使用 |
|---|---|---|
| 期刊范围 | 已读取 Elsevier 官方期刊介绍 | 按计算与信息科学交叉领域的方法及系统研究定位 |
| 已发表论文的组织方式 | 已核对 Chung 等（2025）、PaperEval（2025）和 Wei 与 Chen（2026）的可见引言与章节说明 | 参考相关工作、方法、评价、讨论、结论的组织逻辑；具体对照见 `05_IPM_WRITING_COMPARISON_ZH.md` |
| 本项目的目录选择 | 编辑建议，不是已核实的期刊条款 | 八节目录、单列任务定义、实验设计与结果分开，均可根据篇幅调整 |
| 投稿硬性规范 | 尚未取得当前官方作者指南正文 | 摘要上限、Highlights、匿名方式、格式和提交材料等仍待核对 |

Elsevier 官方期刊介绍将 IP&M 定位在计算与信息科学的交叉领域，接收研究、方法、综述和系统设计相关的 critical application 稿件。CiteWeave 应按具有方法贡献和实证评价的原创研究组织，不是综述论文，也不是软件使用说明。[官方期刊定位](https://shop.elsevier.com/journals/information-processing-and-management/0306-4573)

结构参考来自已发表论文，而非声称期刊强制统一目录：Chung 等（2025）的计算型文献综述论文采用引言、相关工作、数据与方法、案例研究、讨论、结论的组织方式。它提示我们将信息分析任务、方法和学术/实践意义清晰分开。[IP&M 论文](https://www.sciencedirect.com/science/article/pii/S0306457325001864)

**访问限制：** 本次官方 Guide for Authors 返回 403，浏览器也未取得指南正文。因此，当前不能确认摘要字数、关键词数量、Highlights 是否必交、审稿匿名方式、正式参考文献格式或篇幅上限。第三方模板、旧版作者手册和仿冒域名均未作为现行要求依据。下列目录和篇幅都是本项目的编辑建议，不是期刊硬性规定。[待核对的官方指南](https://www.sciencedirect.com/journal/information-processing-and-management/publish/guide-for-authors)

## 2. 论文定位

暂定英文标题：**CiteWeave: Auditable bibliometric synthesis with executable graph evidence and dependency-aware review**

中文工作标题：**CiteWeave：融合可执行图证据与依赖感知审核的可审计文献计量综合**

核心叙事：文献计量研究由研究者串联数据获取、分析、解释和写作；自动化逐步覆盖计算、检索和长文生成，但这些环节之间的证据对应与纠错关系仍需明确建模和评价。CiteWeave 将计算证据、研究声明与审核修订组织成可检查的处理流程。

任务边界：研究语料中的领域结构和演化，不泛指所有系统综述、元分析或自主科学发现。图上的删除实验是结构扰动，不等于现实世界的因果干预。审计记录可帮助发现错误，但不保证数据和解释天然正确。

## 3. 完整目录与章节职责

| 章节 | 内容与职责 | 建议英文词数 | 本轮状态 |
|---|---|---:|---|
| Abstract / Keywords | 问题、方法、实际主要结果和适用边界；最后完成 | 暂以约 200–250 词规划，非已核实上限 | 不编写结果型摘要 |
| 1. Introduction | 传统流程 → 自动化演进 → 剩余接口问题 → CiteWeave → 贡献 | 750–950 | 起草 |
| 2. Related work | 将文献按研究流程和自动化能力组织，不按国内/国外重复分组 | 900–1,200 | 起草 |
| 3. Task formulation and research questions | 输入、输出、两类图、质量维度、三个 RQ | 450–650 | 起草 |
| 4. The CiteWeave framework | 解释系统如何逐项回应问题，而非罗列模块 | 1,300–1,700 | 详细提纲 |
| 5. Evaluation design | 数据、冻结协议、条件、预算、人工实验、统计和伦理 | 1,200–1,600 | 详细提纲 |
| 6. Results | 按 RQ 报告全部条件、失败和不确定性 | 1,000–1,400 | 等待真实结果 |
| 7. Discussion | 机制解释、信息处理意义、实践意义和有效性边界 | 800–1,100 | 提纲，等待结果 |
| 8. Conclusion | 回答 RQ，限定结论范围 | 180–250 | 等待结果 |

篇幅是可调整的编辑预算。正文预计约 6,600–8,900 英文词，不含参考文献；这不是 IP&M 字数限制。

### 1. Introduction

1. 文献计量研究服务于领域结构、合作关系和主题变化的理解。
2. 传统研究者使用数据库、清洗脚本和分析工具，人工完成方法选择与解释整合；承认既有工具已实现实质自动化。
3. 文本检索与科研问答进一步自动获取和综合文献证据。
4. 长文智能体自动完成提纲、分节写作、整合和审阅；承认这些不是我们的首创。
5. 留下的任务是维持数据、结构计算、声明和纠错之间的对应关系；用一个桥接判断贯穿解释。
6. 介绍 CiteWeave，说明计算证据、受约束综合和依赖感知审核如何相互衔接。
7. 列出方法与评价设计贡献，不写尚未观测的效果。

### 2. Related work

- **2.1 Bibliometric workflows and computational literature review**：CiteSpace、VOSviewer、bibliometrix、方法指南，以及 IP&M 的计算型文献综述工作。不能说传统工具只能画图或没有完整工作流。
- **2.2 Retrieval-based scholarly synthesis and automated review writing**：PaperQA2、OpenScholar、STORM、AutoSurvey、LiRA。区分问答、百科式长文、科学综述与本文的文献计量综合任务。
- **2.3 Graph-based retrieval and executable evidence**：GraphRAG、任务依赖的图收益、程序辅助计算。区分检索、摘要、执行和语言表达，不把 JSON 格式或调用图算法当作首创。
- **2.4 Review, feedback, and human–AI collaboration**：机器审阅、人类转交与多审核者协作。研究差异落在依赖对象、修订范围和留出验证，而非仅有“人参与”。
- **2.5 Positioning of this study**：用功能关注点表总结已报道能力；未核对全文的功能记为“未在本次阅读中确认”，不画武断的缺失叉号。

不设独立长篇 Background。传统研究背景放在 1 和 2.1，方法所需定义放在 3。

### 3. Task formulation and research questions

- **3.1 Task and scope**：协议、语料、构图参数、来源摘录与写作任务作为输入；带证据链接的领域综合及审计记录作为输出。
- **3.2 Relational graphs and evidence dependencies**：科学关系图描述作者/机构/主题关系；依赖图描述 source/evidence/operator/claim/paragraph 的派生关系。两者不能混为同一种图。
- **3.3 Research questions**：采用下表，作为论文组织标签，不替换冻结假设 ID。

| RQ | 问题 | 既有实验对应 | 不能预设的答案 |
|---|---|---|---|
| RQ1 | 图程序在什么任务复杂度与规模下改善结构答案及其证据支持？ | 主实验 H1–H4/N1、独立复制、复杂度扩展 C1–C4 | 不能预设图优于 dense，或任何简单题差异都等于机制成功 |
| RQ2 | 依赖感知审核能否有效修正受影响声明，并在质量约束下改善审核资源使用？ | 真人 claim 双审/裁决、pre/post 独立评价、审核策略回放、来源检索器晋升 | 回放不等于因果劳动节省；零真人不能报告人机互补 |
| RQ3 | 计算证据与审核流程能否改善同证据文章的支持度、可追溯性和研究效用？ | 8 主题 × 3 条件文章；A1/A2/A3 与领域具体性门禁 | 质量非劣不等于质量更高；追溯性优势不等于全面超过人类 |

### 4. The CiteWeave framework

- 4.1 Protocol-driven acquisition and canonical data：数据来源、范围、身份与分析资格。
- 4.2 Executable graph evidence：注册算子、参数、输出、证据 ID、执行验证；明确当前任务程序预先规定，不声称自由问题规划已验证。
- 4.3 Evidence-conditioned synthesis：现象卡、跨现象组织、解释边界与声明绑定。
- 4.4 Dependency-aware review and controlled revision：上游反馈如何传播；仅修改绑定段落；限定显式依赖覆盖。
- 4.5 Feedback validation and audit controls：规则支持、留出迁移、回归门禁与审核者资格。不是模型在线微调的同义词。

### 5. Evaluation design

- 5.1 Datasets, graph scope, and prospective amendments。
- 5.2 Graph reasoning conditions and computational budgets。
- 5.3 Independent replication and matched complexity–scale analysis。
- 5.4 Real-reviewer study and controlled revision assessment。
- 5.5 Same-evidence article assessment。
- 5.6 Outcomes, inference, failures, and reproducibility。
- 5.7 Human-participant ethics, consent, compensation, and conflicts：仅按真实情况填写，不能推定伦理已获批准。

主实验 800、复制 120、扩展新增 312 次请求与 672 个扩展逻辑格必须区别。不能把复用格当独立新样本。统计以冻结协议及最新修正案为准。

**必须披露的机制边界：** 当前 graph_program 上下文由注册算子轨迹和支持证据装配而成，代码保留 oracle-support 来源。它评估给定执行证据下的综合能力，不构成自由规划或端到端非 oracle 检索的证明。独立区分 query/hierarchical retrieval 与 program 条件；加入非图脚本分析工具对照是否需要，属于提交前研究设计讨论，不能在看结果后悄悄改冻结实验。

### 6–8. Results / Discussion / Conclusion

结果按 RQ1、RQ2、RQ3 排序，并单列数据完整性和失败；讨论再说明机制、研究效用和适用范围，不重复结果表。保留简单任务无优势、强基线更好或人审无收益的可能结果。

作者身份污染适合作为数据完整性案例，而非替代主实验。当前 Large 是最多 2,000 候选节点的注册图；未截断图只在实际运行并核验的分析中单独报告。

## 4. 图表与补充材料规划

- Figure 1：传统工作流与 CiteWeave 的证据/反馈流；不是产品 UI 截图。
- Figure 2：同一研究判断的计算、声明和审核依赖示例。
- Table 1：相关研究的已报道关注点与本文问题。
- Table 2：数据规模、构图范围、语料角色与身份修复记录。
- Table 3：实验条件、信息访问、计算工具、预算及假设对应。
- Figure 3：复杂度 × 规模效应和区间；有真实结果后生成。
- Figure 4 / Table 4：真人审核质量与时间、文章专家评价；不得用模拟数据占位成真实图。
- Supplement：冻结/修正案索引、详细算子定义、提示词、审核表、重试与缺失、完整逐主题结果。

## 5. 当前可写与待写内容

本轮只写英文第 1–3 节及参考文献。第 4–8 节保留写作任务，不制造结果、专家人数、伦理审批或数据公开承诺。

最新完整机器实验、人审和文章结局均未在本稿中使用。旧实验和开发 pilot 仅用于作者理解动机，不混入正式结果，不在引言中挑选最好数字。

如果真人实验无法完成，须重新评估论文范围：可保留人审接口的设计描述，但不能把互补监督和人工节省作为已验证贡献。

## 6. 投稿前核对

1. 通过可访问的官方 Guide for Authors 核对当前格式及投稿类型；使用作者年引用只是此稿编辑选择，尚非指南核验结论。
2. 完成最接近系统的全文对照，不将“本文关注不同”写成“别人没有”。
3. 将预印本版本与后续正式出版版本核对，更新引用信息。
4. 主文和附件去除本地路径、身份信息、密钥和会泄露匿名作者的链接；是否需要匿名按现行指南确定。
5. 如实写研究中的模型使用和论文准备中的 AI 使用，两者分别说明。Elsevier 当前期刊 AI 政策允许在作者控制下辅助研究整理和写作，要求核验与披露；本草稿不能自动视为作者已经审核。[期刊 AI 政策](https://www.elsevier.com/en-au/about/policies-and-standards/generative-ai-policies-for-journals)
6. 本次检索见 SocLitGen（DOI 10.1016/j.ipm.2026.104885），页面卷期为 2026 年 11 月；未核实首次在线日期，暂列相关候选，不写入截至 2026-08-27 的已发表工作或作为既成的期刊结构依据。
