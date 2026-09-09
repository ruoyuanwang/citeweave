# CiteWeave：融合可执行图证据与依赖感知审核的可审计文献计量综合

> 工作稿 v0.4，2026-09-08。RQ1图实验已完成并写入；RQ2真人结果和RQ3同证据人类文章比较尚未产生，因此当前不可投稿，也不主张已经实现人机互补或超过人类写作。相关工作对照已经扩展，但覆盖完整性仍需作者核验。

## 1. 引言

理解一个研究领域，不仅需要找到与问题相关的论文，还需要分析研究主题如何发展、不同研究群体如何合作，以及不同知识分支之间如何形成联系。文献计量分析通过发表数量等指标和学术记录之间的关系分析，为回答这些问题提供了定量方法。然而，文献计量研究并不是机械地生成一组图表，而是涉及研究问题、数据覆盖范围、分析技术与结果解释等一系列方法选择（Donthu et al., 2021）。本文关注文献计量综合，即结合计算得到的文献计量证据和相关原始文献，形成对某一研究领域的文字分析。

在传统流程中，研究者首先制定检索策略，从学术数据库获取记录，处理数据不一致问题，随后开展统计和网络分析，阅读代表性文献，并将不同分析结果整合为研究报告。已有工具已经实现了其中大量环节的自动化。例如，CiteSpace 支持科学文献时间模式的分析，VOSviewer 支持文献计量图谱的构建与探索，bibliometrix 则将数据处理和科学知识图谱分析组织为可扩展的分析流程（Aria & Cuccurullo, 2017；Chen, 2006；van Eck & Waltman, 2010）。这些工具使分析过程日益可计算、可执行，但研究者仍需判断：某项分析结果究竟能够支持怎样的文字解释，以及分析发生变化后，哪些解释也应随之修改。

自动化随后进一步扩展到文献检索与内容综合。PaperQA2 将文献搜索、带引用的内容总结和矛盾检测组织为智能体工作流程；OpenScholar 将科学文献检索、带引用的回答和迭代反馈相结合（Asai et al., 2024；Skarlinski et al., 2024）。面向长文生成的系统则开始自动组织整篇文章：AutoSurvey 协调检索、提纲构建、分节写作和修订，LiRA 将提纲、写作、编辑与审核交给不同的专门智能体（Go et al., 2026；Wang et al., 2024）。因此，值得进一步研究的问题不仅是系统能否检索文献并生成结构完整的综述，还包括生成的论述如何持续对应其所依赖的证据及数据处理过程。

这一问题在涉及网络结构的解释中尤为具体。例如，若要判断“某一主题连接了原本相互分离的研究群体”，就需要明确语料范围、图的构建方式、“连接”的操作性定义，以及相应的计算方法。若进一步判断网络是否依赖某个桥接节点，则可能需要在删除节点或边后重新计算网络结构。相关文献片段可以帮助解释计算结果，但仅仅检索到这些片段，并不能证明相应的图结构性质。反过来，即使图计算正确，也不能据此直接确定主题的实质含义，更不能自动得出科学发展过程中的因果解释。计算依据和领域解释需要相互衔接，同时保留各自的适用边界。

基于图的检索和程序辅助推理为解决这一问题提供了互补基础。GraphRAG 利用图结构和社区摘要支持语料集合层面的综合，程序辅助语言模型则将计算交给可执行程序完成（Edge et al., 2024；Gao et al., 2023）。不过，系统使用了图或调用了工具，并不意味着它会在所有任务上获得更好的表现。已有 GraphRAG 比较研究明确考察了不同任务条件下图方法的收益（Xiang et al., 2025）。对于文献计量综合，这意味着应当区分相关证据的检索、结构分析的执行以及结果的语言表达，也需要同时考察简单查询、聚合计算和结构扰动等不同复杂度的任务。

审核与修订提出了另一项要求。审核者可能发现实体识别错误、来源不适用、计算无效，或解释超出了证据范围。这类反馈的影响可能不止于当前被审核的句子。如果多个判断依赖同一项证据，仅修改当前句子，就可能使报告其他部分保留不一致的内容。因此，本文将审核视为对相互关联的研究对象进行检查和修改的过程，并从纠错质量与审核投入两个方面考察其作用。

为此，本文提出 CiteWeave，将研究方案驱动的文献计量处理、可执行图证据、受证据约束的写作与依赖感知审核连接起来。系统通过预先登记的图操作产生可检查的计算结果，并将这些结果与文献来源及适用限制共同组织起来，用于形成有明确证据边界的解释和文章。审核操作与具体证据和研究声明关联，使系统能够识别受影响的段落并实施受控修订。候选反馈规则在应用于原案例之外的任务前，还需通过支持性检查与留出验证。该框架的目标是使研究解释的形成依据和修订过程可以被检查，而不是依靠图指标替代研究者的学术判断。

围绕上述问题，本文开展以下三个方面的研究：

（1）构建计算与解释之间的证据关联机制，将文献计量计算结果组织为可检查的证据对象，并与来源和文字声明关联，使结构性解释的计算依据及适用范围能够被追溯。

（2）设计依赖感知的审核与修订机制，将审核决定关联到具体研究对象，依据已记录的依赖关系识别受影响内容，并区分局部纠正与需要留出验证的可复用反馈。

（3）建立分层评价方案，分别考察结构任务表现、审核辅助纠错和文章研究效用，分析方法在不同任务复杂度、图规模与审核投入下的效果及适用条件。

## 2. 相关工作

### 2.1. 文献计量流程与计算型文献综述

文献计量研究既包括衡量研究产出的指标，也包括对论文、作者、机构和术语之间关系的分析。已有方法指南强调，应根据研究目的选择分析技术，并结合具体研究领域解释结果（Donthu et al., 2021）。因此，完成一项计算与说明该计算如何帮助理解一个领域，是相互关联但并不等同的任务。

科学知识图谱工具已覆盖上述流程中的重要环节。CiteSpace 通过科学文献的时间分析考察变化与新兴模式（Chen, 2006）；VOSviewer 侧重文献计量图谱的构建与探索（van Eck & Waltman, 2010）；bibliometrix 提供集成的分析流程，并可通过 R 生态进行扩展（Aria & Cuccurullo, 2017）。这些工作为分析自动化提供了基础。本文进一步关注分析结果与自动生成的研究声明及后续纠错之间的明确关联。

计算型文献综述的研究也不限于传统文献计量网络。Chung 等（2025）将面向特定领域、随时间更新的语言表示与聚类和新兴性指标相结合，用于分析人工智能研究的演化。该研究展示了如何利用计算分析形成对研究领域的实质性解释。CiteWeave 进一步关注分析结果、来源材料与文字解释之间的支持关系，以及证据纠正对后续依赖项的影响。

### 2.2. 基于检索的学术综合与自动综述写作

科学文献检索系统已从返回相关论文列表，发展为能够进一步综合内容的系统。PaperQA2 在文献问答、带引用的主题总结和矛盾检测等任务上评价智能体，并进行了与人类研究者的比较（Skarlinski et al., 2024）。OpenScholar 将科学文献片段集合、检索机制和反馈驱动的生成过程结合起来，同时评价引用归属与回答质量（Asai et al., 2024）。这些研究已将来源归属、迭代综合和人工评价纳入学术内容生成的研究范围。

另一类系统进一步组织完整长文的生成。STORM 通过多视角问题与检索构建提纲，并在评价中纳入有经验的维基百科编辑者的反馈（Shao et al., 2024）。AutoSurvey 将文献获取、提纲构建、小节生成与修订组织为综述写作流程（Wang et al., 2024）。LiRA 则采用专门智能体完成科学文献综述的组织、写作、编辑和审核（Go et al., 2026）。这些工作说明，规划和修订应当作为综合过程的组成部分，而非仅依赖一次生成。

SurveyGen 提供了更直接的实证参照：其数据包含4200余篇人类撰写的综述及242143条被引参考文献，并比较了全自动与有人类指导的生成方式（Bao et al., 2025）。该研究发现，全自动流程仍存在引用质量较低和批判性分析有限的问题。因此，本文的文章评价将表层组织与声明支持、领域解释和批判性综合分开测量。它同时提高了人类对照的要求：少量、非同证据的人类文章只适合诊断文体差异，不能证明系统具有同证据质量优势。

不过，这些系统所面向的任务并不完全相同。科学问题回答、百科式文章、叙述性文献综述与文献计量领域分析，所需要的证据既有重叠，也存在差别。本文重点研究那些需要对明确范围的学术语料进行处理才能获得支持的声明，例如社区聚合结果或网络连通性的变化。相应的比较重点不仅是文章的长度或组织结构，还包括在可比的计算与来源证据条件下，能否忠实使用证据、追溯其来源，并保持修订的一致性。

### 2.3. 基于图的检索与可执行证据

GraphRAG 构建实体图及社区摘要，支持针对整个文档集合的提问（Edge et al., 2024）。它通过多层次的信息组织，扩展了仅检索孤立文本片段的处理方式。然而，从文本中抽取的实体图与文献计量网络并不等同。共同作者关系或术语共现关系都有明确的构建规则，其解释也取决于这些规则。因此，将基于图的综合用于文献计量任务，需要明确边的含义、图的范围及分析参数。

GRAG 通过检索文本子图，并向模型提供互补的文本视图与图视图来支持多跳推理（Hu et al., 2025）。GraphRAG-Bench 则进一步在可包含多跳、数学或编程要求的领域任务上，评价图构建、检索、推理连贯性和最终回答（Xiao et al., 2025）。因此，“把图与文本一起提供”或“增加问题复杂度”本身都不足以构成创新。它们反而使本文需要回答一个更严格的识别问题：图的价值是否只在答案依赖预先登记的结构运算时出现，并且在扁平条件获得同一计算结果后是否仍然存在。本文设置的简单—复杂任务对照、同计算flat-program控制和原始溯源消融，正是为这一较窄的问题服务。

不同任务条件下的比较同样必要。Xiang 等（2025）考察了基于图的检索在不同任务条件下的适用性。在本文任务中，直接属性查询可以与全局聚合、删除操作后的连通性计算形成对照。使任务涉及的实体保持匹配，有助于区分计算需求差异与研究对象差异。此外，比较还需要控制各条件获得的信息及计算结果，避免将额外的信息供给误认为某种表示方式的收益。

可执行计算构成另一个比较维度。PAL 使用程序作为中间表示，并依靠解释器获得计算结果（Gao et al., 2023）。CiteWeave 在文献计量证据流程中应用程序辅助计算的思想，调用预先登记的图操作，并保留其输出，用于后续综合与核验。本文关注计算结果、底层图记录与来源解释对回答和文章中研究判断的支持作用。

### 2.4. 审核、反馈与人机协作

审核既可以由自动化系统执行，也可以由人类执行。PaperEval 采用多智能体方法，从多个方面评价论文的新颖性与原创性（Huang et al., 2025）。LiRA 也将审核纳入写作流程（Go et al., 2026）。这些工作将自动评价纳入学术信息处理流程，而人工审核还涉及独立判断、审核分歧与实际投入等问题。

人机协作研究已经考察何时由模型独立处理、何时与人协作，以及何时将决策交给人类。LECODU 联合建模了与多位用户的协作和决策转交，并考虑参与人数与相应成本（Zhang et al., 2025）。在这些协作与转交研究的基础上，CiteWeave 进一步关注审核决定与受影响研究对象之间的关系，包括来源、图计算结果、研究声明和文章段落。

近期研究使这一边界更加明确。学习转交方法已经能够利用少量上下文适应未见过的专家（Tailor et al., 2024），并显式建模人类与模型错误之间的依赖（Wei et al., 2024）。HyPER 直接优化人类偏好标签与语言模型偏好标签之间的路由（Miranda et al., 2025）；贝叶斯在线共识估计则在没有真实标签时，用后验不确定性权衡查询人类的成本（Showalter et al., 2024）。因此，审核者能力估计、成本感知路由和顺序查询都不能单独作为 CiteWeave 的创新点。本文提出的贡献是将它们用于声明—证据依赖系统：审核动作具有明确干预对象，能够传播到预先登记的依赖项，并约束修订范围。

让人参与也不会自动使评价独立或可信。一项预注册标注实验发现，向标注者展示模型建议并未减少标注时间，却改变了标签分布，还可能抬高基于“人类已确认”标签得到的模型评价（Schroeder et al., 2025）。对106项实验的元分析同样发现，人机组合平均不如人类单独与AI单独两者中的较优者，而且效果随任务显著变化（Vaccaro et al., 2024）。这些结果要求本文采用盲态独立首审，区分human-only、AI-only与human–AI条件，保存不可变交互日志，并直接测量质量、时间、分歧和成本。

这一关注点引出了两个需要分别验证的问题。首先，当上游证据被否定时，系统应能识别其他具有明确依赖关系、因而需要重新检查的声明。其次，局部纠正是否适用于其他研究，需要独立验证。因此，框架将局部修订与可复用反馈规则的验证分开处理，并将人工纠错质量、审核分歧、有效审核时间和规则向留出主题的迁移纳入评价。

### 2.5. 本研究的定位

上述研究已经在分析自动化、学术检索、长文写作、图辅助综合、可执行推理与审核等方面取得进展。CiteWeave 研究的是如何将这些能力连接起来，以支持一个特定任务。其基本组织单元是研究声明，以及支持该声明的数据、计算和来源证据，同时记录声明在被纠正时涉及的依赖关系。

本文围绕三个衔接环节展开：从检索记录到结构计算证据，从证据到有明确适用边界的解释，以及从审核决定到一致修订。通过对这些环节进行受控比较，研究将区分证据供给、计算执行与审核机制各自的作用，并考察它们能否共同改善文献计量综合的质量。

## 3. 任务定义与研究问题

### 3.1. 任务及适用范围

文献计量综合的输入包括研究方案、范围明确的学术记录集合、图构建与分析参数、可获得的原始文献片段，以及写作要求。输出是研究领域分析报告，其中的经验性声明应当关联相应支持材料，并保留有关分析和审核决定的记录。该任务不要求复现一篇既有人类论文，也不假定文献元数据和摘要已经包含系统综述或实质性因果解释所需的全部证据。

例如，某个术语出现频次增加，可以支持关于该术语在样本语料中变化的描述，但不能单独证明相应技术的性能有所提高。类似地，删除节点后网络连通性发生变化，描述的是所构建网络的性质，并不是对某位研究者、某个机构或某个主题在现实世界中消失后会发生什么的估计。这些区别同时约束内容生成与审核。

### 3.2. 关系图与证据依赖

本文区分关系图与推导依赖图。关系图依据明确的构建规则，表示文献计量实体及其关系；推导依赖图则记录来源、证据材料、算子输出、研究声明与段落之间的依赖关系。执行一项预先登记的分析会产生一个证据对象，其解释取决于输入与参数。将该对象关联到某项声明，记录的是拟议的支持关系，并不能仅凭这条连接证明声明在语义上成立。

当某个上游对象被判定无效时，系统可以将已记录的下游依赖项标记为待审核对象，从而界定受控修订的范围，并尽量保留未受影响的文字。这一能力仅覆盖已经表示出来的依赖关系，未记录的语义依赖仍可能无法被发现。因此，实体有效性、计算正确性、声明的证据支持及领域适切性，需要分别检查。在构建错误的图上重复得到相同计算结果，并不能保证科学解释有效。

本研究的图程序条件使用预先登记的执行轨迹和选定的支持性图记录，其中包含利用基准证据整理得到的支持材料。该条件考察预设程序与支持信息下的证据利用，不用于评价无预设支持的自主检索和分析规划。评价方案同时设置查询驱动检索、层次化检索、仅提供算子结果及信息等价的扁平程序条件，以区分计算结果与底层记录访问的作用。大图采用候选节点数量上限，相关分析的适用范围限定于这一构建规则下的网络。

### 3.3. 研究问题

本研究围绕以下三个问题组织。

RQ1：在何种任务复杂度与图规模条件下，相较于基于检索的替代方法，可执行图证据能够改善结构性问题的回答质量及证据支持？该问题区分简单查询、聚合、结构扰动和时间—结构任务，并将证据检索、计算结果是否可得，以及结果的表示形式分开考察。

RQ2：依赖感知的人工审核如何影响研究声明的纠错，并形成怎样的质量与审核投入权衡？该问题同时涉及具有证据支持和缺乏支持的声明，既考察受影响段落的即时修订，也考察反馈在原审核案例之外的有效性，并区分策略回放与实际人工干预的评价结果。

RQ3：在使用相同证据的条件下，计算证据与审核机制能否提高所生成文章的证据支持、可追溯性和研究效用？该问题考察局部任务表现能否进一步转化为有用的学术综合，并分别评价可追溯性、领域针对性和研究效用。

三项研究问题分别考察结构任务、审核修订与文章综合三个层面的表现，以分析计算证据与人工审核的作用及适用条件。

## 4. CiteWeave框架

### 4.1. 方案驱动的数据与执行链

CiteWeave从版本化研究方案开始，固定主题、时间范围、检索式、图类型、规模条件、任务模板和允许的解释。规范化语料、图、benchmark、tokenizer、模型配置和执行计划均用SHA-256绑定；修改通过追加修正案完成，旧工件和失败尝试不被覆盖。这不能消除数据库覆盖偏差，但能恢复一项结论实际使用的样本和变换。

### 4.2. 可执行图证据

系统把证据检索与结构计算分开。注册任务包括跨社区加权多跳、全局桥边选择后的删边反事实、社区总重要性与外连比例对照、加权度枢纽删除后的连通性和替代枢纽重算，以及时间增长与拓扑位置的跨层连接。确定性图程序在冻结图上执行算子，输出结构答案、算子轨迹和底层图记录标识；语言模型负责综合这些工件，而不是从PNG近似完成图计算。

层级图检索只提供组织后的图证据；flat program以非图布局提供与graph program相同的计算内容；operator only提供可承载答案的轨迹但移除底层图provenance。由此分别识别检索、外部图计算、序列化形式和原始证据的作用。

### 4.3. 文章综合与审核

每个复杂现象对象绑定问题、轨迹、图证据、相关文献、替代解释和禁止推断。文章编译使用五类现象及三类注册跨现象综合；graph与flat编译条件共享相同现象、来源、章节顺序、调用数和字数预算，唯一差异是依赖边是否显式提供。

真人审核以claim及其可见依赖为对象，记录事实支持、解释校准、替代解释、证据充分性、决定性证据、失效依赖、动作和理由。正式方案采用独立双审和仅分歧第三裁决。失效上游对象沿显式依赖传播至claim和段落；修订仅能改动hash绑定的受影响段落，未受影响正文保持字节不变。序贯VOI策略会在每次结果揭示后更新依赖Beta后验，按严重度、预计新增下游覆盖和结果前耗时估计重新分配剩余预算。开发服务器还会在较晚发现共享依赖无效时重新打开早先已审核claim，但它与正式金标采集隔离。

## 5. 评价设计

图实验覆盖8个独立主题，前4个为主面板，后4个为独立复制；图规模上限为100、300和2,000节点。主面板为100题×8条件=800次调用，条件包括flat BM25、flat hybrid、BGE-M3 dense、query graph、hierarchical GraphRAG、operator only、flat program和graph program。复制面板新增120次调用。复杂度扩展以48个语义锚点匹配简单/复杂任务和三种规模，共672个逻辑格，其中312个新增调用。机制补充为168题×3条件=504格，其中216个新增调用。

统计推断以主题而非题目、claim或审核者为独立单位，报告主题等权效应、主题簇bootstrap和精确sign-flip；注册检验族使用Holm校正。解析失败计错，传输失败保留，完整cell不重采样。真人部分要求6名合格审核者、能力校准、盲化材料、服务端计时、双审和裁决；当前0名真人返回，因此只构成前瞻设计。同证据文章计划比较graph生成、one-shot和独立人类作者，并把claim支持、研究效用、可追溯性和领域具体性设为不同门禁。

## 6. 结果

### 6.1. 图任务结果

主面板800/800、复制120/120、复杂度扩展312/312新增调用和机制补充216/216新增调用均完成；全部逻辑格通过终止审计，terminal解析失败和主记录重复均为0。

100题主面板中，flat BM25、flat hybrid和BGE-M3 dense准确率均为0，query graph为0.08，hierarchical GraphRAG为0.22，operator only为0.56，flat program为0.82，graph program为0.88。Graph program相对flat hybrid的注册效应为+0.876，层级GraphRAG相对flat hybrid为+0.250；主面板Small到Large的规模交互为0。

| 条件 | 题数 | 准确率 | 平均证据F1 |
|---|---:|---:|---:|
| Flat BM25 | 100 | 0.00 | 0.047 |
| Flat hybrid | 100 | 0.00 | 0.062 |
| BGE-M3 dense | 100 | 0.00 | 0.006 |
| Query graph retrieval | 100 | 0.08 | 0.197 |
| Hierarchical GraphRAG | 100 | 0.22 | 0.241 |
| Operator only | 100 | 0.56 | 0.121 |
| Flat program | 100 | 0.82 | 0.434 |
| Graph program | 100 | 0.88 | 0.404 |

复杂度匹配实验给出了更清楚的边界。48个简单任务上graph program、层级GraphRAG和dense均为1.00；120个复杂任务上graph program为1.00、层级GraphRAG为0.28，flat hybrid和dense均为0。Graph program相对dense的复杂性选择效应在8/8主题均为+1.00，Holm校正精确p=0.0078125；简单题差异在±0.05内等效。规模放大效应在8/8主题均为0，p=1.0。因此在当前图上，复杂计算需求而不是简单扩大节点数决定了图程序收益。

### 6.2. 图机制识别

Graph program与相同计算内容的flat program准确率只差+0.00595，并通过±0.05等效检验，说明图形JSON布局本身没有独立收益。相对operator only，保留原始图provenance使证据F1提高+0.291，Holm校正p=0.0078125；答案准确率提高+0.20并通过非劣门。四类轨迹已携带完整答案的任务上原始provenance不改变答案，但在故意保留不完整轨迹的桥边删除题上带来+1.00准确率，8/8主题一致，Holm校正p=0.0078125。可防守的结论是“确定性图计算加必要时的原始provenance”，不是“图格式天然更好”或“LLM自主发现了图程序”。

### 6.3. 文章与人审状态

正式16个机器文章请求均返回正文，但16/16未通过原门禁：全部超过2,700–3,300词，graph稿8/8触及8,000 completion-token上限，6/8缺完整Conclusion，其中2篇也缺Limitations。层级解析诊断表明16/16实际都在Results和Discussion使用五类现象；graph稿8/8命中三类注册综合，one-shot仅4/8命中至少两类。这只能说明结构覆盖更强，不能覆盖正式失败。

两主题matched compiler开发得到4篇2,835–3,187词稿和每篇23–38条可审核claim，但9/36次分节调用触及token上限、两篇出现截断证据ID，严格联合门禁仍为0/4。非LLM诊断显示机器段落长度变异系数0.15、人类参考0.71，校准词6.52 vs 2.15/千词，数值18.22 vs 48.38/千词，提示模板化展开和领域事实密度不足。当前token-safe恢复因API账户无可用余额保持0/36有效响应。

依赖传播、局部修订、能力路由、离线VOI和实时VOI路径均通过确定性或合成测试；开发特征面板有80条claim、132个文章内依赖，最大fanout为20。独立专家路径现已把文章中的条件无关匿名证据标识解析到三条件共用、哈希绑定的图答案、算子轨迹、解释边界和来源摘要，并能按服务器可见时间不可变地采集整体、claim和成对偏好判断。但真实首审、裁决、VOI和独立专家评分均为0，RQ2和RQ3的人类部分没有结果。

## 7. 讨论

结果支持一个条件性结论：图程序在必须执行全局选择、聚合、干预、重算和跨层连接时有效，在匹配的简单查询上没有优势，图变大本身也没有放大效果。机制消融进一步否定了“图形布局即创新”，把贡献限定为外部确定性图计算和可核验provenance。由于多数复杂题的轨迹已经暴露可确定性派生的答案字段，这也不是自由规划意义上的LLM图推理。

局部答案优势没有自动转化为合格长文。显式依赖提高跨现象组织和可审核claim数量，却同时暴露篇幅失控、截断、段落节奏过度整齐和领域解释不足。正式文章比较必须控制同证据和长度，并分别评价claim支持、领域具体性、可追溯性和研究效用。

人审系统的实质进步是让较晚的人类发现能够改变后续队列、传播修订范围并撤销早期accept，而不是保存一条反馈记录。但软件路径可运行不等于人审有效；没有完整真人标签、结果前耗时估计和随机live比较时，不能声称减少劳动。

限制包括：只有8个主题簇；部分条件出现地板/天花板分离；图是候选截断图；实体规范化和边语义仍可能出错；任务使用注册程序而非自主规划；只测试一个provider/model配置；依赖传播只能覆盖显式记录的关系；真人与同证据人类文章证据仍缺失。

## 8. 结论

当前证据回答了RQ1，但尚未回答RQ2和RQ3。8主题结果表明，注册的确定性图程序在复杂结构任务上产生一致优势，在简单任务上无优势，且规模扩大未进一步放大效果；同计算量消融表明真正有效的是图计算内容及轨迹不足时的原始provenance，而不是图形序列化。依赖感知人审和序贯VOI已经实现为可重放机制，但必须取得真实人审和同证据专家文章结果，才能构成完整三部分投稿。若无法补齐，应把论文范围收缩为已经完成的图计算研究，把人审和文章实验明确列为未来工作。

## 参考文献

Aria, M., & Cuccurullo, C. (2017). bibliometrix: An R-tool for comprehensive science mapping analysis. *Journal of Informetrics, 11*(4), 959–975. https://doi.org/10.1016/j.joi.2017.08.007

Asai, A., He, J., Shao, R., Shi, W., Singh, A., Chang, J. C., Lo, K., Soldaini, L., Feldman, S., D'Arcy, M., Wadden, D., Latzke, M., Tian, M., Ji, P., Liu, S., Tong, H., Wu, B., Xiong, Y., Zettlemoyer, L., . . . Hajishirzi, H. (2024). *OpenScholar: Synthesizing scientific literature with retrieval-augmented LMs* (Version 1) [Preprint]. arXiv. https://arxiv.org/abs/2411.14199v1

Bao, T., Nayeem, M. T., Rafiei, D., & Zhang, C. (2025). SurveyGen: Quality-aware scientific survey generation with large language models. *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 2712–2736. https://doi.org/10.18653/v1/2025.emnlp-main.136

Chen, C. (2006). CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature. *Journal of the American Society for Information Science and Technology, 57*(3), 359–377. https://doi.org/10.1002/asi.20317

Chung, J., Jeong, B., Park, Y.-J., Jeong, H., Lee, J., Yoon, J., & Choi, J. (2025). Harnessing language models for computational literature review of emerging AI topics. *Information Processing & Management, 62*(6), Article 104245. https://doi.org/10.1016/j.ipm.2025.104245

Donthu, N., Kumar, S., Mukherjee, D., Pandey, N., & Lim, W. M. (2021). How to conduct a bibliometric analysis: An overview and guidelines. *Journal of Business Research, 133*, 285–296. https://doi.org/10.1016/j.jbusres.2021.04.070

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., & Larson, J. (2024). *From local to global: A graph RAG approach to query-focused summarization* [Preprint]. arXiv. https://arxiv.org/abs/2404.16130

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., & Neubig, G. (2023). PAL: Program-aided language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, & J. Scarlett (Eds.), *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 10764–10799). PMLR. https://proceedings.mlr.press/v202/gao23f.html

Go, G. H. T., Ly, K., Søgaard, A., Tabatabaei, S. A., de Rijke, M., & Chen, X. (2026). LiRA: A multi-agent framework for reliable and readable literature review generation. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(47), 40456–40464. https://doi.org/10.1609/aaai.v40i47.41489

Hu, Y., Lei, Z., Zhang, Z., Pan, B., Ling, C., & Zhao, L. (2025). GRAG: Graph retrieval-augmented generation. *Findings of the Association for Computational Linguistics: NAACL 2025*, 4145–4157. https://doi.org/10.18653/v1/2025.findings-naacl.232

Huang, S., Wang, Q., Lu, W., Liu, L., Xu, Z., & Huang, Y. (2025). PaperEval: A universal, quantitative, and explainable paper evaluation method powered by a multi-agent system. *Information Processing & Management, 62*(6), Article 104225. https://doi.org/10.1016/j.ipm.2025.104225

Miranda, L. J. V., Wang, Y., Elazar, Y., Kumar, S., Pyatkin, V., Brahman, F., Smith, N. A., Hajishirzi, H., & Dasigi, P. (2025). Hybrid preferences: Learning to route instances for human vs. AI feedback. *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 7162–7200. https://doi.org/10.18653/v1/2025.acl-long.355

Schroeder, H., Roy, D., & Kabbara, J. (2025). Just put a human in the loop? Investigating LLM-assisted annotation for subjective tasks. *Findings of the Association for Computational Linguistics: ACL 2025*, 25771–25795. https://doi.org/10.18653/v1/2025.findings-acl.1323

Shao, Y., Jiang, Y., Kanell, T., Xu, P., Khattab, O., & Lam, M. (2024). Assisting in writing Wikipedia-like articles from scratch with large language models. In K. Duh, H. Gomez, & S. Bethard (Eds.), *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 6252–6278). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.347

Showalter, S., Boyd, A. J., Smyth, P., & Steyvers, M. (2024). Bayesian online learning for consensus prediction. *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, PMLR 238*, 2539–2547. https://proceedings.mlr.press/v238/showalter24a.html

Skarlinski, M. D., Cox, S., Laurent, J. M., Braza, J. D., Hinks, M., Hammerling, M. J., Ponnapati, M., Rodriques, S. G., & White, A. D. (2024). *Language agents achieve superhuman synthesis of scientific knowledge* [Preprint]. arXiv. https://arxiv.org/abs/2409.13740

Tailor, D., Patra, A., Verma, R., Manggala, P., & Nalisnick, E. (2024). Learning to defer to a population: A meta-learning approach. *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, PMLR 238*, 3475–3483. https://proceedings.mlr.press/v238/tailor24a.html

Vaccaro, M., Almaatouq, A., & Malone, T. (2024). When combinations of humans and AI are useful: A systematic review and meta-analysis. *Nature Human Behaviour, 8*, 2293–2303. https://doi.org/10.1038/s41562-024-02024-1

van Eck, N. J., & Waltman, L. (2010). Software survey: VOSviewer, a computer program for bibliometric mapping. *Scientometrics, 84*(2), 523–538. https://doi.org/10.1007/s11192-009-0146-3

Wang, Y., Guo, Q., Yao, W., Zhang, H., Zhang, X., Wu, Z., Zhang, M., Dai, X., Zhang, M., Wen, Q., Ye, W., Zhang, S., & Zhang, Y. (2024). AutoSurvey: Large language models can automatically write surveys. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, & C. Zhang (Eds.), *Advances in Neural Information Processing Systems* (Vol. 37, pp. 115119–115145). Curran Associates. https://doi.org/10.52202/079017-3655

Wei, Z., Cao, Y., & Feng, L. (2024). Exploiting human-AI dependence for learning to defer. *Proceedings of the 41st International Conference on Machine Learning, PMLR 235*, 52484–52499. https://proceedings.mlr.press/v235/wei24a.html

Xiang, Z., Wu, C., Zhang, Q., Chen, S., Hong, Z., Huang, X., & Su, J. (2025). *When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented generation* [Preprint]. arXiv. https://arxiv.org/abs/2506.05690

Xiao, Y., Dong, J., Zhou, C., Dong, S., Zhang, Q., Yin, D., Sun, X., & Huang, X. (2025). *GraphRAG-Bench: Challenging domain-specific reasoning for evaluating graph retrieval-augmented generation* (Version 3) [Preprint]. arXiv. https://arxiv.org/abs/2506.02404

Zhang, Z., Ai, W., Wells, K., Rosewarne, D., Do, T.-T., & Carneiro, G. (2025). Learning to complement and to defer to multiple users. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, & G. Varol (Eds.), *Computer Vision – ECCV 2024* (Vol. 15114, pp. 144–162). Springer. https://doi.org/10.1007/978-3-031-72992-8_9
