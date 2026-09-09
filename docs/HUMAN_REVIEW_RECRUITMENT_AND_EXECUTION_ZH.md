# CiteWeave 真人审核招募与执行手册

版本：2026-09-08。本文只把已冻结协议转成可执行的人事与运行步骤，不修改样本、假设、随机化、评分或统计规则，也不以研究者、LLM或模拟数据替代真人结果。

## 1. 需要招募的不是“一组万能审核者”

为避免同一人员既生产文章、又审核自己的修改、再评价最终文章，正式研究至少分为四个角色池。人员可以跨主题，但不得跨越有利益冲突的角色边界。

| 角色池 | 最低覆盖 | 任务 | 必须隔离 |
|---|---:|---|---|
| R1 来源与机制审核者 | warmup正式roster恰好6人；每人至少2个领域 | 120个来源相关性warmup、能力校准、后续96例2×2互补审核 | 不得看到检索条件、排序、gold answer、其他审核者结论或held-out正确性 |
| R2 文章claim审核者 | 每主题至少3名无冲突且领域合格者 | 8主题×20 claims，320次独立首审；分歧由第三人裁决 | 不得参与对应文章生成；第三人看不到首审身份和票据 |
| A 同证据人类作者 | 每主题至少1名，共8篇文章 | 使用与机器条件相同的证据包、写作brief、字数目标与期限独立写作 | 不得看到任何机器稿、机器评分、review worklist或最终评价 |
| E 独立文章专家 | 每主题至少4名可分配者，其中至少2名领域专家；每篇至少3个评分 | 对CiteWeave、one-shot与human_same_evidence共24篇文章做盲评及claim核验 | 不得参与系统开发、对应文章写作或claim修订；不得看到条件标签和其他评分 |

R1和R2可以是同一批人中的合格子集，但其warmup材料永远排除出确认性结果。A与E必须分离；任何作者、合作者、同课题组或其他已声明冲突均在随机分配前排除。

## 2. 当前缺口与关键路径

当前机器状态不是“人审脚本没写好”，而是正式人员表为空：

- 来源相关性：120个盲化pack已存在，缺6名真实审核者及冲突声明。
- 互补审核：96个held-out任务身份已从正式benchmark结果盲冻结；必须等三个机器面板全部干净terminal后才能物化候选回答，再冻结能力后验与四臂分配。
- 文章claim审核：必须先生成8篇CiteWeave正式稿；之后物化160个claim pack，缺每主题3名真实审核者。
- 同证据文章比较：缺8名独立人类作者、24篇三条件文章，以及满足主题覆盖的独立专家池。
- 所有真实参与者研究在招募前应取得所属机构要求的伦理/IRB审查或书面豁免判断，并确定知情同意、报酬、数据保存和退出规则；项目不得自行假定“只是专家评测所以无需审查”。

执行关键路径如下：

```text
人员预招募与冲突筛查
        │
        ├── R1来源warmup → 双审/裁决 → 能力后验冻结 → retriever held-out promotion
        │                                             │
正式Graph实验 terminal ──→ 候选输出 ≥96例 ────────────┴→ 2×2互补审核
        │
        └── 冻结writer packs → 机器稿 + 同证据人类稿
                                  │
                                  ├── R2 claim双审/裁决 → 依赖传播修订
                                  └── E对24篇文章盲评与claim核验
```

## 3. 招募筛查表

身份信息与联系方式不得放入Git仓库。协调者在独立受控文件中保存真实身份映射，工程内只使用随机生成的`reviewer_id`或`evaluator_id`。每位候选人先填写以下字段：

- 编码ID；真实姓名和联系方式只保存在仓库外映射表。
- 可审核的主题和领域；每项注明资格依据，例如博士训练、论文或实际研究经历。
- 不可审核的主题及冲突类型：作者身份、近年合作、同机构/课题组、竞争关系、经济利益或直接参与本项目。
- 可承担角色：R1、R2、A或E。
- 是否独立于CiteWeave开发和文章生产。
- 可工作时间窗口；不得先收集正式答案再补写资格或冲突信息。
- 已完成知情同意和伦理流程的记录编号，不在工程目录保存签字原件。

R1的工程内roster每人至少需要：

```json
{
  "reviewer_id": "R-opaque-id",
  "eligible_domains": ["domain_a", "domain_b"],
  "conflicted_domains": [],
  "conflicts_declared": true
}
```

R2按已有模板填写`eligible_topics`、`conflicted_topics`、`domain_qualified`与独立性字段。E除主题资格与冲突外，必须显式填写`conflicts_declared`、`independent_from_article_production`、`independent_from_system_development`和`review_or_revision_topics`。系统会按主题排除文章生产者、参加过claim审核/修订者以及系统开发者；缺字段不会被当作“默认独立”。不得为了通过readiness虚构领域资格或把同一人复制成多个ID。

## 4. 工作量与分配规则

### R1：来源校准与能力建模

- 120个source–phenomenon pair，每人目标30题，共180个判断。
- 至少60/120题由两名审核者独立完成。
- 每人至少12个已裁决warmup案例、覆盖至少2个领域，才能进入能力模型。
- direct/contextual/irrelevant或evidence role出现分歧时，由该领域第三名合格者盲裁。
- warmup标签只训练evidence retriever/reranker和审核者能力后验，不进入确认性2×2结果。

来源相关性只是能力向量的一维，不能用于声称系统已识别图核验或过度推断方面的专家。相同6名R1审核者还需完成训练专用的多维校准panel：

- 96个受控校准包，覆盖8主题与`graph_answer_consistency`、`causal_overreach`、`counterevidence_coverage`、`revision_safety`四类问题。
- 96次首审加48次独立重叠，共144次主判断；分歧交结果前预指定的第三人盲裁。
- 每名审核者每类问题至少3例，且在首审、重叠审和潜在裁决的全部已承诺角色中不得重复看到同一个底层图案例。
- 主判断只根据私有受控注入金标计算能力；第三人裁决用于一致性与分歧诊断，不定义金标，也不参与自身能力评分。
- 跨领域路由使用`reviewer × issue_type`层级；只有global/prior证据的后验不得称为个性化能力。

### 2×2互补审核

- 至少96个真实候选输出，8主题每主题至少12例。
- 每主题四个arm各至少3例；每位审核者覆盖四臂，但永远不重复看到同一case。
- 至少50%案例双审，分歧第三人裁决。
- 能力路由只能使用held-out揭盲前冻结的`reviewer × domain × issue_type`后验。
- 人数或能力不足时返回`blocked_insufficient_reviewers`，不得静默退化为自动接受或随便指派。

### R2：文章claim审核

- 每主题20条claim，五类图现象每类至少2条；共160个claim。
- 每条两名独立首审，共320次首审；只有完全一致才直接定案。
- 四项二元判断、action、决定性证据、失效依赖、replacement和transfer guard任一不一致，都交第三人独立裁决。
- 无效`source/graph evidence`沿`evidence → claim → paragraph`传播，只能修改受影响且hash绑定的段落；其他字节必须保持不变。

### A与E：文章质量比较

- A按8个主题各写1篇`human_same_evidence`文章，不看机器稿。
- 三条件共24篇文章：CiteWeave、one-shot、human_same_evidence。
- E对文章做8个整体维度评分；每篇至少3名评分者。
- 每篇抽取20个claim，每条至少双评，分歧第三人裁决。
- 主结论以主题为独立重复，不把数百个claim或评分冒充数百个独立科学主题。
- 前瞻功效审计显示，8主题对A1的20个百分点claim正确性提升约有0.819功效，但对A2/A3标准化topic效应1.0只有约0.565功效。A2未通过只能写“未建立非劣”，不能写“等价”；若计划主张广泛的人类平价或超越，必须另外预注册至少4个独立主题的复制。

## 5. 推荐的实际招募规模

最低人数不等于稳健排班。为承受冲突、退出和第三人裁决，建议目标为：

- R1/R2候选池：10–12人，其中从合格候选中在分配前结果盲选定恰好6人进入R1 warmup正式roster；每个主题最好至少4人可用。
- A同证据作者：8人，最好每人只写一个主题。
- E独立专家池：12–16人，保证每主题至少4人可分配且至少2名领域专家。

这些是排班目标，不改变冻结协议的最低门槛。若无法达到，系统必须报告相应主题`blocked`，不得降低门槛后仍称为确认性实验。

## 6. 参与者邀请文本

> 我们正在开展一项文献计量与人机协作研究，评估结构化图证据、文本检索和人类审核如何共同支持科学综述。任务将在盲化界面中完成，主要包括判断来源是否支持给定解释、核验文章claim或对匿名文章进行专家评分。系统记录服务器端可见页面时间和不可逆提交，不收集与研究无关的数据。参与前会提供知情同意、预计工作量、报酬、退出方式、数据保存期限和冲突声明。您不会被要求评价自己撰写、合作或有其他利益冲突的材料。

邀请中不得声称CiteWeave优于人类，也不得透露预期方向、条件名称或中途结果。

## 7. 执行与安全操作

1. 协调者在仓库外完成伦理文件、真实身份映射和报酬记录。
2. 使用编码ID填写对应roster；先运行readiness，确认人数、领域和冲突覆盖。
3. 资格通过后冻结roster哈希，再生成随机分配；不得先看题目或标签后调整人员资格。
4. 为每名审核者生成一次性明文访问token；工程内只保存SHA-256。明文token只通过单独安全渠道交付。
5. 审核服务默认只监听`127.0.0.1`。远程参与必须使用机构VPN或经审查的TLS反向代理，不能把开发服务裸露到公网。
6. 服务器计时只累计页面真实可见且有heartbeat的时间；客户端自报时长不进入正式劳动指标。
7. 返回结果先做完整性和schema验证，再生成裁决worklist；不得用多数票静默合并。
8. 所有原始返回、裁决、roster、分配、服务日志和分析结果分别哈希冻结，保留失败现场。

来源warmup的准备入口为：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_source_relevance_panel.py `
  --packet-root experiments\human_review_v2\source_relevance_warmup_v4 `
  --reviewer-roster experiments\human_review_v2\source_relevance_warmup_v4\reviewer_roster.json `
  --output-root experiments\human_review_v2\source_relevance_assignments_v1
```

主审访问凭据发放前，必须先为60个双审案例逐一锁定潜在第三裁决者；此步骤检测到任何首审返回时都会拒绝执行：

```powershell
.\.venv\Scripts\python.exe scripts\run_source_relevance_warmup_outcomes.py `
  commit-adjudicators `
  --assignment-root experiments\human_review_v2\source_relevance_assignments_v1 `
  --reviewer-roster experiments\human_review_v2\source_relevance_warmup_v4\reviewer_roster.json `
  --commitment experiments\human_review_v2\source_relevance_assignments_v1\adjudicator_commitment.json
```

六名审核者完成180次首审后，先验证返回身份、任务归属、packet哈希和服务器计时，再只为真正存在分歧的案例生成盲裁panel：

```powershell
.\.venv\Scripts\python.exe scripts\run_source_relevance_warmup_outcomes.py `
  prepare-adjudication `
  --assignment-root experiments\human_review_v2\source_relevance_assignments_v1 `
  --commitment experiments\human_review_v2\source_relevance_assignments_v1\adjudicator_commitment.json `
  --primary-validation experiments\human_review_v2\source_relevance_assignments_v1\primary_validation.json `
  --adjudication-root experiments\human_review_v2\source_relevance_adjudication_v1
```

应为该盲裁panel另行生成访问token；裁决者只看到原始source packet，不看到两名首审者的身份或答案。裁决完成后冻结120个解析标签、120个独立能力观测、reviewer registry和层级Beta能力后验：

```powershell
.\.venv\Scripts\python.exe scripts\run_source_relevance_warmup_outcomes.py `
  finalize `
  --assignment-root experiments\human_review_v2\source_relevance_assignments_v1 `
  --commitment experiments\human_review_v2\source_relevance_assignments_v1\adjudicator_commitment.json `
  --primary-validation experiments\human_review_v2\source_relevance_assignments_v1\primary_validation.json `
  --adjudication-root experiments\human_review_v2\source_relevance_adjudication_v1 `
  --output-root experiments\human_review_v2\source_relevance_resolved_v1
```

60个单审标签只作为带`single_review_training_only`标记的检索训练数据，不用于给同一审核者打能力分，也不能进入retriever promotion门禁。能力后验只使用60个双审共识/第三人裁决案例形成的120个首审观测；生成的`reviewer_registry.json`与`capability_freeze.json`是后续互补审核readiness的正式输入。

多维校准盲包已经冻结在`experiments\human_review_v2\oversight_calibration_v1`。填入同一批6名真实审核者的roster后，按硬约束生成分配：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_oversight_calibration_panel.py `
  --packet-root experiments\human_review_v2\oversight_calibration_v1 `
  --reviewer-roster experiments\human_review_v2\source_relevance_warmup_v4\reviewer_roster.json `
  --output-root experiments\human_review_v2\oversight_calibration_assignments_v1
```

完成144次主判断后，先验证返回并生成只包含真实分歧的盲裁panel：

```powershell
.\.venv\Scripts\python.exe scripts\run_oversight_calibration_outcomes.py validate-primary `
  --assignment-root experiments\human_review_v2\oversight_calibration_assignments_v1 `
  --output experiments\human_review_v2\oversight_calibration_assignments_v1\primary_validation.json

.\.venv\Scripts\python.exe scripts\run_oversight_calibration_outcomes.py prepare-adjudication `
  --assignment-root experiments\human_review_v2\oversight_calibration_assignments_v1 `
  --primary-validation experiments\human_review_v2\oversight_calibration_assignments_v1\primary_validation.json `
  --output-root experiments\human_review_v2\oversight_calibration_adjudication_v1
```

盲裁完成后才可生成训练专用的多维能力观测；不得查看私有金标后修改分配：

```powershell
.\.venv\Scripts\python.exe scripts\run_oversight_calibration_outcomes.py finalize `
  --assignment-root experiments\human_review_v2\oversight_calibration_assignments_v1 `
  --packet-root experiments\human_review_v2\oversight_calibration_v1 `
  --reviewer-roster experiments\human_review_v2\source_relevance_warmup_v4\reviewer_roster.json `
  --primary-validation experiments\human_review_v2\oversight_calibration_assignments_v1\primary_validation.json `
  --adjudication-root experiments\human_review_v2\oversight_calibration_adjudication_v1 `
  --output-root experiments\human_review_v2\oversight_calibration_resolved_v1
```

来源与多维校准都完成后，先把同一6人的120+144条能力观测合并冻结。该步骤会检查五类问题覆盖、领域覆盖、roster身份和冲突声明；任一缺失即阻断：

```powershell
.\.venv\Scripts\python.exe scripts\freeze_integrated_oversight_capabilities.py `
  --source-observations experiments\human_review_v2\source_relevance_resolved_v1\reviewer_observations.json `
  --calibration-observations experiments\human_review_v2\oversight_calibration_resolved_v1\calibration_observations.json `
  --reviewer-roster experiments\human_review_v2\source_relevance_warmup_v4\reviewer_roster.json `
  --output-root experiments\human_review_v2\integrated_capability_v1
```

原v1面板虽已物化，但金标为90个`qualify`和6个`reject`，只保留为自然输出分布诊断。正式真人干预使用结果前修正案008冻结的v2平衡真实候选面板；其任务身份与v1相同，每主题6个`qualify`和6个`reject`。以下构建已完成，不得覆盖重跑：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_balanced_oversight_heldout_packets.py select `
  --base-selection experiments\human_review_v2\oversight_heldout_selection_v1.json `
  --output experiments\human_review_v2\oversight_heldout_selection_v2_balanced.json

.\.venv\Scripts\python.exe scripts\prepare_balanced_oversight_heldout_packets.py materialize `
  --selection experiments\human_review_v2\oversight_heldout_selection_v2_balanced.json `
  --terminal-audit experiments\graph_discovery_v2\formal_v3_primary_terminal_audit.json `
  --terminal-audit experiments\graph_discovery_v2\formal_v3_replication_terminal_audit.json `
  --terminal-audit experiments\graph_discovery_v2\formal_v3_complexity_extension_terminal_audit.json `
  --output-root experiments\human_review_v2\oversight_heldout_factorial_v2_balanced
```

随后执行全局约束四臂分配，并在发放访问凭据前运行严格readiness v2：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_factorial_oversight_assignment.py `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --reviewer-registry experiments\human_review_v2\integrated_capability_v1\reviewer_registry.json `
  --capability-freeze experiments\human_review_v2\integrated_capability_v1\capability_freeze.json `
  --combined-observations experiments\human_review_v2\integrated_capability_v1\combined_reviewer_observations.json `
  --output-root experiments\human_review_v2\oversight_factorial_assignments_v1

.\.venv\Scripts\python.exe scripts\audit_factorial_oversight_readiness_v2.py `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --reviewer-registry experiments\human_review_v2\integrated_capability_v1\reviewer_registry.json `
  --capability-freeze experiments\human_review_v2\integrated_capability_v1\capability_freeze.json `
  --combined-observations experiments\human_review_v2\integrated_capability_v1\combined_reviewer_observations.json `
  --assignment-root experiments\human_review_v2\oversight_factorial_assignments_v1 `
  --amendment experiments\human_review_v2\review_policy_amendment_005_integrated_capability_and_factorial_execution.yml `
  --amendment-freeze experiments\human_review_v2\review_policy_amendment_005_integrated_capability_and_factorial_execution_freeze.json `
  --previous-amendment experiments\human_review_v2\review_policy_amendment_004_multidimensional_calibration.yml `
  --output experiments\human_review_v2\oversight_factorial_assignments_v1\readiness_v2.json
```

真人完成首审后，依次验证、生成真实分歧盲裁、再汇总最终case级结果。不得跳过中间文件或手工合并投票：

```powershell
.\.venv\Scripts\python.exe scripts\run_oversight_factorial_outcomes.py validate-primary `
  --assignment-root experiments\human_review_v2\oversight_factorial_assignments_v1 `
  --output experiments\human_review_v2\oversight_factorial_assignments_v1\primary_validation.json

.\.venv\Scripts\python.exe scripts\run_oversight_factorial_outcomes.py prepare-adjudication `
  --assignment-root experiments\human_review_v2\oversight_factorial_assignments_v1 `
  --primary-validation experiments\human_review_v2\oversight_factorial_assignments_v1\primary_validation.json `
  --output-root experiments\human_review_v2\oversight_factorial_adjudication_v1

.\.venv\Scripts\python.exe scripts\run_oversight_factorial_outcomes.py finalize `
  --assignment-root experiments\human_review_v2\oversight_factorial_assignments_v1 `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --primary-validation experiments\human_review_v2\oversight_factorial_assignments_v1\primary_validation.json `
  --adjudication-root experiments\human_review_v2\oversight_factorial_adjudication_v1 `
  --output experiments\human_review_v2\oversight_factorial_outcomes_v1.json
```

最终结果必须同时运行两套预先冻结、但回答不同问题的分析。原dataset-level O1/O2/O3检验跨主题可迁移性；新增B1/B2/B3严格复现原始2×2 blocked allocation，用50000次随机分配检验冻结96例基准内的因果效应。B检验不能替代或挽救O检验：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_complementary_oversight_factorial.py `
  --input experiments\human_review_v2\oversight_factorial_outcomes_v1.json `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --assignment-manifest experiments\human_review_v2\oversight_factorial_assignments_v1\assignment_manifest.json `
  --amendment experiments\human_review_v2\review_policy_amendment_003_factorial_cluster_inference.yml `
  --amendment-freeze experiments\human_review_v2\review_policy_amendment_003_factorial_cluster_inference_freeze.json `
  --output experiments\human_review_v2\oversight_factorial_transportability_analysis_v1.json

.\.venv\Scripts\python.exe scripts\analyze_oversight_randomization.py `
  --outcomes experiments\human_review_v2\oversight_factorial_outcomes_v1.json `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --assignment-manifest experiments\human_review_v2\oversight_factorial_assignments_v1\assignment_manifest.json `
  --amendment experiments\human_review_v2\review_policy_amendment_006_finite_benchmark_randomization.yml `
  --amendment-freeze experiments\human_review_v2\review_policy_amendment_006_finite_benchmark_randomization_freeze.json `
  --output experiments\human_review_v2\oversight_factorial_randomization_analysis_v1.json
```

修正案008发现“未审核AI直接接受”的四分类基线在当前gold定义下是定义性零分，因此撤回旧C1的确认性地位。下列分析仍需运行以报告团队相对单次辅助人工首审的C2、挽救/伤害案例和成本，但不得单凭它声称广义`human-AI complementarity`；该表述还需要后续同包AI-only自审对照：

```powershell
.\.venv\Scripts\python.exe scripts\analyze_oversight_complementarity.py `
  --packet-manifest experiments\human_review_v2\oversight_heldout_factorial_v2_balanced\manifest.json `
  --assignment-root experiments\human_review_v2\oversight_factorial_assignments_v1 `
  --primary-validation experiments\human_review_v2\oversight_factorial_assignments_v1\primary_validation.json `
  --final-outcomes experiments\human_review_v2\oversight_factorial_outcomes_v1.json `
  --amendment experiments\human_review_v2\review_policy_amendment_007_complementarity_baselines.yml `
  --amendment-freeze experiments\human_review_v2\review_policy_amendment_007_complementarity_baselines_freeze.json `
  --output experiments\human_review_v2\oversight_complementarity_analysis_v1.json
```

生成访问凭据时必须把终端中仅显示一次的明文token安全交付给参与者，不写入文档、聊天记录或版本控制：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_review_access.py `
  --manifest <assignment-manifest.json> `
  --output <access-hashes.json>
```

本地审核服务：

```powershell
.\.venv\Scripts\python.exe scripts\serve_human_review.py `
  --packet-root <frozen-assignment-root> `
  --access <access-hashes.json> `
  --host 127.0.0.1 `
  --port 8765
```

路径占位符必须替换为readiness生成并冻结的真实工件，不能照抄占位符启动正式实验。

24篇同证据文章、claim inventories、production records和专家roster全部通过门禁后，先生成原始专家盲包，再补充三条件共用的匿名证据查看器与正式采集manifest：

```powershell
.\.venv\Scripts\python.exe scripts\build_article_expert_packets.py `
  --intake <article_intake.json> `
  --evaluator-roster <evaluator_roster.json> `
  --output-dir <expert_packet_root>

.\.venv\Scripts\python.exe scripts\prepare_article_expert_collection.py `
  --intake <article_intake.json> `
  --packet-manifest <expert_packet_root\manifest.json> `
  --output-dir <expert_collection_root>
```

第二步会验证修正案007及全部实现哈希，并拒绝无法从文章匿名`EV-*`解析到图现象/算子轨迹或来源摘要的claim。之后为专家manifest单独生成访问token，并启动专用服务；该服务和claim修订服务不是同一个角色池：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_review_access.py `
  --manifest <expert_packet_root\manifest.json> `
  --output <expert_access_hashes.json>

.\.venv\Scripts\python.exe scripts\serve_article_expert_review.py `
  --collection-manifest <expert_collection_root\collection_manifest.json> `
  --access <expert_access_hashes.json> `
  --host 127.0.0.1 `
  --port 8767
```

专家看到的是opaque topic/article/evidence code、统一图和匿名证据内容；不会收到条件、作者、自动分数、检索排名或其他专家答案。整体评分、claim核验和成对偏好均采用服务器可见heartbeat计时并在首次提交后冻结。

全部主审完成后，先验证任务注册表、返回身份、哈希和计时并恢复私有条件映射；存在任何claim分歧时输出只能停在`awaiting_blind_claim_adjudication`，不能直接送入分析器：

```powershell
.\.venv\Scripts\python.exe scripts\validate_article_expert_primary_returns.py `
  --collection-manifest <expert_collection_root\collection_manifest.json> `
  --protocol experiments\article_quality_v2\expert_evaluation_protocol.yml `
  --output <expert_collection_root\primary_validation.json>
```

生成的`adjudication_worklist`只列争议claim和必须排除的两名首审者，不向裁决者暴露首审答案。只有盲裁返回被合并并再次通过完整性门禁后，才允许调用`analyze_article_expert_evaluation.py`。

使用冻结worklist生成第三人盲裁collection。路由器只在该主题原先已通过资格/冲突门禁的专家中选择，硬排除两名首审者，并按当前裁决负载与结果前固定哈希确定分配：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_article_expert_adjudication.py `
  --primary-validation <expert_collection_root\primary_validation.json> `
  --primary-collection-manifest <expert_collection_root\collection_manifest.json> `
  --output-dir <expert_adjudication_root>
```

为裁决manifest另行生成访问token，再用同一个`serve_article_expert_review.py`启动；该collection只启用claim任务，且不会包含两名首审者的身份或答案。全部盲裁提交后合并：

```powershell
.\.venv\Scripts\python.exe scripts\finalize_article_expert_adjudication.py `
  --primary-validation <expert_collection_root\primary_validation.json> `
  --adjudication-collection-manifest <expert_adjudication_root\collection_manifest.json> `
  --output <expert_collection_root\analysis_input.json>
```

只有输出状态为`analysis_input_ready_after_blind_adjudication`且裁决完整性全部通过时，才能进入冻结分析器。若主审完全无分歧，则使用`analysis_input_ready_no_claim_disagreements`输出，无须人为制造裁决任务。

## 8. 停止条件与允许的论文表述

出现以下任一情况立即停止对应阶段：roster覆盖不足、冲突未声明、分配哈希漂移、同一人重复看到同一case、访问token泄露、计时来源不合规、返回缺失、裁决人不独立、正式材料被条件标签污染，或任何模拟/LLM结果进入真人结果目录。

只有真实数据完成且通过冻结分析后，才能分别检验：

- typed dependency-aware review是否在质量非劣前提下降低审核时间；
- adversarial two-sided packet是否提高最终正确性；
- capability router是否提高每审核分钟正确结局；
- CiteWeave文章是否相对one-shot提高claim正确且有证据支持的比例；
- CiteWeave相对同证据人类文章是否达到研究效用非劣且证据可追溯性更高。

在此之前，只能表述为“系统、协议和门禁已实现，真实人类效应尚未测得”。
