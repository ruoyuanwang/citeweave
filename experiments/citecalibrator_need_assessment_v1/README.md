# CiteCalibrator 训练必要性实验

## 当前结论

114 个既有自然 claim case 已由两个上下文完全隔离的 LLM judge 独立审查。按输出前冻结的判定协议，动作一致性 Cohen's κ 为 0.879，核心字段完全共识覆盖率为 89.5%（102/114）；可靠性、问题规模和规则缺口三道门槛全部通过。

这支持进入 CiteCalibrator 模型训练实验：当前确定性规则不能替代 claim–evidence 的语义判断。它还不能证明“微调必然优于 prompt-only 模型”，因为当前结果属于 development proxy gold，只有两个既有主题，而且两名 judge 属于同一模型家族。正式结论仍需 topic-disjoint 测试集、人工抽检/金标以及 zero-shot、few-shot、SFT 的同底模对照。

本轮没有重新生成文章，也没有调用外部 LLM API。两名 Codex 子代理只负责对现有 114 个 case 作 LLM-as-judge 审查。

## 已复用的现有产物

| 产物 | 数量 | 在本实验中的用途 |
|---|---:|---|
| claim-ready development 文章 | 4 篇、2 个主题 | 构建自然 pilot case |
| PH-bearing sentence candidates | 114 条 | LLM-as-judge 的评测单位 |
| graph phenomena | 10 个唯一现象 | 关联 verified answer、operator trace 和 interpretation contract |
| robustness variants | 8 个主题 × 18 个变体 | 为 case 提供参数敏感性证据 |
| 旧版 machine generation | 16 篇、8 个主题 | 全部为 `generation_quality_gate_failed`，只记录工程失败 |
| 既有整篇文章 LLM 盲评 | 8 个主题 | 仅作文章质量背景，不转换为 claim-level gold |

四篇 pilot 文章在 benchmark 构建前均已完成 9 次 provider calls，其中包含 Results 和 Discussion 的既有修复调用。因此，本实验测量的是当前生成/修复流程之后仍然存在的 claim-level 问题。

## Benchmark

### Pilot benchmark

- `pilot_benchmark/cases.jsonl`：114 个 case，来自 4 篇现有文章。
- 63 条来自 2 篇通过机器门禁的文章；51 条来自 2 篇未通过门禁的文章，分别保存在 `qualified_article_cases.jsonl` 和 `failed_article_cases.jsonl`。
- 每个 case 包含原句、段落上下文、PH/REF、verified answer、operator trace、解释边界、robustness profile、允许使用的 evidence ID 和 lineage hash。
- reviewer 看不到规则输出、case condition、controlled challenge gold 或其他 judge 的结果。
- 这些文章和主题在开发中已被查看，因此禁止把结果解释为自然部署缺陷率。

### Controlled challenge

- `controlled_challenge/cases.jsonl`：40 个 case，由 10 个唯一图现象各构造 4 种受控情况。
- 标签构成：10 accept、20 qualify、10 reject。
- 它只验证评测链路能识别显式 supported、causal overclaim、semantic reification 和 fabricated numeric；人为构造的缺陷比例不能证明训练必要性。

## LLM-as-judge 方案

判定协议在两个 judge 输出前冻结，见 `llm_judging_v1/judge_protocol.yml` 和 `judge_protocol_freeze.json`。

### 隔离

- 两名子代理均以 `fork_context=false` 启动，不继承父上下文。
- A/B 获得内容和 SHA-256 完全相同、但物理路径分离的 case、rubric 和 protocol 副本。
- 两边输出目录分离；审查过程中不允许访问另一 judge、规则预测、private condition map、challenge gold、父对话结论、网页或外部 API。
- A 完成后也没有把结果发给 B；两边全部结束后才运行聚合器。
- `assignment_manifest.json` 和 `dispatch_manifest.json` 固化了分配及隔离设置。

### 判定标签

- `accept`：原样保留。
- `qualify`：核心结果可保留，但必须给出证据内可直接使用的最小改写。
- `reject`：核心事实、来源依赖或解释无法保留。
- `abstain`：case 证据不足或存在无法解决的冲突。

每条还判断 factual support、interpretation calibration、alternative adequacy、evidence sufficiency 和 minor/major/critical severity。只有两名 judge 在这 6 个核心字段上全部相同才形成共识；其他 case 保持 unresolved，不强行裁决。风险标签只在核心字段共识后取并集，不参与四个主要指标。

### 四个主要指标

1. `inter_judge_action_kappa`：两名 judge 的动作一致性，可靠性门槛为 κ ≥ 0.70。
2. `consensus_major_critical_defect_rate`：共识 case 中 major/critical 的 qualify/reject 比例，问题规模门槛为 ≥ 0.08。
3. `rule_major_critical_defect_recall`：规则对上述共识严重缺陷的召回率，规则缺口条件为 < 0.80。
4. `rule_accept_false_intervention_rate`：规则对共识 accept 的误干预率，规则缺口条件为 > 0.10。

可靠性还要求核心字段共识覆盖率 ≥ 0.80。只有可靠性、问题规模、规则缺口三道门槛同时通过，才支持 development-stage 的训练假设。

## 实验结果

### Judge 可靠性

| 项目 | 结果 |
|---|---:|
| Judge A | 54 accept / 60 qualify |
| Judge B | 51 accept / 61 qualify / 2 abstain |
| action raw agreement | 93.9% |
| action Cohen's κ | **0.879** |
| 6 个核心字段完全共识 | **102/114（89.5%）** |
| unresolved | 12/114（10.5%） |

### 共识结果与规则对照

102 条共识 case 中有 50 条 accept、52 条 major qualify；没有共识 reject 或 critical。52 条缺陷的共同模式都是：claim 把一个 REF 写成其事实或解释的来源，但该 REF excerpt 不支持相应内容；PH 中的结构性事实本身通常仍然成立，因此需要删除错误依赖或改成 PH-bounded 表述，而不是整句拒绝。

| LLM 共识 | 规则 accept | 规则 qualify | 合计 |
|---|---:|---:|---:|
| accept | 8 | 42 | 50 |
| major qualify | 52 | 0 | 52 |

由此得到四个冻结指标：

| 指标 | 结果 | 门槛 | 是否通过 |
|---|---:|---:|---|
| inter-judge action κ | **0.879** | ≥ 0.70 | 是 |
| consensus major/critical defect rate | **51.0%** | ≥ 8% | 是 |
| rule major/critical defect recall | **0.0%** | < 80% 表示存在缺口 | 存在缺口 |
| rule accept false-intervention rate | **84.0%** | > 10% 表示存在缺口 | 存在缺口 |

规则不是随机变差，而是使用了错误代理：它把“是否出现某类 token/REF”当作“证据是否充分”，无法判断 REF excerpt 是否真正蕴含 claim。因此，它漏掉全部 52 条错误来源依赖，同时误改 42/50 条仅依赖 PH 且证据充分的 claim。这是重复、可学习、需要 claim–evidence 对齐的语义任务，适合模型化；继续增加关键词规则会在 recall 与误干预之间来回摆动。

按主题，两个 development topic 均出现共识 major 缺陷（23/55 与 29/47）。在通过机器门禁文章的 55 条共识 case 中仍有 22 条 major 缺陷，而且两篇合格文章均受影响；这说明现有 article-level gate 不能替代 claim-level calibrator。以上比例只描述该 development pilot，不能外推为部署发生率。

规则在 40 条 controlled challenge 上得到完美结果，只说明显式模板链路正确；它与自然 case 上的失败恰好说明 challenge 不能替代真实生成输出。

## “需要训练”究竟证明到哪一步

当前已经证明：

1. 当前产出中存在 judge 高一致性确认的、机器门禁后仍可能残留的语义来源依赖问题；
2. 确定性规则在严重缺陷召回与正确 claim 误干预两个方向同时失败；
3. 问题需要联合读取 claim、PH、REF excerpt 和解释边界，因此有必要进入模型训练实验，而不是继续把规则当最终方案。

当前尚未证明：监督训练本身优于直接调用一个 prompt-only 模型。下一阶段应在完全未见 topic 上，用同一 base model 和同一解码预算比较：no-intervention、当前规则、zero-shot、few-shot、SFT。训练成立的预注册标准建议为：相对最佳非训练模型，major/critical recall 至少提升 0.10，accept false-intervention 不超过 0.10，并且 action macro-F1 同时提升；所有最终指标以人工 gold 为准，LLM 共识仅用于开发与筛选。

## 可复现入口

- 准备隔离输入：`scripts/prepare_citecalibrator_llm_judges.py`
- 聚合两名 judge：`scripts/aggregate_citecalibrator_llm_judges.py`
- 完整实验审计：`scripts/audit_citecalibrator_experiment.py`
- 聚合结果：`llm_judging_v1/aggregate/analysis.json`
- 未强制裁决的分歧：`llm_judging_v1/aggregate/disagreements.jsonl`
