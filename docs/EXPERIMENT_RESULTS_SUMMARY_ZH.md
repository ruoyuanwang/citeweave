# CiteWeave 当前实验结果总结

更新日期：2026-08-17  
结果冻结时间：2026-08-07  
正式图实验运行标识：`formal_v2_nonthinking_20260806`

## 1. 一句话结论

当前正式实验已经完成闭环，最新只读审计 108/108 项通过。多阶段 CiteWeave 报告相对同证据的一次性生成明显降低了不受支持声明；Graph RAG 相对无检索有很大优势，但相对扁平结构化证据没有优势；自适应审核在本次 36 个锁定样本上减少了 44.4% 的审核请求且未观察到不安全自动放行，但样本量不足以满足预设的生产级置信门槛。

## 2. 实验状态与统计范围

- 冻结状态以 `experiments/formal_completion_audit_current.json` 为准：该文件在 2026-08-07 记录 `all_complete=true`，108 项完整、0 项缺失、0 项无效。重新审计时还需要操作系统对 `report_resolved` 和 `graph_resolved` 镜像目录具有读取权限；ACL 拒绝访问属于环境问题，应与实验产物缺失区分。
- `experiments/formal_completion_audit.json` 是实验尚未完成时的历史快照，不能用于描述当前状态。
- 共 8 个完整自然年主题：2 个开发主题只用于查询和 Judge 校准，6 个锁定主题进入正式比较统计。
- 统计使用 6 个锁定 topic cluster、10,000 次 topic-cluster bootstrap、随机种子 `20260806`。
- 图实验的 3 个预注册比较使用 topic-level 精确符号翻转检验，并进行 Holm 多重比较校正。
- 最终报告由本地接受产物确定性生成，不调用 API；其 provenance manifest 记录了 1,029 个结构化来源文件的 SHA-256。

## 3. 数据规模

| 数据集 | 角色 | 年份 | 收到记录 | OpenAlex 唯一记录 | Canonical works |
|---|---|---:|---:|---:|---:|
| gnn_drug_discovery_2017_2023 | 开发 | 2017–2023 | 456 | 456 | 456 |
| crispr_extracellular_vesicles_2015_2022 | 开发 | 2015–2022 | 259 | 259 | 259 |
| machine_learning_climate_change_2008_2022 | 锁定 | 2008–2022 | 7,720 | 7,686 | 7,675 |
| climate_change_risks_1990_2021 | 锁定 | 1990–2021 | 152,865 | 152,030 | 151,562 |
| digital_twins_healthcare_2012_2024 | 锁定 | 2012–2023 | 796 | 796 | 796 |
| plant_heat_drought_2008_2021 | 锁定 | 2008–2021 | 1,604 | 1,599 | 1,597 |
| gene_editing_als_2004_2024 | 锁定 | 2004–2023 | 211 | 211 | 211 |
| global_microplastics_2004_2019 | 锁定 | 2004–2019 | 4,011 | 3,995 | 3,973 |
| **合计** | — | — | **167,922** | **167,032** | **166,529** |

口径说明：OpenAlex 唯一记录是源身份去重后的 staged 输入；Canonical works 是经过 DOI 或规范化“标题+年份”精确去重后真正用于下游统计的文献数。6 个锁定主题对应的两个总数分别为 166,317 和 165,814，相差 503。最终英文报告数据表里的 `Processed` 表示进入处理流程的源记录，不等于最终 canonical 分析样本量。

## 4. 报告生成实验

两个系统条件使用完全相同的冻结 Evidence Bundle：

- `citeweave_full`：编辑计划、完整初稿、内部审阅、修订稿，共 4 次调用；
- `structured_one_shot`：同一证据上的单次结构化生成，共 1 次调用；
- `published_human_reference`：同主题的已发表人类文献计量研究，只是质量参照，不是同语料复现，也不是声明级 gold truth。

| 比较 | A 条件 | A UCR | A 完整度 | B 条件 | B UCR | B 完整度 | A 胜/平/负 |
|---|---|---:|---:|---|---:|---:|---:|
| full_vs_oneshot | CiteWeave Full | 1.7% | 5.000 | Structured One-shot | 8.3% | 4.000 | 6/0/0 |
| full_vs_human | CiteWeave Full | 4.5% | 5.000 | Published Human | 6.6% | 3.083 | 6/0/0 |
| oneshot_vs_human | Structured One-shot | 10.9% | 4.000 | Published Human | 0.0% | 3.000 | 6/0/0 |

关键解释：

- Full 相对 One-shot 的 UCR 绝对降低 6.6 个百分点，95% CI 为 2.9–10.2 个百分点，不跨 0；这是报告实验里最清楚的主结果。
- Full 相对人类参考的 UCR 低 2.1 个百分点，但 95% CI 为 -5.3–3.6 个百分点，跨 0，因此不能声称其事实支持度显著优于人类参考。
- Full 的完整度比人类参考高 1.917 分，且 6/6 主题获总体偏好；但这比较的是“同主题输出质量”，不是对同一检索语料的严格复现。
- One-shot 虽在总体偏好和完整度上胜过人类参考，但 UCR 明显更高。这说明总体偏好、覆盖度与事实支持度必须并列报告，不能只选一个有利指标。

## 5. 图证据实验

正式文本面板比较同一个 `deepseek-v4-pro` 在三种证据条件下的表现：`graph_rag`、`flat_structured`、`no_rag`。6 个锁定主题共 174 个文本问答项。Figure/VLM 是仅看可视图的跨模型扩展，只覆盖 35 个 network-size 项，不属于严格同模型主比较。

| 比较 | Graph RAG UCR | 对照 UCR | Graph RAG 完整度 | 对照完整度 | Graph 胜/平/负 |
|---|---:|---:|---:|---:|---:|
| graph_vs_no | 1.5% | 89.6% | 4.839 | 1.851 | 143/27/4 |
| graph_vs_flat | 1.4% | 0.8% | 4.839 | 4.908 | 2/167/5 |
| graph_vs_figure | 0.0% | 0.0% | 5.000 | 5.000 | 0/35/0 |

关键解释：

- 相对无检索，Graph RAG 的 UCR 降低 88.1 个百分点，95% CI 为 86.4–89.7 个百分点，效果量很大。
- 但 `graph_vs_no` 的原始 p=0.0312，经 3 项 Holm 校正后 p=0.0938，未达到 family-wise 0.05 显著性阈值。原因之一是独立统计单元只有 6 个主题。
- Graph RAG 相对扁平结构化证据并未提升：UCR 反而高 0.6 个百分点，差异区间跨 0，174 项中 167 项打平。当前证据支持“检索/结构化 grounding 很重要”，不支持“图表示本身优于同信息量的扁平结构”。
- Figure/VLM 的 35 项全平只说明可见图中的网络规模可被同样正确读取，不能外推到最高频节点、最强边、聚类归属或错误前提等完整任务族。

## 6. 自适应审核实验

正式统计只使用 6 个锁定主题，每个主题 6 项，共 36 项。Human Proxy 只能在系统给出风险提示后查看最多 500 字符的被标记片段，并执行一次最多 500 字符的局部接受、拒绝、替换、删除或附加 caveat；不能浏览、调用工具、查看完整产物、改查询或重跑分析。

| 条件 | 审核请求率 | 最终质量通过率 | 不安全自动放行率 |
|---|---:|---:|---:|
| baseline_original | 0.0% | 75.0% | 25.0% |
| always_review | 100.0% | 77.8% | 不定义 |
| static_review | 66.7% | 77.8% | 8.3% |
| adaptive_review | 55.6% | 77.8% | 0.0% |

三个审核策略都把错误率从 25.0% 降到 22.2%，绝对只降低 2.8 个百分点。Adaptive 相对 Always-review 少请求 44.4 个百分点的审核，并在 16 个自动放行项中观察到 0 个错误。

但按 `experiments/BENCHMARK.md` 的预设 RQ2 门槛，生产级成功还要求自动放行精度的单侧 95% Clopper–Pearson 下界至少为 0.95。16/16 正确对应的下界约为 0.829，因此当前结果只能说“观察样本中未出现不安全自动放行”，不能说“已经证明达到生产安全门槛”。此外，Human Proxy 不是实际用户，不能据此推断真实审核时长、信任或可用性。

## 7. 当前最稳妥的论文级结论

1. 多阶段、有内部审阅的报告工作流比同证据的一次性生成更可靠，且该差异在 6 个锁定主题上方向一致。
2. Grounding 相对无检索显著改善事实支持和完整度；然而在小主题数和多重校正下，统计显著性证据仍有限。
3. 图结构并未优于信息等价的扁平结构化证据，图的价值需要在更依赖关系路径的任务上继续验证。
4. 自适应审核显示了减少人工请求的潜力，但质量提升很小，且置信下界未过预设门槛；应扩充自动放行审计样本并开展真实用户研究。
5. 所有对人类论文的比较都应写成 topic-aligned quality comparison，不能写成 exact replication 或“超过人类”的普遍结论。

## 8. 权威结果文件

- 当前完成性：`experiments/formal_completion_audit_current.json`
- 冻结数据协议：`experiments/formal_datasets_openalex_title_abstract.yml`
- 正式统计输入：`experiments/formal_statistics_manifest.json`
- 正式统计结果：`experiments/formal_statistics/formal_statistics.json`
- 表格指标：`experiments/formal_statistics/formal_metrics.csv`
- Holm 结果：`experiments/formal_statistics/graph_holm.csv`
- 最终英文报告：`experiments/final_report/end_to_end_report.md`
- Judge 冻结协议：`experiments/judge_calibration_freeze.json`
- Human Proxy 协议：`experiments/human_proxy_protocol.yml`
