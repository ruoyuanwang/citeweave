# CiteWeave Formal V3 正式结果解释

更新时间：2026-09-08

## 结论摘要

正式结果支持“复杂任务触发图能力”，不支持“图越大优势越强”。在8个独立主题、120个复杂任务上，Graph Program准确率为1.00，flat hybrid与BGE-M3 dense均为0，层级GraphRAG为0.28；在48个匹配简单任务上，Graph Program、层级GraphRAG与BGE-M3 dense均为1.00。注册的复杂性选择效应C1为+1.00，8/8主题同方向，exact dataset sign-flip `p=0.00390625`、Holm调整后`p=0.0078125`。注册的规模放大效应C2为0，`p=1.0`。

随后完成的504-cell同计算量机制补充进一步给出边界：Graph Program与携带完全相同计算内容的flat program在准确率上等效，差值仅+0.006，满足±0.05 TOST；这否定了“JSON图形布局本身是创新”的解释。相对只给operator trace的条件，Graph Program在复杂任务上evidence F1提高+0.291、答案准确率提高+0.20，8/8主题一致；对轨迹不完整的桥边反事实，额外原始证据带来+1.00准确率，而四类轨迹完整任务差值为0。M2与M3的Holm调整`p`均为0.0078125。

因此论文最稳妥的主张是：当答案需要多阶段图算子、反事实重计算或跨层聚合时，预计算Graph Program对普通扁平检索形成稳定优势；当任务可由局部证据直接回答时，图程序与强dense基线等效。真正有效的是“确定性图计算 + 可追溯原始证据”，不是图状序列化；图规模本身也不是已证实的调节变量。

## 主要描述结果

| 条件 | 简单任务准确率（n=48） | 复杂任务准确率（n=120） | 复杂任务 evidence F1 |
|---|---:|---:|---:|
| Flat hybrid | 0.81 | 0.00 | 0.09 |
| BGE-M3 dense | 1.00 | 0.00 | 0.01 |
| Hierarchical GraphRAG | 1.00 | 0.28 | 0.24 |
| Graph Program | 1.00 | 1.00 | 0.41 |

复杂任务分型显示，Graph Program在五类任务上均为1.00。层级GraphRAG在桥边删除反事实、社区角色对照和删枢纽韧性上分别为0.62、0.42和0.33，但在多跳连接与时间—结构联合上均为0。这说明“仅分层取回图记录”可以帮助一部分全局/反事实问题，却不能稳定替代显式算子执行。

主面板100题的总体描述也一致：Graph Program准确率0.88、flat program 0.82、operator-only 0.56、hierarchical GraphRAG 0.22、graph-query retrieval 0.08，三种扁平检索为0。注册H1中Graph Program相对flat hybrid的主题等权准确率差为+0.876，Holm调整`p=1.55e-25`；H2中hierarchical GraphRAG相对flat hybrid为+0.25，Holm调整`p=3.81e-6`。

## 应当如何解释

1. **任务复杂性是有效边界。** C1达到所有主题+1.00，而简单任务上Graph Program与BGE-M3 dense在±0.05界值内等效。这比笼统的“有图优于无图”更有论文价值。
2. **显式图计算强于只检索图结构。** 层级GraphRAG有增益但远未达到Graph Program；系统创新应定位为检索、算子执行、反事实重计算和provenance回链的组合，而非图文本序列化。
3. **扩大规模不是充分条件。** 主实验、独立复制和8主题合并的small-to-large交互都为0；C2也为0。不能声称图越大优势越明显。更准确的设计原则是扩大图以制造局部检索不足，再用需要全局计算的任务识别图程序价值。
4. **当前出现明显地板/天花板。** 复杂任务上扁平基线全错、Graph Program全对，混合logistic模型出现boundary/separation。exact dataset-level检验仍有效，但下一轮基准应增加中间难度、噪声、不完整图与开放式解释，避免只证明确定性算子能够输出其已计算答案。

## 同计算量机制补充

| 条件 | 简单任务准确率（n=48） | 复杂任务准确率（n=120） | 复杂任务 evidence F1 |
|---|---:|---:|---:|
| Operator only | 1.00 | 0.80 | 0.12 |
| Flat program | 1.00 | 0.99 | 0.41 |
| Graph program | 1.00 | 1.00 | 0.41 |

- M1：Graph Program减flat program的主题等权准确率差为+0.00595，bootstrap区间0至0.01786；±0.05双单侧exact检验均`p=0.00390625`，判定等效。
- M2：Graph Program减operator-only的复杂任务evidence F1为+0.291，八个主题效应为+0.276至+0.344；Holm调整`p=0.0078125`。答案准确率差为+0.20且通过-0.05非劣门。
- M3：桥边反事实的Graph Program减operator-only准确率为+1.00，四类完整轨迹任务差为0，选择性差异+1.00，Holm调整`p=0.0078125`。

这些结果把贡献拆成两层：图计算使复杂答案可获得；原始记录/provenance在轨迹不完整时补足答案，并显著提高证据回链。把同样内容从图JSON改成flat JSON几乎不改变准确率。

## 尚不能主张的内容

- 不能把结果表述为LLM从原始大图中“自主发现”答案。生成前审计显示120个复杂任务中96个的完整答案字段可由operator trace直接派生；Graph Program的正确定位是“先执行图程序，再由LLM进行受约束综合”。
- 不能把Graph Program相对flat program的差异写成图序列化优势；注册机制检验已证明两者在±0.05内等效。原始记录/provenance相对operator-only提高证据F1并补足轨迹不完整的桥边反事实，但它可以用flat结构承载。
- 不能声称规模放大效应；正式结果明确不支持。
- 不能从exact answer外推到科学解释质量、领域洞察或文章质量。后者仍需claim级真人审核和独立专家盲评。

## 下一轮更强的任务设计

- 在固定图程序之外加入证据冲突、图构造不确定性和多种可接受解释，要求模型比较假设而非抄录唯一数值答案。
- 让关键答案依赖至少两种图层、一个反事实重计算和一个来源语义约束，同时不给出完整终态字段，只提供可复核中间结果与原始证据。
- 增加受控缺边、别名噪声、社区不稳定和时间窗口变动，评分从exact answer扩展到稳健性判断、校准、证据充分性和替代解释质量。
- 保留简单任务作为negative control；新复杂任务应调到强基线非零、Graph Program非满分的区间，减少地板/天花板。
- 把机制补充中的flat program作为必要强基线：只有Graph Program在答案非劣的同时提高evidence provenance，才能声称原始图结构有独立价值。

## 对应正式工件

- 主面板分析：`experiments/graph_discovery_v2/formal_v3_analysis.json`
- 独立规模复制：`experiments/graph_discovery_v2/formal_v3_scale_replication_analysis.json`
- 8主题复杂度分析：`experiments/graph_discovery_v2/formal_v3_complexity_extension_analysis.json`
- 复杂度合并结果：`experiments/graph_discovery_v2/formal_v3_complexity_extension_merged/`
- 答案暴露审计：`experiments/graph_discovery_v2/formal_v3_graph_program_answer_exposure_audit.json`
- 机制补充协议：`experiments/graph_discovery_v2/formal_v3_mechanism_supplement_protocol.yml`
- 机制补充504-cell合并结果：`experiments/graph_discovery_v2/formal_v3_mechanism_supplement_merged.json`
- 注册机制分析：`experiments/graph_discovery_v2/formal_v3_mechanism_supplement_analysis.json`
