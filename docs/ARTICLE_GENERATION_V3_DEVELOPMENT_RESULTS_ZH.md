# CiteWeave 文章生成 V3 开发实验结果

更新时间：2026-09-08

## 结论

单靠提示模型“写2700–3100词”不能稳定控制科学长文。可行方向是把文章生成变成依赖图驱动的编译过程：按章节依赖顺序生成，Results/Discussion执行受约束的完整性修复，最后由确定性句子预算器实现篇幅硬约束。早期两主题开发样例最终达到3217和3240词并通过机器结构门禁，但每篇只有15条独立可审核claim。最新matched compiler v1把人审接口写进生成契约后，4/4篇达到23–38条候选并通过claim-review readiness；然而9/36次分节调用触及token上限，严格联合门禁仍为0/4。所有这些都是查看过先前失败后的开发证据，不能替代新的确认性实验或真人质量评价。

## 逐阶段结果

### 正式单次生成

8主题×2机器条件的16个冻结单次请求全部获得响应，但原质量门禁16/16失败。层级解析纠偏后，CiteWeave稿8/8包含全部三类预注册跨现象综合，而one-shot只有4/8达到至少两类；与此同时16/16超长，CiteWeave 8/8触及8000 completion-token上限，6/8缺完整Conclusion。

### 单次预算提示开发v1

两个主题×`budgeted_one_shot/budgeted_graph_plan`共4次请求，2/4通过。图规划条件两篇均自然停止且落在允许篇幅，但其中一篇Discussion没有再次显式写PH编号；one-shot有一篇仍触及长度上限并产生截断ID。

### 单次检查点开发v2

增加累计篇幅检查点和Discussion首句PH配对后，4/4稿均自然停止，七章完整，Results/Discussion/Conclusion均覆盖五类现象，三类综合全部命中且无非法ID。仍只有2/4通过全门禁：另外两篇分别为2674和2386词，低于2700词。该结果证明结构控制可由显式依赖计划稳定实现，但单次模型的词数控制仍不可靠。

### 图依赖文章编译器v1

编译器按`Methods → Results → Discussion → Limitations → Conclusion → Introduction → Abstract`生成七个不可变分节，再按正常文章顺序确定性组装。第一篇3096词并通过全部全局门禁；第二篇3209词，唯一失败是Results在分节token上限处截断一个REF标识。两篇均保留五类现象和全部三类综合。

### 分节完整性修复

对两篇的Results和Discussion各执行一次独立修复调用。四个修复请求消除了非法ID并保留全部结构，但LLM把“压缩”写成扩展：组装稿变为3333和3544词，0/2通过篇幅门。这说明模型式压缩不能承担硬长度约束。

### 确定性句子预算优化

最终预算器只删除不含PH/REF的完整句子，禁止删除每段首句、任何证据句或导致段落少于两句的内容，并记录每个删除句的位置、词数与SHA256。结果如下：

| 主题 | 优化前 | 优化后 | 删除句/词 | PH/REF集合 | 全部门禁 |
|---|---:|---:|---:|---|---|
| 可解释医学影像 | 3333 | 3240 | 2句/93词 | 完全保持 | 通过 |
| 钠离子电池正极 | 3544 | 3217 | 8句/327词 | 完全保持 | 通过 |

预算器只证明格式、长度与显式依赖不变，不能证明删除没有损害隐含语义、论证流畅或领域效用；这些必须由真人盲审判断。

## 与真人claim审核的接口审计

把两篇早期最终开发稿送入现有claim抽样器后，均被正确阻断：每篇只有15个带PH标识的独立可审核句子，其中Results 8个、Discussion 7个，低于冻结协议要求的20个；五类现象本身均超过每类2条的覆盖下限。这个缺口没有通过机械拆句掩盖，而是在后续matched compiler中前瞻性要求Results每段前两句和Discussion每段前两句形成带PH+REF的原子claim。

matched compiler v1的四篇文章分别得到23、38、25和28条候选，4/4通过readiness；Results候选为9/21/15/16，Discussion为14/17/10/12，每类PH均不少于2条。这首次证明生成器能稳定产出足够密度的人工审核对象。只有通过该readiness，系统才会物化双审与预承诺第三裁决包。由此，人审不再是文章生成完后随意加意见，而是生成器必须满足的下游接口和实验门禁。

## matched flat/graph编译器开发v1

两主题分别生成`flat_article_compiler`和`graph_dependency_compiler`，每篇严格9次调用：7个基础章节调用，加Results和Discussion各一次预承诺修复。两组共享writer input、五类PH、全部REF、三类综合、章节slot、调用顺序、word预算、确定性预算器；唯一操纵是graph组显式携带dependency edges，flat组只给同信息的独立记录与synthesis membership。

| 主题 | 条件 | 词数 | 可审核claim | 机器门禁 | 人审readiness | 9次全stop |
|---|---|---:|---:|---|---|---|
| 可解释医学影像 | flat | 2992 | 23 | 失败：截断REF | 通过 | 否 |
| 可解释医学影像 | graph | 3187 | 38 | 通过 | 通过 | 否 |
| 钠离子电池正极 | flat | 2835 | 25 | 通过 | 通过 | 否 |
| 钠离子电池正极 | graph | 3051 | 28 | 失败：截断PH | 通过 | 否 |

全部36次调用均收到响应，0传输失败；27次`finish_reason=stop`、9次为`length`。两篇机器失败稿各只失败于一个未注册的半截证据标识，分别是`REF-729c0c5fe068`和`PH-ef`。graph相对flat的可审核claim差值在两个主题上为+15和+3，均值+9；但样本只有两个开发主题，且graph平均多消耗约7204个prompt token和606个completion token，因此不能报告为有效性优势或显著结果。

v1没有达到“4/4篇机器门禁、人审门禁、9次stop全部通过”的预承诺资格线，所以没有启动8主题确认性文章实验。结果文件为`experiments/article_quality_v3/matched_article_compiler_claim_ready_development_v1_analysis.json`。

## token-safe开发v2与当前停止点

v1诊断后另行冻结了只改变分节`max_tokens`的v2：基础章节上限调整为Abstract 700、Introduction 1000、Methods 1200、Results 2400、Discussion 1900、Limitations 800、Conclusion 600；两次修复为2800/2300。其余prompt、任务、调用数、条件差异、门禁均保持不变。首次请求在获得任何有效响应前返回HTTP 402，因此v2当前为0/36，不能据此判断更高上限是否解决自然停止问题，也不能晋升确认性实验。

## 章节最小披露与低成本开发v3

v1的36次调用实际消耗474,063个prompt token。逐请求审计发现，完整writer input在九个章节节点重复发送，单个请求约4.4万–7.5万字符；其中约3.0万–3.6万字符来自source记录，包含对Methods、Conclusion、Abstract等章节没有直接作用的完整摘要和检索排序元数据。

新增确定性`section_writer_view`后，所有章节仍保留完整五类PH记录、全部REF标识、写作约束、非视觉图元数据和原始writer input哈希；Introduction、Results、Discussion及对应修复继续保留全部摘要原文，其他章节只保留REF、题名、年份和匹配关键词。所有章节删除retrieval query/rank、BM25 rank、引用量和质量门禁账本，且flat/graph两组使用完全相同的view。

对原36个调用用原有依赖上下文逐一重构后，请求正文从2,038,568降至1,377,636字符，减少32.42%；36/36个view通过PH规范化等价、REF全集、摘要哈希和无图像访问审计。按旧调用的字符/token比例估算约为320,365 prompt token，但该数字不是provider tokenizer实测或账单保证。

token-safe上限与最小披露组合已作为development v3在0个响应时独立冻结，4篇×9调用plan dry-run为ready。它不会恢复或覆盖v2的402失败记录；只有v3四篇全部通过完整联合门槛，才会生成八主题确认性计划。

## 对系统创新的影响

文章生成不再是“一次长prompt加一段图解释”，而是一个可审计的论证编译器：

1. 五类图程序输出形成PH节点，来源形成REF节点，三类跨现象命题形成综合节点。
2. 章节是依赖DAG中的下游节点；Discussion读取已冻结Results，Conclusion读取Results/Discussion/Limitations，Abstract最后读取全部章节。
3. 自动门禁先处理篇幅、章节、证据ID、现象覆盖和依赖闭合；真人不浪费时间检查机器可确定的格式错误。
4. 真人审核针对source、operator、claim和paragraph依赖边；否定上游证据会自动传播成下游修订worklist，而不是把审核意见简单追加到上下文。
5. 修改只允许发生在hash绑定的受影响子图，修订后再由独立专家判断收益和旁损伤。
6. 文章必须先通过claim-review readiness；目前两篇开发稿均因15<20而被阻断，不能仅凭机器质量门通过就进入真人分配。

## 下一轮确认性设计

不能直接把两篇开发成功稿写进论文主结果。建议先实现并冻结三个机器条件：

- `checkpoint_one_shot`：一次生成，作为生态基线。
- `flat_article_compiler`：与图编译器相同的分节、修复调用数和预算器，但依赖计划使用扁平列表。
- `graph_dependency_compiler`：使用PH–REF–synthesis–section依赖DAG。

flat与graph编译器必须完全匹配9次模型调用、模型、温度、section token预算、确定性预算器和writer input；唯一差异是依赖表示与调度。八主题确认性比较首先检验graph相对flat的跨现象综合、claim支持率和修订传播准确性，而不是再次证明“多调用优于单调用”。one-shot用于成本/生态参照，不承担同计算量机制主检验。

四类文章（上述三机器条件加同证据真人稿）进入独立专家盲评。长度需在评审前满足主题内1.10倍与条件均值1.05倍门限；主指标仍为claim支持、领域具体性、跨证据综合、研究效用和可追溯性。自动门禁通过不能替代真人结论。

## 关键工件

- 单次开发v1：`experiments/article_quality_v3/development_length_control_pilot_v1/`
- 单次检查点v2：`experiments/article_quality_v3/development_length_control_pilot_v2/`
- 图依赖编译器：`experiments/article_quality_v3/article_compiler_development_pilot_v1/`
- 分节修复：`experiments/article_quality_v3/article_compiler_repair_development_pilot_v1/`
- 确定性预算输出：`experiments/article_quality_v3/article_budget_optimizer_development_v1/`
- 编译器实现：`src/citeweave/article_compiler.py`
- 分节修复实现：`src/citeweave/article_section_repair.py`
- 预算优化实现：`src/citeweave/article_budget_optimizer.py`
- 人审接口readiness：`experiments/article_quality_v3/article_budget_optimizer_development_v1/oversight_readiness.json`
