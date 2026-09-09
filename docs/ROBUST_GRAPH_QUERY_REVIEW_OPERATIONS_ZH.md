# 确认性 GraphRAG 候选主题盲审执行手册

## 当前状态

8个优先候选已通过自动数据门槛，每个候选已有100条哈希抽样的匿名题名—摘要记录。当前没有注册真人、没有人工判断、没有选定主题，也没有生成确认性扰动、gold或模型响应。

冻结输入：

- 协议：`experiments/graph_discovery_v2/robust_graph_synthesis_confirmation_v1_protocol.yml`
- 候选审计：`experiments/graph_discovery_v2/robust_graph_synthesis_confirmation_v1_candidate_audit.json`
- 空名单模板：`experiments/graph_discovery_v2/robust_graph_synthesis_confirmation_v1_query_reviewer_roster.json`
- 审核链修正案：`experiments/graph_discovery_v2/robust_graph_synthesis_confirmation_v1_amendment_003_query_review.yml`
- 访问交付修正案：`experiments/graph_discovery_v2/robust_graph_synthesis_confirmation_v1_amendment_004_access_delivery.yml`

## 1. 招募与伦理前置

在联系参与者前，研究负责人应根据所在机构要求取得伦理审批、豁免或“不属于人体研究”的书面判定，并冻结知情说明、补偿标准、退出规则和数据保留期限。代码不能替代这一步。

每个候选必须至少有3名冲突排除且具对应领域资格的真人：2名主审、1名可能的第三裁决者。同一人可以覆盖多个确有资格的领域，但不能因缺人虚报资格。资格可由相关学位、论文、项目或可审计的专业经历证明；实名和联系方式应保存在仓库外的受限映射中，实验文件只使用稳定的假名编号。

不要修改冻结的空模板。复制为新的填充名单，例如：

```powershell
Copy-Item -LiteralPath experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_reviewer_roster.json -Destination experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_reviewer_roster_filled_v1.json
```

在新文件中将`status`改为`real_query_reviewers_registered`，按模板字段填写`reviewers`。必须保留`real_human_attestation=true`、`synthetic_or_proxy_reviewer=false`、资格领域和候选冲突声明。完成后先冻结填充名单的SHA256；不得根据审核结果换人或修改冲突。

## 2. 生成盲法分配

```powershell
.venv\Scripts\python.exe scripts\prepare_robust_graph_query_review.py `
  --protocol experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol.yml `
  --protocol-freeze experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol_freeze.json `
  --candidate-audit experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_candidate_audit.json `
  --roster experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_reviewer_roster_filled_v1.json `
  --output-dir experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1
```

分配器会拒绝少于3名合格无冲突审核者的候选。审核者包中不含数据集ID、候选优先级、图统计、扰动、任务gold、模型结果、其他审核者身份或答案。私有分配manifest在收集期间只能由协调者访问。

## 3. 生成并交付访问令牌

在受信任的本地终端运行：

```powershell
.venv\Scripts\python.exe scripts\prepare_review_access.py `
  --manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_reviewer_roster_filled_v1.json `
  --output experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\access_hashes.json
```

命令只在终端显示一次明文令牌，`access_hashes.json`仅存SHA256。明文应逐人通过安全渠道交付，不得写入仓库、共享文档、实验包或聊天记录。访问哈希文件也不应进入公开候选包。

## 4. 主审收集

```powershell
.venv\Scripts\python.exe scripts\serve_robust_graph_query_review.py `
  --manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\collection_manifest.json `
  --access experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\access_hashes.json `
  --host 127.0.0.1 --port 8771
```

网页逐条显示两个注册概念、题名、摘要和年份。相关仅表示两个概念都是该记录的科学核心；“文本不足”不是模糊的相关标签。服务端拥有任务字段、会话和可见时间，提交后不可修改或重复。

全部主审完成后验证：

```powershell
.venv\Scripts\python.exe scripts\validate_robust_graph_query_review.py `
  --collection-manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\collection_manifest.json `
  --output experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\primary_validation.json
```

## 5. 分歧第三裁决

如果状态为`awaiting_blind_query_adjudication`，生成只包含分歧项的新集合：

```powershell
.venv\Scripts\python.exe scripts\prepare_robust_graph_query_adjudication.py `
  --primary-validation experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\primary_validation.json `
  --collection-manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\collection_manifest.json `
  --output-dir experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_adjudication_v1
```

用同一网页服务启动该目录的`adjudication_manifest.json`。第三人看不到主审身份和答案，且不能选择“无法判断”；每个分歧必须形成一个决定性标签。主审一致项禁止事后裁决。

## 6. 资格判定与停止规则

无分歧时省略`--adjudication-manifest`；存在分歧时传入裁决manifest：

```powershell
.venv\Scripts\python.exe scripts\finalize_robust_graph_query_review.py `
  --primary-validation experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\primary_validation.json `
  --collection-manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\collection_manifest.json `
  --adjudication-manifest experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_adjudication_v1\adjudication_manifest.json `
  --output experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_selection_v1.json
```

每个候选的100条最终标签计算相关率，门槛固定为0.80。若当前8个全部合格，按冻结优先级选中8个；若不足8个，选中列表保持为空，只能依次采集和审核预注册备用候选；若12个仍不足8个则停止，不允许临时换题。

在`selected_topics_ready_for_post_selection_freeze`出现且选择文件再次冻结前，严禁生成确认性扰动、查看确定性gold、构造200个模型请求或调用provider。

## 7. 选择后的确认面板构建与执行

只有上一步状态为`selected_topics_ready_for_post_selection_freeze`且选择文件已经单独冻结后，才可运行：

```powershell
.venv\Scripts\python.exe scripts\build_robust_graph_confirmation_panel.py `
  --protocol experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol.yml `
  --protocol-freeze experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_protocol_freeze.json `
  --selection experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_selection_v1.json `
  --review-collection experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_query_review_collection_v1\collection_manifest.json `
  --candidate-audit experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_candidate_audit.json `
  --initialization experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_initialization.json `
  --tokenizer-manifest experiments\graph_discovery_v2\formal_v3_execution_prerequisites\deepseek_v4_tokenizer_manifest.json `
  --output-root experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_panel
```

该命令只做确定性构建和冻结，不调用模型。成功后应对`post_selection_input_freeze.json`做独立复验，再先执行不带`--execute`的计划检查：

```powershell
.venv\Scripts\python.exe scripts\run_robust_graph_confirmation_panel.py `
  --freeze experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_panel\post_selection_input_freeze.json `
  --output-root experiments\graph_discovery_v2\robust_graph_synthesis_confirmation_v1_execution
```

独立复验通过且provider余额可用后，原命令增加`--execute`。执行器会重建全部200个请求哈希、重新核验原始数据和144个图状态并做余额预检；不得绕过专用执行器直接调用通用runner。
