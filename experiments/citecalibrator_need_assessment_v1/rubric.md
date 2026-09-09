# CiteCalibrator claim judge rubric v1

## 审核顺序

对每条 case 按以下顺序判断，不能先看规则或模型输出：

1. **证据是否足够作出判断？** 若 evidence packet 缺少决定性信息、存在无法消解的冲突，选择 `abstain`。
2. **核心事实是否与 verified answer、operator trace 和来源一致？** 核心数值、实体、路径或证据依赖错误且不能靠限定语修复时，选择 `reject`。
3. **解释强度是否越过 interpretation contract？** 因果化、机制化、全领域外推或遗漏必要的语料/参数限定，但核心观察仍成立时，选择 `qualify`，并给出最小修订。
4. **原句是否可原样保留？** 事实与解释均得到支持、证据充分且无实质修订必要时，选择 `accept`。

## 严重度

- `minor`：不改变主要事实、范围或可靠性理解的局部问题；
- `major`：会改变读者对结果范围、稳健性或机制的理解；
- `critical`：核心事实、来源依赖或主要结论错误。

训练必要性分析只把 `qualify/reject` 且严重度为 `major/critical` 的 case 计为主要缺陷。

## 最小修订规则

`qualify` 必须给出一条可直接替换原 claim 的最小修订。修订可以删去因果或机制断言、收窄到当前语料/图构建、补充参数敏感性限制；不得引入 evidence packet 中没有的新事实。若无法通过最小修订保留核心内容，应使用 `reject`。

## 证据引用

`decisive_evidence_ids` 只能选择 case 的 `allowed_evidence_ids`。`unsupported_spans` 必须是原 claim 的原文片段。发现引用不存在或依赖失效时，同时记录 `invalid_dependency_ids`。

## 双人首审与盲裁

每条 case 必须由两名不同审核者独立提交。以下任一关键字段不一致即进入第三人盲裁：四项布尔判断、action、risk types 或 severity。第三人只看原始 case，不看前两人的答案；其完整判断成为最终 gold。无分歧 case 直接采用首审共识。

审核界面和文件不得显示文章生成条件、规则预测、challenge intervention、其他审核者答案或任何预期标签。
