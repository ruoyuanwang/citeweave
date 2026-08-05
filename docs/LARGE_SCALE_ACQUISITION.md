# 大规模元数据采集设计与运行手册

## 1. 目标与完成定义

大规模采集不是“请求成功若干页”，而是对一个冻结的检索协议证明：

1. 日期分片无重叠、无空洞；
2. 每个分片的来源报告数等于实际取得数；
3. 每个原始响应页已经持久化并通过 SHA-256 校验；
4. 合并后全局唯一记录数等于各分片预计总数；
5. 进程中断后只重做最小必要范围；
6. API 密钥、游标和临时响应不会写入最终公开查询描述；
7. `citeweave harvest-accept` 的五项检查全部通过。

这里的“全量”仍然是相对于数据源、检索式、日期范围、文献类型、语言和
采集时间的源内全量，不等于多个数据库合并后的绝对召回。

## 2. 为什么采用日期自适应分片

一个跨多年热门关键词可能返回几十万至数百万条结果。单一游标链存在三个问题：

- 任何一次长期中断都会使恢复范围过大；
- Crossref 游标约五分钟不用即可能失效；
- 活跃年份的数据在采集期间可能变化，导致总数漂移。

系统先对完整日期范围执行只返回计数的轻量请求。若预计数高于
`target_slice_records`，就按日期中点二分并分别重新计数，直到每个叶分片小于目标
规模或已经缩小到单日。日期边界使用闭区间，右侧分片从左侧结束日的下一天开始，
因此分片天然互斥。零结果日期范围同样保留在计划中，用于证明完整日期覆盖。

建议值：

| 数据规模 | `target_slice_records` | 说明 |
|---|---:|---|
| 1 万以下 | 10,000 | 一个或少量分片 |
| 1 万—10 万 | 10,000–25,000 | 便于快速恢复和核对 |
| 10 万—100 万 | 20,000–50,000 | 平衡计数请求与恢复成本 |
| 百万以上 | 25,000–100,000 | 优先考虑官方快照或下载工具 |

## 3. 存储结构

```text
project/
├─ raw/harvest/<YYYYMMDD-YYYYMMDD>/
│  └─ <source>-page-000001-<hash>.json.gz
├─ staged/
│  └─ source_records.jsonl.gz
└─ audit/
   ├─ harvest_manifest.json
   ├─ acquisition_manifest.json
   ├─ harvest.lock
   └─ state.json
```

每个 API 页先编码为确定性 JSON，计算未压缩内容的 SHA-256，再以 gzip 原子写入。
只有原始页写入成功后，才更新 `harvest_manifest.json` 中的游标、页号、记录数、路径、
哈希和压缩字节数。若进程恰好在两步之间退出，最多留下一个未被清单引用的孤立文件，
不会把未落盘页面误标为完成。

所有分片完成后，系统逐页读取压缩响应，通过磁盘 SQLite 主键表进行全局去重，
流式写入 `source_records.jsonl.gz.tmp`。合并成功后再原子替换正式文件，因此不需要
在内存中保留几十万条记录或几十万个标识符。

## 4. 中断、游标和并发策略

- **Europe PMC / OpenAlex**：每页保存 `nextCursorMark` / `next_cursor`，重启后直接从该
  游标继续。
- **Crossref**：服务端滚动游标可能连续多页返回相同 token，不能用字符串相等判断
  停滞；以页面长度和预计总数判断终点。由于游标可能过期，进程级中断后只重启当前
  未完成日期分片，已完成分片不重复下载。
- **进程锁**：`audit/harvest.lock` 使用排他创建，阻止两个采集进程写同一项目。
  锁中记录 PID；若进程已经不存在，下一次运行自动清除陈旧锁。
- **受控暂停**：`--page-budget N` 可在提交 N 个页面后留下合法的 partial 检查点，
  用于故障演练、分时下载或人工检查；再次执行同一命令即可恢复。

## 5. 限速与重试

所有 bulk 请求经过全局节流器。默认安全速率为：

- Crossref 公共池：0.8 请求/秒；
- Crossref polite 池（配置 `CROSSREF_MAILTO`）：2.5 请求/秒；
- Europe PMC：4 请求/秒；
- OpenAlex：5 请求/秒。

HTTP 429、5xx、连接错误和 JSON 解码错误使用指数退避并加入随机抖动；优先服从
`Retry-After`。请求尝试数和重试数进入最终采集警告。每次失败发生在页提交之前，
因此恢复时不会跳过未持久化的数据。

Crossref 当前官方列表请求限制为公共池 1 请求/秒、polite 池 3 请求/秒；响应头中的
限制值仍是运行时权威。参考：
[Crossref access and authentication](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)。

## 6. 数据源选择

### Crossref

适合几十万级主题元数据采集。每页最多 1,000 条，完整记录通常直接包含作者、机构、
来源、摘要（若出版商提供）、参考文献和被引用次数等字段。大结果集使用 cursor，
最后一页以返回条数少于 `rows` 判定。参考：
[Crossref large result sets](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/)。

### Europe PMC

检索接口每页最多 1,000 条，`resultType=core` 返回完整出版物核心元数据。
参考文献列表位于逐文献的独立接口中，不属于搜索页响应。为避免把“核心元数据全量”
和“参考文献富集全量”混成一个不可恢复任务，bulk 搜索阶段要求
`include_references=false`；参考文献应作为独立、可断点续传的队列运行。
参考：[Europe PMC REST API](https://dev.europepmc.org/RestfulWebService)。

初始化 Europe PMC bulk 项目时使用 `--no-references` 即可明确选择核心元数据阶段。

### OpenAlex

普通分页只能访问前 10,000 条，大结果必须使用 cursor；当前每页最多 100 条。
免费 API key 每日大约允许 1,000 次搜索、即 100,000 条搜索结果。几十万以上的筛选
任务可跨日恢复；全库或百万级本地分析应使用官方 OpenAlex CLI 或季度快照。
参考：[OpenAlex cursor paging](https://developers.openalex.org/guides/page-through-results)、
[OpenAlex authentication and pricing](https://developers.openalex.org/guides/authentication)、
[OpenAlex downloads](https://developers.openalex.org/download/overview)。

## 7. 命令

创建 bulk 项目：

```powershell
.\.venv\Scripts\citeweave.exe init runs\bulk-study `
  --title "Bibliometric metadata 2020-2025" `
  --keyword bibliometric --mode phrase `
  --from 2020 --to 2025 --source crossref `
  --bulk --target-slice-records 10000
```

采集、受控暂停、恢复和验收：

```powershell
.\.venv\Scripts\citeweave.exe harvest runs\bulk-study --page-budget 2
.\.venv\Scripts\citeweave.exe harvest runs\bulk-study
.\.venv\Scripts\citeweave.exe harvest-accept runs\bulk-study
```

完整项目也可以在 `project.yml` 中设置 `acquisition.mode: bulk` 后执行 `citeweave run`。
采集器会复用相同的压缩 staged 文件进入规范化阶段，不再额外构造一个来源记录列表。

## 8. 已验证的故障模型

自动测试使用合成 API 完成 109,800 条记录、100 个以上响应页的全量采集，并验证：

- 自适应日期分片和连续覆盖；
- 压缩原始页哈希；
- SQLite 磁盘去重和流式 JSONL；
- Crossref 未完成分片重启；
- Crossref 相同滚动游标 token；
- Europe PMC 精确游标恢复；
- 429 与 503 重试；
- 并发进程锁和陈旧锁恢复。
