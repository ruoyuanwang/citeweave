# 大规模元数据清洗、去重与结构化运行手册

## 1. 交付目标

本模块把 `staged/source_records.jsonl(.gz)` 转换为可审计、可断点恢复、可直接供可视化使用的 Parquet 数据集。它不是把“流式输入”重新收集成 Python 列表，而是：

1. 每次只在 Python 中规范化固定数量的来源记录；
2. 每批原子写入分区 Parquet，并在成功后更新检查点；
3. 用 DuckDB 对所有分区执行全局去重、关系重映射和磁盘溢写；
4. 严格执行研究协议的年份范围，并把排除记录保留在隔离账本；
5. 生成完整统计表和候选优先的网络边表；
6. 独立校验文件哈希、主键、外键、输入守恒与可视化表契约。

该阶段的完成标准是 `citeweave process-accept <project>` 六项检查全部通过，而不是“程序没有报错”。

## 2. 命令

所有命令必须在项目虚拟环境中执行：

```powershell
.\.venv\Scripts\citeweave.exe process runs\bulk-study
.\.venv\Scripts\citeweave.exe process-accept runs\bulk-study
```

常用控制项：

```powershell
# 改变单批上限；中断后必须使用相同值恢复
.\.venv\Scripts\citeweave.exe process runs\bulk-study --chunk-size 2000

# 从 processing_manifest.json 恢复
.\.venv\Scripts\citeweave.exe process runs\bulk-study --resume

# 提交两个批次后受控暂停，用于故障演练；再次执行即可恢复
.\.venv\Scripts\citeweave.exe process runs\bulk-study --batch-budget 2

# 放弃旧分区并从 staged 文件重新规范化
.\.venv\Scripts\citeweave.exe process runs\bulk-study --no-resume

# 规范化规则未变时，仅从已有分区重建规范表、质量表和可视化表
.\.venv\Scripts\citeweave.exe process runs\bulk-study --refinalize
```

`project.yml` 中的处理策略：

```yaml
processing:
  mode: disk
  chunk_size: 1000
  duckdb_memory_limit: 4GB
  candidate_pool_size: null   # null = max(visualization_max_nodes * 8, 400)
  edge_row_limit: 200000
  keep_partitions: true
```

## 3. 数据分层与恢复语义

```text
staged/source_records.jsonl.gz
  └─ canonical/_parts/<table>/part-000001.parquet
       └─ DuckDB 全局归并
            ├─ canonical/*.parquet
            ├─ canonical/visualization/*.parquet
            ├─ quality/excluded_records.parquet
            ├─ quality/processing_report.json
            ├─ canonical/schema.json
            └─ audit/processing_manifest.json
```

- 每个分区先写入 `.tmp`，成功后原子替换正式文件。
- 一个批次的所有表完成后才推进 `records_processed`。
- 崩溃发生在检查点前时，恢复会安全覆盖同一批次，不会重复提交。
- gzip 无法可靠随机定位，因此普通恢复会顺序跳过已提交记录；不会将跳过记录载入列表。
- `processing.lock` 防止两个进程同时写同一项目；进程消失后陈旧锁自动回收。
- `--refinalize` 要求每张分区表的批次数与清单一致。
- 清单记录 pipeline/cleaning rules 版本、Python/Pandas/DuckDB 版本和全部输出参数；清洗规则变化时拒绝混用旧分区，年份或网络阈值变化时要求 `--refinalize`。

## 4. 固化在代码中的清洗规则

### 文本与标识符

- HTML 标签移除、HTML entity 解码、Unicode NFKC、连续空白折叠；
- DOI 转小写并移除 `doi:`/resolver 前缀，格式不合法时存为 null；
- ORCID 格式校验，存在时优先作为作者标识；
- Crossref 无 ORCID 作者使用“规范姓名 + 主机构”生成保守标识，降低常见姓名误合并；
- Crossref `author.sequence=first` 只代表作者顺序，不推断通讯作者；
- 机构名称从自由文本中保守提取组织成分，原始记录哈希始终保留。

### 作品去重

优先级固定且确定：

1. 精确规范 DOI；
2. 无 DOI 时，长度至少 20 个字符的精确规范标题 + 年份；
3. 短标题或缺标题记录使用来源记录哈希，不自动合并。

候选组内保留元数据字段最完整的记录；平局依次使用来源记录哈希、作品 ID 和输入顺序。所有被移除作品的作者关系、关键词、主题、来源记录和引用关系均重映射到保留作品，不能直接丢弃。

当前规则有意偏保守。ORCID/ROR 候选合并、跨机构作者迁移和机器学习消歧应作为带人工审批的后续实体解析层，不能由 LLM 无日志修改。

### 研究范围

主分析表只保留 canonical publication year 位于 `project.yml` 的 `year_from`–`year_to` 的作品。来源检索过滤与最终出版年不一致时：

- 不静默删除；
- 来源记录写入 `quality/excluded_records.parquet`；
- 记录 `missing_publication_year` 或 `outside_protocol_year_range`；
- 校验 `纳入来源记录数 + 隔离来源记录数 = staged 输入记录数`；
- 所有以作品为外键的关系同步过滤，孤儿数必须为零。

## 5. 规范表与可视化表

### 规范星型/关系模型

| 表 | 粒度 | 主要用途 |
|---|---|---|
| `works` | 每篇纳入作品一行 | 年度、类型、来源、引文统计 |
| `authors` | 每个保守作者实体一行 | 作者节点与标签 |
| `institutions` | 每个机构实体一行 | 机构节点与国家字段 |
| `sources` | 每个出版来源一行 | 期刊/来源生产力 |
| `authorships` | 作品–作者–机构关系 | 合著和机构合作 |
| `keywords` | 作品–关键词关系 | 主题共现 |
| `topics` | 作品–来源主题关系 | OpenAlex/Europe PMC 主题 |
| `references` | citing–cited 唯一有向边 | 直接引文、共被引、耦合 |
| `provenance` | 来源记录–保留作品映射 | 回溯原始记录 |
| `duplicates` | 被移除–保留作品映射 | 去重审计 |

所有列、主键和外键写入 `canonical/schema.json`。

### 可视化就绪表

`canonical/visualization/` 包含：

- `annual_output`
- `document_types`
- `languages`
- `source_productivity`
- `author_productivity`
- `institution_productivity`
- `keyword_occurrences`
- `topic_occurrences`
- `reference_impact`
- `coauthor_edges`
- `institution_collaboration_edges`
- `keyword_cooccurrence_edges`
- `cocitation_edges`
- `direct_citation_edges`

描述统计和出现次数使用全部规范记录，不抽样。网络先在完整关系表上计算实体出现次数，再选择有限候选实体，最后才展开实体对；因此限制的是展示候选空间，不是基础计数。候选池、边上限和每个输出文件的行数、路径、SHA-256 均写入清单。

## 6. 缺少来源关键词时的处理

如果来源关键词覆盖率低于 50%，系统会自动运行磁盘式、全语料 TF-IDF：

1. 从所有纳入作品的标题和摘要生成 unigram/bigram；
2. 删除英文停用词、纯数字和通用学术措辞；
3. 对全语料计算 document frequency；
4. 选取最多 5,000 个候选词；
5. 为每篇作品保留分数最高的 5 个词。

该过程不抽样，也不把整个稀疏矩阵放入 Python 内存。派生词使用 `keyword_type=derived_tfidf_full_corpus`，绝不伪装成作者关键词；方法参数、覆盖率、词表大小和派生行数写入质量报告。

## 7. 自动质量门禁

`quality/processing_report.json` 至少证明：

- staged 输入记录全部得到纳入或隔离解释；
- 作品、作者、机构、来源主键唯一；
- 所有作品侧外键孤儿数为 0；
- 规范年份全部位于协议范围；
- DOI、标题、摘要、年份、语言、来源、出版商、引文等字段覆盖率可量化；
- 去重规则和关系重映射规则已记录；
- 关键词派生是否执行及其参数已记录。

`process-accept` 再独立读取 Parquet，而不是信任质量报告自身，重新计算主外键、行数和哈希。

## 8. 已完成的真实压力验收

验收项目：`runs/bulk-crossref-bibliometric`

- staged 输入：46,336 条 Crossref 来源记录；
- 批次：47（每批 1,000）；
- 原始来源参考关系：1,737,970；
- 规范唯一引用边：约 160 万；
- 协议：2020–2025；
- 来源检索与 canonical year 不一致的 2026 记录：632 条，全部进入隔离账本；
- 主分析作品：45,704；
- 来源关键词覆盖率：0，因此自动生成约 21.8 万条显式标记的全语料 TF-IDF 关系；
- Python 规范化阶段观察到的常驻内存约 370 MB；
- `process-accept`：6/6 通过；
- 27 项自动测试与 Ruff 静态检查通过。

权威运行证据位于：

- `audit/processing_manifest.json`
- `audit/processing_acceptance_console.txt`
- `quality/processing_report.json`
- `quality/excluded_records.parquet`
- `canonical/schema.json`

上述验收证明了本机对“数万作品 + 百万级关系”的完整链路。百万作品级能力仍应在目标硬件上做单独容量测试；系统设计允许 DuckDB 磁盘溢写和固定 Python 批次，但不把未经实际测量的规模声明为已验证。

可用以下命令在目标机器上执行包含规范化器的全链路容量测试，而不仅是 SQL 聚合微基准：

```powershell
.\.venv\Scripts\citeweave.exe benchmark-processing runs\processing-benchmark-100k `
  --documents 100000 --references-per-document 10 --chunk-size 2000
```

该命令只允许写入一个尚无 `project.yml` 的新目录，以避免覆盖真实研究项目；完成后会生成 `audit/processing_benchmark.json` 和完整的 `process-accept` 证据。

本机已完成该基准：

- 100,000 篇作品；
- 1,000,000 条唯一参考关系；
- 200,000 条作者关系；
- 50 个分区批次，每批 2,000；
- 完整处理耗时 244.397 秒；
- 质量门禁通过；
- 独立验收 6/6 通过。

权威结果为 `runs/processing-benchmark-100k/audit/processing_benchmark.json`。这证明当前实现已验证到十万作品/百万关系；它仍不等于百万作品已验证。
