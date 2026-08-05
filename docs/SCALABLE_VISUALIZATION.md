# BibAgent 大规模文献计量可视化

## 1. 运行

必须使用项目虚拟环境：

```powershell
.\.venv\Scripts\bibagent.exe visualize runs\your-project
.\.venv\Scripts\bibagent.exe visualize-accept runs\your-project
```

HTTP API 同时提供：

- `POST /v1/projects/visualize?project_dir=...`
- `GET /v1/projects/visualize/accept?project_dir=...`

## 2. 为什么不直接画全量网络

年度产出、来源、作者、机构、文献类型和引文分布由 DuckDB 扫描全量 Parquet 计算，不抽样。关系网络不把全部实体做成 `N × N` 矩阵，而采用候选优先的稀疏流程：

1. 从处理阶段生成的稀疏边端点确定可连接候选项。
2. 根据图类型的最低出现次数和候选池目标，自适应计算阈值。
3. 使用关联强度 `c_ij / (w_i × w_j)` 标准化关系。
4. 用出现次数与总连接强度联合排序，并按社区分配节点配额。
5. 保留最大生成森林与每个节点最强的若干条边。
6. 语义网络使用两级 VOS 风格社区布局；多组件作者网络使用 ForceAtlas2 LinLog。
7. 多个固定随机种子分别布局，以节点碰撞、边长和社区分离指标选择最佳结果。
8. 节点、边、标签和社区都有硬上限，所有实际阈值写入方法清单。

这与 VOSviewer 推荐的关联强度、阈值选择、标签控制和多起点布局思想一致，同时保留了可审计的 Python 实现。VOSviewer 官方资料：[功能概览](https://www.vosviewer.com/features/highlights)、[用户手册](https://www.vosviewer.com/documentation/Manual_VOSviewer_1.6.4.pdf)。

书目耦合使用另一条候选优先路径：先按引文与参考文献信息选出有限候选文献，再仅对候选文献的共享参考文献做 DuckDB 稀疏自连接；高频泛化参考文献和共享数不足的关系会被过滤。该路径不会创建全语料文献对矩阵。

## 3. 图表

当前生成：

- 年度科学产出
- 高产来源、作者和机构
- 文献类型构成
- 引文分布和高被引文献
- Bradford 来源分区
- 关键词年度演化热图
- 作者合作网络
- 机构合作网络
- 关键词共现网络
- 语料内直接引文网络
- 参考文献共被引网络
- 文献书目耦合网络
- 关键词平均发表年份覆盖图
- 关键词密度图

输入没有相应关系时，图会被标记为跳过，不会伪造连接。

## 4. 可复现产物

`figures/`：

- 每张图的 PNG 与可编辑 SVG
- `figure_manifest.json`：文件哈希、尺寸、事实、选择参数、布局指标和运行耗时

`analyses/visualization/`：

- `{network}_nodes.parquet`：最终节点、社区、重要性、`x/y` 坐标
- `{network}_edges.parquet`：最终稀疏边及关联强度
- `{network}_method.json`：阈值、候选规模、边压缩、布局与质量指标

`audit/visualization_acceptance.json`：

- 文件存在性和哈希
- PNG/SVG 成对性与最小尺寸
- 图像非空检查
- 节点上限和稀疏矩阵契约
- 网络底表与方法卡完整性
- 布局节点碰撞上限

自动验收不调用 VLM。开发阶段由 Codex 查看真实 PNG 并多轮调整算法；这是本阶段刻意保留的人类视觉回路。

## 5. 已完成验收

真实 Crossref 项目：

- 45,704 篇文献、1,609,865 条参考关系
- 17 张 PNG/SVG 图
- 六类核心网络全部生成
- 最终可视化运行约 25 秒
- 没有创建全邻接矩阵
- 自动验收全部通过

合成压力项目：

- 100,000 篇文献、1,000,000 条参考关系
- 16 张 PNG/SVG 图
- 运行约 16 秒
- 进程内存约 0.5 GB
- 合成语料没有语料内直接引文边，因此该图按合同跳过
- 自动验收全部通过

验收结果分别位于：

- `runs/bulk-crossref-bibliometric/audit/visualization_acceptance.json`
- `runs/processing-benchmark-100k/audit/visualization_acceptance.json`
