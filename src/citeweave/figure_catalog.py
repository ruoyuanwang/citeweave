from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")


PAPER_FIGURE_ORDER = (
    "annual_publications",
    "document_types",
    "top_sources",
    "bradford_sources",
    "top_authors",
    "top_institutions",
    "network_coauthorship",
    "network_institution_collaboration",
    "citation_distribution",
    "top_cited_documents",
    "keyword_trends",
    "network_keyword_cooccurrence",
    "network_keyword_overlay",
    "network_keyword_density",
    "network_cocitation",
    "network_bibliographic_coupling",
    "network_citation",
)


FIGURE_CAPTIONS = {
    "annual_publications": "年度科学产出趋势",
    "document_types": "文献类型构成",
    "top_sources": "主要来源期刊与出版载体",
    "bradford_sources": "Bradford 来源分区",
    "top_authors": "高产作者分布",
    "top_institutions": "高产机构分布",
    "network_coauthorship": "作者合作网络",
    "network_institution_collaboration": "机构合作网络",
    "citation_distribution": "文献被引次数分布",
    "top_cited_documents": "高被引文献",
    "keyword_trends": "高频关键词年度演化",
    "network_keyword_cooccurrence": "关键词共现网络",
    "network_keyword_overlay": "关键词时间叠加图",
    "network_keyword_density": "关键词密度图",
    "network_cocitation": "参考文献共被引网络",
    "network_citation": "语料库直接引文网络",
    "network_bibliographic_coupling": "文献耦合网络",
}


# Anchors are deterministic placement hints for legacy manuscripts that predate
# explicit figure directives.  They describe the first paragraph that starts
# the substantive interpretation of each figure, not an earlier cross-reference.
FIGURE_ANCHORS = {
    "annual_publications": "**年度产出趋势。**",
    "document_types": "**文献类型结构。**",
    "top_sources": "**来源排名。**",
    "bradford_sources": "**Bradford分区。**",
    "top_authors": "**高产作者与高产机构。**",
    "top_institutions": "在机构层面",
    "network_coauthorship": "**作者合作网络的结构特征。**",
    "network_institution_collaboration": "**机构合作网络的结构特征。**",
    "citation_distribution": "本语料库的引文影响分布呈现",
    "top_cited_documents": "高被引文献的题名与摘要主题呈现",
    "keyword_trends": "对全球频次最高的关键词按年度统计",
    "network_keyword_cooccurrence": "在关键词共现网络中",
    "network_keyword_overlay": "**时间覆盖与元数据边界**",
    "network_keyword_density": "**时间覆盖与元数据边界**",
    "network_cocitation": "共被引网络以语料库中",
    "network_citation": "共被引网络与文献耦合网络在本语料库",
    "network_bibliographic_coupling": "文献耦合网络基于候选文献间",
}


# A concise, non-interpretive reading guide is placed directly below every
# figure in Word.  The following manuscript paragraph then provides the
# evidence-backed expansion, producing a conventional figure -> reading ->
# discussion sequence without asking an LLM to invent bridging claims.
FIGURE_READING_GUIDES = {
    "annual_publications": (
        "以年份为横轴、文献量为纵轴呈现年度科学产出，便于比较年度增量、"
        "总体增长幅度以及末年数据可能存在的截断。"
    ),
    "document_types": (
        "比较语料库中主要文献类型的数量构成，较长的条形表示该类型在当前"
        "元数据快照中占比更高。"
    ),
    "top_sources": (
        "按发文量排列主要来源期刊与出版载体，用于识别产出集中的头部来源"
        "及其相对梯度。"
    ),
    "bradford_sources": (
        "对来源数量与文献产出进行Bradford分区，用于比较核心区与外围区在"
        "来源规模和产出集中度上的差异。"
    ),
    "top_authors": (
        "按发文量展示高产作者，条形长度编码作者在去重语料库中的文献数量，"
        "用于观察头部作者与长尾作者的产出差异。"
    ),
    "top_institutions": (
        "按发文量展示高产机构，便于比较机构层面的产出梯度，并识别名称"
        "规范化可能影响排名的条目。"
    ),
    "network_coauthorship": (
        "节点表示作者、连线表示合作关系；节点大小编码出现频次，颜色表示"
        "社区，图中仅保留阈值筛选和边约简后的核心合作结构。"
    ),
    "network_institution_collaboration": (
        "节点表示机构、连线表示机构间合作；节点大小编码出现频次，颜色表示"
        "社区，用于辨识核心机构群及跨机构连接。"
    ),
    "citation_distribution": (
        "展示文献被引次数的频数分布，并对极端高值进行显示截断，使主体区间"
        "的右偏形态与低被引文献比例保持可读。"
    ),
    "top_cited_documents": (
        "按来源报告的被引次数排列高被引文献，用于比较当前数据源与检索时点"
        "下的引用可见度，而非直接评价研究质量。"
    ),
    "keyword_trends": (
        "以年份为横轴比较高频关键词的年度文献量，曲线的上升、平台或回落"
        "分别提示不同的时间变化轨迹。"
    ),
    "network_keyword_cooccurrence": (
        "节点表示关键词、连线表示共现关系；节点大小编码出现频次，颜色表示"
        "社区，空间邻近反映归一化后的关联结构。"
    ),
    "network_keyword_overlay": (
        "沿用关键词共现网络的位置，以颜色映射关键词的平均发表年份，用于"
        "辨识相对较早与较新的概念区域。"
    ),
    "network_keyword_density": (
        "在关键词网络布局上叠加平滑密度，颜色强度综合节点权重与空间邻近性，"
        "用于识别概念活动较集中的区域。"
    ),
    "network_cocitation": (
        "节点表示被共同引用的参考文献，连线表示共被引关系；核心社区用于"
        "刻画当前语料库共享的历史知识基础。"
    ),
    "network_citation": (
        "节点表示语料库文献，连线表示直接引用关系；稀疏化后的社区呈现当前"
        "语料库内部可观察到的引文连接结构。"
    ),
    "network_bibliographic_coupling": (
        "节点表示语料库文献，连线表示共享参考文献形成的耦合关系；相对紧密"
        "的社区用于近似观察当前研究前沿。"
    ),
}


FIGURE_READING_EXPANSIONS = {
    "network_keyword_overlay": (
        "该视图只改变节点的时间颜色编码，不改变共现网络的节点位置与连接关系；"
        "因此颜色差异应解释为相对时间位置，而不能据此推断主题之间的因果演化。"
    ),
    "network_keyword_density": (
        "密度热点反映筛选后关键词在当前布局与平滑参数下的局部集中程度；它适合"
        "定位概念活动区域，但不代表未显示关键词不存在，也不能替代节点级频次比较。"
    ),
    "network_citation": (
        "直接引文网络描述语料库文献之间可观测的引用方向，与共被引网络的共享"
        "知识基础、文献耦合网络的共享参考文献含义不同，三类网络不可相互替代。"
    ),
}


FIGURE_SECTIONS = {
    "annual_publications": "3.1",
    "document_types": "3.1",
    "top_sources": "3.1",
    "bradford_sources": "3.1",
    "top_authors": "3.2",
    "top_institutions": "3.2",
    "network_coauthorship": "3.2",
    "network_institution_collaboration": "3.2",
    "citation_distribution": "3.3",
    "top_cited_documents": "3.3",
    "keyword_trends": "3.4",
    "network_keyword_cooccurrence": "3.4",
    "network_keyword_overlay": "3.4",
    "network_keyword_density": "3.4",
    "thematic_map": "3.4",
    "three_field_map": "3.4",
    "network_cocitation": "3.5",
    "network_citation": "3.5",
    "network_bibliographic_coupling": "3.5",
}


def figure_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item["name"])
    return str(item.name)


def order_figures(figures: Iterable[T]) -> list[T]:
    order = {name: index for index, name in enumerate(PAPER_FIGURE_ORDER)}
    return sorted(figures, key=lambda item: (order.get(figure_name(item), 10_000), figure_name(item)))


def figure_caption(name: str) -> str:
    return FIGURE_CAPTIONS.get(name, name.replace("_", " ").title())


def figure_anchor(name: str) -> str:
    return FIGURE_ANCHORS.get(name, "")


def figure_reading_guide(name: str) -> str:
    return FIGURE_READING_GUIDES.get(
        name,
        f"展示{figure_caption(name)}的确定性分析结果，后文给出证据约束下的扩展解读。",
    )


def figure_reading_expansion(name: str) -> str:
    return FIGURE_READING_EXPANSIONS.get(name, "")


def figure_section(name: str) -> str:
    return FIGURE_SECTIONS.get(name, "3.5")
