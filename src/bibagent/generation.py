from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .evidence import EvidenceBundle
from .exceptions import ConfigurationError, EvidenceError
from .figure_catalog import PAPER_FIGURE_ORDER, figure_caption
from .io import atomic_write_bytes, write_json
from .models import ProjectConfig

EVIDENCE_CITATION_RE = re.compile(r"\[[^\]]*E\d{3}[^\]]*\]")
EVIDENCE_ID_RE = re.compile(r"E\d{3}")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)%?")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


@dataclass
class GenerationResult:
    manuscript: str
    review: str | None
    validation: dict[str, Any]
    model: str
    quality: dict[str, Any] | None = None


@dataclass(frozen=True)
class SectionContract:
    slug: str
    heading: str
    claim_types: frozenset[str]
    rhetorical_moves: tuple[str, ...]
    minimum_characters: int
    max_tokens: int


class DeepSeekClient:
    def __init__(self, api_key: str | None, base_url: str, model: str):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ConfigurationError("DEEPSEEK_API_KEY is required for LLM generation.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=180, follow_redirects=True)

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 7000,
    ) -> str:
        response: httpx.Response | None = None
        for attempt in range(7):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "thinking": {"type": "disabled"},
                        "stream": False,
                    },
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 6:
                        break
                    retry_after = response.headers.get("retry-after")
                    delay = min(float(retry_after) if retry_after else 2**attempt, 45)
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    break
                payload = response.json()
                return payload["choices"][0]["message"]["content"].strip()
            except httpx.HTTPError:
                if attempt == 6:
                    raise
                time.sleep(min(2**attempt, 30))
        if response is None:
            raise ConfigurationError("DeepSeek API did not return a response.")
        sanitized = response.text.replace(self.api_key, "***")
        raise ConfigurationError(
            f"DeepSeek API returned {response.status_code} after retries: {sanitized[:600]}"
        )


def _strip_structural_numbers(text: str) -> str:
    text = re.sub(
        r"(?i)(?:图|表|figure|table)\s*\d+(?:\.\d+)?",
        "",
        text,
    )
    text = re.sub(r"(?i)\bRQ\s*\d+\b", "", text)
    text = re.sub(r"§\s*\d+(?:\.\d+)*", "", text)
    text = re.sub(r"第\s*\d+(?:\.\d+)+\s*(?:节|部分|章)", "", text)
    return text


SYSTEM_PROMPT = """你是一名严谨的文献计量研究者和学术编辑。你的唯一事实来源是用户提供的
结构化证据包。必须遵循以下规则：
1. 每个实证性句子末尾使用一个或多个 [E000] 格式的证据编号。
2. 不得创造数字、作者、期刊、机构、DOI、算法结果或参考文献。
3. 不得自行计算证据包未明确列出的百分比、增长率或其他派生数字。
4. 不得把共现、共被引、合作或中心性解释成因果关系。
5. 不得把聚类标签解释成论文全文结论；若依据只是标题/摘要，要明确限定。
6. “显著”只能用于证据中明确提供统计检验的情况。
7. 区分数据源覆盖、元数据缺失、检索范围和真实学科现象。
8. 每段应完成一个清晰的论证动作，避免把图题改写成机械清单。
9. 输出中文 Markdown，语言自然、专业、克制，接近高质量人工论文。
10. 只输出指定文稿，不输出写作说明、虚构引文或参考文献列表。"""


COMMON_TYPES = frozenset(
    {
        "corpus_size",
        "acquisition_completeness",
        "metadata_coverage",
        "time_span",
        "method_reference",
    }
)

SECTION_CONTRACTS = (
    SectionContract(
        "introduction",
        "## 1 引言",
        COMMON_TYPES | {"work_content"},
        (
            "说明研究对象为何值得进行知识结构梳理",
            "用纳入文献的标题和摘要证据概括邻近研究，而非虚构领域史",
            "指出现有研究在覆盖范围、知识结构或方法组合上的具体空白",
            "提出四个可由本项目分析回答的研究问题",
            "说明本研究在数据审计、绩效分析、科学知识图谱和证据约束解释上的贡献",
        ),
        1800,
        3200,
    ),
    SectionContract(
        "methods",
        "## 2 数据与方法",
        COMMON_TYPES
        | {
            "annual_output_series",
            "coauthorship_structure",
            "institution_collaboration_structure",
            "keyword_cooccurrence_structure",
            "cocitation_structure",
            "bibliographic_coupling_structure",
            "figure",
        },
        (
            "设置2.1研究设计与数据源、2.2检索与纳入、2.3清洗与规范化、2.4绩效指标、2.5科学知识图谱、2.6可视化与可复现性",
            "交代检索式构成、年份边界、数据源、采集终止条件和完整性的相对含义",
            "解释去重、缺失值、实体规范化、完整计数及其选择理由",
            "报告网络单元、候选阈值、归一化、聚类、布局和显示子图规则",
            "区分分析全集与可视化筛选，说明研究不涉及人类受试者数据",
        ),
        2300,
        3900,
    ),
    SectionContract(
        "results_performance",
        "### 3.1 文献产出、类型与来源结构",
        {
            "corpus_size",
            "time_span",
            "annual_peak",
            "growth",
            "annual_output_series",
            "document_type_distribution",
            "top_source",
            "top_source_ranking",
            "bradford_zone_distribution",
            "figure",
        },
        (
            "依次解读年度产出、文献类型、来源排名和Bradford分区",
            "明确提及对应图号，并给出图中可核验的实体或数值",
            "每一组图均按观察、证据、有限解释和边界四步展开",
            "讨论不完整年度、来源缺失和完整计数对结论的影响",
        ),
        1400,
        2600,
    ),
    SectionContract(
        "results_constituents",
        "### 3.2 作者、机构与合作的社会结构",
        {
            "top_author",
            "top_author_ranking",
            "top_institution",
            "top_institution_ranking",
            "coauthorship_structure",
            "institution_collaboration_structure",
            "three_field_relations",
            "figure",
        },
        (
            "结合高产作者、高产机构、作者合作、机构合作和三字段关系解读社会结构",
            "比较生产力排名与网络位置，不能把二者等同",
            "给出代表性节点、聚类规模和网络稀疏性证据",
            "说明作者消歧、机构名称变体和显示阈值的边界",
        ),
        1500,
        2800,
    ),
    SectionContract(
        "results_impact",
        "### 3.3 引文影响与高被引文献",
        {
            "citation_summary",
            "zero_citation_share",
            "citation_distribution_detail",
            "top_cited_documents",
            "work_content",
            "figure",
        },
        (
            "解读均值、中位数、上分位和零被引比例所反映的偏斜",
            "介绍若干高被引文献的题名和摘要主题",
            "区分引用可见度、研究质量和新近性",
            "明确引文窗口与数据源时点限制",
        ),
        1300,
        2500,
    ),
    SectionContract(
        "results_conceptual",
        "### 3.4 概念结构与主题演化",
        {
            "keyword_cooccurrence_structure",
            "keyword_temporal_dynamics",
            "three_field_relations",
            "work_content",
            "figure",
        },
        (
            "结合关键词趋势、共现网络、主题图谱和三字段关系解释概念结构",
            "说明高频、连接度、聚类和时间变化各自回答的问题",
            "使用代表性文献摘要检验主题标签是否具有内容对应",
            "避免将作者关键词直接等同于研究结论",
        ),
        1600,
        2900,
    ),
    SectionContract(
        "results_intellectual",
        "### 3.5 知识基础与研究前沿",
        {
            "cocitation_structure",
            "bibliographic_coupling_structure",
            "top_cited_documents",
            "work_content",
            "figure",
        },
        (
            "区分共被引所表示的共享知识基础与文献耦合所表示的共享参考文献",
            "报告网络规模、代表性节点和聚类，并联系题名/摘要证据",
            "解释两种网络为何可能呈现不同结构",
            "声明网络结果受参考文献覆盖、候选筛选和参数化影响",
        ),
        1500,
        2800,
    ),
    SectionContract(
        "discussion",
        "## 4 讨论",
        frozenset(),
        (
            "设置4.1主要发现、4.2与既有研究的对照、4.3社会/概念/知识结构的综合解释、4.4研究与实践启示、4.5稳健性边界与未来议程",
            "逐一回应引言中的研究问题，但不要逐句复制结果",
            "至少综合两类分析形成一个跨结果判断",
            "只用代表性文献的标题和摘要进行对照，并明确证据层级",
            "提出由结果和局限直接导出的未来研究问题",
        ),
        2600,
        4300,
    ),
    SectionContract(
        "limitations",
        "## 5 局限",
        frozenset(),
        (
            "分别讨论数据源、检索词、元数据、引文窗口、实体消歧、网络参数、摘要证据和可视化筛选",
            "每项局限说明可能影响的分析、偏差方向和可执行缓解方案",
            "避免泛泛而谈",
        ),
        1000,
        2100,
    ),
    SectionContract(
        "conclusion",
        "## 6 结论",
        frozenset(),
        (
            "用紧凑段落回答四个研究问题",
            "区分已证实的描述性结论、受边界约束的解释和未来工作",
            "不得引入结果章节未出现的新事实",
        ),
        700,
        2600,
    ),
)


EDITORIAL_SYSTEM = """你是文献计量论文的执行编辑。先规划，后写作。输出一份中文Markdown
编辑计划：研究问题、贡献、章节论证目标、每节所用证据类型、图表归属、跨章节衔接、
风险和审校清单。不得撰写正文，不得创造证据包以外的事实。"""

REVIEWER_SYSTEMS = {
    "method": """你是文献计量方法审稿人。逐节检查检索可复现性、数据清洗、计数方法、
网络单元、阈值、归一化、聚类、显示筛选、引文窗口及解释边界。输出按优先级排序的
具体修改意见，每条包含章节、缺陷、后果和修复动作。""",
    "evidence": """你是证据与引用审计员。检查每个实证判断是否有相邻[E000]证据，
是否越过标题/摘要证据层级，是否出现因果化、无依据比较、虚构数字/实体/DOI，
以及图表叙述与证据是否一致。输出可执行的逐节修复清单。""",
    "narrative": """你是中文学术期刊编辑。按高质量文献计量论文标准检查研究空白、
贡献、研究问题、段落论证、图表解释深度、跨结果综合、既有研究对照、启示、未来议程、
重复和语言生硬。输出逐节修复清单，不补写事实。""",
}


def _editorial_prompt(config: ProjectConfig, evidence: EvidenceBundle) -> str:
    protocol = config.protocol
    contracts = [
        {
            "heading": item.heading,
            "moves": item.rhetorical_moves,
            "minimum_characters": item.minimum_characters,
        }
        for item in SECTION_CONTRACTS
    ]
    return f"""研究主题：{protocol.title}
检索关键词：{protocol.keywords}
年份范围：{protocol.year_from}–{protocol.year_to}
数据源：{protocol.source.value}

章节合同：
{json.dumps(contracts, ensure_ascii=False, indent=2)}

核心证据：
{evidence.prompt_packet(max_abstract_chars=350, claim_types=COMMON_TYPES | {"work_content"})}
"""


def _section_prompt(
    config: ProjectConfig,
    evidence: EvidenceBundle,
    contract: SectionContract,
    editorial_plan: str,
    *,
    result_context: str = "",
    review: str = "",
    original: str = "",
) -> str:
    claim_types = set(contract.claim_types)
    if not claim_types:
        claim_types = {item.claim_type for item in evidence.items}
    packet = evidence.prompt_packet(
        max_abstract_chars=1100,
        claim_types=claim_types,
    )
    mode = (
        "根据审稿意见修订原节。只输出修订后的完整目标章节。"
        if original
        else "撰写目标章节。只输出该章节，不要输出其他章节。"
    )
    figure_catalog = "\n".join(
        f"- 图{number}：{figure_caption(stem)}（{stem}）"
        for number, stem in enumerate(PAPER_FIGURE_ORDER, start=1)
    )
    return f"""研究主题：{config.protocol.title}
目标章节：{contract.heading}
最低有效字符数：{contract.minimum_characters}
必须完成的论证动作：
{chr(10).join(f"- {move}" for move in contract.rhetorical_moves)}

编辑计划：
{editorial_plan}

前置结果上下文（只能用于综合，不替代证据）：
{result_context or "无"}

审稿意见：
{review or "无"}

原章节：
{original or "无"}

目标证据包：
{packet}

{mode}
要求：保留目标标题；每个含事实、数字、实体、比较或网络解释的句子末尾添加证据编号；
可用图号与图名如下：
{figure_catalog}
结果章节中，每张实际讨论的图必须有一个独立的直接图解段落：段首写“图N显示/展示”，
该段只解释一张图并引用对应的figure证据；扩展讨论另起一段。不要把多张图集中罗列，
也不要生成参考文献列表。"""


def validate_manuscript(
    text: str, evidence: EvidenceBundle, *, strict_structure: bool = True
) -> dict[str, Any]:
    valid_ids = {item.evidence_id for item in evidence.items}
    used_ids = set(EVIDENCE_ID_RE.findall(text))
    invalid_ids = sorted(used_ids - valid_ids)
    malformed_evidence_tokens = sorted(set(re.findall(r"\[E\d{3}(?!\])", text)))
    observed_dois = {doi.lower().rstrip(".,;") for doi in DOI_RE.findall(text)}
    allowed_dois: set[str] = set()

    allowed_numbers: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            allowed_numbers.add(str(value))
            if isinstance(value, float):
                allowed_numbers.add(f"{value:.1f}")
                allowed_numbers.add(f"{value:.2f}")
                if value.is_integer():
                    allowed_numbers.add(str(int(value)))
            if isinstance(value, int):
                allowed_numbers.add(f"{value:,}")
        elif isinstance(value, str):
            allowed_dois.update(doi.lower().rstrip(".,;") for doi in DOI_RE.findall(value))
            for match in NUMBER_RE.findall(value):
                allowed_numbers.add(match.rstrip("%"))
                allowed_numbers.add(match.rstrip("%").replace(",", ""))

    for item in evidence.items:
        collect(item.value)
        for match in NUMBER_RE.findall(item.statement):
            allowed_numbers.add(match.rstrip("%"))

    # Remove headings and evidence citation blocks before scanning empirical numbers.
    stripped_source = text.split("## 参考文献", 1)[0]
    stripped = "\n".join(
        line for line in stripped_source.splitlines() if not line.lstrip().startswith("#")
    )
    stripped = EVIDENCE_CITATION_RE.sub("", stripped)
    # Formal references are checked against the DOI allow-list separately. DOI
    # fragments must not be mistaken for unsupported empirical measurements.
    stripped = DOI_RE.sub("", stripped)
    stripped = _strip_structural_numbers(stripped)
    observed = NUMBER_RE.findall(stripped)
    unsupported_numbers = []
    for number in observed:
        normalized = number.rstrip("%").replace(",", "")
        candidates = {normalized}
        try:
            numeric = float(normalized)
            candidates.update({str(int(numeric)) if numeric.is_integer() else str(numeric)})
        except ValueError:
            pass
        if not candidates.intersection({value.replace(",", "") for value in allowed_numbers}):
            # Section numbers and ordered-list markers are structural, not empirical.
            if normalized in {"1", "2", "3", "4", "5", "6"}:
                continue
            unsupported_numbers.append(number)
    unsupported_dois = sorted(observed_dois - allowed_dois)
    uncited_numeric_sentences = []
    # A bibliography is evidence metadata rather than an empirical claim. Keep
    # it under DOI validation, but exclude it from sentence-level claim binding.
    claim_text = text.split("## 参考文献", 1)[0]
    protected_text = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", claim_text.replace("\n", " "))
    sentences = re.findall(
        r".*?(?:[。！？](?:\s*\[[^\]]*E\d{3}[^\]]*\])*\s*|$)",
        protected_text,
    )
    for sentence in sentences:
        cleaned_sentence = sentence.replace("<DECIMAL>", ".").strip()
        if not cleaned_sentence or cleaned_sentence.startswith("#"):
            continue
        if re.search(r"\bRQ\d+\b", cleaned_sentence, re.IGNORECASE) and (
            "？" in cleaned_sentence or "?" in cleaned_sentence
        ):
            continue
        empirical_numbers = [
            value
            for value in NUMBER_RE.findall(
                _strip_structural_numbers(EVIDENCE_CITATION_RE.sub("", cleaned_sentence))
            )
            if value.rstrip("%") not in {"1", "2", "3", "4", "5", "6"}
        ]
        if empirical_numbers and not EVIDENCE_CITATION_RE.search(cleaned_sentence):
            uncited_numeric_sentences.append(cleaned_sentence[:200])
    required_sections = ["摘要", "数据与方法", "结果", "讨论", "局限", "结论"]
    missing_sections = [section for section in required_sections if section not in text]
    incomplete_paragraphs = _find_incomplete_paragraphs(text)
    quality = evaluate_manuscript_quality(text, evidence)
    minimum_content_pass = quality["passed"] if strict_structure else bool(text.strip())
    result = {
        "valid": (
            not invalid_ids
            and not malformed_evidence_tokens
            and not unsupported_numbers
            and not unsupported_dois
            and not uncited_numeric_sentences
            and not incomplete_paragraphs
            and (not missing_sections or not strict_structure)
            and minimum_content_pass
        ),
        "used_evidence_ids": sorted(used_ids),
        "evidence_coverage": len(used_ids) / len(valid_ids) if valid_ids else 0.0,
        "invalid_evidence_ids": invalid_ids,
        "malformed_evidence_tokens": malformed_evidence_tokens,
        "unsupported_numbers": sorted(set(unsupported_numbers)),
        "uncited_or_unapproved_dois": unsupported_dois,
        "uncited_numeric_sentences": uncited_numeric_sentences,
        "incomplete_paragraphs": incomplete_paragraphs,
        "missing_sections": missing_sections,
        "minimum_content_pass": minimum_content_pass,
        "journal_readiness": quality,
        "paragraphs": len([part for part in text.split("\n\n") if part.strip()]),
        "characters": len(text),
    }
    return result


def _section_text(text: str, heading: str, next_heading: str | None = None) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    if next_heading is None:
        return text[start:]
    end = text.find(next_heading, start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def evaluate_manuscript_quality(
    text: str,
    evidence: EvidenceBundle,
) -> dict[str, Any]:
    """Evaluate depth and structure against the frozen bibliometric benchmark."""
    paragraphs = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip() and not part.lstrip().startswith(("#", "|"))
    ]
    references_text = _section_text(text, "## 参考文献")
    reference_entries = re.findall(r"(?m)^\d+\.\s+.+$", references_text)
    figure_mentions = {
        int(value) for value in re.findall(r"图\s*(\d{1,2})", text) if 1 <= int(value) <= 16
    }
    used_ids = set(EVIDENCE_ID_RE.findall(text))
    result_text = _section_text(text, "## 3 结果", "## 4 讨论")
    discussion_text = _section_text(text, "## 4 讨论", "## 5 局限")
    methods_text = _section_text(text, "## 2 数据与方法", "## 3 结果")
    introduction_text = _section_text(text, "## 1 引言", "## 2 数据与方法")

    method_moves = {
        "data_source": any(term in methods_text for term in ("数据源", "数据库")),
        "query": "检索" in methods_text,
        "completeness": "完整" in methods_text,
        "normalization": any(term in methods_text for term in ("规范化", "归一化")),
        "counting": "计数" in methods_text,
        "threshold": any(term in methods_text for term in ("阈值", "候选")),
        "clustering": any(term in methods_text for term in ("聚类", "社区检测")),
        "visual_filter": any(term in methods_text for term in ("显示子图", "可视化筛选")),
        "reproducibility": any(term in methods_text for term in ("可复现", "参数")),
    }
    discussion_moves = {
        "principal_findings": "主要发现" in discussion_text,
        "prior_work_comparison": any(
            term in discussion_text for term in ("既有研究", "已有研究", "代表性文献")
        ),
        "cross_result_synthesis": any(
            term in discussion_text for term in ("综合来看", "共同表明", "结合", "相互印证")
        ),
        "implications": any(term in discussion_text for term in ("启示", "意义")),
        "future_agenda": any(
            term in discussion_text for term in ("未来研究", "后续研究", "研究议程")
        ),
        "evidence_boundary": any(
            term in discussion_text for term in ("标题和摘要", "标题与摘要", "证据层级")
        ),
    }
    required_subsections = [
        "### 2.1",
        "### 2.2",
        "### 2.3",
        "### 2.4",
        "### 2.5",
        "### 3.1",
        "### 3.2",
        "### 3.3",
        "### 3.4",
        "### 3.5",
        "### 4.1",
        "### 4.2",
        "### 4.3",
        "### 4.4",
        "### 4.5",
    ]
    checks = {
        "characters_at_least_12000": len(text) >= 12000,
        "substantive_paragraphs_at_least_18": len(paragraphs) >= 18,
        "subsections_at_least_15": sum(marker in text for marker in required_subsections) >= 15,
        "figure_mentions_at_least_12": len(figure_mentions) >= 12,
        "evidence_items_at_least_20": len(used_ids) >= min(20, len(evidence.items)),
        "references_at_least_8": len(reference_entries) >= 8,
        "introduction_at_least_1500": len(introduction_text) >= 1500,
        "methods_at_least_2000": len(methods_text) >= 2000,
        "results_at_least_4500": len(result_text) >= 4500,
        "discussion_at_least_2200": len(discussion_text) >= 2200,
        "research_questions_present": any(
            term in introduction_text for term in ("研究问题", "RQ1", "RQ 1")
        ),
        "method_moves_complete": all(method_moves.values()),
        "discussion_moves_complete": all(discussion_moves.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "characters": len(text),
        "substantive_paragraphs": len(paragraphs),
        "subsections": sum(marker in text for marker in required_subsections),
        "figure_mentions": sorted(figure_mentions),
        "unique_evidence_items": len(used_ids),
        "reference_entries": len(reference_entries),
        "section_characters": {
            "introduction": len(introduction_text),
            "methods": len(methods_text),
            "results": len(result_text),
            "discussion": len(discussion_text),
        },
        "method_moves": method_moves,
        "discussion_moves": discussion_moves,
    }


def _save_stage(stage_dir: Path | None, name: str, content: str) -> None:
    if stage_dir is None:
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(stage_dir / name, content.encode("utf-8"))


def _reference_list(text: str, evidence: EvidenceBundle) -> str:
    used_ids = set(EVIDENCE_ID_RE.findall(text))
    entries: list[str] = []
    for item in evidence.items:
        if item.evidence_id not in used_ids:
            continue
        if item.claim_type == "method_reference":
            value = item.value
            citation = value["citation"].rstrip(".")
            entries.append(f"{citation}. https://doi.org/{value['doi']} [{item.evidence_id}]")
        elif item.claim_type == "work_content":
            value = item.value
            authors = value.get("authors") or ["作者信息未提供"]
            author_text = ", ".join(authors[:6])
            if len(authors) > 6:
                author_text += ", et al."
            year = value.get("year") or "年份未提供"
            source = value.get("source") or "来源信息未提供"
            doi = value.get("doi")
            suffix = f" https://doi.org/{doi}" if doi else ""
            entries.append(
                f"{author_text}. ({year}). {value['title']}. *{source}*.{suffix} "
                f"[{item.evidence_id}]"
            )
    return "\n".join(
        ["## 参考文献", *[f"{index}. {entry}" for index, entry in enumerate(entries, 1)]]
    )


def _assemble_manuscript(
    config: ProjectConfig,
    abstract: str,
    sections: dict[str, str],
    evidence: EvidenceBundle,
) -> str:
    topic = "、".join(config.protocol.keywords) or config.protocol.title
    title = (
        f"{topic}主题研究的知识结构与演化：基于"
        f"{config.protocol.source.value}（{config.protocol.year_from}—"
        f"{config.protocol.year_to}）的可审计文献计量分析"
    )
    keyword_values = list(
        dict.fromkeys(
            [
                *config.protocol.keywords,
                "文献计量",
                "科学知识图谱",
                "共被引分析",
                "文献耦合",
                "研究前沿",
            ]
        )
    )
    keywords = f"**关键词**：{'；'.join(keyword_values)}"
    main_text = "\n\n".join(
        [
            f"# {title}",
            abstract.strip(),
            keywords,
            sections["introduction"].strip(),
            sections["methods"].strip(),
            "## 3 结果",
            sections["results_performance"].strip(),
            sections["results_constituents"].strip(),
            sections["results_impact"].strip(),
            sections["results_conceptual"].strip(),
            sections["results_intellectual"].strip(),
            sections["discussion"].strip(),
            sections["limitations"].strip(),
            sections["conclusion"].strip(),
        ]
    )
    main_text = _normalize_evidence_tokens(main_text)
    main_text = _bind_paragraph_numeric_claims(main_text, evidence)
    return f"{main_text}\n\n{_reference_list(main_text, evidence)}\n"


def _validation_failures(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "invalid_evidence_ids": validation["invalid_evidence_ids"],
        "malformed_evidence_tokens": validation.get("malformed_evidence_tokens", []),
        "unsupported_numbers": validation["unsupported_numbers"],
        "uncited_numeric_sentences": validation["uncited_numeric_sentences"],
        "incomplete_paragraphs": validation.get("incomplete_paragraphs", []),
        "uncited_or_unapproved_dois": validation["uncited_or_unapproved_dois"],
    }


def _normalize_evidence_tokens(text: str) -> str:
    text = re.sub(r"\[(E\d{3})(?!\])", r"[\1]", text)
    return re.sub(
        r"(?<![A-Za-z0-9\[])(E\d{3})(?![A-Za-z0-9\]])",
        r"[\1]",
        text,
    )


def _find_incomplete_paragraphs(text: str) -> list[str]:
    body = text.split("## 参考文献", 1)[0]
    incomplete: list[str] = []
    for paragraph in body.split("\n\n"):
        value = paragraph.strip()
        if (
            len(value) <= 100
            or value.startswith(("#", "|", "**关键词**"))
            or value[-1] in "。！？；：）】]"
        ):
            continue
        incomplete.append(value[-240:])
    return incomplete


def repair_incomplete_paragraphs(
    client: DeepSeekClient,
    evidence: EvidenceBundle,
    text: str,
) -> tuple[str, int]:
    """Repair only paragraphs truncated at a generation token boundary."""
    repaired_text = text
    repairs = 0
    for paragraph in text.split("\n\n"):
        value = paragraph.strip()
        if not _find_incomplete_paragraphs(value):
            continue
        evidence_ids = set(EVIDENCE_ID_RE.findall(value))
        packet = evidence.prompt_packet(
            max_abstract_chars=250,
            evidence_ids=evidence_ids or None,
        )
        repaired = client.complete(
            system=SYSTEM_PROMPT,
            user=(
                "下列段落在生成边界处被截断。只输出修复后的完整段落；保留原有事实、"
                "数字与证据编号，只补全被截断的句子并给出克制的边界解释。不得新增"
                "证据包之外的数字、实体、算法结果或参考文献。\n\n"
                f"原段落：\n{value}\n\n可用证据：\n{packet}"
            ),
            temperature=0.0,
            max_tokens=2800,
        )
        repaired = _normalize_evidence_tokens(repaired.strip())
        repaired_text = repaired_text.replace(paragraph, repaired, 1)
        repairs += 1
    return repaired_text, repairs


def _bind_paragraph_numeric_claims(text: str, evidence: EvidenceBundle) -> str:
    """Attach paragraph evidence to otherwise uncited, already-allowed numbers.

    This is deliberately conservative: it never introduces evidence from outside
    the paragraph, except for an exact figure-number to figure-evidence mapping.
    Unsupported numbers are still rejected by ``validate_manuscript``.
    """
    valid_ids = {item.evidence_id for item in evidence.items}
    figure_evidence_by_stem: dict[str, str] = {}
    for item in evidence.items:
        if item.claim_type == "figure":
            figure_evidence_by_stem[Path(item.artifact_path).stem] = item.evidence_id
    figure_ids = {
        number: figure_evidence_by_stem[stem]
        for number, stem in enumerate(PAPER_FIGURE_ORDER, start=1)
        if stem in figure_evidence_by_stem
    }

    rebuilt: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph_ids = [item for item in EVIDENCE_ID_RE.findall(paragraph) if item in valid_ids]
        if not paragraph_ids or paragraph.lstrip().startswith("#"):
            rebuilt.append(paragraph)
            continue
        fallback = "".join(f"[{item}]" for item in dict.fromkeys(paragraph_ids))

        def bind(
            match: re.Match[str],
            fallback_citations: str = fallback,
        ) -> str:
            sentence = match.group("sentence")
            punctuation = match.group("punctuation")
            if EVIDENCE_CITATION_RE.search(sentence):
                return match.group(0)
            numeric_values = NUMBER_RE.findall(_strip_structural_numbers(sentence))
            if not numeric_values:
                return match.group(0)
            mentioned_figures = [
                int(value)
                for value in re.findall(r"图\s*(\d{1,2})", sentence)
                if int(value) in figure_ids
            ]
            citations = (
                "".join(f"[{figure_ids[number]}]" for number in dict.fromkeys(mentioned_figures))
                if mentioned_figures
                else fallback_citations
            )
            return f"{sentence}{citations}{punctuation}"

        rebuilt.append(
            re.sub(
                r"(?P<sentence>[^。！？\n]+)(?P<punctuation>[。！？])",
                bind,
                paragraph,
            )
        )
    return "\n\n".join(rebuilt)


def _repair_section(
    client: DeepSeekClient,
    config: ProjectConfig,
    evidence: EvidenceBundle,
    contract: SectionContract,
    editorial_plan: str,
    section: str,
    *,
    result_context: str,
    attempts: int = 2,
) -> tuple[str, dict[str, Any], int]:
    section = _normalize_evidence_tokens(section)
    validation = validate_manuscript(section, evidence, strict_structure=False)
    if (
        validation["uncited_numeric_sentences"]
        and not validation["invalid_evidence_ids"]
        and not validation["unsupported_numbers"]
        and not validation["uncited_or_unapproved_dois"]
    ):
        # Final deterministic assembly binds these sentences only to evidence
        # already cited in the same paragraph (or the exact figure evidence).
        return section, validation, 0
    attempt = 0
    while not validation["valid"] and attempt < attempts:
        attempt += 1
        repair_review = (
            "程序化证据审计未通过。逐项修复以下问题："
            f"\n{json.dumps(_validation_failures(validation), ensure_ascii=False, indent=2)}"
            "\n不得自行计算比例、倍数、合计或外推值；若证据包没有精确值，删除该数字和"
            "由它推出的判断。为保留的年份、实体、数量和比较添加相邻证据编号。"
            f"修订后仍应不少于{contract.minimum_characters}个有效字符，并保持原有论证深度。"
        )
        section = client.complete(
            system=SYSTEM_PROMPT,
            user=_section_prompt(
                config,
                evidence,
                contract,
                editorial_plan,
                result_context=result_context,
                review=repair_review,
                original=section,
            ),
            temperature=0.0,
            max_tokens=contract.max_tokens,
        )
        section = _normalize_evidence_tokens(section)
        validation = validate_manuscript(section, evidence, strict_structure=False)
    return section, validation, attempt


def _repair_abstract(
    client: DeepSeekClient,
    evidence: EvidenceBundle,
    abstract: str,
    body: str,
    *,
    attempts: int = 2,
) -> tuple[str, dict[str, Any], int]:
    validation = validate_manuscript(abstract, evidence, strict_structure=False)
    attempt = 0
    while not validation["valid"] and attempt < attempts:
        attempt += 1
        abstract = client.complete(
            system=SYSTEM_PROMPT,
            user=(
                "修订下列结构式摘要，只输出完整摘要。保留“## 摘要”以及**目的**、"
                "**数据与方法**、**结果**、**结论**、**限制**。不得新增事实；删除证据包"
                "没有直接给出的比例、合计、外推和比较，并为所有实证数字添加相邻证据编号。"
                f"\n\n审计错误：\n"
                f"{json.dumps(_validation_failures(validation), ensure_ascii=False, indent=2)}"
                f"\n\n原摘要：\n{abstract}\n\n已验证正文：\n{body}\n\n证据包：\n"
                f"{evidence.prompt_packet(max_abstract_chars=300)}"
            ),
            temperature=0.0,
            max_tokens=1800,
        )
        validation = validate_manuscript(abstract, evidence, strict_structure=False)
    return abstract, validation, attempt


def generate_manuscript(
    config: ProjectConfig,
    evidence: EvidenceBundle,
    *,
    api_key: str | None = None,
    review_rounds: int = 1,
    candidate_dir: Path | None = None,
    stage_dir: Path | None = None,
) -> GenerationResult:
    """Generate a paper through planning, section drafting, review, and repair."""
    client = DeepSeekClient(api_key, config.llm_base_url, config.llm_model)
    stages = stage_dir or candidate_dir

    editorial_plan = client.complete(
        system=EDITORIAL_SYSTEM,
        user=_editorial_prompt(config, evidence),
        temperature=0.1,
        max_tokens=3500,
    )
    _save_stage(stages, "01_editorial_plan.md", editorial_plan)

    sections: dict[str, str] = {}
    result_context = ""
    for index, contract in enumerate(SECTION_CONTRACTS, start=1):
        context = (
            result_context if contract.slug in {"discussion", "limitations", "conclusion"} else ""
        )
        section = client.complete(
            system=SYSTEM_PROMPT,
            user=_section_prompt(
                config,
                evidence,
                contract,
                editorial_plan,
                result_context=context,
            ),
            temperature=0.15,
            max_tokens=contract.max_tokens,
        )
        sections[contract.slug] = section
        if contract.slug.startswith("results_"):
            result_context = "\n\n".join(
                value for key, value in sections.items() if key.startswith("results_")
            )
        _save_stage(stages, f"02_draft_{index:02d}_{contract.slug}.md", section)

    combined_review = ""
    for round_index in range(review_rounds):
        provisional = "\n\n".join(sections.values())
        reviews = []
        for reviewer, system in REVIEWER_SYSTEMS.items():
            review = client.complete(
                system=system,
                user=(
                    f"编辑计划：\n{editorial_plan}\n\n待审文稿：\n{provisional}\n\n"
                    "证据索引：\n"
                    f"{evidence.prompt_packet(max_abstract_chars=280)}"
                ),
                temperature=0.0,
                max_tokens=3600,
            )
            reviews.append(f"## {reviewer}\n\n{review}")
            _save_stage(
                stages,
                f"03_review_{round_index + 1:02d}_{reviewer}.md",
                review,
            )
        combined_review = "\n\n".join(reviews)

        revised: dict[str, str] = {}
        result_context = "\n\n".join(
            sections[contract.slug]
            for contract in SECTION_CONTRACTS
            if contract.slug.startswith("results_")
        )
        for index, contract in enumerate(SECTION_CONTRACTS, start=1):
            context = (
                result_context
                if contract.slug in {"discussion", "limitations", "conclusion"}
                else ""
            )
            revised_section = client.complete(
                system=SYSTEM_PROMPT,
                user=_section_prompt(
                    config,
                    evidence,
                    contract,
                    editorial_plan,
                    result_context=context,
                    review=combined_review,
                    original=sections[contract.slug],
                ),
                temperature=0.1,
                max_tokens=contract.max_tokens,
            )
            revised[contract.slug] = revised_section
            _save_stage(
                stages,
                f"04_revision_{round_index + 1:02d}_{index:02d}_{contract.slug}.md",
                revised_section,
            )
        sections = revised

    result_context = "\n\n".join(
        sections[contract.slug]
        for contract in SECTION_CONTRACTS
        if contract.slug.startswith("results_")
    )
    section_repair_rounds = 0
    for index, contract in enumerate(SECTION_CONTRACTS, start=1):
        context = (
            result_context if contract.slug in {"discussion", "limitations", "conclusion"} else ""
        )
        repaired, section_validation, attempts = _repair_section(
            client,
            config,
            evidence,
            contract,
            editorial_plan,
            sections[contract.slug],
            result_context=context,
        )
        sections[contract.slug] = repaired
        section_repair_rounds += attempts
        if attempts:
            _save_stage(
                stages,
                f"05_evidence_repair_{index:02d}_{contract.slug}.md",
                repaired,
            )
            if stages:
                write_json(
                    stages / f"05_evidence_repair_{index:02d}_{contract.slug}.json",
                    section_validation,
                )

    body_for_abstract = "\n\n".join(sections.values())
    abstract = client.complete(
        system=SYSTEM_PROMPT,
        user=(
            "根据下列已完成正文撰写结构式中文摘要。只输出“## 摘要”及五个段落："
            "**目的**、**数据与方法**、**结果**、**结论**、**限制**。"
            "摘要中的每个实证句必须保留正文所用证据编号，不得新增事实或数字；"
            "总长度650至1000个中文字符。\n\n"
            f"正文：\n{body_for_abstract}\n\n"
            f"核心证据：\n{evidence.prompt_packet(max_abstract_chars=250)}"
        ),
        temperature=0.0,
        max_tokens=1800,
    )
    abstract, abstract_validation, abstract_repairs = _repair_abstract(
        client,
        evidence,
        abstract,
        body_for_abstract,
    )
    section_repair_rounds += abstract_repairs
    _save_stage(stages, "06_structured_abstract.md", abstract)
    if stages:
        write_json(stages / "06_abstract_validation.json", abstract_validation)

    draft = _assemble_manuscript(config, abstract, sections, evidence)
    draft, boundary_repairs = repair_incomplete_paragraphs(client, evidence, draft)
    section_repair_rounds += boundary_repairs
    if boundary_repairs:
        _save_stage(stages, "07_boundary_repaired_candidate.md", draft)
    _save_stage(stages, "07_assembled_candidate.md", draft)
    if candidate_dir and candidate_dir != stages:
        _save_stage(candidate_dir, "candidate-00-draft.md", draft)

    validation = validate_manuscript(draft, evidence)
    validation["repair_rounds"] = section_repair_rounds
    if stages:
        write_json(stages / "validation_final.json", validation)
        write_json(
            stages / "manuscript_quality.json",
            validation["journal_readiness"],
        )
    if not validation["valid"]:
        raise EvidenceError(
            "Generated manuscript failed evidence validation: "
            + json.dumps(validation, ensure_ascii=False)
        )
    return GenerationResult(
        draft,
        combined_review or None,
        validation,
        config.llm_model,
        quality=validation["journal_readiness"],
    )


def finalize_staged_manuscript(
    config: ProjectConfig,
    evidence: EvidenceBundle,
    stage_dir: Path,
    *,
    api_key: str | None = None,
) -> GenerationResult:
    """Resume from reviewed section artifacts and run evidence-safe finalization."""
    plan_path = stage_dir / "01_editorial_plan.md"
    if not plan_path.exists():
        raise FileNotFoundError(f"Editorial plan is missing: {plan_path}")
    editorial_plan = plan_path.read_text(encoding="utf-8")
    client = DeepSeekClient(api_key, config.llm_base_url, config.llm_model)
    sections: dict[str, str] = {}
    for contract in SECTION_CONTRACTS:
        candidates = sorted(stage_dir.glob(f"05_evidence_repair_*_{contract.slug}.md"))
        if not candidates:
            candidates = sorted(stage_dir.glob(f"04_revision_*_{contract.slug}.md"))
        if not candidates:
            candidates = sorted(stage_dir.glob(f"02_draft_*_{contract.slug}.md"))
        if not candidates:
            raise FileNotFoundError(f"No staged section found for {contract.slug} in {stage_dir}")
        sections[contract.slug] = candidates[-1].read_text(encoding="utf-8")

    result_context = "\n\n".join(
        sections[contract.slug]
        for contract in SECTION_CONTRACTS
        if contract.slug.startswith("results_")
    )
    repair_rounds = 0
    for index, contract in enumerate(SECTION_CONTRACTS, start=1):
        repaired, section_validation, attempts = _repair_section(
            client,
            config,
            evidence,
            contract,
            editorial_plan,
            sections[contract.slug],
            result_context=(
                result_context
                if contract.slug in {"discussion", "limitations", "conclusion"}
                else ""
            ),
            attempts=3,
        )
        sections[contract.slug] = repaired
        repair_rounds += attempts
        _save_stage(
            stage_dir,
            f"05_evidence_repair_{index:02d}_{contract.slug}.md",
            repaired,
        )
        write_json(
            stage_dir / f"05_evidence_repair_{index:02d}_{contract.slug}.json",
            section_validation,
        )

    body = "\n\n".join(sections.values())
    abstract = client.complete(
        system=SYSTEM_PROMPT,
        user=(
            "根据下列已通过分节证据审计的正文撰写结构式中文摘要。只输出“## 摘要”及"
            "**目的**、**数据与方法**、**结果**、**结论**、**限制**五段。不得新增事实、"
            "比例、合计或外推；每个实证句必须使用正文已有的证据编号。"
            f"\n\n正文：\n{body}\n\n证据包：\n"
            f"{evidence.prompt_packet(max_abstract_chars=250)}"
        ),
        temperature=0.0,
        max_tokens=1800,
    )
    abstract, abstract_validation, abstract_attempts = _repair_abstract(
        client,
        evidence,
        abstract,
        body,
        attempts=3,
    )
    repair_rounds += abstract_attempts
    _save_stage(stage_dir, "06_structured_abstract.md", abstract)
    write_json(stage_dir / "06_abstract_validation.json", abstract_validation)

    manuscript = _assemble_manuscript(config, abstract, sections, evidence)
    manuscript, boundary_repairs = repair_incomplete_paragraphs(client, evidence, manuscript)
    repair_rounds += boundary_repairs
    if boundary_repairs:
        _save_stage(stage_dir, "07_boundary_repaired_candidate.md", manuscript)
    validation = validate_manuscript(manuscript, evidence)
    validation["repair_rounds"] = repair_rounds
    _save_stage(stage_dir, "07_assembled_candidate.md", manuscript)
    write_json(stage_dir / "validation_final.json", validation)
    write_json(stage_dir / "manuscript_quality.json", validation["journal_readiness"])
    if not validation["valid"]:
        raise EvidenceError(
            "Staged manuscript failed final validation: "
            + json.dumps(validation, ensure_ascii=False)
        )
    reviews = "\n\n".join(
        path.read_text(encoding="utf-8") for path in sorted(stage_dir.glob("03_review_*_*.md"))
    )
    return GenerationResult(
        manuscript,
        reviews or None,
        validation,
        config.llm_model,
        quality=validation["journal_readiness"],
    )


def save_generation(result: GenerationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_dir / "manuscript.md", result.manuscript.encode("utf-8"))
    if result.review:
        atomic_write_bytes(out_dir / "llm_review.md", result.review.encode("utf-8"))
    if result.quality:
        write_json(out_dir / "manuscript_quality.json", result.quality)
    write_json(
        out_dir / "generation_manifest.json",
        {
            "model": result.model,
            "validation": result.validation,
            "pipeline": "staged-bibliometric-writing-v2",
        },
    )
