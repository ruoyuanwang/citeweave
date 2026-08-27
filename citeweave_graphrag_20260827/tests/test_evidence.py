from datetime import UTC, datetime

from citeweave.analytics import analyze
from citeweave.evidence import bind_claims, build_evidence
from citeweave.generation import (
    _bind_paragraph_numeric_claims,
    _normalize_evidence_tokens,
    evaluate_manuscript_quality,
    validate_manuscript,
)
from citeweave.models import AcquisitionManifest, SourceName
from citeweave.transform import Canonicalizer


def _bundle(crossref_records, graph_explanations=None):
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    analyses = analyze(tables, network_candidate_pool=100)
    manifest = AcquisitionManifest(
        source=SourceName.crossref,
        query={"q": "test"},
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        expected_records=18,
        received_records=18,
        unique_records=18,
        pages=1,
        complete=True,
    )
    return build_evidence(
        manifest,
        tables,
        analyses,
        [],
        graph_explanations=graph_explanations,
    )


def test_verified_graph_explanation_enters_evidence_bundle(crossref_records):
    explanations = [
        {
            "status": "complete",
            "figure_name": "network_keyword_cooccurrence",
            "network_name": "keyword_cooccurrence",
            "mode": "graph_rag",
            "verified_claims": [
                {
                    "claim_id": "GC001",
                    "type": "cross_community",
                    "statement": "已核验的克制表述。",
                    "model_statement": "不应进入写作证据包的模型原句。",
                    "verified": True,
                }
            ],
            "rejected_claims": [{"reason": "unsupported evidence edge"}],
            "verification": {"reported_claims": 2, "verified_claims": 1},
            "caveats": ["共现不表示因果"],
        }
    ]

    evidence = _bundle(crossref_records, explanations)
    item = next(item for item in evidence.items if item.claim_type == "figure_interpretation")

    assert item.value["mode"] == "graph_rag"
    assert item.value["verified_claims"][0]["claim_id"] == "GC001"
    assert "model_statement" not in item.value["verified_claims"][0]
    assert item.artifact_path == "evidence/graph_explanations.json"


def test_validated_text(crossref_records):
    evidence = _bundle(crossref_records)
    corpus = next(item for item in evidence.items if item.claim_type == "corpus_size")
    result = validate_manuscript(
        f"本研究纳入 {corpus.value} 篇文献。[{corpus.evidence_id}]",
        evidence,
        strict_structure=False,
    )
    assert result["valid"]


def test_rejects_unknown_number_and_evidence(crossref_records):
    evidence = _bundle(crossref_records)
    result = validate_manuscript(
        "本研究纳入 987654 篇文献。[E999]", evidence, strict_structure=False
    )
    assert not result["valid"]
    assert result["invalid_evidence_ids"] == ["E999"]
    assert result["unsupported_numbers"] == ["987654"]


def test_accepts_citation_ranges_and_decimal_values(crossref_records):
    evidence = _bundle(crossref_records)
    numeric_items = [
        item for item in evidence.items if any(char.isdigit() for char in str(item.value))
    ][:2]
    first, second = numeric_items
    text = (
        "## 2.1 结果\n\n"
        f"该语料包含 {first.value} 项；相关值由证据给出"
        f"[{first.evidence_id}–{second.evidence_id}]。"
    )
    result = validate_manuscript(text, evidence, strict_structure=False)
    assert result["invalid_evidence_ids"] == []
    assert not any(value in {"1", "2.1"} for value in result["unsupported_numbers"])


def test_claims_have_raw_to_claim_evidence_path(crossref_records):
    evidence = _bundle(crossref_records)
    first = evidence.items[0]
    ledger = bind_claims(
        evidence,
        f"语料规模由规范数据表计算得出。[{first.evidence_id}]",
    )
    claim_id = ledger.iloc[0]["claim_id"]
    import networkx as nx

    assert nx.has_path(evidence.graph, "raw_snapshot", claim_id)


def test_normalizes_bare_evidence_tokens_without_double_wrapping():
    text = "该判断由E044支持，另一项已有[E043]。"

    assert _normalize_evidence_tokens(text) == "该判断由[E044]支持，另一项已有[E043]。"


def test_binds_uncited_numeric_sentence_to_paragraph_evidence(crossref_records):
    evidence = _bundle(crossref_records)
    corpus = next(item for item in evidence.items if item.claim_type == "corpus_size")
    text = f"语料规模依据[{corpus.evidence_id}]确定。规范化后共保留{corpus.value}篇文献。"

    bound = _bind_paragraph_numeric_claims(text, evidence)

    assert f"{corpus.value}篇文献[{corpus.evidence_id}]。" in bound


def test_journal_readiness_gate_rejects_shallow_manuscript(crossref_records):
    evidence = _bundle(crossref_records)
    quality = evaluate_manuscript_quality(
        "# 标题\n\n## 摘要\n\n简短摘要。\n\n## 1 引言\n\n简短引言。",
        evidence,
    )

    assert not quality["passed"]
    assert not quality["checks"]["characters_at_least_12000"]
    assert not quality["checks"]["method_moves_complete"]
