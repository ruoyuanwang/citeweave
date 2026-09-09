from pathlib import Path

from citeweave.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "docs" / "ipm_paper" / "02_MANUSCRIPT_EN.md"
CHINESE = ROOT / "docs" / "ipm_paper" / "04_MANUSCRIPT_ZH.md"


def test_manuscripts_transcribe_registered_graph_results_and_failures() -> None:
    main = read_json(
        ROOT / "experiments" / "graph_discovery_v2" / "formal_v3_analysis.json"
    )
    complexity = read_json(
        ROOT
        / "experiments"
        / "graph_discovery_v2"
        / "formal_v3_complexity_extension_analysis.json"
    )
    mechanism = read_json(
        ROOT
        / "experiments"
        / "graph_discovery_v2"
        / "formal_v3_mechanism_supplement_analysis.json"
    )
    english = ENGLISH.read_text(encoding="utf-8")
    chinese = CHINESE.read_text(encoding="utf-8")

    assert main["records"] == 800
    assert main["condition_summaries"]["graph_program"]["accuracy"] == 0.88
    assert main["primary_hypotheses"]["H3"]["estimate"] == 0
    assert "Graph program | 100 | 0.88 | 0.404" in english
    assert "Graph program | 100 | 0.88 | 0.404" in chinese

    c1 = complexity["registered_contrasts"]["C1_complexity_selectivity"]
    c2 = complexity["registered_contrasts"][
        "C2_scale_amplification_of_complexity_selectivity"
    ]
    assert c1["cluster_bootstrap"]["estimate"] == 1
    assert c1["multiplicity"]["holm_adjusted_p_value"] == 0.0078125
    assert c2["cluster_bootstrap"]["estimate"] == 0
    assert "Holm-adjusted exact p = 0.0078125" in english
    assert "Holm校正精确p=0.0078125" in chinese

    assert mechanism["M1_same_computation_representation"]["equivalence"][
        "equivalent_at_0_05"
    ]
    assert mechanism["M2_raw_provenance_value"]["evidence_f1_superiority"][
        "estimate"
    ] > 0.29
    assert mechanism["M3_incomplete_trace_selectivity"]["estimate"] == 1
    assert "Graph-shaped serialization was therefore not" in english
    assert "图形JSON布局本身没有独立收益" in chinese

    for text in (english, chinese):
        assert "16/16" in text
        assert "0/36" in text
        assert "0.15" in text and "0.71" in text
        assert "No real" in text or "真实首审" in text


def test_manuscript_audits_bind_current_files_and_keep_submission_blocked() -> None:
    english_check = read_json(ROOT / "docs" / "ipm_paper" / "draft_checks.json")
    chinese_check = read_json(ROOT / "docs" / "ipm_paper" / "draft_checks_zh.json")
    assert english_check["manuscript_sha256"] == sha256_file(ENGLISH)
    assert chinese_check["manuscript_sha256"] == sha256_file(CHINESE)
    assert english_check["submission_ready"] is False
    assert chinese_check["submission_ready"] is False
    assert "*Planned section" not in ENGLISH.read_text(encoding="utf-8")
    assert "*Reserved for complete" not in ENGLISH.read_text(encoding="utf-8")


def test_related_work_sets_conservative_novelty_boundaries() -> None:
    english = ENGLISH.read_text(encoding="utf-8")
    chinese = CHINESE.read_text(encoding="utf-8")
    english_check = read_json(ROOT / "docs" / "ipm_paper" / "draft_checks.json")
    chinese_check = read_json(ROOT / "docs" / "ipm_paper" / "draft_checks_zh.json")

    for marker in (
        "SurveyGen",
        "GraphRAG-Bench",
        "GRAG: Graph retrieval-augmented generation",
        "Hybrid preferences",
        "Bayesian online learning for consensus prediction",
        "Just put a human in the loop?",
        "When combinations of humans and AI are useful",
    ):
        assert marker.casefold() in english.casefold()
        assert marker.casefold() in chinese.casefold()

    assert "neither presenting a graph beside text nor increasing question complexity" in english
    assert "都不足以构成创新" in chinese
    assert "cannot be treated as isolated CiteWeave contributions" in english
    assert "不能单独作为 CiteWeave 的创新点" in chinese
    assert english_check["reference_count"] == 24
    assert chinese_check["reference_count"] == 24
    assert english_check["checks"]["expanded_related_work_positioning_verified"]
    assert chinese_check["checks"]["expanded_related_work_positioning_verified"]
