from pathlib import Path

from bibagent.analytics import analyze
from bibagent.transform import Canonicalizer
from bibagent.visualization import render_all


def test_analysis_networks(crossref_records):
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    bundle = analyze(tables, network_candidate_pool=100)
    assert bundle.summary["documents"] == 18
    assert bundle.summary["authors"] == 5
    assert len(bundle.annual) == 6
    assert not bundle.networks["coauthorship"].edges.empty
    assert not bundle.networks["keyword_cooccurrence"].edges.empty
    assert not bundle.networks["cocitation"].edges.empty
    assert not bundle.networks["bibliographic_coupling"].edges.empty


def test_render_figures(crossref_records, tmp_path: Path):
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    bundle = analyze(tables, network_candidate_pool=100)
    figures = render_all(
        tables,
        bundle,
        tmp_path,
        max_nodes=40,
        label_budget=12,
        seed=42,
    )
    names = {figure.name for figure in figures}
    assert "annual_publications" in names
    assert "network_keyword_cooccurrence" in names
    assert "network_coauthorship" in names
    assert "thematic_map" in names
    assert all(figure.qa["minimum_dimension_pass"] for figure in figures)
