from pathlib import Path

import pandas as pd

from citeweave.analytics import NetworkResult, analyze
from citeweave.interoperability import export_all, export_vosviewer
from citeweave.models import SourceName
from citeweave.transform import Canonicalizer


def test_vosviewer_export_uses_stable_numeric_ids(tmp_path: Path) -> None:
    network = NetworkResult(
        "keywords",
        pd.DataFrame(
            [
                {"id": "x", "label": "Alpha", "occurrences": 3, "cluster": 1},
                {"id": "y", "label": "Beta", "occurrences": 2, "cluster": 1},
            ]
        ),
        pd.DataFrame([{"source": "x", "target": "y", "weight": 2, "association_strength": 0.3}]),
        {},
    )
    manifest = export_vosviewer(network, tmp_path)
    assert manifest["nodes"] == 2
    assert (
        (tmp_path / manifest["map"])
        .read_text(encoding="utf-8")
        .startswith("id\tlabel\tweight<Occurrences>")
    )
    assert "\n1\t2\t2\n" in (tmp_path / manifest["network"]).read_text(encoding="utf-8")


def test_all_exports_cover_corpus(tmp_path: Path, crossref_records) -> None:
    sample_tables = Canonicalizer(SourceName.crossref).canonicalize(crossref_records)
    analyses = analyze(sample_tables, network_candidate_pool=50)
    manifest = export_all(sample_tables, analyses, tmp_path)
    assert manifest["bibliometrix"]["rows"] == len(sample_tables.works)
    exported = pd.read_csv(tmp_path / "bibliometrix_data.csv")
    assert len(exported) == len(sample_tables.works)
    assert {"AU", "TI", "SO", "PY", "DI", "DE", "CR", "TC"}.issubset(exported.columns)
