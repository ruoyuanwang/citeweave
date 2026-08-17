import json
from pathlib import Path

from citeweave.connectors.import_file import ImportFileConnector
from citeweave.models import SearchProtocol, SourceName
from citeweave.transform import Canonicalizer


def test_import_wos_csv_is_complete_and_canonical(tmp_path: Path) -> None:
    source = tmp_path / "wos.csv"
    source.write_text(
        "UT,TI,AB,PY,SO,AU,DE,DI,TC,CR\n"
        "WOS:1,Large language model bibliometric study,Study abstract,2024,"
        'Journal A,"Zhang, Y; Smith, J","AI; Bibliometrics",10.1000/test,7,'
        '"Doe J, 2020, Journal B, 10.1000/ref"\n'
        "WOS:2,Unrelated paper,Other,2019,Journal B,Lee K,Other,,0,\n",
        encoding="utf-8",
    )
    protocol = SearchProtocol(
        title="Imported bibliometrics",
        keywords=["large language model", "bibliometric"],
        year_from=2022,
        year_to=2025,
        source=SourceName.import_file,
        input_file=source,
    )
    connector = ImportFileConnector(tmp_path / "raw")
    try:
        result = connector.acquire(protocol)
    finally:
        connector.close()
    assert result.manifest.complete
    assert result.manifest.received_records == 2
    assert result.manifest.unique_records == 1
    tables = Canonicalizer(SourceName.import_file).canonicalize(result.records)
    assert len(tables.works) == 1
    assert tables.works.iloc[0]["doi"] == "10.1000/test"
    assert len(tables.references) == 1
    assert tables.provenance.iloc[0]["source"] == "import_file"


def test_import_ris_and_bibtex(tmp_path: Path) -> None:
    ris = tmp_path / "sample.ris"
    ris.write_text(
        "TY  - JOUR\nTI  - Bibliometric agent systems\n"
        "AU  - Doe, Jane\nPY  - 2023\nJO  - Test Journal\n"
        "KW  - bibliometric\nDO  - 10.1000/ris\nER  -\n",
        encoding="utf-8",
    )
    bib = tmp_path / "sample.bib"
    bib.write_text(
        "@article{x, title={Bibliometric agent systems}, author={Doe, Jane}, "
        "year={2023}, journal={Test Journal}, keywords={bibliometric}, "
        "doi={10.1000/bib}}",
        encoding="utf-8",
    )
    for path, fmt in ((ris, "ris"), (bib, "bibtex")):
        protocol = SearchProtocol(
            title="Imported bibliometrics",
            keywords=["bibliometric"],
            year_from=2022,
            year_to=2025,
            source=SourceName.import_file,
            input_file=path,
            input_format=fmt,
        )
        connector = ImportFileConnector(tmp_path / f"raw-{fmt}")
        try:
            result = connector.acquire(protocol)
        finally:
            connector.close()
        assert result.manifest.complete
        assert result.manifest.unique_records == 1


def test_import_native_crossref_jsonl_preserves_references(tmp_path: Path) -> None:
    source = tmp_path / "crossref.jsonl"
    row = {
        "DOI": "10.1000/native",
        "title": ["Graph retrieval study"],
        "abstract": "Graph retrieval benchmark.",
        "published": {"date-parts": [[2024, 3, 1]]},
        "type": "journal-article",
        "reference": [{"DOI": "10.1000/reference"}],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    protocol = SearchProtocol(
        title="Native Crossref import",
        keywords=["graph", "retrieval"],
        query_mode="all",
        year_from=2022,
        year_to=2025,
        source=SourceName.import_file,
        input_file=source,
        input_format="jsonl",
    )
    connector = ImportFileConnector(tmp_path / "raw-jsonl")
    try:
        result = connector.acquire(protocol)
    finally:
        connector.close()

    assert result.manifest.complete
    tables = Canonicalizer(SourceName.import_file).canonicalize(result.records)
    assert tables.works.iloc[0]["doi"] == "10.1000/native"
    assert tables.references.iloc[0]["cited_work_id"] == "doi:10.1000/reference"
