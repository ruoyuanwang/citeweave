from datetime import date

from citeweave.bulk_acquisition import BulkSourceAdapter
from citeweave.connectors.crossref import CrossrefConnector
from citeweave.connectors.openalex import OpenAlexConnector
from citeweave.models import SearchProtocol, SourceName


def test_crossref_standard_cursor_uses_documented_parameters(tmp_path):
    connector = CrossrefConnector(tmp_path, max_retries=0)
    protocol = SearchProtocol(
        title="Graph retrieval benchmark",
        keywords=["graph retrieval augmented generation"],
        query_mode="phrase",
        year_from=2022,
        year_to=2025,
        source=SourceName.crossref,
        max_records=300,
    )

    params = connector._params(protocol)

    assert params["cursor"] == "*"
    assert params["rows"] == 1000
    assert "cursor-max" not in params


def test_openalex_bulk_adapter_uses_current_per_page_contract(tmp_path):
    connector = OpenAlexConnector(tmp_path, api_key="test-key", max_retries=0)
    protocol = SearchProtocol(
        title="OpenAlex contract test",
        keywords=["graph neural network", "drug discovery"],
        year_from=2020,
        year_to=2021,
        source=SourceName.openalex,
        language="en",
        document_types=["article"],
    )

    params = BulkSourceAdapter(connector)._params(
        protocol,
        date(2020, 1, 1),
        date(2021, 12, 31),
        page_size=100,
        cursor="*",
    )
    connector.close()

    assert params["per_page"] == 100
    assert "per-page" not in params
    assert params["api_key"] == "test-key"
    assert "language:en" in params["filter"]
    assert params["search"] == '"graph neural network" AND "drug discovery"'


def test_openalex_connector_uses_configured_key_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEWEAVE_TEST_OPENALEX_KEY", "configured-key")
    connector = OpenAlexConnector(
        tmp_path,
        api_key_env="CITEWEAVE_TEST_OPENALEX_KEY",
        max_retries=0,
    )

    assert connector.api_key == "configured-key"
    connector.close()


def test_openalex_bulk_adapter_preserves_frozen_boolean_expression(tmp_path):
    connector = OpenAlexConnector(tmp_path, api_key="test-key", max_retries=0)
    protocol = SearchProtocol(
        title="Boolean synonym contract",
        keywords=["climate change", "risk"],
        query_mode="all",
        search_expression=(
            '"climate change" AND '
            '("climate risk" OR "risk assessment" OR vulnerability)'
        ),
        year_from=1990,
        year_to=2021,
        source=SourceName.openalex,
        document_types=["article"],
    )

    params = BulkSourceAdapter(connector)._params(
        protocol,
        date(1990, 1, 1),
        date(2021, 12, 31),
        page_size=100,
        cursor="*",
    )
    connector.close()

    assert params["search"] == protocol.search_expression


def test_openalex_bulk_adapter_scopes_search_to_title_and_abstract(tmp_path):
    connector = OpenAlexConnector(tmp_path, api_key="test-key", max_retries=0)
    protocol = SearchProtocol(
        title="Title and abstract contract",
        keywords=["digital twin", "healthcare"],
        query_mode="all",
        search_expression='"digital twin" AND (healthcare OR medical)',
        search_scope="title_abstract",
        year_from=2012,
        year_to=2023,
        source=SourceName.openalex,
        document_types=["article"],
    )

    params = BulkSourceAdapter(connector)._params(
        protocol,
        date(2012, 1, 1),
        date(2023, 12, 31),
        page_size=100,
        cursor="*",
    )
    connector.close()

    assert "search" not in params
    assert (
        'title_and_abstract.search:"digital twin" AND (healthcare OR medical)'
        in params["filter"]
    )
