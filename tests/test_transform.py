import pytest

from citeweave.transform import (
    Canonicalizer,
    _date_parts_crossref,
    derive_keywords,
    normalize_doi,
)


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1234/ABC.1") == "10.1234/abc.1"
    assert normalize_doi("not-a-doi") is None


@pytest.mark.parametrize(
    "missing", [None, "", "  ", "None", "null", "NaN", "https://openalex.org/None"]
)
def test_openalex_unknown_authors_are_work_position_scoped(missing):
    records = [
        {
            "id": f"https://openalex.org/W{number}",
            "title": f"Work {number}",
            "authorships": [
                {"author": {"id": missing, "display_name": "Same name"}},
                {"author": {"display_name": "Same name"}},
            ],
        }
        for number in (1, 2)
    ]
    tables = Canonicalizer("openalex").canonicalize(records)
    assert len(tables.authors) == 4
    assert tables.authorships.author_id.nunique() == 4
    assert tables.authorships.author_id.str.startswith("openalex-author-occurrence:").all()
    repeated = Canonicalizer("openalex").canonicalize(records + records)
    assert set(repeated.authors.author_id) == set(tables.authors.author_id)


def test_openalex_known_author_identities_are_preserved():
    records = [
        {
            "id": f"https://openalex.org/W{number}",
            "title": f"Work {number}",
            "authorships": [
                {"author": {"id": "https://openalex.org/A123", "display_name": "Known"}},
                {"author": {"id": None, "orcid": "https://orcid.org/0000-0002-1825-0097"}},
            ],
        }
        for number in (1, 2)
    ]
    tables = Canonicalizer("openalex").canonicalize(records)
    assert set(tables.authors.author_id) == {"openalex-author:A123", "orcid:0000-0002-1825-0097"}
    assert tables.authorships.author_id.nunique() == 2


def test_crossref_canonical_tables(crossref_records):
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    assert len(tables.works) == 18
    assert len(tables.authors) == 5
    assert len(tables.authorships) == 36
    assert tables.references["cited_work_id"].nunique() == 6
    assert tables.keywords["work_id"].nunique() == 18
    assert tables.works["doi"].notna().all()


def test_derived_keywords_when_source_keywords_missing(crossref_records):
    for record in crossref_records:
        record["subject"] = []
    tables = Canonicalizer("crossref").canonicalize(crossref_records)
    keywords = derive_keywords(tables.works, tables.keywords)
    assert not keywords.empty
    assert set(keywords["keyword_type"]) == {"derived_tfidf"}


def test_crossref_year_prefers_published_date_over_later_print_issue():
    item = {
        "published": {"date-parts": [[2025, 12, 23]]},
        "published-online": {"date-parts": [[2025, 12, 23]]},
        "published-print": {"date-parts": [[2026, 2, 28]]},
        "issued": {"date-parts": [[2025, 12, 23]]},
    }

    assert _date_parts_crossref(item) == (2025, "2025-12-23")
