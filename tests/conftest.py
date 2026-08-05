from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture()
def crossref_records() -> list[dict[str, Any]]:
    records = []
    journals = ["Journal of Evidence Systems", "Scientometrics Lab"]
    authors = [
        ("Ada", "Chen"),
        ("Ming", "Li"),
        ("Sara", "Khan"),
        ("Omar", "Singh"),
        ("Elena", "Garcia"),
    ]
    subjects = [
        ["bibliometrics", "science mapping", "artificial intelligence"],
        ["bibliometrics", "large language models", "research evaluation"],
        ["science mapping", "knowledge graph", "research evaluation"],
    ]
    for index in range(18):
        year = 2020 + index % 6
        selected = [authors[index % len(authors)], authors[(index + 1) % len(authors)]]
        references = [
            {
                "DOI": f"10.9999/reference.{ref}",
                "article-title": f"Foundational reference {ref}",
                "author": authors[ref % len(authors)][1],
                "year": str(2010 + ref),
            }
            for ref in (index % 6, (index + 1) % 6, (index + 2) % 6)
        ]
        records.append(
            {
                "DOI": f"10.1234/example.{index}",
                "URL": f"https://doi.org/10.1234/example.{index}",
                "title": [f"Evidence-first bibliometric systems study {index}"],
                "abstract": (
                    "<jats:p>This study examines bibliometric science mapping, "
                    "knowledge networks, and artificial intelligence.</jats:p>"
                ),
                "published": {"date-parts": [[year, 5, 1]]},
                "type": "journal-article",
                "container-title": [journals[index % 2]],
                "ISSN": [f"1234-56{index % 2:02d}"],
                "author": [
                    {
                        "given": given,
                        "family": family,
                        "affiliation": [{"name": f"University {family[0]}"}],
                    }
                    for given, family in selected
                ],
                "subject": subjects[index % len(subjects)],
                "reference": references,
                "references-count": len(references),
                "is-referenced-by-count": index * 2,
                "publisher": "Evidence Press",
            }
        )
    return records
