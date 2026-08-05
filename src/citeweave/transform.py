from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .io import sha256_bytes
from .models import SourceName

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s-]+", re.UNICODE)
ORG_TOKEN_RE = re.compile(
    r"\b(university|universit[yéèà]|hospital|institute|institut|academy|college|"
    r"school|centre|center|laboratory|foundation|clinic)\b",
    re.IGNORECASE,
)
GENERIC_INDEX_TERMS = {
    "humans",
    "human",
    "male",
    "female",
    "adult",
    "middle aged",
    "aged",
    "young adult",
    "adolescent",
    "child",
    "animals",
    "journal article",
}
KEYWORD_ALIASES = {
    "artificial intelligence": "Artificial intelligence",
    "large language model": "Large language models",
    "large language models": "Large language models",
    "chatgpt": "ChatGPT",
    "bibliometric analysis": "Bibliometric analysis",
    "bibliometrics": "Bibliometrics",
    "machine learning": "Machine learning",
    "deep learning": "Deep learning",
    "natural language processing": "Natural language processing",
    "generative artificial intelligence": "Generative artificial intelligence",
}
CANONICAL_COLUMNS = {
    "works": [
        "work_id",
        "doi",
        "external_id",
        "title",
        "abstract",
        "year",
        "publication_date",
        "document_type",
        "language",
        "source_id",
        "volume",
        "issue",
        "pages",
        "publisher",
        "cited_by_count",
        "reference_count",
        "is_retracted",
        "source_record_hash",
    ],
    "authors": ["author_id", "name", "given_name", "family_name", "orcid"],
    "institutions": ["institution_id", "name", "ror", "country_code"],
    "authorships": [
        "work_id",
        "author_id",
        "institution_id",
        "position",
        "is_corresponding",
    ],
    "sources": ["source_id", "name", "issn", "publisher", "source_type"],
    "keywords": ["work_id", "keyword", "keyword_type", "score"],
    "topics": ["work_id", "topic_id", "topic", "score", "field"],
    "references": [
        "citing_work_id",
        "cited_work_id",
        "cited_doi",
        "cited_title",
        "cited_author",
        "cited_year",
        "source",
    ],
    "provenance": [
        "work_id",
        "source",
        "source_record_id",
        "source_record_hash",
    ],
}


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(TAG_RE.sub(" ", str(value)))
    text = unicodedata.normalize("NFKC", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text or None


def normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.rstrip(" .;,")
    return text if DOI_RE.match(text) else None


def normalize_orcid(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().replace("https://orcid.org/", "")
    return text if re.fullmatch(r"\d{4}-\d{4}-\d{4}-[\dX]{4}", text) else None


def normalize_title(value: Any) -> str:
    text = clean_text(value) or ""
    text = NON_WORD_RE.sub(" ", text.casefold())
    return SPACE_RE.sub(" ", text).strip()


def stable_id(prefix: str, *parts: Any) -> str:
    text = "\x1f".join(str(part or "").casefold().strip() for part in parts)
    return f"{prefix}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:20]}"


def normalize_affiliation(value: Any) -> str | None:
    """Extract a conservative organization label from a free-text affiliation."""
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"\b\S+@\S+\b", "", text)
    parts = [SPACE_RE.sub(" ", part).strip(" .;") for part in text.split(",") if part.strip()]
    matched = [part for part in parts if ORG_TOKEN_RE.search(part)]
    if not matched:
        return text[:180]
    priority = ("university", "hospital", "institute", "academy", "college", "center", "centre")
    for token in priority:
        for part in matched:
            if token in part.casefold():
                return part[:180]
    return matched[-1][:180]


def normalize_document_type(values: Any) -> str | None:
    if isinstance(values, str):
        values = [values]
    labels = [clean_text(value) for value in (values or [])]
    labels = [label for label in labels if label]
    lowered = [label.casefold() for label in labels]
    priorities = [
        ("systematic review", "Systematic review"),
        ("meta-analysis", "Meta-analysis"),
        ("review", "Review"),
        ("clinical trial", "Clinical trial"),
        ("preprint", "Preprint"),
        ("research article", "Research article"),
        ("journal article", "Journal article"),
        ("editorial", "Editorial"),
        ("letter", "Letter"),
        ("case reports", "Case report"),
    ]
    for token, canonical in priorities:
        if any(token in label for label in lowered):
            return canonical
    return labels[0] if labels else None


def normalize_keyword(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = SPACE_RE.sub(" ", text.casefold().replace("_", " ")).strip(" .;,")
    if normalized in GENERIC_INDEX_TERMS:
        return None
    if normalized in KEYWORD_ALIASES:
        return KEYWORD_ALIASES[normalized]
    if len(text) <= 5 and text.isupper():
        return text
    return normalized[:1].upper() + normalized[1:]


def _date_parts_crossref(item: dict[str, Any]) -> tuple[int | None, str | None]:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        obj = item.get(key)
        if not obj:
            continue
        parts = obj.get("date-parts") if isinstance(obj, dict) else None
        if parts and parts[0]:
            values = parts[0]
            year = int(values[0])
            date = "-".join(
                [f"{year:04d}"]
                + ([f"{int(values[1]):02d}"] if len(values) > 1 else [])
                + ([f"{int(values[2]):02d}"] if len(values) > 2 else [])
            )
            return year, date
    return None, None


@dataclass
class CanonicalTables:
    works: pd.DataFrame
    authors: pd.DataFrame
    institutions: pd.DataFrame
    authorships: pd.DataFrame
    sources: pd.DataFrame
    keywords: pd.DataFrame
    topics: pd.DataFrame
    references: pd.DataFrame
    provenance: pd.DataFrame
    duplicates: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "works": self.works,
            "authors": self.authors,
            "institutions": self.institutions,
            "authorships": self.authorships,
            "sources": self.sources,
            "keywords": self.keywords,
            "topics": self.topics,
            "references": self.references,
            "provenance": self.provenance,
            "duplicates": self.duplicates,
        }


class Canonicalizer:
    def __init__(self, source: SourceName | str):
        self.source = SourceName(source)

    def canonicalize(self, records: Iterable[dict[str, Any]]) -> CanonicalTables:
        if self.source in {SourceName.crossref, SourceName.import_file}:
            rows = self._crossref(records)
        elif self.source == SourceName.openalex:
            rows = self._openalex(records)
        elif self.source == SourceName.europe_pmc:
            rows = self._europe_pmc(records)
        else:
            raise NotImplementedError(f"canonicalizer for {self.source}")
        return self._finalize(rows)

    @staticmethod
    def _empty_rows() -> dict[str, list[dict[str, Any]]]:
        return defaultdict(list)

    def _crossref(self, records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out = self._empty_rows()
        provenance_source = self.source.value
        for item in records:
            raw = json.dumps(item, ensure_ascii=False, sort_keys=True).encode()
            raw_hash = sha256_bytes(raw)
            doi = normalize_doi(item.get("DOI"))
            title = clean_text((item.get("title") or [None])[0])
            year, published_date = _date_parts_crossref(item)
            first_author = (item.get("author") or [{}])[0]
            work_id = (
                f"doi:{doi}"
                if doi
                else stable_id("work", normalize_title(title), year, first_author.get("family"))
            )
            container = clean_text((item.get("container-title") or [None])[0])
            issn_values = item.get("ISSN") or []
            source_id = (
                f"issn:{min(issn_values).lower()}"
                if issn_values
                else stable_id("source", container)
            )
            out["works"].append(
                {
                    "work_id": work_id,
                    "doi": doi,
                    "external_id": item.get("URL"),
                    "title": title,
                    "abstract": clean_text(item.get("abstract")),
                    "year": year,
                    "publication_date": published_date,
                    "document_type": item.get("type"),
                    "language": item.get("language"),
                    "source_id": source_id,
                    "volume": item.get("volume"),
                    "issue": item.get("issue"),
                    "pages": item.get("page"),
                    "publisher": clean_text(item.get("publisher")),
                    "cited_by_count": item.get("is-referenced-by-count"),
                    "reference_count": item.get("references-count"),
                    "is_retracted": bool(item.get("update-to")),
                    "source_record_hash": raw_hash,
                }
            )
            out["sources"].append(
                {
                    "source_id": source_id,
                    "name": container,
                    "issn": "|".join(issn_values) or None,
                    "publisher": clean_text(item.get("publisher")),
                    "source_type": item.get("type"),
                }
            )
            for position, author in enumerate(item.get("author") or [], start=1):
                full_name = clean_text(
                    " ".join(filter(None, [author.get("given"), author.get("family")]))
                )
                orcid = normalize_orcid(author.get("ORCID"))
                affiliations = author.get("affiliation") or []
                primary_affiliation = (
                    normalize_affiliation((affiliations[0] or {}).get("name"))
                    if affiliations
                    else None
                )
                author_id = (
                    f"orcid:{orcid}"
                    if orcid
                    else stable_id("author", full_name, primary_affiliation)
                )
                out["authors"].append(
                    {
                        "author_id": author_id,
                        "name": full_name,
                        "given_name": clean_text(author.get("given")),
                        "family_name": clean_text(author.get("family")),
                        "orcid": orcid,
                    }
                )
                if not affiliations:
                    out["authorships"].append(
                        {
                            "work_id": work_id,
                            "author_id": author_id,
                            "institution_id": None,
                            "position": position,
                            "is_corresponding": False,
                        }
                    )
                for affiliation in affiliations:
                    name = normalize_affiliation(affiliation.get("name"))
                    institution_id = stable_id("institution", name) if name else None
                    if name:
                        out["institutions"].append(
                            {
                                "institution_id": institution_id,
                                "name": name,
                                "ror": None,
                                "country_code": None,
                            }
                        )
                    out["authorships"].append(
                        {
                            "work_id": work_id,
                            "author_id": author_id,
                            "institution_id": institution_id,
                            "position": position,
                            # Crossref's sequence means first/additional author,
                            # not corresponding-author status.
                            "is_corresponding": False,
                        }
                    )
            for keyword in item.get("subject") or []:
                text = clean_text(keyword)
                if text:
                    out["keywords"].append(
                        {
                            "work_id": work_id,
                            "keyword": text,
                            "keyword_type": (
                                "imported_keyword"
                                if self.source == SourceName.import_file
                                else "crossref_subject"
                            ),
                        }
                    )
            for reference in item.get("reference") or []:
                cited_doi = normalize_doi(reference.get("DOI"))
                cited_title = clean_text(reference.get("article-title"))
                cited_id = (
                    f"doi:{cited_doi}"
                    if cited_doi
                    else stable_id(
                        "reference",
                        cited_title,
                        reference.get("year"),
                        reference.get("author"),
                    )
                )
                out["references"].append(
                    {
                        "citing_work_id": work_id,
                        "cited_work_id": cited_id,
                        "cited_doi": cited_doi,
                        "cited_title": cited_title,
                        "cited_author": clean_text(reference.get("author")),
                        "cited_year": reference.get("year"),
                        "source": provenance_source,
                    }
                )
            out["provenance"].append(
                {
                    "work_id": work_id,
                    "source": provenance_source,
                    "source_record_id": doi or item.get("URL"),
                    "source_record_hash": raw_hash,
                }
            )
        return out

    def _openalex(self, records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out = self._empty_rows()
        for item in records:
            raw_hash = sha256_bytes(
                json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            doi = normalize_doi(item.get("doi"))
            work_id = f"doi:{doi}" if doi else f"openalex:{str(item.get('id', '')).split('/')[-1]}"
            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            source_id = (
                f"issn:{source.get('issn_l')}"
                if source.get("issn_l")
                else f"openalex-source:{str(source.get('id', '')).split('/')[-1]}"
            )
            abstract_index = item.get("abstract_inverted_index") or {}
            abstract = None
            if abstract_index:
                positions = [
                    (position, token)
                    for token, token_positions in abstract_index.items()
                    for position in token_positions
                ]
                abstract = " ".join(token for _, token in sorted(positions))
            out["works"].append(
                {
                    "work_id": work_id,
                    "doi": doi,
                    "external_id": item.get("id"),
                    "title": clean_text(item.get("title") or item.get("display_name")),
                    "abstract": clean_text(abstract),
                    "year": item.get("publication_year"),
                    "publication_date": item.get("publication_date"),
                    "document_type": item.get("type"),
                    "language": item.get("language"),
                    "source_id": source_id,
                    "volume": (item.get("biblio") or {}).get("volume"),
                    "issue": (item.get("biblio") or {}).get("issue"),
                    "pages": (item.get("biblio") or {}).get("first_page"),
                    "publisher": clean_text(source.get("host_organization_name")),
                    "cited_by_count": item.get("cited_by_count"),
                    "reference_count": item.get("referenced_works_count"),
                    "is_retracted": bool(item.get("is_retracted")),
                    "source_record_hash": raw_hash,
                }
            )
            out["sources"].append(
                {
                    "source_id": source_id,
                    "name": clean_text(source.get("display_name")),
                    "issn": "|".join(source.get("issn") or []) or None,
                    "publisher": clean_text(source.get("host_organization_name")),
                    "source_type": source.get("type"),
                }
            )
            for position, authorship in enumerate(item.get("authorships") or [], start=1):
                author = authorship.get("author") or {}
                orcid = normalize_orcid(author.get("orcid"))
                author_id = (
                    f"orcid:{orcid}"
                    if orcid
                    else f"openalex-author:{str(author.get('id', '')).split('/')[-1]}"
                )
                out["authors"].append(
                    {
                        "author_id": author_id,
                        "name": clean_text(author.get("display_name")),
                        "given_name": None,
                        "family_name": None,
                        "orcid": orcid,
                    }
                )
                institutions = authorship.get("institutions") or [None]
                for institution in institutions:
                    institution_id = None
                    if institution:
                        ror = institution.get("ror")
                        institution_id = (
                            f"ror:{str(ror).rstrip('/').split('/')[-1]}"
                            if ror
                            else f"openalex-institution:{str(institution.get('id', '')).split('/')[-1]}"
                        )
                        out["institutions"].append(
                            {
                                "institution_id": institution_id,
                                "name": clean_text(institution.get("display_name")),
                                "ror": ror,
                                "country_code": institution.get("country_code"),
                            }
                        )
                    out["authorships"].append(
                        {
                            "work_id": work_id,
                            "author_id": author_id,
                            "institution_id": institution_id,
                            "position": position,
                            "is_corresponding": bool(authorship.get("is_corresponding")),
                        }
                    )
            for keyword in item.get("keywords") or []:
                text = clean_text(keyword.get("display_name"))
                if text:
                    out["keywords"].append(
                        {"work_id": work_id, "keyword": text, "keyword_type": "openalex"}
                    )
            for topic in item.get("topics") or []:
                out["topics"].append(
                    {
                        "work_id": work_id,
                        "topic_id": topic.get("id"),
                        "topic": clean_text(topic.get("display_name")),
                        "score": topic.get("score"),
                        "field": clean_text((topic.get("field") or {}).get("display_name")),
                    }
                )
            for cited in item.get("referenced_works") or []:
                out["references"].append(
                    {
                        "citing_work_id": work_id,
                        "cited_work_id": f"openalex:{str(cited).split('/')[-1]}",
                        "cited_doi": None,
                        "cited_title": None,
                        "cited_author": None,
                        "cited_year": None,
                        "source": "openalex",
                    }
                )
            out["provenance"].append(
                {
                    "work_id": work_id,
                    "source": "openalex",
                    "source_record_id": item.get("id"),
                    "source_record_hash": raw_hash,
                }
            )
        return out

    def _europe_pmc(self, records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out = self._empty_rows()
        for item in records:
            raw_hash = sha256_bytes(
                json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            doi = normalize_doi(item.get("doi"))
            external = f"{item.get('source')}:{item.get('id')}"
            work_id = f"doi:{doi}" if doi else f"epmc:{external}"
            journal = (item.get("journalInfo") or {}).get("journal") or {}
            issn = journal.get("issn")
            source_id = f"issn:{issn}" if issn else stable_id("source", journal.get("title"))
            out["works"].append(
                {
                    "work_id": work_id,
                    "doi": doi,
                    "external_id": external,
                    "title": clean_text(item.get("title")),
                    "abstract": clean_text(item.get("abstractText")),
                    "year": int(item["pubYear"])
                    if str(item.get("pubYear", "")).isdigit()
                    else None,
                    "publication_date": item.get("firstPublicationDate"),
                    "document_type": normalize_document_type(
                        (item.get("pubTypeList") or {}).get("pubType")
                    ),
                    "language": item.get("language"),
                    "source_id": source_id,
                    "volume": (item.get("journalInfo") or {}).get("volume"),
                    "issue": (item.get("journalInfo") or {}).get("issue"),
                    "pages": item.get("pageInfo"),
                    "publisher": None,
                    "cited_by_count": item.get("citedByCount"),
                    "reference_count": len(item.get("_references") or [])
                    if item.get("hasReferences") == "Y"
                    else 0,
                    "is_retracted": bool(item.get("isRetracted")),
                    "source_record_hash": raw_hash,
                }
            )
            out["sources"].append(
                {
                    "source_id": source_id,
                    "name": clean_text(journal.get("title")),
                    "issn": issn,
                    "publisher": None,
                    "source_type": "journal",
                }
            )
            author_list = (item.get("authorList") or {}).get("author") or []
            for position, author in enumerate(author_list, start=1):
                full_name = clean_text(author.get("fullName"))
                orcid = normalize_orcid(
                    next(
                        (
                            author_identifier.get("value")
                            for author_identifier in (
                                (author.get("authorIdList") or {}).get("authorId") or []
                            )
                            if isinstance(author_identifier, dict)
                            and str(author_identifier.get("type", "")).upper() == "ORCID"
                        ),
                        None,
                    )
                )
                author_id = f"orcid:{orcid}" if orcid else stable_id("author", full_name)
                out["authors"].append(
                    {
                        "author_id": author_id,
                        "name": full_name,
                        "given_name": clean_text(author.get("firstName")),
                        "family_name": clean_text(author.get("lastName")),
                        "orcid": orcid,
                    }
                )
                affiliations = author.get("authorAffiliationDetailsList") or {}
                affiliations = affiliations.get("authorAffiliation") or [None]
                for affiliation in affiliations:
                    name = normalize_affiliation((affiliation or {}).get("affiliation"))
                    institution_id = stable_id("institution", name) if name else None
                    if name:
                        out["institutions"].append(
                            {
                                "institution_id": institution_id,
                                "name": name,
                                "ror": None,
                                "country_code": None,
                            }
                        )
                    out["authorships"].append(
                        {
                            "work_id": work_id,
                            "author_id": author_id,
                            "institution_id": institution_id,
                            "position": position,
                            "is_corresponding": False,
                        }
                    )
            for mesh in (item.get("meshHeadingList") or {}).get("meshHeading") or []:
                descriptor = mesh.get("descriptorName")
                text = normalize_keyword(
                    descriptor.get("#text") if isinstance(descriptor, dict) else descriptor
                )
                if text:
                    out["keywords"].append(
                        {"work_id": work_id, "keyword": text, "keyword_type": "mesh"}
                    )
            for keyword in (item.get("keywordList") or {}).get("keyword") or []:
                text = normalize_keyword(
                    keyword.get("#text") if isinstance(keyword, dict) else keyword
                )
                if text:
                    out["keywords"].append(
                        {"work_id": work_id, "keyword": text, "keyword_type": "author"}
                    )
            for reference in item.get("_references") or []:
                cited_doi = normalize_doi(reference.get("doi"))
                cited_external = (
                    f"{reference.get('source')}:{reference.get('id')}"
                    if reference.get("source") and reference.get("id")
                    else None
                )
                cited_id = (
                    f"doi:{cited_doi}"
                    if cited_doi
                    else (
                        f"epmc:{cited_external}"
                        if cited_external
                        else stable_id(
                            "reference",
                            reference.get("title"),
                            reference.get("authorString"),
                            reference.get("pubYear"),
                        )
                    )
                )
                out["references"].append(
                    {
                        "citing_work_id": work_id,
                        "cited_work_id": cited_id,
                        "cited_doi": cited_doi,
                        "cited_title": clean_text(reference.get("title")),
                        "cited_author": clean_text(reference.get("authorString")),
                        "cited_year": reference.get("pubYear"),
                        "source": "europe_pmc",
                    }
                )
            out["provenance"].append(
                {
                    "work_id": work_id,
                    "source": "europe_pmc",
                    "source_record_id": external,
                    "source_record_hash": raw_hash,
                }
            )
        return out

    @staticmethod
    def _finalize(rows: dict[str, list[dict[str, Any]]]) -> CanonicalTables:
        frames = {
            name: pd.DataFrame(rows.get(name, []), columns=columns)
            for name, columns in CANONICAL_COLUMNS.items()
        }
        works = frames.get("works", pd.DataFrame())
        duplicates: list[dict[str, Any]] = []
        work_id_map: dict[str, str] = {}
        if not works.empty:
            works["title_normalized"] = works["title"].map(normalize_title)
            works["dedup_key"] = works.apply(
                lambda row: (
                    f"doi:{row['doi']}"
                    if pd.notna(row.get("doi")) and str(row.get("doi")).strip()
                    else (
                        stable_id(
                            "dedup",
                            row.get("title_normalized"),
                            row.get("year") if pd.notna(row.get("year")) else "",
                        )
                        if len(str(row.get("title_normalized") or "")) >= 20
                        else f"record:{row.get('source_record_hash')}"
                    )
                ),
                axis=1,
            )
            keep_indices = []
            for _, group in works.groupby("dedup_key", dropna=False, sort=False):
                ranked = group.assign(
                    _score=group[["title", "abstract", "doi", "source_id"]].notna().sum(axis=1)
                ).sort_values("_score", ascending=False)
                winner = ranked.iloc[0]
                keep_indices.append(winner.name)
                for _, loser in ranked.iloc[1:].iterrows():
                    work_id_map[str(loser["work_id"])] = str(winner["work_id"])
                    duplicates.append(
                        {
                            "kept_work_id": winner["work_id"],
                            "removed_work_id": loser["work_id"],
                            "rule": "exact_doi_or_normalized_title_year",
                        }
                    )
            works = works.loc[keep_indices].drop(columns=["dedup_key"]).reset_index(drop=True)
        valid_work_ids = set(works.get("work_id", []))
        for name in ("authorships", "keywords", "topics", "provenance"):
            frame = frames.get(name, pd.DataFrame())
            if not frame.empty and "work_id" in frame:
                if work_id_map:
                    frame = frame.copy()
                    frame["work_id"] = frame["work_id"].map(
                        lambda value: work_id_map.get(str(value), value)
                    )
                frames[name] = frame[frame["work_id"].isin(valid_work_ids)].drop_duplicates()
        keywords = frames.get("keywords", pd.DataFrame())
        if not keywords.empty:
            keywords["keyword_raw"] = keywords["keyword"]
            keywords["keyword"] = keywords["keyword"].map(normalize_keyword)
            keywords = keywords.dropna(subset=["keyword"]).drop_duplicates(
                ["work_id", "keyword", "keyword_type"]
            )
            frames["keywords"] = keywords
        references = frames.get("references", pd.DataFrame())
        if not references.empty:
            if work_id_map:
                references = references.copy()
                references["citing_work_id"] = references["citing_work_id"].map(
                    lambda value: work_id_map.get(str(value), value)
                )
                references["cited_work_id"] = references["cited_work_id"].map(
                    lambda value: work_id_map.get(str(value), value)
                )
            references = references[
                references["citing_work_id"].isin(valid_work_ids)
            ].drop_duplicates(["citing_work_id", "cited_work_id"])
        return CanonicalTables(
            works=works,
            authors=frames.get("authors", pd.DataFrame()).drop_duplicates("author_id"),
            institutions=frames.get("institutions", pd.DataFrame()).drop_duplicates(
                "institution_id"
            ),
            authorships=frames.get("authorships", pd.DataFrame()),
            sources=frames.get("sources", pd.DataFrame()).drop_duplicates("source_id"),
            keywords=frames.get("keywords", pd.DataFrame()),
            topics=frames["topics"],
            references=references,
            provenance=frames.get("provenance", pd.DataFrame()),
            duplicates=pd.DataFrame(
                duplicates,
                columns=["kept_work_id", "removed_work_id", "rule"],
            ),
        )


def derive_keywords(
    works: pd.DataFrame,
    existing_keywords: pd.DataFrame,
    *,
    top_per_document: int = 5,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Add explicitly-labeled TF-IDF terms when source keywords are sparse."""
    if works.empty:
        return existing_keywords
    coverage = (
        existing_keywords["work_id"].nunique() / len(works) if not existing_keywords.empty else 0.0
    )
    if coverage >= 0.5:
        return existing_keywords
    documents = (works["title"].fillna("") + ". " + works["abstract"].fillna("")).str.strip()
    valid = documents.str.len() >= 10
    if valid.sum() < 2:
        return existing_keywords
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if valid.sum() >= 10 else 1,
        max_df=0.85 if valid.sum() >= 10 else 1.0,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(documents[valid])
    except ValueError as exc:
        if "no terms remain" not in str(exc).lower():
            raise
        # Small or stylistically homogeneous corpora can make every useful term
        # exceed max_df. Preserve the derivation path with a relaxed, explicit fallback.
        vectorizer.set_params(min_df=1, max_df=1.0)
        matrix = vectorizer.fit_transform(documents[valid])
    terms = np.asarray(vectorizer.get_feature_names_out())
    derived = []
    for row_number, work_index in enumerate(works.index[valid]):
        row = matrix.getrow(row_number)
        if row.nnz == 0:
            continue
        order = np.argsort(row.data)[-top_per_document:][::-1]
        for local in order:
            derived.append(
                {
                    "work_id": works.loc[work_index, "work_id"],
                    "keyword": str(terms[row.indices[local]]),
                    "keyword_type": "derived_tfidf",
                    "score": float(row.data[local]),
                }
            )
    derived_frame = pd.DataFrame(derived)
    if existing_keywords.empty:
        return derived_frame
    return pd.concat(
        [existing_keywords, derived_frame], ignore_index=True, sort=False
    ).drop_duplicates(["work_id", "keyword", "keyword_type"])
