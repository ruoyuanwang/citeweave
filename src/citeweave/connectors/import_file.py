from __future__ import annotations

import csv
import json
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..exceptions import AcquisitionError
from ..io import sha256_file
from ..models import SearchProtocol, SourceName
from .base import AcquisitionResult, BaseConnector

YEAR_RE = re.compile(r"\b(18|19|20|21)\d{2}\b")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _split(value: Any, separators: str = r";|\|") -> list[str]:
    return [part.strip() for part in re.split(separators, str(value or "")) if part.strip()]


def _first(row: dict[str, Any], *names: str) -> Any:
    folded = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        value = folded.get(name.casefold())
        if value not in (None, ""):
            return value
    return None


def _year(value: Any) -> int | None:
    match = YEAR_RE.search(str(value or ""))
    return int(match.group()) if match else None


def _author(name: str) -> dict[str, Any]:
    name = name.strip()
    if "," in name:
        family, given = (part.strip() for part in name.split(",", 1))
    else:
        parts = name.split()
        family, given = (parts[-1], " ".join(parts[:-1])) if parts else ("", "")
    return {"given": given or None, "family": family or None, "affiliation": []}


def _reference(text: str) -> dict[str, Any]:
    doi_match = DOI_RE.search(text)
    year = _year(text)
    author = text.split(",", 1)[0].strip() or None
    return {
        "unstructured": text,
        "DOI": doi_match.group().rstrip(".,;)") if doi_match else None,
        "year": str(year) if year else None,
        "author": author,
    }


def _as_crossref(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    """Map common WoS, Scopus and generic CSV columns to one lossless staging shape."""
    title = _first(row, "TI", "Title", "Document Title", "Article Title")
    abstract = _first(row, "AB", "Abstract", "Abstract Note")
    year = _year(_first(row, "PY", "Year", "Publication Year", "Y1"))
    doi = _first(row, "DI", "DOI", "Digital Object Identifier")
    source = _first(row, "SO", "Source title", "Publication Name", "Journal", "JF", "JO")
    author_text = _first(row, "AU", "Authors", "Author", "Author(s)")
    keywords = _split(_first(row, "DE", "Author Keywords", "Keywords", "KW")) + _split(
        _first(row, "ID", "Index Keywords", "Keywords Plus")
    )
    refs = _split(_first(row, "CR", "References", "Cited References"), r";|\n")
    affiliations = _split(_first(row, "C1", "Affiliations", "Author Address"), r";|\|")
    authors = [_author(name) for name in _split(author_text)]
    if affiliations and authors:
        authors[0]["affiliation"] = [{"name": name} for name in affiliations]
    issn = _split(_first(row, "SN", "ISSN"), r";|\||,")
    cited = _first(row, "TC", "Cited by", "Times Cited", "is-referenced-by-count")
    ref_count = _first(row, "NR", "References Count", "references-count")
    record_id = _first(row, "UT", "EID", "Accession Number", "URL", "UR")
    return {
        "DOI": doi,
        "URL": record_id or f"import-row:{row_number}",
        "title": [title] if title else [],
        "abstract": abstract,
        "published": {"date-parts": [[year]]} if year else None,
        "container-title": [source] if source else [],
        "author": authors,
        "subject": list(dict.fromkeys(keywords)),
        "reference": [_reference(value) for value in refs],
        "ISSN": issn,
        "publisher": _first(row, "PU", "Publisher"),
        "type": _first(row, "DT", "Document Type", "Type", "TY"),
        "language": _first(row, "LA", "Language", "Language of Original Document"),
        "volume": _first(row, "VL", "Volume"),
        "issue": _first(row, "IS", "Issue"),
        "page": _first(row, "Pages", "Page start", "BP", "SP"),
        "is-referenced-by-count": int(float(cited))
        if str(cited or "").replace(".", "", 1).isdigit()
        else None,
        "references-count": int(float(ref_count))
        if str(ref_count or "").replace(".", "", 1).isdigit()
        else len(refs),
        "_import_row": row_number,
        "_import_fields": row,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.casefold() in {".tsv", ".txt"} else csv.excel
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, dialect=dialect)]


def _tagged_records(lines: Iterable[str], end_tag: str = "ER") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    last_tag: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        match = re.match(r"^([A-Z][A-Z0-9])(?:  |- )?(.*)$", line)
        if match:
            tag, value = match.groups()
            if tag == end_tag:
                if current:
                    records.append(current)
                current, last_tag = {}, None
                continue
            last_tag = tag
            if tag in {"AU", "AF", "KW", "DE", "ID", "CR", "C1"}:
                current[tag] = "; ".join(filter(None, [current.get(tag), value.strip()]))
            else:
                current[tag] = value.strip()
        elif last_tag and line.strip():
            current[last_tag] = f"{current.get(last_tag, '')} {line.strip()}".strip()
    if current:
        records.append(current)
    return records


def _read_ris(path: Path) -> list[dict[str, Any]]:
    raw = _tagged_records(path.read_text(encoding="utf-8-sig", errors="replace").splitlines())
    aliases = {
        "T1": "TI",
        "JF": "SO",
        "JO": "SO",
        "T2": "SO",
        "Y1": "PY",
        "N2": "AB",
        "DO": "DI",
    }
    return [{aliases.get(key, key): value for key, value in row.items()} for row in raw]


def _read_bibtex(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    entries: list[str] = []
    start = 0
    depth = 0
    active = False
    for index, char in enumerate(text):
        if char == "@" and depth == 0:
            start, active = index, True
        elif active and char == "{":
            depth += 1
        elif active and char == "}":
            depth -= 1
            if depth == 0:
                entries.append(text[start : index + 1])
                active = False
    rows = []
    for entry in entries:
        kind = re.match(r"@\s*(\w+)", entry)
        fields: dict[str, Any] = {"TY": kind.group(1) if kind else None}
        body = entry[entry.find(",") + 1 : -1]
        for match in re.finditer(
            r"(?is)(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|([^,\n]+))\s*,?",
            body,
        ):
            key = match.group(1).casefold()
            value = next(value for value in match.groups()[1:] if value is not None).strip()
            mapping = {
                "title": "TI",
                "abstract": "AB",
                "year": "PY",
                "doi": "DI",
                "journal": "SO",
                "booktitle": "SO",
                "author": "AU",
                "keywords": "DE",
                "issn": "SN",
                "publisher": "PU",
                "volume": "VL",
                "number": "IS",
                "pages": "Pages",
                "url": "URL",
                "language": "LA",
            }
            fields[mapping.get(key, key)] = value.replace(" and ", "; ")
        rows.append(fields)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


class ImportFileConnector(BaseConnector):
    source_name = SourceName.import_file.value

    def acquire(self, protocol: SearchProtocol) -> AcquisitionResult:
        if protocol.input_file is None:
            raise AcquisitionError("input_file is required")
        source_path = protocol.input_file.expanduser().resolve()
        if not source_path.is_file():
            raise AcquisitionError(f"Input file does not exist: {source_path}")
        fmt = protocol.input_format
        if fmt == "auto":
            fmt = {
                ".csv": "csv",
                ".tsv": "csv",
                ".ris": "ris",
                ".bib": "bibtex",
                ".bibtex": "bibtex",
                ".jsonl": "jsonl",
            }.get(source_path.suffix.casefold(), "wos")
        readers = {
            "csv": _read_csv,
            "ris": _read_ris,
            "bibtex": _read_bibtex,
            "wos": _read_ris,
            "jsonl": _read_jsonl,
        }
        try:
            raw_rows = readers[fmt](source_path)
        except Exception as exc:
            raise AcquisitionError(f"Failed to parse {fmt} file: {exc}") from exc
        staged = (
            raw_rows
            if fmt == "jsonl"
            else [_as_crossref(row, index) for index, row in enumerate(raw_rows, 1)]
        )

        terms = [term.casefold() for term in protocol.keywords]
        filtered = []
        for item in staged:
            year = _year(item.get("published"))
            haystack = " ".join(
                [
                    *(item.get("title") or []),
                    item.get("abstract") or "",
                    *(item.get("subject") or []),
                ]
            ).casefold()
            term_match = (
                all(term in haystack for term in terms)
                if protocol.query_mode == "all"
                else any(term in haystack for term in terms)
            )
            if protocol.query_mode == "phrase":
                term_match = " ".join(terms) in haystack
            if year and protocol.year_from <= year <= protocol.year_to and term_match:
                filtered.append(item)

        digest = sha256_file(source_path)
        raw_copy = self.raw_dir / f"import-{digest[:12]}{source_path.suffix.casefold()}"
        if not raw_copy.exists():
            shutil.copyfile(source_path, raw_copy)
        manifest = self._new_manifest(
            self.source_name,
            {
                "input_file": str(source_path),
                "input_sha256": digest,
                "format": fmt,
                "keywords": protocol.keywords,
                "year_from": protocol.year_from,
                "year_to": protocol.year_to,
            },
        )
        manifest.expected_records = len(filtered)
        manifest.received_records = len(raw_rows)
        manifest.unique_records = len(filtered)
        manifest.pages = 1
        manifest.raw_sha256 = [digest]
        manifest.finished_at = datetime.now(UTC)
        manifest.drift = 0
        manifest.complete = True
        manifest.warnings.append(
            f"Parsed all {len(raw_rows)} source records; {len(filtered)} matched the protocol."
        )
        return AcquisitionResult(filtered, manifest, [raw_copy])
