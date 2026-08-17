from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "human_outputs"
METHOD_PATTERN = re.compile(
    r"\b(methods?|materials?|methodology|data|search strategy|retrieval|inclusion|"
    r"exclusion|statistical analysis|bibliometric analysis)\b",
    flags=re.IGNORECASE,
)


def _name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element if _name(item) == name), None)


def _descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element.iter() if _name(item) == name), None)


def _method_sections(root: ET.Element) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for section in (item for item in root.iter() if _name(item) == "sec"):
        title = _text(_child(section, "title"))
        sec_type = section.attrib.get("sec-type", "").replace("-", " ")
        if METHOD_PATTERN.search(title) or METHOD_PATTERN.search(sec_type):
            content = _text(section)
            if content:
                selected.append((title or sec_type or "Methods", content))
    return selected


def _tables_and_figures(root: ET.Element) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for element in root.iter():
        kind = _name(element)
        if kind not in {"table-wrap", "fig"}:
            continue
        label = _text(_child(element, "label"))
        caption = _text(_child(element, "caption"))
        body = ""
        if kind == "table-wrap":
            table = _descendant(element, "table")
            if table is not None:
                rows = []
                for row in (item for item in table.iter() if _name(item) == "tr"):
                    cells = [
                        _text(cell)
                        for cell in row
                        if _name(cell) in {"td", "th"} and _text(cell)
                    ]
                    if cells:
                        rows.append(" | ".join(cells))
                body = "\n".join(rows) or _text(table)
        content = " | ".join(value for value in (label, caption, body) if value)
        if content:
            records.append((kind, label or kind, content))
    return records


def extract_reference_evidence(xml_bytes: bytes) -> tuple[str, dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    article_meta = _descendant(root, "article-meta")
    if article_meta is None:
        raise ValueError("XML lacks article-meta")
    title = _text(_descendant(article_meta, "article-title"))
    abstract = _text(_descendant(article_meta, "abstract"))
    methods = _method_sections(root)
    displays = _tables_and_figures(root)
    lines = [
        f"# Evidence for: {title}",
        "",
        (
            "This packet is paired only with the corresponding published human reference. "
            "Its corpus statistics are not Gold for the independently retrieved system report."
        ),
        "",
    ]
    if abstract:
        lines.extend(["## Abstract and scope", "", abstract, ""])
    lines.extend(["## Methods and retrieval evidence", ""])
    if methods:
        for heading, content in methods:
            lines.extend([f"### {heading}", "", content, ""])
    else:
        lines.extend(["No separately titled methods section was available in the XML.", ""])
    lines.extend(["## Tables and figure captions", ""])
    if displays:
        for kind, label, content in displays:
            lines.extend([f"### {kind}: {label}", "", content, ""])
    else:
        lines.extend(["No table or figure caption was available in the XML.", ""])
    report = "\n".join(lines).rstrip() + "\n"
    metadata = {
        "methods_sections": len(methods),
        "tables_and_figures": len(displays),
        "characters": len(report),
    }
    return report, metadata


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_directory(directory: Path) -> str:
    source = directory / "source.xml"
    output = directory / "reference_evidence.md"
    manifest_path = directory / "evidence_manifest.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or manifest_path.exists():
        if not output.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"Incomplete evidence output: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_xml_sha256") != _sha(source)
            or manifest.get("reference_evidence_sha256") != _sha(output)
        ):
            raise RuntimeError(f"Hash-mismatched evidence output: {directory}")
        return "skipped"
    evidence, extraction = extract_reference_evidence(source.read_bytes())
    output.write_text(evidence, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "reference_id": directory.name,
        "source_xml_sha256": _sha(source),
        "reference_evidence_sha256": _sha(output),
        "evidence_role": "candidate-specific evidence; not system numeric Gold",
        "extraction": extraction,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--id", action="append", default=[])
    args = parser.parse_args()
    directories = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.id:
        selected = set(args.id)
        directories = [path for path in directories if path.name in selected]
        missing = selected - {path.name for path in directories}
        if missing:
            raise SystemExit(f"Unknown reference ids: {sorted(missing)}")
    for directory in directories:
        print(f"{prepare_directory(directory)}: {directory.name}")


if __name__ == "__main__":
    main()
