from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "human_references.yml"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "human_outputs"
EUROPE_PMC_FULL_TEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
SECTION_PATTERN = re.compile(
    r"\b(results?|discussions?|conclusions?|concluding remarks)\b",
    flags=re.IGNORECASE,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element.iter() if _local_name(item.tag) == name), None)


def _normalized_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _direct_content_blocks(element: ET.Element) -> list[str]:
    blocks: list[str] = []
    for child in element:
        name = _local_name(child.tag)
        if name in {"title", "sec"}:
            continue
        text = _normalized_text(child)
        if text:
            blocks.append(text)
    return blocks


def _render_nested_section(element: ET.Element, level: int) -> list[str]:
    lines: list[str] = []
    title_element = next(iter(_children(element, "title")), None)
    title = _normalized_text(title_element)
    if title:
        lines.extend([f"{'#' * min(level, 6)} {title}", ""])
    for block in _direct_content_blocks(element):
        lines.extend([block, ""])
    for child in _children(element, "sec"):
        lines.extend(_render_nested_section(child, level + 1))
    return lines


def _section_matches(element: ET.Element) -> bool:
    title = _normalized_text(next(iter(_children(element, "title")), None))
    sec_type = element.attrib.get("sec-type", "").replace("-", " ")
    return bool(SECTION_PATTERN.search(title) or SECTION_PATTERN.search(sec_type))


def _selected_sections(body: ET.Element | None) -> list[ET.Element]:
    if body is None:
        return []
    selected: list[ET.Element] = []

    def visit(section: ET.Element, ancestor_selected: bool = False) -> None:
        matched = _section_matches(section)
        if matched and not ancestor_selected:
            selected.append(section)
        for child in _children(section, "sec"):
            visit(child, ancestor_selected or matched)

    for section in _children(body, "sec"):
        visit(section)
    return selected


def _render_abstract(abstract: ET.Element) -> list[str]:
    lines = ["## Abstract", ""]
    blocks = _direct_content_blocks(abstract)
    if not blocks:
        text = _normalized_text(abstract)
        if text:
            blocks = [text]
    for block in blocks:
        lines.extend([block, ""])
    for section in _children(abstract, "sec"):
        lines.extend(_render_nested_section(section, 3))
    return lines


def extract_reference_report(xml_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
    root = ET.fromstring(xml_bytes)
    article_meta = _first_descendant(root, "article-meta")
    if article_meta is None:
        raise ValueError("Europe PMC XML does not contain article-meta")
    article_title = _normalized_text(_first_descendant(article_meta, "article-title"))
    if not article_title:
        raise ValueError("Europe PMC XML does not contain an article title")

    abstract = _first_descendant(article_meta, "abstract")
    abstract_text = _normalized_text(abstract)
    body = _first_descendant(root, "body")
    selected = _selected_sections(body)

    lines = [f"# {article_title}", ""]
    extracted = [
        {
            "kind": "title",
            "title": "Article Title",
            "characters": len(article_title),
        }
    ]
    if abstract is not None and abstract_text:
        lines.extend(_render_abstract(abstract))
        extracted.append(
            {
                "kind": "abstract",
                "title": "Abstract",
                "characters": len(abstract_text),
            }
        )

    for section in selected:
        section_title = _normalized_text(next(iter(_children(section, "title")), None))
        if not section_title:
            section_title = section.attrib.get("sec-type", "Selected Section").replace("-", " ")
        lines.extend(_render_nested_section(section, 2))
        extracted.append(
            {
                "kind": "article_section",
                "title": section_title,
                "characters": len(_normalized_text(section)),
            }
        )

    if not selected:
        raise ValueError("No Results, Discussion, or Conclusion section was found")
    report = "\n".join(lines).rstrip() + "\n"
    return report, extracted


def _download_xml(
    client: httpx.Client,
    url: str,
    *,
    retries: int,
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            ET.fromstring(content)
            return content, str(response.url)
        except (httpx.HTTPError, ET.ParseError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise RuntimeError(f"Failed to download valid XML from {url}: {last_error}") from last_error


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot validate existing manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Existing manifest is not a JSON object: {path}")
    return value


def _validate_existing(output_dir: Path) -> bool:
    source_path = output_dir / "source.xml"
    report_path = output_dir / "reference_report.md"
    manifest_path = output_dir / "manifest.json"
    expected = {source_path, report_path, manifest_path}
    if not output_dir.exists():
        return False
    if not output_dir.is_dir() or not all(path.is_file() for path in expected):
        raise RuntimeError(
            f"Refusing to overwrite incomplete or unexpected existing output: {output_dir}"
        )
    manifest = _load_manifest(manifest_path)
    hashes = manifest.get("sha256", {})
    source_matches = hashes.get("source_xml") == _sha256_file(source_path)
    report_matches = hashes.get("reference_report") == _sha256_file(report_path)
    if not source_matches or not report_matches:
        raise RuntimeError(f"Refusing to overwrite hash-mismatched existing output: {output_dir}")
    return True


def _atomic_write_output(
    output_dir: Path,
    *,
    source_xml: bytes,
    report: str,
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        (temporary / "source.xml").write_bytes(source_xml)
        (temporary / "reference_report.md").write_bytes(report.encode("utf-8"))
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise RuntimeError(
                f"Output appeared during preparation; refusing overwrite: {output_dir}"
            )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_reference(
    reference: dict[str, Any],
    *,
    output_root: Path,
    client: httpx.Client,
    retries: int,
) -> str:
    output_dir = output_root / reference["id"]
    if _validate_existing(output_dir):
        return "skipped"

    pmcid = str(reference["pmcid"]).upper()
    request_url = EUROPE_PMC_FULL_TEXT_URL.format(pmcid=pmcid)
    retrieved_at = datetime.now(UTC)
    source_xml, final_url = _download_xml(client, request_url, retries=retries)
    report, extracted_sections = extract_reference_report(source_xml)
    report_bytes = report.encode("utf-8")
    manifest = {
        "schema_version": "1.0",
        "reference_id": reference["id"],
        "pmcid": pmcid,
        "source_url": final_url,
        "retrieval_utc": retrieved_at.isoformat(),
        "reference_only": True,
        "independent_query_required": True,
        "human_numeric_results_are_gold": False,
        "sha256": {
            "source_xml": _sha256_bytes(source_xml),
            "reference_report": _sha256_bytes(report_bytes),
        },
        "characters": {
            "source_xml": len(source_xml.decode("utf-8", errors="replace")),
            "reference_report": len(report),
        },
        "extracted_sections": extracted_sections,
    }
    _atomic_write_output(
        output_dir,
        source_xml=source_xml,
        report=report,
        manifest=manifest,
    )
    return "created"


def _load_references(path: Path) -> list[dict[str, Any]]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    references = registry.get("references", [])
    if not isinstance(references, list) or not references:
        raise ValueError(f"No references found in {path}")
    if registry.get("independent_query_required") is not True:
        raise ValueError("Reference registry must require an independent CiteWeave query")
    if registry.get("human_numeric_results_are_gold") is not False:
        raise ValueError("Human numerical results must not be registered as gold")
    return references


def _select_references(
    references: Iterable[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    values = list(references)
    if not selected_ids:
        return values
    by_id = {item["id"]: item for item in values}
    missing = selected_ids - set(by_id)
    if missing:
        raise ValueError(f"Unknown reference ids: {sorted(missing)}")
    return [item for item in values if item["id"] in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract frozen human-reference report sections from Europe PMC."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--id", action="append", default=[], help="Prepare only this reference id.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")

    references = _select_references(_load_references(args.registry), set(args.id))
    headers = {
        "Accept": "application/xml",
        "User-Agent": "CiteWeave/0.1 (human-reference artifact preparation)",
    }
    with httpx.Client(timeout=args.timeout, follow_redirects=True, headers=headers) as client:
        for reference in references:
            status = prepare_reference(
                reference,
                output_root=args.output_root,
                client=client,
                retries=args.retries,
            )
            print(f"{status}: {reference['id']}")


if __name__ == "__main__":
    main()
