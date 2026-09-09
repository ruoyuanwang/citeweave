from __future__ import annotations

from typing import Any

from .article_compiler_v3 import PROMPT_VERSION as PREVIOUS_PROMPT_VERSION
from .article_compiler_v3 import (
    build_compiler_repair_request as _build_repair,
)
from .article_compiler_v3 import (
    build_compiler_section_request as _build_section,
)
from .article_context_compaction import audit_section_writer_view, section_writer_view

PROMPT_VERSION = "matched-article-compiler-claim-ready-compact-context-20260908"


def _view(writer_input: dict[str, Any], section: str) -> dict[str, Any]:
    view = section_writer_view(writer_input, section=section)
    audit = audit_section_writer_view(writer_input, view)
    if not audit["passed"]:
        raise RuntimeError(f"Section writer view failed integrity audit: {section}")
    return view


def _versioned(request: dict[str, Any]) -> dict[str, Any]:
    request["messages"][1]["content"] = request["messages"][1]["content"].replace(
        PREVIOUS_PROMPT_VERSION, PROMPT_VERSION
    )
    return request


def build_compiler_section_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    section: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    return _versioned(
        _build_section(
            _view(writer_input, section),
            condition=condition,
            section=section,
            compiled_sections=compiled_sections,
            model=model,
        )
    )


def build_compiler_repair_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    section: str,
    original_body: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    return _versioned(
        _build_repair(
            _view(writer_input, section),
            condition=condition,
            section=section,
            original_body=original_body,
            compiled_sections=compiled_sections,
            model=model,
        )
    )
