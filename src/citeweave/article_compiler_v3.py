from __future__ import annotations

from typing import Any

from .article_compiler_v2 import (
    PROMPT_VERSION as PREVIOUS_PROMPT_VERSION,
)
from .article_compiler_v2 import (
    build_compiler_repair_request as _build_repair,
)
from .article_compiler_v2 import (
    build_compiler_section_request as _build_section,
)

PROMPT_VERSION = "matched-article-compiler-claim-ready-token-safe-development-20260908"
SECTION_MAX_TOKENS = {
    "Abstract": 700,
    "Introduction": 1000,
    "Methods": 1200,
    "Results": 2400,
    "Discussion": 1900,
    "Limitations": 800,
    "Conclusion": 600,
}
REPAIR_MAX_TOKENS = {"Results": 2800, "Discussion": 2300}


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
    request = _build_section(
        writer_input,
        condition=condition,
        section=section,
        compiled_sections=compiled_sections,
        model=model,
    )
    request["max_tokens"] = SECTION_MAX_TOKENS[section]
    return _versioned(request)


def build_compiler_repair_request(
    writer_input: dict[str, Any],
    *,
    condition: str,
    section: str,
    original_body: str,
    compiled_sections: dict[str, str],
    model: str = "deepseek-v4-pro",
) -> dict[str, Any]:
    request = _build_repair(
        writer_input,
        condition=condition,
        section=section,
        original_body=original_body,
        compiled_sections=compiled_sections,
        model=model,
    )
    request["max_tokens"] = REPAIR_MAX_TOKENS[section]
    return _versioned(request)
