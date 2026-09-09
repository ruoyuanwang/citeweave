from __future__ import annotations

try:
    import run_article_compiler_v3_pilot as runner
except ModuleNotFoundError:
    import scripts.run_article_compiler_v3_pilot as runner

from citeweave.article_compiler_v4 import (
    build_compiler_repair_request,
    build_compiler_section_request,
)


def main() -> None:
    runner.build_compiler_section_request = build_compiler_section_request
    runner.build_compiler_repair_request = build_compiler_repair_request
    runner.main()


if __name__ == "__main__":
    main()
