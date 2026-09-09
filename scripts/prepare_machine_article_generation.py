from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_generation import build_machine_article_generation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    args = parser.parse_args()
    plan = build_machine_article_generation_plan(
        args.writer_input_manifest,
        output_dir=args.output_dir,
        model=args.model,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
