from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.article_writer_inputs import build_text_only_writer_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_text_only_writer_inputs(
        args.source_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
