from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.source_relevance_packets import (
    audit_source_relevance_packets,
    build_source_relevance_packets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-pack-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build_source_relevance_packets(
        manifest_path=args.writer_pack_manifest,
        output_root=args.output_root,
    )
    audit = audit_source_relevance_packets(args.output_root)
    write_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    raise SystemExit(0 if audit["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
