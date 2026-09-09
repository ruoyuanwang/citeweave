from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import write_json
from citeweave.source_review_assignment import (
    assess_source_warmup_panel_readiness,
    validate_source_warmup_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--reviewer-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol.yml"
        ),
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/source_relevance_warmup_protocol_freeze.json"
        ),
    )
    parser.add_argument(
        "--execution-amendment",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity.yml"
        ),
    )
    parser.add_argument(
        "--execution-amendment-freeze",
        type=Path,
        default=Path(
            "experiments/human_review_v2/"
            "source_relevance_warmup_amendment_001_execution_integrity_freeze.json"
        ),
    )
    args = parser.parse_args()
    validate_source_warmup_protocol(
        protocol_path=args.protocol,
        protocol_freeze_path=args.protocol_freeze,
        packet_root=args.packet_root,
        amendment_path=args.execution_amendment,
        amendment_freeze_path=args.execution_amendment_freeze,
    )
    readiness = assess_source_warmup_panel_readiness(
        packet_root=args.packet_root,
        reviewer_roster_path=args.reviewer_roster,
    )
    write_json(args.output, readiness)
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    raise SystemExit(0 if readiness["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
