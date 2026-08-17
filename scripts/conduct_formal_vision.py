from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.formal_vision_conductor import (
    RUN_ID,
    VisionConductorError,
    import_returns,
    prepare_assignments,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and import independent visible-only Codex Figure/VLM assignments."
        )
    )
    parser.add_argument("action", choices=("prepare", "import"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "experiments" / "formal_datasets_openalex_title_abstract.yml",
    )
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=ROOT / "experiments" / "vision_packets",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT / "experiments" / "formal_workspaces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "vision_outputs",
    )
    parser.add_argument(
        "--exchange-root",
        type=Path,
        default=ROOT / "experiments" / "formal_vision_exchange",
    )
    parser.add_argument(
        "--returns-root",
        type=Path,
        default=ROOT / "experiments" / "formal_vision_returns",
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--run-score",
        action="store_true",
        help=(
            "After a complete validated import, run score_formal_graph_runs.py "
            "--include-vision for all eight topics."
        ),
    )
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            if args.run_score:
                parser.error("--run-score is only valid with the import action")
            result = prepare_assignments(
                registry_path=args.registry.resolve(),
                packet_root=args.packet_root.resolve(),
                workspace_root=args.workspace_root.resolve(),
                output_root=args.output_root.resolve(),
                exchange_root=args.exchange_root.resolve(),
                returns_root=args.returns_root.resolve(),
            )
        else:
            result = import_returns(
                registry_path=args.registry.resolve(),
                packet_root=args.packet_root.resolve(),
                workspace_root=args.workspace_root.resolve(),
                output_root=args.output_root.resolve(),
                returns_root=args.returns_root.resolve(),
                run_score=args.run_score,
                run_id=args.run_id,
            )
    except VisionConductorError as exc:
        raise SystemExit(f"Formal Figure/VLM conductor refused exchange: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
