from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.formal_adaptive_review import (
    FormalAdaptiveConfig,
    FormalAdaptiveReviewRunner,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Advance the model-free formal Always/Static/Adaptive review state machine. "
            "A constrained Codex Human Proxy handles feedback packets; an independent "
            "Evaluation Judge handles evaluation packets."
        )
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT / "experiments" / "human_references.yml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "formal_adaptive_review",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-confirmations", type=int, default=2)
    parser.add_argument("--auto-accept-threshold", type=float, default=0.95)
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--static-detector-threshold", type=float, default=0.50)
    parser.add_argument(
        "--development-calibration",
        action="store_true",
        help=(
            "Run only the two development topics and mark all outputs as calibration, "
            "never as formal results."
        ),
    )
    args = parser.parse_args()
    runner = FormalAdaptiveReviewRunner(
        cases_path=args.cases,
        reference_registry=args.references,
        output_root=args.output_root,
        config=FormalAdaptiveConfig(
            experiment_mode=(
                "development_calibration"
                if args.development_calibration
                else "formal"
            ),
            seed=args.seed,
            minimum_confirmations=args.minimum_confirmations,
            auto_accept_threshold=args.auto_accept_threshold,
            audit_rate=args.audit_rate,
            static_detector_threshold=args.static_detector_threshold,
        ),
    )
    print(json.dumps(runner.advance_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
