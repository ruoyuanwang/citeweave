from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeweave.io import read_json
from citeweave.oversight_ai_self_review import (
    build_ai_self_review_plan,
    freeze_ai_self_review_plan,
    run_ai_self_review_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--packet-manifest", type=Path, required=True)
    build.add_argument("--tokenizer-manifest", type=Path, required=True)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--plan-freeze", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-freeze", type=Path, required=True)
    run.add_argument("--packet-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--api-key-file", type=Path, default=Path("apikey.md"))
    run.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    run.add_argument("--base-url", default="https://api.deepseek.com")
    run.add_argument("--retry-failed-call", action="store_true")
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        plan = build_ai_self_review_plan(
            packet_manifest_path=args.packet_manifest,
            tokenizer_manifest_path=args.tokenizer_manifest,
            output_path=args.plan,
        )
        freeze = freeze_ai_self_review_plan(args.plan, args.plan_freeze)
        result = {"plan": plan, "freeze": freeze}
    elif not args.execute:
        result = {
            "status": "dry_run",
            "plan": read_json(args.plan),
            "message": "Pass --execute only after the provider balance is available.",
        }
    else:
        result = run_ai_self_review_plan(
            plan_path=args.plan,
            plan_freeze_path=args.plan_freeze,
            packet_manifest_path=args.packet_manifest,
            output_root=args.output_root,
            api_key_file=args.api_key_file,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            retry_failed_call=args.retry_failed_call,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
