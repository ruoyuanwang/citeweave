from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from citeweave.live_voi_review import create_live_voi_review_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve development-only outcome-adaptive article review."
    )
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--access", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    app = create_live_voi_review_app(args.packet_root, args.access)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
