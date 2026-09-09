from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from citeweave.query_relevance_review_ui import create_query_relevance_review_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--access", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    app = create_query_relevance_review_app(args.manifest, args.access)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
