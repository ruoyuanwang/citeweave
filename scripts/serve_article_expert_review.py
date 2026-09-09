from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from citeweave.article_expert_review_ui import create_article_expert_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--access", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    app = create_article_expert_app(args.collection_manifest, args.access)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
