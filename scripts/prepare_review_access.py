from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

from citeweave.io import read_json, write_json


def prepare_access(manifest_path: Path, output_path: Path) -> dict[str, str]:
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite existing reviewer access hashes")
    rows = read_json(manifest_path).get("reviewers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reviewer manifest must contain a nonempty reviewers list")
    reviewer_ids = []
    for row in rows:
        reviewer_id = row.get("reviewer_id") if isinstance(row, dict) else row
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("Every reviewer requires a nonempty identity")
        reviewer_ids.append(reviewer_id)
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError("Duplicate reviewer identities are prohibited")
    invitations = {}
    hashes = {}
    for reviewer_id in reviewer_ids:
        token = secrets.token_urlsafe(32)
        invitations[reviewer_id] = token
        hashes[reviewer_id] = hashlib.sha256(token.encode()).hexdigest()
    write_json(
        output_path,
        {
            "schema_version": 1,
            "reviewer_token_sha256": hashes,
            "note": "Plaintext tokens were printed once and are not stored here.",
        },
    )
    return invitations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    invitations = prepare_access(args.manifest, args.output)
    print(json.dumps({"reviewer_tokens": invitations}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
