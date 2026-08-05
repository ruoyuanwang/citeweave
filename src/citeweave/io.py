from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .models import ProjectConfig

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def slugify(value: str, limit: int = 64) -> str:
    value = SAFE_NAME.sub("-", value.strip()).strip("-._").lower()
    return (value or "project")[:limit]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    atomic_write_bytes(path, payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically write JSONL without constructing a second corpus-sized buffer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_config(path: Path) -> ProjectConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)


def save_config(path: Path, config: ProjectConfig) -> None:
    data = config.model_dump(mode="json")
    payload = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
    atomic_write_bytes(path, payload)
