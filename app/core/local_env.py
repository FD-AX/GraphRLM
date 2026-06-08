from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILES = (".env.local", ".env")


def load_local_env(
    project_root: Path,
    filenames: tuple[str, ...] = DEFAULT_ENV_FILES,
    override: bool = False,
) -> list[str]:
    loaded_keys: list[str] = []
    for filename in filenames:
        path = project_root / filename
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            loaded_keys.append(key)
    return loaded_keys


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip().lstrip("\ufeff")
    if not key:
        return None
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]
    return key, value
