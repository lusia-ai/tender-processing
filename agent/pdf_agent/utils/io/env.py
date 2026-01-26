from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv

_LOADED_ENV_FILES: Set[Path] = set()


def load_env(env_file: Optional[str]) -> None:
    if not env_file:
        return
    path = Path(env_file).expanduser().resolve()
    if path in _LOADED_ENV_FILES:
        return
    load_dotenv(dotenv_path=path, override=False)
    _LOADED_ENV_FILES.add(path)


def require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {key}")
    return value
