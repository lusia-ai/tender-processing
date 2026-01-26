from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

_PROMPT_CACHE: Dict[str, str] = {}
_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path = _PROMPTS_DIR / name
    text = path.read_text()
    _PROMPT_CACHE[name] = text
    return text


def render_prompt(name: str, **kwargs: Any) -> str:
    text = load_prompt(name)
    for key, value in kwargs.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text
