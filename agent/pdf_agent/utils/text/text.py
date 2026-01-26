from __future__ import annotations

import re


def preview_text(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
