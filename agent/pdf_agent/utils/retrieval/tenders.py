from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_tender_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if raw.startswith("uploaded:"):
        raw = raw.split(":", 1)[1]
    raw = Path(raw).name
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]
    raw = raw.replace(" ", "")
    if not raw:
        return None
    return raw


def infer_tender_id_from_sources(sources: List[Dict[str, Any]]) -> Optional[str]:
    for src in sources:
        src_id = normalize_tender_id(src.get("source"))
        if src_id:
            return src_id
    return None


def infer_tender_year(tender_id: Optional[str]) -> Optional[int]:
    if not tender_id:
        return None
    match = re.search(r"-(\d{4})", tender_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
