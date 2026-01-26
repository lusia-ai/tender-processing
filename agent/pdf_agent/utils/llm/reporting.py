from __future__ import annotations

from typing import Optional


def infer_report_format(report_format: Optional[str], query: str) -> str:
    if report_format:
        value = report_format.strip().lower()
        if value in {"json", "table", "text"}:
            return value
    lowered = (query or "").lower()
    if "json" in lowered:
        return "json"
    if "table" in lowered:
        return "table"
    return "text"
