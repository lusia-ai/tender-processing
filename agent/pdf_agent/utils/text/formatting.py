from __future__ import annotations

from typing import Any, Dict, List


def format_company_profile(profile: Dict[str, Any]) -> str:
    if not profile:
        return "not found"
    lines = []
    for key in ("name", "description", "country", "website", "notes"):
        value = profile.get(key)
        if value:
            lines.append(f"{key}: {value}")
    capabilities = profile.get("capabilities", [])
    if isinstance(capabilities, list) and capabilities:
        lines.append("capabilities:")
        for cap in capabilities:
            if not isinstance(cap, dict):
                continue
            detail = " | ".join(
                [
                    str(cap.get("capability")) if cap.get("capability") else "",
                    str(cap.get("capacity")) if cap.get("capacity") else "",
                    str(cap.get("unit")) if cap.get("unit") else "",
                ]
            ).strip(" |")
            if detail:
                lines.append(f"- {detail}")
    certifications = profile.get("certifications", [])
    if isinstance(certifications, list) and certifications:
        lines.append("certifications:")
        for item in certifications:
            if not isinstance(item, dict):
                continue
            detail = " | ".join(
                [
                    str(item.get("certification")) if item.get("certification") else "",
                    str(item.get("status")) if item.get("status") else "",
                    str(item.get("expires")) if item.get("expires") else "",
                ]
            ).strip(" |")
            if detail:
                lines.append(f"- {detail}")
    return "\n".join(lines) if lines else "not found"


def format_company_compliance(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "not found"
    lines = []
    for item in items:
        parts = []
        for key in ("category", "status", "notes"):
            if item.get(key):
                parts.append(str(item.get(key)))
        if parts:
            lines.append(" - ".join(parts))
    return "\n".join(lines) if lines else "not found"


def format_tender_metadata(meta: Dict[str, Any]) -> str:
    if not meta:
        return "not found"
    lines = []
    for key in (
        "tender_id",
        "title",
        "primary_title",
        "buyer_name",
        "buyer_country",
        "publication_date",
        "deadline",
        "value",
        "currency",
    ):
        if meta.get(key) is not None:
            lines.append(f"{key}: {meta.get(key)}")
    return "\n".join(lines) if lines else "not found"


def format_tender_lots(lots: List[Dict[str, Any]]) -> str:
    if not lots:
        return "not found"
    lines = []
    for lot in lots:
        parts = []
        if lot.get("lot_id"):
            parts.append(f"lot_id={lot.get('lot_id')}")
        if lot.get("title"):
            parts.append(str(lot.get("title")))
        if lot.get("description"):
            parts.append(str(lot.get("description")))
        if lot.get("value"):
            parts.append(f"value={lot.get('value')} {lot.get('currency') or ''}".strip())
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else "not found"


def format_tender_activity(activity: List[Dict[str, Any]]) -> str:
    if not activity:
        return "not found"
    lines = []
    for item in activity:
        parts = []
        if item.get("tender_id"):
            parts.append(str(item.get("tender_id")))
        if item.get("status"):
            parts.append(str(item.get("status")))
        if item.get("notes"):
            parts.append(str(item.get("notes")))
        line = " | ".join(parts)
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else "not found"
