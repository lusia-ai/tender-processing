from __future__ import annotations

import re
from numbers import Number
from typing import Any, Dict, List, Optional

from pdf_agent.db.company_db import get_company_deliveries
from pdf_agent.utils.retrieval.tenders import normalize_tender_id


def parse_amount(value: str) -> Optional[int]:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def extract_experience_threshold(tender_context: str) -> Optional[int]:
    if not tender_context:
        return None
    lines = tender_context.splitlines()
    candidates: list[int] = []
    for line in lines:
        lowered = line.lower()
        if "experience" in lowered or "deliver" in lowered:
            match = re.search(r"([0-9][0-9\s.,]{5,})\s*pln", line, re.IGNORECASE)
            if match:
                amount = parse_amount(match.group(1))
                if amount:
                    candidates.append(amount)
    if candidates:
        return max(candidates)
    match = re.search(r"([0-9][0-9\s.,]{5,})\s*pln", tender_context, re.IGNORECASE)
    if match:
        return parse_amount(match.group(1))
    return None


def delivery_total_in_years(deliveries: List[Dict[str, Any]], start_year: int, end_year: int) -> int:
    total = 0
    for item in deliveries:
        delivered_at = item.get("delivered_at")
        year = None
        if delivered_at is None:
            continue
        if hasattr(delivered_at, "year"):
            year = delivered_at.year
        else:
            try:
                year = int(str(delivered_at)[:4])
            except ValueError:
                year = None
        if year is None:
            continue
        if year < start_year or year > end_year:
            continue
        value = item.get("value")
        if isinstance(value, Number):
            total += int(value)
            continue
        if value is None:
            continue
        parsed = parse_amount(str(value))
        if parsed:
            total += parsed
    return total


def sum_delivery_values(deliveries: List[Dict[str, Any]]) -> int:
    total = 0
    for item in deliveries:
        value = item.get("value")
        if isinstance(value, Number):
            total += int(value)
            continue
        if value is None:
            continue
        parsed = parse_amount(str(value))
        if parsed:
            total += parsed
    return total


def extract_delivery_keywords(tender_context: str, query: str) -> List[str]:
    text = f"{tender_context}\n{query}".lower()
    keywords = [
        "op230",
        "superheater",
        "steam",
        "pressure parts",
        "przegrzew",
        "reduktor",
        "przeklad",
        "gearbox",
        "motoreduktor",
        "boiler",
        "kotl",
    ]
    found = []
    for key in keywords:
        if key in text:
            found.append(key)
    tender_id = None
    match = re.search(r"ted_\d{1,}-\d{4}_[A-Za-z]{2,}", text)
    if match:
        tender_id = match.group(0)
        found.append(tender_id)
    return list(dict.fromkeys(found))


def dedupe_deliveries(deliveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for item in deliveries:
        key = (
            item.get("title"),
            item.get("customer"),
            item.get("delivered_at"),
            item.get("value"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def collect_delivery_matches(tender_context: str, user_query: str) -> List[Dict[str, Any]]:
    deliveries: List[Dict[str, Any]] = []
    tender_id = None
    match = re.search(r"ted_\d{1,}-\d{4}_[A-Za-z]{2,}", f"{tender_context} {user_query}")
    if match:
        tender_id = match.group(0)
    if tender_id:
        deliveries.extend(get_company_deliveries(tender_id=normalize_tender_id(tender_id), limit=10))
    for keyword in extract_delivery_keywords(tender_context, user_query):
        deliveries.extend(get_company_deliveries(keyword=keyword, limit=6))
    return dedupe_deliveries(deliveries)


def format_delivery_matches(deliveries: List[Dict[str, Any]]) -> str:
    if not deliveries:
        return "No matching deliveries found."
    lines: List[str] = []
    for item in deliveries:
        parts: List[str] = []
        if item.get("delivered_at"):
            parts.append(str(item.get("delivered_at")))
        if item.get("title"):
            parts.append(str(item.get("title")))
        if item.get("customer"):
            parts.append(str(item.get("customer")))
        if item.get("value"):
            currency = item.get("currency") or ""
            parts.append(f"{item.get('value')} {currency}".strip())
        if parts:
            lines.append(" - ".join(parts))
    return "\n".join(lines)
