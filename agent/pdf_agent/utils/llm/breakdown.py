from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pdf_agent.prompts.loader import render_prompt


def _tender_breakdown_section_titles(language: str) -> List[str]:
    return [
        "Buyer & procedure",
        "Scope (what is being procured)",
        "Lots & deliverables",
        "Timeline & deadlines",
        "Award criteria & auction",
        "Qualification requirements",
        "Securities & payment terms",
        "Submission details",
        "Contacts",
        "Not covered in the notice (SIWZ)",
    ]


def _tender_breakdown_schema() -> str:
    example = {
        "report_type": "tender_breakdown",
        "title": "Tender breakdown title",
        "sections": {
            "Buyer & procedure": "Short bullets or sentences.",
            "Scope (what is being procured)": "Short bullets or sentences.",
        },
        "assistant_followup": "Ask how you can help next.",
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _tender_report_schema() -> str:
    example = {
        "report_type": "tender_report",
        "scope": "Short scope summary.",
        "key_requirements": [
            {"item": "Requirement", "source": "file#page"}
        ],
        "timeline": [
            {"item": "Event", "date": "YYYY-MM-DD", "source": "file#page"}
        ],
        "evaluation_criteria": [
            {"criterion": "Price", "weight": "100%", "source": "file#page"}
        ],
        "payment_terms": [
            {"term": "Payment term", "source": "file#page"}
        ],
        "risks": [
            {"risk": "Risk", "source": "file#page"}
        ],
        "sources": ["file#page"],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def _ensure_json_report(raw: str, llm: ChatOpenAI, schema: str) -> Tuple[Optional[Any], str]:
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*", "", stripped).strip()
            stripped = stripped.rstrip("`").rstrip()
        return stripped
    cleaned = _strip_code_fence(raw)
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass
    prompt = [
        SystemMessage(content=render_prompt("json_fix.txt", schema=schema)),
        HumanMessage(content=raw),
    ]
    response = llm.invoke(prompt)
    fixed = _strip_code_fence(response.content.strip())
    try:
        return json.loads(fixed), fixed
    except json.JSONDecodeError:
        return None, fixed


def _sections_from_mapping(mapping: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for title, content in mapping.items():
        bullets: List[str] = []
        if isinstance(content, list):
            bullets = [str(item).strip() for item in content if str(item).strip()]
        elif isinstance(content, str):
            bullets = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            bullets = [str(content).strip()] if content is not None else []
        sections.append({"title": title, "bullets": bullets, "sources": []})
    return _normalize_sections(sections, language)


def _normalize_section_title(title: str) -> str:
    if not title:
        return ""
    cleaned = re.sub(r"^[\d\s\)\.\-]+", "", str(title)).strip()
    cleaned = cleaned.replace("**", "").replace("__", "").strip()
    return cleaned


def _normalize_sections(sections: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for sec in sections or []:
        if not isinstance(sec, dict):
            continue
        title = _normalize_section_title(str(sec.get("title", "")))
        bullets: List[str] = []
        raw_bullets = sec.get("bullets")
        if isinstance(raw_bullets, list):
            bullets = [str(item).strip() for item in raw_bullets if str(item).strip()]
        elif isinstance(sec.get("content"), str):
            bullets = [line.strip() for line in str(sec.get("content")).splitlines() if line.strip()]
        elif isinstance(sec.get("text"), str):
            bullets = [line.strip() for line in str(sec.get("text")).splitlines() if line.strip()]
        cleaned: List[str] = []
        for item in bullets:
            item = re.sub(r"^[\-•*]+\s*", "", item).strip()
            item = item.replace("**", "").replace("__", "").strip()
            if not item:
                continue
            if re.fullmatch(r"\d+", item):
                continue
            cleaned.append(item)
        sources = sec.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        normalized.append({"title": title, "bullets": cleaned, "sources": sources})
    return normalized


def _split_numbered_bullets_into_sections(sections: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for sec in sections:
        bullets = sec.get("bullets", []) if isinstance(sec, dict) else []
        if not bullets:
            output.append(sec)
            continue
        current = None
        for bullet in bullets:
            match = re.match(r"^\s*\d+[\)\.\-]\s+(.*)", bullet)
            if match:
                if current:
                    output.append(current)
                title = _normalize_section_title(match.group(1).strip())
                current = {"title": title, "bullets": [], "sources": sec.get("sources", [])}
                continue
            if current is None:
                current = {"title": sec.get("title", ""), "bullets": [], "sources": sec.get("sources", [])}
            current["bullets"].append(bullet)
        if current:
            output.append(current)
    return _normalize_sections(output, language)


def _sectionize_breakdown_text(text: str, language: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not text:
        title = "Tender breakdown"
        return title, []
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    title = ""
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines:
        if not title and re.search(r"tender breakdown", line, re.IGNORECASE):
            title = _normalize_section_title(line)
            continue
        heading_match = re.match(r"^\d+[\)\.\-]\s+(.*)", line)
        if heading_match:
            heading = _normalize_section_title(heading_match.group(1))
            if current:
                sections.append(current)
            current = {"title": heading, "bullets": [], "sources": []}
            continue
        if line.endswith(":") and len(line.split()) < 10:
            heading = _normalize_section_title(line.rstrip(":"))
            if current:
                sections.append(current)
            current = {"title": heading, "bullets": [], "sources": []}
            continue
        if current is None:
            summary_title = "Summary"
            current = {"title": summary_title, "bullets": [], "sources": []}
        bullet = re.sub(r"^[\-•*]+\s*", "", line).strip()
        if bullet:
            current["bullets"].append(bullet)
    if current:
        sections.append(current)
    return title or "Tender breakdown", _normalize_sections(sections, language)


def _render_tender_breakdown(report: Dict[str, Any], language: str) -> str:
    if not isinstance(report, dict):
        return ""
    title = report.get("title") if isinstance(report.get("title"), str) else ""
    sections = report.get("sections", [])
    if isinstance(sections, dict):
        sections = _sections_from_mapping(sections, language)
    if not isinstance(sections, list):
        sections = []
    lines: List[str] = []
    if title:
        lines.append(title)
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        sec_title = sec.get("title") or ""
        if sec_title:
            lines.append(sec_title)
        for bullet in sec.get("bullets", []) or []:
            lines.append(f"- {bullet}")
    followup = report.get("assistant_followup") if isinstance(report.get("assistant_followup"), str) else ""
    if followup:
        lines.append(followup)
    return "\n".join(lines).strip()


def _restructure_breakdown_text(text: str, language: str, llm: ChatOpenAI) -> Dict[str, Any]:
    if not text:
        return {}
    schema = _tender_breakdown_schema()
    prompt = [
        SystemMessage(content=render_prompt("tender_breakdown_fix.txt", schema=schema)),
        HumanMessage(content=text),
    ]
    response = llm.invoke(prompt)
    raw = response.content.strip()
    parsed, _ = _ensure_json_report(raw, llm, schema)
    return parsed if isinstance(parsed, dict) else {}
