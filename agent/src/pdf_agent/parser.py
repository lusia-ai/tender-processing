from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""


@dataclass
class PagePreview:
    page_number: int
    char_count: int
    preview: str


@dataclass
class ParsedPDF:
    path: Path
    page_count: int
    total_chars: int
    previews: List[PagePreview]


@dataclass
class PageText:
    page_number: int
    text: str


def parse_pdf(path: Path, max_pages: Optional[int] = 5, preview_chars: int = 800) -> ParsedPDF:
    """Extract lightweight previews and counts from the PDF."""
    if not path.exists():
        raise PDFParseError(f"PDF not found: {path}")
    if max_pages is not None and max_pages <= 0:
        max_pages = None

    try:
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]

            previews: List[PagePreview] = []
            total_chars = 0

            for idx, page in enumerate(pages, start=1):
                text = page.extract_text() or ""
                cleaned = _normalize_whitespace(text)
                total_chars += len(cleaned)
                previews.append(
                    PagePreview(
                        page_number=idx,
                        char_count=len(cleaned),
                        preview=cleaned[:preview_chars],
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to parse PDF: {path}") from exc

    return ParsedPDF(
        path=path,
        page_count=page_count,
        total_chars=total_chars,
        previews=previews,
    )


def extract_pages_text(path: Path, normalize_whitespace: bool = True) -> List[PageText]:
    """Return full text for every page (optionally normalized)."""
    if not path.exists():
        raise PDFParseError(f"PDF not found: {path}")
    pages: List[PageText] = []
    try:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                text = _normalize_whitespace(raw) if normalize_whitespace else raw
                pages.append(PageText(page_number=idx, text=text))
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to parse PDF: {path}") from exc
    return pages


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace without losing sentence boundaries."""
    return " ".join(text.split())
