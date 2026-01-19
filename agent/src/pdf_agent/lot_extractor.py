from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pdf_agent.parser import extract_pages_text

try:
    import camelot  # type: ignore
except Exception:  # noqa: BLE001
    camelot = None

PAGE_MARKER = re.compile(r"<<<PAGE:(\d+)>>>")


@dataclass
class LotRecord:
    source: str
    lot_number: str
    title: str
    description: str
    estimated_value: Optional[str]
    estimated_currency: Optional[str]
    qualification_min_value: Optional[str]
    qualification_currency: Optional[str]
    pages: List[int]


def main() -> None:
    args = _build_args().parse_args()
    input_paths = sorted(Path(args.input_dir).expanduser().resolve().glob("*.pdf"))
    if not input_paths:
        raise SystemExit(f"No PDFs found in {args.input_dir}")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[LotRecord] = []
    for pdf in input_paths:
        records.extend(extract_lots(pdf, use_tables=not args.no_tables))

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = {
                "chunk_id": _lot_chunk_id(record),
                "text": _lot_text(record),
                "type": "lot",
                **asdict(record),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(records)} lot records to {out_path}")
    if records:
        print(f"Example: {_lot_text(records[0])[:400]}")


def extract_lots(path: Path, use_tables: bool = True) -> List[LotRecord]:
    pages = extract_pages_text(path, normalize_whitespace=False)
    full_text = _merge_pages(pages)
    lot_blocks = _extract_lot_blocks(full_text)

    qualification_map = _extract_qualification_values(full_text)
    if use_tables and camelot is not None:
        qualification_map = _merge_qualification_maps(qualification_map, _extract_table_values(path))

    records: List[LotRecord] = []
    for block in lot_blocks:
        lot_number = _extract_lot_number(block) or "all"
        title = _extract_section_text(block, "II.2.1. Title")
        description = _extract_section_text(block, "II.2.4. Description of the procurement")
        est_value, est_currency = _extract_value(block, "II.2.6. Estimated value")
        pages_in_block = _pages_in_block(block)

        qual_value, qual_currency = qualification_map.get(lot_number, (None, None))

        records.append(
            LotRecord(
                source=path.name,
                lot_number=lot_number,
                title=title,
                description=description,
                estimated_value=est_value,
                estimated_currency=est_currency,
                qualification_min_value=qual_value,
                qualification_currency=qual_currency,
                pages=pages_in_block,
            )
        )

    if not records:
        pages_in_block = list(range(1, len(pages) + 1))
        records.append(
            LotRecord(
                source=path.name,
                lot_number="all",
                title="",
                description="",
                estimated_value=None,
                estimated_currency=None,
                qualification_min_value=None,
                qualification_currency=None,
                pages=pages_in_block,
            )
        )

    return records


def _merge_pages(pages) -> str:
    merged: List[str] = []
    for page in pages:
        merged.append(f"<<<PAGE:{page.page_number}>>>")
        merged.append(page.text or "")
    return "\n".join(merged)


def _extract_lot_blocks(text: str) -> List[str]:
    pattern = re.compile(
        r"II\.2\.\s*Description(?P<body>.*?)(?=II\.2\.\s*Description|Section\s+III:|Section\s+IV:|Section\s+V:|Section\s+VI:|$)",
        re.S,
    )
    return [match.group(0) for match in pattern.finditer(text)]


def _extract_lot_number(block: str) -> Optional[str]:
    patterns = [
        r"Lot\s+No\s*:?\s*(\d+)",
        r"Zadanie\s+nr\s*(\d+)",
        r"Cz[eę]ść\s*(\d+)",
    ]
    for pat in patterns:
        match = re.search(pat, block, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_section_text(block: str, header: str) -> str:
    pattern = re.compile(
        re.escape(header)
        + r"\s*(.*?)(?=\nII\.2\.|\nSection\s+III:|\nSection\s+IV:|\nSection\s+V:|\nSection\s+VI:|$)",
        re.S,
    )
    match = pattern.search(block)
    if not match:
        return ""
    return _clean_text(match.group(1))


def _extract_value(block: str, header: str) -> tuple[Optional[str], Optional[str]]:
    pattern = re.compile(re.escape(header) + r"\s*(.*?)(?=\nII\.2\.|\nSection\s+III:|$)", re.S)
    match = pattern.search(block)
    if not match:
        return None, None
    fragment = _clean_text(match.group(1))
    val_match = re.search(r"([0-9][0-9\s.,]*)\s*(PLN|EUR|USD|GBP)", fragment)
    if val_match:
        return val_match.group(1).strip(), val_match.group(2)
    return None, None


def _extract_qualification_values(text: str) -> Dict[str, tuple[str, Optional[str]]]:
    results: Dict[str, tuple[str, Optional[str]]] = {}
    pattern = re.compile(
        r"Dla\s+zadania\s+nr\s*(\d+)\s*-\s*([0-9][0-9\s.,]*)\s*(PLN|EUR|USD|GBP)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        results[match.group(1)] = (match.group(2).strip(), match.group(3))
    return results


def _extract_table_values(path: Path) -> Dict[str, tuple[str, Optional[str]]]:
    if camelot is None:
        return {}
    results: Dict[str, tuple[str, Optional[str]]] = {}
    try:
        tables = camelot.read_pdf(str(path), pages="all", flavor="stream")
    except Exception:  # noqa: BLE001
        return results

    for table in tables:
        df = table.df
        for _, row in df.iterrows():
            line = " ".join(str(x) for x in row if str(x).strip())
            match = re.search(
                r"(zadania|lot|część)\s*(nr)?\s*(\d+)\s*[-–:]?\s*([0-9][0-9\s.,]*)\s*(PLN|EUR|USD|GBP)",
                line,
                re.IGNORECASE,
            )
            if match:
                results[match.group(3)] = (match.group(4).strip(), match.group(5))
    return results


def _merge_qualification_maps(
    primary: Dict[str, tuple[str, Optional[str]]],
    secondary: Dict[str, tuple[str, Optional[str]]],
) -> Dict[str, tuple[str, Optional[str]]]:
    merged = dict(primary)
    for key, value in secondary.items():
        merged.setdefault(key, value)
    return merged


def _pages_in_block(block: str) -> List[int]:
    pages = [int(m.group(1)) for m in PAGE_MARKER.finditer(block)]
    if not pages:
        return []
    return sorted(set(pages))


def _clean_text(text: str) -> str:
    text = PAGE_MARKER.sub(" ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def _lot_chunk_id(record: LotRecord) -> str:
    base = record.source.replace(".pdf", "")
    return f"{base}_lot_{record.lot_number}"


def _lot_text(record: LotRecord) -> str:
    parts = [f"Tender: {record.source}", f"Lot: {record.lot_number}"]
    if record.title:
        parts.append(f"Title: {record.title}")
    if record.description:
        parts.append(f"Description: {record.description}")
    if record.estimated_value:
        currency = record.estimated_currency or ""
        parts.append(f"Estimated value: {record.estimated_value} {currency}".strip())
    if record.qualification_min_value:
        currency = record.qualification_currency or ""
        parts.append(
            f"Qualification requirement (min past delivery value): {record.qualification_min_value} {currency}".strip()
        )
    return "\n".join(parts)


def _build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract lot-level records from tenders (with optional table parsing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "tenders" / "raw"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "tenders" / "processed" / "lots.jsonl"),
    )
    parser.add_argument("--no-tables", action="store_true", help="Skip Camelot table extraction")
    return parser


if __name__ == "__main__":
    main()
